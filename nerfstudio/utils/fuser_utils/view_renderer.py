import logging
import os
import shutil
from pathlib import Path

import imageio
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from tqdm import tqdm, trange

from my_models import MyNerfactoModelConfig

# from mu_models import MyNerfactoWoAppModelConfig
from utils import complete_transformation


class ViewRenderer:
    def __init__(self, model_method, load_dir, load_step=None, log_num_rays_per_chunk=15, device="cuda") -> None:
        self.model_method = model_method
        if load_step is None or not os.path.exists(load_path := load_dir / f"step-{load_step:09d}.ckpt"):
            # load the latest checkpoint
            load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:09d}.ckpt"
        state = torch.load(load_path, map_location=device)
        state = {key[7:]: val for key, val in state["pipeline"].items() if key.startswith("_model")}
        if model_method.endswith("nerfacto"):
            self.model = (
                MyNerfactoModelConfig(eval_num_rays_per_chunk=1 << log_num_rays_per_chunk)
                .setup(
                    scene_box=SceneBox(aabb=state["field.aabb"]),
                    num_train_data=len(state["field.embedding_appearance.embedding.weight"]),
                )
                .to(device)
            )
        # elif model_method.endswith('nerfacto-wo-app'):
        #     self.model = MyNerfactoWoAppModelConfig(eval_num_rays_per_chunk=1 << log_num_rays_per_chunk).setup(scene_box=SceneBox(aabb=state['field.aabb']), num_train_data=None).to(device)
        # elif model_method == 'bayes-nerf':
        #     self.model = MyBayesNeRFModelConfig(eval_num_rays_per_chunk=1 << log_num_rays_per_chunk).setup(scene_box=SceneBox(aabb=state['field.aabb']), num_train_data=None).to(device)
        self.model.load_state_dict(state)
        logging.info("done loading checkpoint from %s", load_path)
        self.model.eval()
        self.device = device

    def render_views(
        self,
        poses,
        cam_info,
        name,
        rel_pose=None,
        rel_scale=1,
        multi_cam=False,
        output_dir=Path("outputs"),
        save_outputs=False,
        animate=0,
    ):
        shutil.rmtree(output_dir / name, ignore_errors=True)
        os.makedirs(output_dir / name)
        poses = complete_transformation(torch.as_tensor(poses, device=self.device))
        for p in cam_info:
            if isinstance(cam_info[p], np.ndarray):
                if cam_info[p].dtype == int:
                    cam_info[p] = torch.as_tensor(cam_info[p], device=self.device)
                else:
                    cam_info[p] = torch.as_tensor(cam_info[p], dtype=torch.float32, device=self.device)
        if rel_pose is None:
            rel_pose = torch.eye(4, device=self.device)
        else:
            rel_pose = complete_transformation(torch.as_tensor(rel_pose, device=self.device))
        rel_scale = torch.diag(
            torch.tensor([rel_scale, rel_scale, rel_scale, 1], dtype=torch.float32, device=self.device)
        )
        meta = {"rel_pose": rel_pose, "rel_scale": rel_scale}
        poses_ = rel_pose @ rel_scale @ poses @ torch.linalg.inv(rel_scale)
        if not multi_cam:
            cams = Cameras(poses_[:, :3].cpu(), **cam_info).to(self.device)
        imgs = []
        print("rendering images...")
        for i in trange(len(poses)):
            id = f"{i:03d}"
            meta[id] = poses[i]
            if multi_cam:
                cam = Cameras(poses_[i, :3], **{p: cam_info[p][i] for p in cam_info}).to(self.device)
                ray_bundle = cam.generate_rays(0)
            else:
                ray_bundle = cams.generate_rays(i)
            outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            imgs.append(outputs["rgb_img"].cpu())
            if save_outputs:
                rgb = outputs["rgb"]
                rgb = rgb.reshape(rgb.shape[:-1] + (-1, 3))
                torch.save(rgb, output_dir / name / f"{id}_rgb.pt")
                if self.model_method.startswith("bayes"):
                    uncertainty = outputs["uncertainty"]
                    torch.save(uncertainty, output_dir / name / f"{id}_uncertainty.pt")
                del outputs["rgb"]
                if self.model_method.startswith("bayes"):
                    del outputs["uncertainty"]
                outputs["c2w"] = poses_[i]
                torch.save(outputs, output_dir / name / f"{id}.pt")
        torch.save(meta, output_dir / f"{name}_in.pt")
        if save_outputs:
            torch.save(imgs, output_dir / f"{name}.pt")
        imgs = [(img.numpy() * 255).astype(np.uint8) for img in imgs]
        print("saving images...")
        for i, img in enumerate(tqdm(imgs)):
            imageio.v3.imwrite(output_dir / name / f"{i:03d}.png", img)
        if animate:
            imageio.v3.imwrite(output_dir / f"{name}.mp4", imgs, fps=animate)

    def query_density(self, pts, rel_trans=None):
        """pts: points in local NeRF coordinates before scene contraction"""
        pts = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
        if pts.shape[-1] != 1:
            pts.unsqueeze_(-1)
        if pts.shape[-2] == 3:
            pts = torch.cat((pts, torch.ones((*pts.shape[:-2], 1, 1), device=self.device)), dim=1)
        if rel_trans is None:
            rel_trans = torch.eye(4, device=self.device)
        else:
            rel_trans = torch.as_tensor(rel_trans, device=self.device)
        pts = rel_trans @ pts
        return self.model.field.density_fn(pts[..., :3, 0])
