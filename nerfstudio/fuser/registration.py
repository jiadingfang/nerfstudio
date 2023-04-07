import argparse
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from nerfstudio.process_data.colmap_utils import CameraModel, read_images_binary, run_colmap
from nerfstudio.process_data.hloc_utils import run_hloc

from nerfstudio.utils.fuser_utils.utils import (
    complete_transformation,
    compute_trans_diff,
    extract_colmap_pose,
    gen_hemispherical_poses,
)
from nerfstudio.utils.fuser_utils.visualizer import Visualizer
from nerfstudio.utils.fuser_utils.view_renderer import ViewRenderer


class Registration:
    def __init__(
        self,
        model_method,
        model_A_dir,
        model_B_dir,
        cam_info,
        c2ws_A,
        c2ws_B,
        exp_name=None,
        downsample=1.0,
        model_step=None,
        n_hemi_views=32,
        render_hemi_views=None,
        fps=0,
        sfm_tool="hloc",
        sfm_wo_training_views=None,
        sfm_w_hemi_views=1.0,
        output_dir="outputs/registration",
        render_views=None,
        run_sfm=None,
        compute_trans=None,
        vis=None,
    ) -> None:
        self.exp_name = exp_name
        self.model_method = model_method
        self.model_A_dir = model_A_dir
        self.model_B_dir = model_B_dir
        self.model_step = model_step
        self.cam_info = cam_info
        self.downsample = downsample
        self.c2ws_A = c2ws_A
        self.c2ws_B = c2ws_B
        self.n_hemi_views = n_hemi_views
        self.render_hemi_views = render_hemi_views
        self.fps = fps
        self.sfm_tool = sfm_tool
        self.sfm_wo_training_views = sfm_wo_training_views
        self.sfm_w_hemi_views = sfm_w_hemi_views
        self.output_dir = output_dir
        self.render_views = render_views
        self.run_sfm = run_sfm
        self.compute_trans = compute_trans
        self.vis = vis

    def run(self):
        print("Registration")

        if self.exp_name:
            name = self.exp_name
        else:
            name = datetime.now().strftime("%m.%d_%H:%M:%S")

        output_dir = Path(self.output_dir) / name
        os.makedirs(output_dir, exist_ok=True)

        if self.run_sfm or self.compute_trans:
            cfg = f"hemi{self.sfm_w_hemi_views:.2f}_train{int(not self.sfm_wo_training_views)}"
            sfm_dir = output_dir / f"sfm_{cfg}"
            log_dict = {}
            for k in [
                "model_method",
                "model_A_dir",
                "model_B_dir",
                "model_step",
                "cam_info",
                "c2ws_A",
                "c2ws_B",
                "n_hemi_views",
                "render_hemi_views",
                "sfm_tool",
            ]:
                attr = getattr(self, k)
                if isinstance(attr, Path):
                    log_dict[k] = str(attr)
                else:
                    log_dict[k] = attr

            # print("log_dict: ", log_dict)
            # print("model_A_dir type: ", type(log_dict["model_A_dir"]))

            with open(output_dir / f"{cfg}.json", "w") as f:
                json.dump(log_dict, f, indent=2)

            model_A_path = Path(self.model_A_dir)
            model_B_path = Path(self.model_B_dir)
            with open(model_A_path.parent / "dataparser_transforms.json") as f:
                transforms = json.load(f)
            s = transforms["scale"]
            S_CA_gt = np.diag((s, s, s, 1)).astype(np.float32)
            T_CA_gt = S_CA_gt @ complete_transformation(np.array(transforms["transform"], dtype=np.float32))
            with open(model_B_path.parent / "dataparser_transforms.json") as f:
                transforms = json.load(f)
            s = transforms["scale"]
            S_CB_gt = np.diag((s, s, s, 1)).astype(np.float32)
            T_CB_gt = S_CB_gt @ complete_transformation(np.array(transforms["transform"], dtype=np.float32))

        if self.c2ws_A:
            with open(self.c2ws_A) as f:
                transforms = json.load(f)
            frames_A = transforms["frames"]
            l_A = len(frames_A)
        else:
            l_A = self.n_hemi_views
        m = (1 + math.sqrt(1 + l_A / 3)) / 2
        m_A = math.ceil(m)
        n_A = math.ceil(12 * (m - 1))
        k_A = m_A * n_A
        if self.c2ws_B:
            with open(self.c2ws_B) as f:
                transforms = json.load(f)
            frames_B = transforms["frames"]
            l_B = len(frames_B)
        else:
            l_B = self.n_hemi_views
        m = (1 + math.sqrt(1 + l_B * 1.3 / 3)) / 2
        m_B = math.ceil(m)
        n_B = math.ceil(12 * (m - 1))
        k_B = m_B * n_B

        if self.render_views:
            c2ws = complete_transformation(np.array(gen_hemispherical_poses(1, np.pi / 6, m=m_A, n=n_A)))
            if self.c2ws_A:
                pose_dict = {
                    int(re.split(r"/|\.|_", frame["file_path"])[-2]): np.array(
                        frame["transform_matrix"], dtype=np.float32
                    )
                    for frame in frames_A
                }
                c2ws_A = T_CA_gt @ np.array([pose_dict[k] for k in sorted(pose_dict.keys())]) @ np.linalg.inv(S_CA_gt)
                if self.render_hemi_views:
                    c2ws_A = np.concatenate((c2ws, c2ws_A))
            else:
                c2ws_A = c2ws
            c2ws = complete_transformation(np.array(gen_hemispherical_poses(1, np.pi / 6, m=m_B, n=n_B)))
            if self.c2ws_B:
                pose_dict = {
                    int(re.split(r"/|\.|_", frame["file_path"])[-2]): np.array(
                        frame["transform_matrix"], dtype=np.float32
                    )
                    for frame in frames_B
                }
                c2ws_B = T_CB_gt @ np.array([pose_dict[k] for k in sorted(pose_dict.keys())]) @ np.linalg.inv(S_CB_gt)
                if self.render_hemi_views:
                    c2ws_B = np.concatenate((c2ws, c2ws_B))
            else:
                c2ws_B = c2ws

            if isinstance(self.cam_info, Path):
                with open(self.cam_info) as f:
                    transforms = json.load(f)
                cam_info = {
                    "fx": transforms["fl_x"],
                    "fy": transforms["fl_y"],
                    "cx": transforms["cx"],
                    "cy": transforms["cy"],
                    "width": transforms["w"],
                    "height": transforms["h"],
                    "distortion_params": np.array(
                        [transforms["k1"], transforms["k2"], 0, 0, transforms["p1"], transforms["p2"]]
                    ),
                }
            elif isinstance(self.cam_info, list):
                cam_info = dict(zip(["fx", "fy", "cx", "cy", "width", "height"], [float(v) for v in self.cam_info]))
            cam_info["height"] = int(cam_info["height"])
            cam_info["width"] = int(cam_info["width"])

            if self.downsample > 1.0:
                cam_info["height"] = int(cam_info["height"] / self.downsample)
                cam_info["width"] = int(cam_info["width"] / self.downsample)
                cam_info["fx"] = cam_info["fx"] / self.downsample
                cam_info["fy"] = cam_info["fy"] / self.downsample
                cam_info["cx"] = cam_info["cx"] / self.downsample
                cam_info["cy"] = cam_info["cy"] / self.downsample

            with torch.no_grad():
                vr = ViewRenderer(self.model_method, model_A_path, load_step=self.model_step)
                vr.render_views(c2ws_A, cam_info, "A", output_dir=output_dir, animate=self.fps)
                vr = ViewRenderer(self.model_method, model_B_path, load_step=self.model_step)
                vr.render_views(c2ws_B, cam_info, "B", output_dir=output_dir, animate=self.fps)

        if self.run_sfm:
            shutil.rmtree(sfm_dir, ignore_errors=True)
            os.mkdir(sfm_dir)
            for s in ["A", "B"]:
                files = os.listdir(output_dir / s)
                has_c2ws = getattr(self, f"c2ws_{s}")
                k = locals()[f"k_{s}"]
                l = locals()[f"l_{s}"]
                c1 = not has_c2ws or self.render_hemi_views
                if c1:
                    ids = set(np.round(np.linspace(0, k - 1, math.floor(self.sfm_w_hemi_views * l))))
                c2 = has_c2ws and self.render_hemi_views and not self.sfm_wo_training_views
                c3 = has_c2ws and not self.render_hemi_views
                for f in files:
                    id = int(f.split(".")[0])
                    if c1 and id in ids or c2 and id >= k or c3:
                        os.symlink((output_dir / s / f).absolute(), sfm_dir / f"{s}_{f}")
            run_func = run_colmap if self.sfm_tool == "colmap" else run_hloc
            run_func(sfm_dir, sfm_dir, CameraModel.OPENCV)

        if self.compute_trans:
            images = read_images_binary(sfm_dir / "sparse/0/images.bin")
            meta_A = torch.load(output_dir / "A_in.pt", map_location="cpu")
            meta_B = torch.load(output_dir / "B_in.pt", map_location="cpu")
            # poses of camAi_A
            poses_A = []
            # poses of camBi_B
            poses_B = []
            # poses of camCi_C
            poses_C = {}
            for im_data in images.values():
                fname = im_data.name
                id = fname[2:5]
                poses_C[fname] = extract_colmap_pose(im_data)
                if fname.startswith("A"):
                    poses_A.append((fname, meta_A[id].numpy()))
                else:
                    poses_B.append((fname, meta_B[id].numpy()))

            n = len(poses_A)
            print(f"Got {n} poses for A from SfM")
            s_lst = []
            for i in range(n - 1):
                for j in range(i + 1, n):
                    tAi_A = poses_A[i][1][:3, 3]
                    tAj_A = poses_A[j][1][:3, 3]
                    tAi_C = poses_C[poses_A[i][0]][:3, 3]
                    tAj_C = poses_C[poses_A[j][0]][:3, 3]
                    s_lst.append(np.linalg.norm(tAi_C - tAj_C) / np.linalg.norm(tAi_A - tAj_A))
            s_AC = np.median(s_lst)
            S_AC = np.diag((s_AC, s_AC, s_AC, 1)).astype(np.float32)
            T_AC_lst = [poses_C[pose[0]] @ S_AC @ np.linalg.inv(pose[1]) for pose in poses_A]
            T_AC = np.median(T_AC_lst, axis=0)
            u, _, vh = np.linalg.svd(T_AC[:3, :3])
            T_AC[:3, :3] = u * s_AC @ vh

            n = len(poses_B)
            print(f"Got {n} poses for B from SfM")
            s_lst = []
            for i in range(n - 1):
                for j in range(i + 1, n):
                    tBi_B = poses_B[i][1][:3, 3]
                    tBj_B = poses_B[j][1][:3, 3]
                    tBi_C = poses_C[poses_B[i][0]][:3, 3]
                    tBj_C = poses_C[poses_B[j][0]][:3, 3]
                    s_lst.append(np.linalg.norm(tBi_C - tBj_C) / np.linalg.norm(tBi_B - tBj_B))
            s_BC = np.median(s_lst)
            S_BC = np.diag((s_BC, s_BC, s_BC, 1)).astype(np.float32)
            T_BC_lst = [poses_C[pose[0]] @ S_BC @ np.linalg.inv(pose[1]) for pose in poses_B]
            T_BC = np.median(T_BC_lst, axis=0)
            u, _, vh = np.linalg.svd(T_BC[:3, :3])
            T_BC[:3, :3] = u * s_BC @ vh

            T_BA = np.linalg.inv(T_AC) @ T_BC
            np.save(output_dir / f"T_BA_{cfg}.npy", T_BA)
            print("pred trans\n", T_BA)

            T_BA_gt = T_CA_gt @ np.linalg.inv(T_CB_gt)
            print("gt trans\n", T_BA_gt)
            r, t, s = compute_trans_diff(T_BA_gt, T_BA)
            print(f"rotation error {r:.3g}")
            print(f"translation error {t:.3g}")
            print(f"scale error {s:.3g}")

            if self.vis:
                vis = Visualizer(show_frame=True)
                # gt
                vis.add_trajectory([T_BA_gt], pose_spec=0, cam_size=0.1, color=(0, 0.7, 0))
                # pred
                vis.add_trajectory([T_BA], pose_spec=0, cam_size=0.1, color=(0.7, 0, 0.7))
                # auxiliary
                # vis.add_trajectory(poses_C.values(), cam_size=0.05, color=(0, 0.7, 0))
                vis.add_trajectory([poses_C[pose[0]] for pose in poses_A], cam_size=0.05, color=(0.7, 0, 0))
                vis.add_trajectory(
                    T_AC @ np.array([pose[1] for pose in poses_A]) @ np.linalg.inv(S_AC),
                    cam_size=0.05,
                    color=(0.7, 0.7, 0),
                )
                vis.add_trajectory([poses_C[pose[0]] for pose in poses_B], cam_size=0.05, color=(0, 0, 0.7))
                vis.add_trajectory(
                    T_BC @ np.array([pose[1] for pose in poses_B]) @ np.linalg.inv(S_BC),
                    cam_size=0.05,
                    color=(0, 0.7, 0.7),
                )
                vis.show()
