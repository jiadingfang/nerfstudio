from dataclasses import dataclass, field
from typing import Type, Union

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.renderers import RGBRenderer

# from nerfstudio.models.bayes_nerf import BayesNeRFModel, BayesNeRFModelConfig
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

# from nerfstudio.models.nerfacto_wo_app import NerfactoWoAppModel, NerfactoWoAppModelConfig
from torchtyping import TensorType
from typing_extensions import Literal


@dataclass
class MyNerfactoModelConfig(NerfactoModelConfig):
    """My Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: MyNerfactoModel)


class MyNerfactoModel(NerfactoModel):
    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)[0]
        field_outputs = self.field(ray_samples)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = field_outputs[FieldHeadNames.RGB]
        img = self.renderer_rgb(rgb=rgb, weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        rgb = torch.cat((torch.empty_like(rgb[..., [0], :]), rgb), dim=-2)
        weights = torch.cat((torch.zeros_like(weights[..., [0], :]), weights), dim=-2)
        deltas = torch.cat((ray_samples.frustums.starts[..., [0], :], ray_samples.deltas), dim=-2)

        outputs = {
            "rgb": rgb,
            "rgb_img": img,
            "accumulation": accumulation,
            "depth": depth,
            "weights": weights,
            "deltas": deltas,
            "directions": ray_samples.frustums.directions[:, 0],
        }
        return outputs


# @dataclass
# class MyNerfactoWoAppModelConfig(NerfactoWoAppModelConfig):
#     """My NerfactoMinus Model Config"""

#     _target: Type = field(default_factory=lambda: MyNerfactoWoAppModel)


# class MyNerfactoWoAppModel(NerfactoWoAppModel):
#     def get_outputs(self, ray_bundle: RayBundle):
#         ray_samples = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)[0]
#         field_outputs = self.field(ray_samples)
#         weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

#         rgb = field_outputs[FieldHeadNames.RGB]
#         img = self.renderer_rgb(rgb=rgb, weights=weights)
#         depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
#         accumulation = self.renderer_accumulation(weights=weights)

#         rgb = torch.cat((torch.empty_like(rgb[..., [0], :]), rgb), dim=-2)
#         weights = torch.cat((torch.zeros_like(weights[..., [0], :]), weights), dim=-2)
#         deltas = torch.cat((ray_samples.frustums.starts[..., [0], :], ray_samples.deltas), dim=-2)

#         outputs = {
#             'rgb': rgb,
#             'rgb_img': img,
#             'accumulation': accumulation,
#             'depth': depth,
#             'weights': weights,
#             'deltas': deltas,
#             'directions': ray_samples.frustums.directions[:, 0]
#         }
#         return outputs


# @dataclass
# class MyBayesNeRFModelConfig(BayesNeRFModelConfig):
#     """My BayesNeRF Model Config"""

#     _target: Type = field(default_factory=lambda: MyBayesNeRFModel)


# class MyBayesNeRFModel(BayesNeRFModel):
#     def get_outputs(self, ray_bundle: RayBundle):
#         ray_samples = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)[0]
#         field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
#         weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

#         rgb = field_outputs[FieldHeadNames.RGB]
#         rgb_img = self.renderer_rgb(rgb=rgb, weights=weights)
#         depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
#         accumulation = self.renderer_accumulation(weights=weights)
#         uncertainty = field_outputs[FieldHeadNames.UNCERTAINTY]
#         uncertainty_img = self.renderer_uncertainty(uncertainty, weights)

#         rgb = torch.cat((torch.empty_like(rgb[..., [0], :]), rgb), dim=-2)
#         uncertainty = torch.cat((torch.empty_like(uncertainty[..., [0], :]), uncertainty), dim=-2)
#         weights = torch.cat((torch.zeros_like(weights[..., [0], :]), weights), dim=-2)
#         deltas = torch.cat((ray_samples.frustums.starts[..., [0], :], ray_samples.deltas), dim=-2)

#         outputs = {
#             'rgb': rgb,
#             "rgb_img": rgb_img,
#             "accumulation": accumulation,
#             "depth": depth,
#             "uncertainty": uncertainty,
#             "uncertainty_img": uncertainty_img,
#             'weights': weights,
#             'deltas': deltas,
#             'directions': ray_samples.frustums.directions[:, 0]
#         }
#         return outputs


class MyRGBRenderer(RGBRenderer):
    """Weighted volumetic rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    @classmethod
    def combine_rgb(
        cls,
        rgb: TensorType["bs":..., "num_samples", 3],
        ws: TensorType["bs":..., "num_samples", 1],
        bg_w: TensorType["bs":..., 1],
        background_color: Union[Literal["random", "last_sample"], TensorType[3]] = "random",
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            weights: Termination probability mass for each sample.
            ws: Weights for each sample. E.g. from IDW.
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        comp_rgb = torch.sum(rgb * ws, dim=-2)

        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        elif background_color == "random":
            background_color = torch.rand_like(comp_rgb)

        comp_rgb += background_color * bg_w

        return comp_rgb

    def forward(
        self,
        rgb: TensorType["bs":..., "num_samples", 3],
        ws: TensorType["bs":..., "num_samples", 1],
        bg_w: TensorType["bs":..., 1],
    ) -> TensorType["bs":..., 3]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample.
            ws: weighted termination probability mass for each sample.

        Returns:
            Outputs of rgb values.
        """

        rgb = self.combine_rgb(rgb, ws, bg_w, background_color=self.background_color)
        return rgb
