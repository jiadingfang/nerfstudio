"""
Script for fusing NeRFs into one.
"""

# pylint: disable=no-member

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import tyro
from rich.console import Console

from nerfstudio.fuser.registration import Registration

CONSOLE = Console(width=120)


@dataclass
class Fuser:
    """
    Load arbitrary number of NeRFs, fuse (i.e. register and blend) them and output the fused NeRF.
    """
    
    model_names: List[str]
    """names of models to blend"""
    model_dirs: List[Path]
    """model checkpoints directories"""
    cam_info: Union[List[Path], List[np.ndarray]]
    """either cam params (fx fy cx cy w h) or path to json"""
    name: Optional[str] = None
    """if present, will continue with the existing named experiment"""
    model_method: str = "nerfacto"
    """Model method used for the NeRFs."""
    model_gt_trans: Optional[str] = None
    """path to npy containing ground-truth transforms from the common world coordinate system to each model\'s local one; can be 'identity'"""
    step: Optional[int] = None
    """model step to load"""
    downscale_factor: Optional[float] = None
    """downsample factor for NeRF rendering"""
    chunk_size: Optional[int] = None
    """number of rays to render at a time"""
    training_poses: Optional[List[Path]] = None
    """paths to training poses defined in models\' local coordinate systems; if present, will be used to render training views and to determine the number of hemispheric poses"""
    n_hemi_poses: int = 32
    """number of hemispheric poses; only applicable when training-poses is not present"""
    render_hemi_views: bool = False
    """use 1.3x hemispheric poses for rendering"""
    fps: Optional[int] = None
    """frames per second for rendering"""
    sfm_tool: str = "hloc"
    """structure-from-motion tool to use for registration, choices are ['hloc', 'colmap']"""
    sfm_wo_training_views: bool = False
    """only applicable when render-hemi-views"""
    sfm_w_hemi_views: float = 1.0
    """ratio of #hemi-views vs. #training-views or n-hemi-poses, within range [0, 1.3]"""
    output_dir: Path = Path("outputs/registration")
    """directory to save outputs"""
    render_views: bool = False
    """whether to render views"""
    run_sfm: bool = False
    """whether to run SfM"""
    compute_trans: bool = False
    """whether to compute transforms"""
    vis: bool = False
    """whether to visualize"""

    def main(self) -> None:
        """Main method"""
        self.register = Registration(
            name=self.name,
            model_method=self.model_method,
            model_names=self.model_names,
            model_gt_trans=self.model_gt_trans,
            model_dirs=self.model_dirs,
            step=self.step,
            cam_info=self.cam_info,
            downscale_factor=self.downscale_factor,
            chunk_size=self.chunk_size,
            training_poses=self.training_poses,
            n_hemi_poses=self.n_hemi_poses,
            render_hemi_views=self.render_hemi_views,
            fps=self.fps,
            sfm_tool=self.sfm_tool,
            sfm_wo_training_views=self.sfm_wo_training_views,
            sfm_w_hemi_views=self.sfm_w_hemi_views,
            output_dir=self.output_dir,
            render_views=self.render_views,
            run_sfm=self.run_sfm,
            compute_trans=self.compute_trans,
            vis=self.vis,
        )
        self.register.run()


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Fuser).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Fuser)  # noqa
