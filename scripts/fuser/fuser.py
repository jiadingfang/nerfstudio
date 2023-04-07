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
import open3d as o3d
import torch
import tyro
from rich.console import Console

from nerfstudio.fuser.registration import Registration


CONSOLE = Console(width=120)


@dataclass
class Fuser:
    """
    Load two NeRFs, fuse (i.e. register and blend) them and output the fused NeRF.
    """

    model_A_dir: Path
    """Path to the first NeRF."""
    model_B_dir: Path
    """Path to the second NeRF."""
    cam_info: Path
    """Path to the camera info file."""
    c2ws_A: Optional[Path] = None
    "path to A's transforms.json; if not present, will use hemispheric poses"
    c2ws_B: Optional[Path] = None
    "path to B's transforms.json; if not present, will use hemispheric poses"
    exp_name: Optional[str] = None
    """Name of the experiment."""
    model_method: str = "nerfacto"
    """Model method used for the NeRFs."""
    model_step: Optional[int] = None
    """Model step to load."""
    downsample: float = 1.0
    """Downsample factor for NeRF rendering."""
    n_hemi_views: int = 32
    """Number of hemispheric views to use."""
    render_hemi_views: bool = False
    """Number of hemispheric views to render."""
    fps: int = 0
    """Frames per second for rendering."""
    sfm_tool: str = "hloc"
    """SfM tool used for the SfM."""
    sfm_wo_training_views: bool = False
    """Whether to include training views in SfM."""
    sfm_w_hemi_views: float = 1.0
    """ratio of hemispheric views compared to training views to use in SfM."""
    output_dir: Path = Path("outputs/registration")
    """Output directory."""
    render_views: bool = False
    """Number of views to render."""
    run_sfm: bool = False
    """Whether to run SfM."""
    compute_trans: bool = False
    """Whether to compute the transformation error statistics."""
    vis: bool = False
    """Whether to visualize the results."""

    def main(self) -> None:
        """Main method"""
        self.register = Registration(
            model_method=self.model_method,
            model_A_dir=self.model_A_dir,
            model_B_dir=self.model_B_dir,
            cam_info=self.cam_info,
            c2ws_A=self.c2ws_A,
            c2ws_B=self.c2ws_B,
            exp_name=self.exp_name,
            downsample=self.downsample,
            model_step=self.model_step,
            n_hemi_views=self.n_hemi_views,
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
