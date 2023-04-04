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


CONSOLE = Console(width=120)


@dataclass
class Fuser:
    """ Load two NeRFs, fuse (i.e. register and blend) them and output the fused NeRF.
    """

    nerf_a_path: Path
    """Path to the first NeRF."""
    nerf_b_path: Path
    """Path to the second NeRF."""

    def main(self) -> None:
        """Main method"""
        print('nerf_a_path', self.nerf_a_path)
        print('nerf_b_path', self.nerf_b_path)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Fuser).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(Commands)  # noqa
