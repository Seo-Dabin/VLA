"""Geometric transformation from Physical AI to NuScenes camera views (PyTorch).

Performs rotation-only remap from f-theta (Physical AI) source cameras to
pinhole (NuScenes) target cameras using precomputed LUTs and grid_sample.

Reference: /mnt/mydisk/alpamayo/src/alpamayo_r1/transform_physicalai_to_nuscenes.py
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_models import (
    NUSCENES_CAMERAS,
    PHYSICALAI_CAMERAS,
    FThetaCameraConfig,
    PinholeCameraConfig,
)


# Single primary source camera for each NuScenes view.
# Same mapping as the numpy reference implementation.
PRIMARY_SOURCE: Dict[str, str] = {
    "CAM_FRONT": "front_wide",
    "CAM_FRONT_LEFT": "cross_left",
    "CAM_FRONT_RIGHT": "cross_right",
    "CAM_BACK_LEFT": "cross_left",
    "CAM_BACK_RIGHT": "cross_right",
}


def build_rotation_only_remap_torch(
    src_cam: FThetaCameraConfig,
    tgt_cam: PinholeCameraConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build remap LUT from f-theta source to pinhole target (rotation only).

    For each target pixel, compute ray direction, rotate into source frame,
    and project using f-theta model.

    Args:
        src_cam: Source f-theta camera (Physical AI).
        tgt_cam: Target pinhole camera (NuScenes).

    Returns:
        grid: Normalized grid for F.grid_sample, shape (1, tgt_H, tgt_W, 2).
        valid: Valid pixel mask, shape (tgt_H, tgt_W).
    """
    tgt_h, tgt_w = tgt_cam.height, tgt_cam.width

    R_src = src_cam.rotation_matrix  # (3, 3)
    R_tgt = tgt_cam.rotation_matrix  # (3, 3)

    # Relative rotation: target cam -> source cam
    R_tgt_to_src = R_src.T @ R_tgt  # (3, 3)

    # Create target pixel grid
    v_coords, u_coords = torch.meshgrid(
        torch.arange(tgt_h, dtype=torch.float64),
        torch.arange(tgt_w, dtype=torch.float64),
        indexing="ij",
    )

    # Get rays in target camera frame
    rays_tgt = tgt_cam.pixel2ray(u_coords, v_coords)  # (H, W, 3)

    # Transform to source camera frame (rotation only)
    rays_src = torch.einsum("ij,hwj->hwi", R_tgt_to_src, rays_tgt)

    # Project to source f-theta image
    u_src, v_src, valid = src_cam.project(rays_src)

    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * u_src.float() / (src_cam.width - 1) - 1.0
    grid_y = 2.0 * v_src.float() / (src_cam.height - 1) - 1.0

    # Set invalid pixels to out-of-bounds
    grid_x = torch.where(valid, grid_x, torch.full_like(grid_x, -2.0))
    grid_y = torch.where(valid, grid_y, torch.full_like(grid_y, -2.0))

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    return grid.float(), valid


class GeometricTransform(nn.Module):
    """Online geometric transformation from Physical AI to NuScenes camera views.

    Precomputes remap LUTs at initialization and applies them via grid_sample.
    Output images are resized to the specified target size.

    Args:
        target_size: Output image size (H, W) after transformation and resize.
    """

    # NuScenes camera order (5 cameras)
    NUSCENES_CAM_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # Physical AI cameras used as sources
    PHYSICALAI_SOURCE_CAMS = ["front_wide", "cross_left", "cross_right"]

    def __init__(self, target_size: Tuple[int, int] = (320, 576)) -> None:
        super().__init__()
        self.target_size = target_size

        # Precompute remap grids for all NuScenes cameras
        for tgt_name in self.NUSCENES_CAM_ORDER:
            src_name = PRIMARY_SOURCE[tgt_name]
            src_cam = PHYSICALAI_CAMERAS[src_name]
            tgt_cam = NUSCENES_CAMERAS[tgt_name]
            grid, valid = build_rotation_only_remap_torch(src_cam, tgt_cam)
            # Register as buffers so they move with the module to GPU
            self.register_buffer(f"grid_{tgt_name}", grid)
            self.register_buffer(f"valid_{tgt_name}", valid)

    def forward(
        self,
        physicalai_images: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Transform Physical AI images to NuScenes 5-camera views.

        Args:
            physicalai_images: Dict mapping Physical AI camera names to
                image tensors of shape (B, 3, H_src, W_src), values in [0, 1] float.

        Returns:
            NuScenes images tensor of shape (B, 5, 3, H_tgt, W_tgt).
        """
        nuscenes_images = []
        B = next(iter(physicalai_images.values())).shape[0]

        for tgt_name in self.NUSCENES_CAM_ORDER:
            src_name = PRIMARY_SOURCE[tgt_name]
            src_img = physicalai_images[src_name]  # (B, 3, H_src, W_src)

            grid = getattr(self, f"grid_{tgt_name}")  # (1, H_tgt_orig, W_tgt_orig, 2)
            grid = grid.expand(B, -1, -1, -1)  # (B, H, W, 2)

            # Apply remap via grid_sample
            warped = F.grid_sample(
                src_img,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # (B, 3, H_tgt_orig, W_tgt_orig)

            # Resize to target size
            if warped.shape[2:] != self.target_size:
                warped = F.interpolate(
                    warped,
                    size=self.target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            nuscenes_images.append(warped)

        return torch.stack(nuscenes_images, dim=1)  # (B, 5, 3, H, W)

    def get_valid_masks(self) -> Dict[str, torch.Tensor]:
        """Return validity masks for each NuScenes camera view.

        Returns:
            Dict mapping camera names to boolean masks (H, W).
        """
        masks = {}
        for tgt_name in self.NUSCENES_CAM_ORDER:
            masks[tgt_name] = getattr(self, f"valid_{tgt_name}")
        return masks
