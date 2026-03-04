"""Plucker Ray positional embedding for camera-aware feature encoding.

Computes 6D Plucker coordinates (direction + moment) for each pixel in
the feature map based on camera intrinsics and extrinsics, then projects
to the feature dimension via a small MLP.

Plucker coordinates: (d_x, d_y, d_z, m_x, m_y, m_z) where
  d = unit ray direction in world frame
  m = cross(origin, d) = moment vector
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PluckerRayPE(nn.Module):
    """Plucker Ray positional embedding module.

    Computes camera-aware positional embeddings from Plucker ray coordinates
    and projects them to the feature dimension via an MLP.

    Args:
        out_dim: Output dimension (should match backbone feature dim).
        hidden_dim: Hidden dimension of the projection MLP.
    """

    def __init__(self, out_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim

        # MLP: 6D Plucker -> hidden -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Add Plucker ray positional embedding to features.

        Args:
            features: Backbone features, shape (B*N, C, fH, fW).
            intrinsics: Camera intrinsic matrices, shape (B*N, 3, 3).
            extrinsics: Camera extrinsic matrices [R|t] (cam-to-world),
                shape (B*N, 4, 4) or (B*N, 3, 4).

        Returns:
            Features with Plucker PE added, shape (B*N, C, fH, fW).
        """
        BN, C, fH, fW = features.shape
        device = features.device
        dtype = features.dtype

        # Compute Plucker coordinates at feature map resolution
        plucker = self._compute_plucker_coords(
            intrinsics, extrinsics, fH, fW, device, dtype
        )  # (BN, fH, fW, 6)

        # Project through MLP
        pe = self.mlp(plucker)  # (BN, fH, fW, C)
        pe = pe.permute(0, 3, 1, 2)  # (BN, C, fH, fW)

        return features + pe

    def _compute_plucker_coords(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        fH: int,
        fW: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute 6D Plucker coordinates for each feature pixel.

        Args:
            intrinsics: Camera K matrices, shape (BN, 3, 3).
            extrinsics: Camera-to-world transforms, shape (BN, 4, 4) or (BN, 3, 4).
            fH: Feature height.
            fW: Feature width.
            device: Computation device.
            dtype: Computation dtype.

        Returns:
            Plucker coordinates, shape (BN, fH, fW, 6).
        """
        BN = intrinsics.shape[0]

        # Create pixel grid at feature map resolution (scaled to image coords)
        # Assume feature map is 1/16 of original image
        scale = 16.0
        v_coords, u_coords = torch.meshgrid(
            torch.arange(fH, device=device, dtype=dtype) * scale + scale / 2,
            torch.arange(fW, device=device, dtype=dtype) * scale + scale / 2,
            indexing="ij",
        )  # (fH, fW)

        # Unproject to camera frame rays
        ones = torch.ones_like(u_coords)
        pixels = torch.stack([u_coords, v_coords, ones], dim=-1)  # (fH, fW, 3)
        pixels = pixels.unsqueeze(0).expand(BN, -1, -1, -1)  # (BN, fH, fW, 3)

        # K_inv @ pixel -> ray in camera frame
        K_inv = torch.inverse(intrinsics.float())  # (BN, 3, 3)
        rays_cam = torch.einsum("bij,bhwj->bhwi", K_inv, pixels)  # (BN, fH, fW, 3)
        rays_cam = F.normalize(rays_cam, dim=-1)

        # Extract rotation and translation from extrinsics
        R = extrinsics[:, :3, :3].float()  # (BN, 3, 3) cam-to-world rotation
        t = extrinsics[:, :3, 3].float()  # (BN, 3) camera origin in world

        # Transform ray directions to world frame
        d = torch.einsum("bij,bhwj->bhwi", R, rays_cam)  # (BN, fH, fW, 3)
        d = F.normalize(d, dim=-1)

        # Camera origin in world frame
        o = t.unsqueeze(1).unsqueeze(1).expand(-1, fH, fW, -1)  # (BN, fH, fW, 3)

        # Moment vector: m = o x d
        m = torch.cross(o, d, dim=-1)  # (BN, fH, fW, 3)

        # Concatenate: (d, m) -> 6D Plucker coordinates
        plucker = torch.cat([d, m], dim=-1).to(dtype)  # (BN, fH, fW, 6)

        return plucker
