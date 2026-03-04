"""Inverse Splat: Lift-Splat from nuScenes cameras to Physical AI camera planes.

Instead of projecting to BEV (Bird's Eye View) as in standard LSS, this module
projects nuScenes camera features to Physical AI target camera planes.

Pipeline:
  1. DepthNet: Predict depth distribution (D bins) + context features (C channels)
  2. Lift: Create frustum, outer product depth_probs x context -> 3D features
  3. Splat: Project 3D points to target camera planes using f-theta projection
  4. Pillar pooling: Aggregate features at same 2D location

Reference: /mnt/mydisk/OmniDrive/mmdetection3d/projects/BEVFusion/bevfusion/depth_lss.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..camera_models import FThetaCameraConfig, PinholeCameraConfig


class InverseSplat(nn.Module):
    """Lift-Splat module projecting to target camera planes.

    Args:
        in_channels: Input feature channels from backbone.
        context_channels: Context feature channels (C).
        depth_bins: Number of depth bins (D).
        image_size: Original image size (H, W).
        feature_size: Feature map size (fH, fW).
        target_feature_size: Target camera plane feature size (tH, tW).
        dbound: Depth range (min, max, step).
    """

    def __init__(
        self,
        in_channels: int = 256,
        context_channels: int = 360,
        depth_bins: int = 64,
        image_size: Tuple[int, int] = (320, 576),
        feature_size: Tuple[int, int] = (20, 36),
        target_feature_size: Tuple[int, int] = (20, 36),
        dbound: Tuple[float, float, float] = (1.0, 60.0, 0.921875),
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.depth_bins = depth_bins
        self.image_size = image_size
        self.feature_size = feature_size
        self.target_feature_size = target_feature_size
        self.dbound = dbound

        # DepthNet: predict depth distribution + context features
        self.depthnet = nn.Conv2d(in_channels, depth_bins + context_channels, 1)

        # Create frustum points
        self.register_buffer("frustum", self._create_frustum())

    def _create_frustum(self) -> torch.Tensor:
        """Create frustum grid of 3D sample points.

        Returns:
            Frustum tensor of shape (D, fH, fW, 3) with (u, v, depth) coordinates.
        """
        iH, iW = self.image_size
        fH, fW = self.feature_size
        d_min, d_max, d_step = self.dbound

        ds = torch.arange(d_min, d_max, d_step, dtype=torch.float32)
        D = ds.shape[0]
        ds = ds.view(D, 1, 1).expand(D, fH, fW)

        # Map feature coordinates back to image coordinates
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float32).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float32).view(1, fH, 1).expand(D, fH, fW)

        frustum = torch.stack([xs, ys, ds], dim=-1)  # (D, fH, fW, 3)
        return frustum

    def _lift(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lift 2D features to 3D using depth distribution.

        Args:
            features: Backbone features, shape (B, N, C_in, fH, fW).
            intrinsics: Camera K matrices, shape (B, N, 3, 3).
            extrinsics: Camera-to-ego transforms, shape (B, N, 4, 4).

        Returns:
            points_3d: 3D point coordinates in ego frame, shape (B, N, D, fH, fW, 3).
            features_3d: 3D features weighted by depth, shape (B, N, D, fH, fW, C).
        """
        B, N, C_in, fH, fW = features.shape
        D = self.frustum.shape[0]

        # Predict depth + context
        x = features.reshape(B * N, C_in, fH, fW)
        x = self.depthnet(x)  # (B*N, D+C, fH, fW)

        depth_logits = x[:, :self.depth_bins]  # (B*N, D, fH, fW)
        depth_probs = depth_logits.softmax(dim=1)

        context = x[:, self.depth_bins:]  # (B*N, C, fH, fW)

        # Outer product: depth_probs * context -> 3D features
        # depth: (B*N, D, fH, fW) -> (B*N, D, 1, fH, fW)
        # context: (B*N, C, fH, fW) -> (B*N, 1, C, fH, fW)
        features_3d = depth_probs.unsqueeze(2) * context.unsqueeze(1)
        # (B*N, D, C, fH, fW) -> (B, N, D, fH, fW, C)
        features_3d = features_3d.view(B, N, D, self.context_channels, fH, fW)
        features_3d = features_3d.permute(0, 1, 2, 4, 5, 3)

        # Compute 3D coordinates in ego frame
        frustum = self.frustum.to(features.device)  # (D, fH, fW, 3)

        u = frustum[..., 0:1]  # (D, fH, fW, 1)
        v = frustum[..., 1:2]
        d = frustum[..., 2:3]
        pixel_homo = torch.cat([u * d, v * d, d], dim=-1)  # (D, fH, fW, 3)

        K_inv = torch.linalg.inv(intrinsics.float())  # (B, N, 3, 3)
        R = extrinsics[:, :, :3, :3].float()  # (B, N, 3, 3)
        t = extrinsics[:, :, :3, 3].float()  # (B, N, 3)

        # Process per source camera to save memory
        ego_list = []
        ph = pixel_homo.unsqueeze(0)  # (1, D, fH, fW, 3)
        for n in range(N):
            cam_pts_n = torch.einsum(
                "bij,dHWj->bidHW",
                K_inv[:, n],  # (B, 3, 3)
                pixel_homo,   # (D, fH, fW, 3)
            )  # (B, 3, D, fH, fW)
            cam_pts_n = cam_pts_n.permute(0, 2, 3, 4, 1)  # (B, D, fH, fW, 3)

            ego_n = torch.einsum(
                "bij,bdHWj->bdHWi",
                R[:, n],      # (B, 3, 3)
                cam_pts_n,    # (B, D, fH, fW, 3)
            )  # (B, D, fH, fW, 3)
            ego_n = ego_n + t[:, n, None, None, None, :]
            ego_list.append(ego_n)

        ego_points = torch.stack(ego_list, dim=1)  # (B, N, D, fH, fW, 3)

        return ego_points, features_3d

    def _splat_to_target_planes(
        self,
        ego_points: torch.Tensor,
        features_3d: torch.Tensor,
        target_cameras: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Splat 3D features to target camera planes.

        Projects ego-frame 3D points to each target camera's image plane
        using its intrinsics/extrinsics and accumulates features via pillar pooling.

        Memory-efficient: processes each source camera independently to avoid
        flattening all N source cameras at once (5x peak memory reduction).

        Args:
            ego_points: 3D points in ego frame, shape (B, N, D, fH, fW, 3).
            features_3d: 3D features, shape (B, N, D, fH, fW, C).
            target_cameras: Dict mapping camera names to (K_target, E_target) tuples
                where K is (B, 3, 3) and E is (B, 4, 4) ego-to-camera transform.

        Returns:
            Dict mapping camera names to feature maps (B, C, tH, tW).
        """
        B, N, D, fH, fW, C = features_3d.shape
        tH, tW = self.target_feature_size
        device = features_3d.device
        scale_w = tW / self.image_size[1]
        scale_h = tH / self.image_size[0]

        result = {}
        for cam_name, (K_tgt, E_tgt) in target_cameras.items():
            R_tgt = E_tgt[:, :3, :3].float()  # (B, 3, 3)
            t_tgt = E_tgt[:, :3, 3].float()  # (B, 3)

            fx = K_tgt[:, 0, 0].unsqueeze(1).unsqueeze(2)
            fy = K_tgt[:, 1, 1].unsqueeze(1).unsqueeze(2)
            cx = K_tgt[:, 0, 2].unsqueeze(1).unsqueeze(2)
            cy = K_tgt[:, 1, 2].unsqueeze(1).unsqueeze(2)

            target_map = torch.zeros(B, C, tH, tW, device=device, dtype=features_3d.dtype)

            # Process each source camera separately to save memory
            for n in range(N):
                # (B, D*fH*fW, 3) and (B, D*fH*fW, C) — views, no copy
                pts_n = ego_points[:, n].reshape(B, -1, 3)
                feat_n = features_3d[:, n].contiguous().reshape(B, -1, C)

                cam_pts = torch.einsum("bij,bmj->bmi", R_tgt, pts_n) + t_tgt.unsqueeze(1)

                z = cam_pts[:, :, 2:3].clamp(min=0.1)
                u = fx * cam_pts[:, :, 0:1] / z + cx
                v = fy * cam_pts[:, :, 1:2] / z + cy

                u_feat = (u * scale_w).long().squeeze(-1)  # (B, M)
                v_feat = (v * scale_h).long().squeeze(-1)

                valid = (
                    (cam_pts[:, :, 2] > 0.1) &
                    (u_feat >= 0) & (u_feat < tW) &
                    (v_feat >= 0) & (v_feat < tH)
                )  # (B, M)

                for b in range(B):
                    valid_mask = valid[b]
                    if valid_mask.sum() == 0:
                        continue
                    uf = u_feat[b][valid_mask]
                    vf = v_feat[b][valid_mask]
                    feats = feat_n[b][valid_mask]  # (K, C)

                    flat_idx = vf * tW + uf  # (K,)
                    flat_idx = flat_idx.unsqueeze(1).expand(-1, C)  # (K, C)

                    target_flat = target_map[b].reshape(C, -1).permute(1, 0)  # (tH*tW, C)
                    target_flat.scatter_add_(0, flat_idx, feats)
                    target_map[b] = target_flat.permute(1, 0).reshape(C, tH, tW)

            result[cam_name] = target_map

        return result

    def forward(
        self,
        features: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_extrinsics: torch.Tensor,
        target_cameras: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Lift nuScenes camera features and splat to target camera planes.

        Args:
            features: Backbone features, shape (B, N, C_in, fH, fW).
            source_intrinsics: NuScenes camera K matrices, shape (B, N, 3, 3).
            source_extrinsics: NuScenes camera-to-ego transforms, shape (B, N, 4, 4).
            target_cameras: Dict mapping Physical AI camera names to
                (K_target, E_target) tuples for projection.

        Returns:
            Dict mapping camera names to feature maps (B, context_channels, tH, tW).
        """
        # Lift to 3D
        ego_points, features_3d = self._lift(features, source_intrinsics, source_extrinsics)

        # Splat to target planes
        return self._splat_to_target_planes(ego_points, features_3d, target_cameras)
