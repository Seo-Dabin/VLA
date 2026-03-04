"""Epipolar-Guided Cross-Attention module for V1.1.

Replaces InverseSplat with geometry-aware attention. For each target camera
pixel, precomputes sample points along epipolar lines in each source camera.
Cross-attention queries (target) attend to sampled source features along
these epipolar lines, enabling implicit depth reasoning.

Attention size per query: 720 x 160 = 115,200
(vs full attention 720 x 3600 = 2.6M -> 22x smaller)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..camera_models import (
    FThetaCameraConfig,
    PinholeCameraConfig,
    NUSCENES_CAMERAS,
    PHYSICALAI_CAMERAS,
)


def precompute_epipolar_samples(
    target_cams: Dict[str, FThetaCameraConfig],
    source_cams: Dict[str, PinholeCameraConfig],
    target_size: Tuple[int, int] = (20, 36),
    source_size: Tuple[int, int] = (20, 36),
    n_samples: int = 32,
    depth_range: Tuple[float, float] = (1.0, 60.0),
    target_original_size: Tuple[int, int] = (1080, 1920),
    source_original_size: Tuple[int, int] = (900, 1600),
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Precompute epipolar sample coordinates for all target-source camera pairs.

    For each target pixel in the feature map, samples points along the
    corresponding epipolar line in each source camera.

    Args:
        target_cams: Dict of target Physical AI camera configs.
        source_cams: Dict of source NuScenes camera configs.
        target_size: Target feature map size (tH, tW).
        source_size: Source feature map size (sH, sW).
        n_samples: Number of depth samples per epipolar line.
        depth_range: (min_depth, max_depth) in meters.
        target_original_size: Original target image size (H, W).
        source_original_size: Original source image size (H, W).

    Returns:
        Dict mapping target camera names to:
            "grid": (tH*tW, n_src, n_samples, 2) normalized coords for grid_sample
            "valid": (tH*tW, n_src) bool mask, has any valid sample per source cam
    """
    tH, tW = target_size
    sH, sW = source_size
    oH_tgt, oW_tgt = target_original_size
    oH_src, oW_src = source_original_size
    d_min, d_max = depth_range

    # Log-uniform depth samples
    depths = torch.exp(torch.linspace(
        math.log(d_min), math.log(d_max), n_samples, dtype=torch.float64
    ))  # (n_samples,)

    source_cam_names = list(source_cams.keys())
    n_src = len(source_cam_names)

    results = {}

    for tgt_name, tgt_cam in target_cams.items():
        n_pixels = tH * tW

        # Target pixel grid at feature resolution, scaled to original PA resolution
        v_indices = torch.arange(tH, dtype=torch.float64)
        u_indices = torch.arange(tW, dtype=torch.float64)
        v_grid, u_grid = torch.meshgrid(v_indices, u_indices, indexing="ij")
        u_pa = (u_grid + 0.5) * (oW_tgt / tW)  # (tH, tW)
        v_pa = (v_grid + 0.5) * (oH_tgt / tH)

        # Unproject target pixels to 3D rays (camera frame)
        rays_cam = tgt_cam.pixel2ray(u_pa, v_pa)  # (tH, tW, 3)

        # Transform to world frame
        R_tgt = tgt_cam.rotation_matrix  # (3, 3) cam-to-ego
        t_tgt = tgt_cam.translation      # (3,) cam position in ego

        rays_world = torch.einsum("ij,hwj->hwi", R_tgt, rays_cam)  # (tH, tW, 3)
        rays_world = F.normalize(rays_world, dim=-1)
        rays_flat = rays_world.reshape(n_pixels, 3)  # (720, 3)

        origin_world = t_tgt.unsqueeze(0).expand(n_pixels, -1)  # (720, 3)

        # Sample 3D points along each ray
        # points_3d: (720, n_samples, 3)
        points_3d = origin_world.unsqueeze(1) + depths.unsqueeze(0).unsqueeze(-1) * rays_flat.unsqueeze(1)

        # Project to each source camera
        all_grids = []
        all_valid = []

        for src_name in source_cam_names:
            src_cam = source_cams[src_name]
            R_src = src_cam.rotation_matrix  # (3, 3) cam-to-ego
            t_src = src_cam.translation      # (3,) cam position in ego

            # World -> source camera frame
            R_src_inv = R_src.T  # ego-to-cam rotation
            # points_cam = R_src_inv @ (points_3d - t_src)
            pts_shifted = points_3d - t_src.unsqueeze(0).unsqueeze(0)  # (720, n_samples, 3)
            pts_cam = torch.einsum("ij,mnj->mni", R_src_inv, pts_shifted)  # (720, n_samples, 3)

            # Project to pixel coordinates
            u_src, v_src, valid = src_cam.project(pts_cam)  # each (720, n_samples)

            # Normalize for grid_sample on source feature map
            # grid_sample expects coords in [-1, 1]
            u_norm = 2.0 * (u_src * sW / oW_src) / sW - 1.0  # = 2*u_src/oW_src - 1
            v_norm = 2.0 * (v_src * sH / oH_src) / sH - 1.0  # = 2*v_src/oH_src - 1

            # Set invalid to out-of-bounds
            u_norm = torch.where(valid, u_norm, torch.full_like(u_norm, -2.0))
            v_norm = torch.where(valid, v_norm, torch.full_like(v_norm, -2.0))

            grid = torch.stack([u_norm, v_norm], dim=-1).float()  # (720, n_samples, 2)
            has_valid = valid.any(dim=-1)  # (720,) - any valid sample for this source

            all_grids.append(grid)
            all_valid.append(has_valid)

        # Stack across source cameras
        epipolar_grid = torch.stack(all_grids, dim=1)   # (720, n_src, n_samples, 2)
        epipolar_valid = torch.stack(all_valid, dim=1)   # (720, n_src)

        results[tgt_name] = {
            "grid": epipolar_grid,
            "valid": epipolar_valid,
        }

    return results


class EpipolarCrossAttentionLayer(nn.Module):
    """Single layer of epipolar-guided cross-attention + FFN.

    Target queries attend to source features sampled along precomputed
    epipolar lines via multi-head cross-attention, followed by FFN.

    Args:
        d_model: Feature dimension.
        n_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Cross-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_queries: torch.Tensor,
        sampled_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: cross-attention + FFN with pre-norm residuals.

        Args:
            target_queries: Target query features, shape (B, Q, C).
            sampled_features: Source features sampled along epipolar lines,
                shape (B, Q, K, C) where K = n_src * n_samples.

        Returns:
            output: Updated target queries, shape (B, Q, C).
            attn_weights: Attention weights, shape (B, n_heads, Q, K).
        """
        B, Q, C = target_queries.shape
        K = sampled_features.shape[2]

        # Pre-norm cross-attention
        q_normed = self.norm1(target_queries)

        # Reshape sampled features for attention: (B, Q, K, C) -> (B, Q*K, C)
        # But we need per-query attention, so keep Q dimension
        # Q projection: (B, Q, C) -> (B, n_heads, Q, d_k)
        q = self.q_proj(q_normed).reshape(B, Q, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # K, V projection: (B, Q, K, C) -> (B, n_heads, Q, K, d_k)
        # Each query has its own set of keys/values from epipolar samples
        k = self.k_proj(sampled_features).reshape(B, Q, K, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)
        v = self.v_proj(sampled_features).reshape(B, Q, K, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)

        # Attention: q (B, H, Q, d_k), k (B, H, Q, K, d_k) -> scores (B, H, Q, K)
        # For per-query attention: scores[b,h,q,:] = q[b,h,q,:] @ k[b,h,q,:,:].T
        scores = torch.einsum("bhqd,bhqkd->bhqk", q, k) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum: attn (B, H, Q, K) x v (B, H, Q, K, d_k) -> (B, H, Q, d_k)
        attn_output = torch.einsum("bhqk,bhqkd->bhqd", attn_weights, v)

        # Reshape back: (B, n_heads, Q, d_k) -> (B, Q, C)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, Q, C)
        attn_output = self.out_proj(attn_output)

        # Residual
        target_queries = target_queries + self.dropout1(attn_output)

        # Pre-norm FFN
        ffn_input = self.norm2(target_queries)
        target_queries = target_queries + self.ffn(ffn_input)

        return target_queries, attn_weights


class EpipolarCrossAttention(nn.Module):
    """Epipolar-Guided Cross-Attention module replacing InverseSplat.

    Produces target feature maps by having learnable target queries
    cross-attend to source features sampled along precomputed epipolar lines.

    Args:
        d_model: Feature dimension (256, matching backbone).
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        n_samples: Points per epipolar line.
        ffn_dim: FFN hidden dimension.
        dropout: Dropout probability.
        target_size: Target feature map (tH, tW).
        source_size: Source feature map (sH, sW).
        depth_range: Depth range for epipolar sampling (min, max) meters.
        target_cameras: List of Physical AI camera names.
        source_cameras: List of NuScenes camera names.
    """

    # NuScenes camera order (must match build_camera_params)
    NUSCENES_CAM_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        n_samples: int = 32,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        target_size: Tuple[int, int] = (20, 36),
        source_size: Tuple[int, int] = (20, 36),
        depth_range: Tuple[float, float] = (1.0, 60.0),
        target_cameras: List[str] = ("front_wide", "cross_left", "cross_right"),
        source_cameras: List[str] = (
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        ),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_samples = n_samples
        self.target_size = target_size
        self.source_size = source_size
        self.target_cameras = list(target_cameras)
        self.source_cameras = list(source_cameras)
        self.n_src = len(self.source_cameras)

        tH, tW = target_size
        n_pixels = tH * tW  # 720

        # Learnable target queries per camera
        self.target_queries = nn.ParameterDict({
            cam: nn.Parameter(torch.randn(1, n_pixels, d_model) * 0.02)
            for cam in self.target_cameras
        })

        # Transformer layers
        self.layers = nn.ModuleList([
            EpipolarCrossAttentionLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output norm
        self.output_norm = nn.LayerNorm(d_model)

        # Precompute epipolar sample grids
        self._precompute_and_register_buffers(
            target_size, source_size, n_samples, depth_range,
        )

    def _precompute_and_register_buffers(
        self,
        target_size: Tuple[int, int],
        source_size: Tuple[int, int],
        n_samples: int,
        depth_range: Tuple[float, float],
    ) -> None:
        """Precompute epipolar sample grids and register as buffers.

        Args:
            target_size: Target feature map size (tH, tW).
            source_size: Source feature map size (sH, sW).
            n_samples: Points per epipolar line.
            depth_range: Depth range for sampling (min, max).
        """
        target_cams = {
            name: PHYSICALAI_CAMERAS[name] for name in self.target_cameras
        }
        source_cams = {
            name: NUSCENES_CAMERAS[name] for name in self.source_cameras
        }

        precomputed = precompute_epipolar_samples(
            target_cams=target_cams,
            source_cams=source_cams,
            target_size=target_size,
            source_size=source_size,
            n_samples=n_samples,
            depth_range=depth_range,
        )

        for tgt_name, data in precomputed.items():
            self.register_buffer(
                f"epipolar_grid_{tgt_name}",
                data["grid"],  # (720, n_src, n_samples, 2)
            )
            self.register_buffer(
                f"epipolar_valid_{tgt_name}",
                data["valid"],  # (720, n_src)
            )

    def _sample_epipolar_features(
        self,
        source_features: torch.Tensor,
        target_cam_name: str,
    ) -> torch.Tensor:
        """Sample source features along precomputed epipolar lines.

        For each target pixel, gathers features from each source camera
        at the precomputed epipolar sample locations.

        Args:
            source_features: Source camera features, shape (B, N_src, C, sH, sW).
            target_cam_name: Name of target camera being processed.

        Returns:
            Sampled features, shape (B, Q, K, C) where Q=tH*tW, K=N_src*n_samples.
        """
        B, N_src, C, sH, sW = source_features.shape
        grid = getattr(self, f"epipolar_grid_{target_cam_name}")  # (Q, N_src, n_samples, 2)
        Q = grid.shape[0]  # 720

        sampled_list = []
        for n in range(N_src):
            # Grid for this source camera: (Q, n_samples, 2)
            cam_grid = grid[:, n]  # (Q, n_samples, 2)

            # Reshape for grid_sample: need (B, Q, n_samples, 2)
            cam_grid_batch = cam_grid.unsqueeze(0).expand(B, -1, -1, -1)

            # Source features for this camera: (B, C, sH, sW)
            src_feat = source_features[:, n]

            # grid_sample: (B, C, sH, sW) with grid (B, Q, n_samples, 2) -> (B, C, Q, n_samples)
            sampled = F.grid_sample(
                src_feat, cam_grid_batch,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )  # (B, C, Q, n_samples)

            sampled_list.append(sampled)

        # Concatenate along sample dimension: (B, C, Q, N_src * n_samples)
        sampled_all = torch.cat(sampled_list, dim=-1)

        # Reshape to (B, Q, K, C)
        sampled_all = sampled_all.permute(0, 2, 3, 1)  # (B, Q, K, C)

        return sampled_all

    def forward(
        self,
        source_features: torch.Tensor,
        target_cam_name: str,
        target_pe: torch.Tensor | None = None,
        target_cam_id_pe: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Produce target feature map via epipolar cross-attention.

        Args:
            source_features: Source camera features with PE,
                shape (B, N_src, C, sH, sW).
            target_cam_name: Which target camera to process.
            target_pe: Optional target 3D PE, shape (1, C, tH, tW).
            target_cam_id_pe: Optional target camera ID PE, shape (1, C).

        Returns:
            target_feature_map: Feature map, shape (B, C, tH, tW).
            attn_weights_list: List of attention weights per layer,
                each shape (B, n_heads, Q, K).
        """
        B = source_features.shape[0]
        tH, tW = self.target_size

        # Initialize target queries
        queries = self.target_queries[target_cam_name].expand(B, -1, -1)  # (B, 720, C)

        # Add target 3D PE
        if target_pe is not None:
            pe_flat = target_pe.flatten(2).permute(0, 2, 1)  # (1, 720, C)
            queries = queries + pe_flat.expand(B, -1, -1)

        # Add target camera ID PE
        if target_cam_id_pe is not None:
            queries = queries + target_cam_id_pe.unsqueeze(1)  # broadcast (1, 1, C) -> (B, 720, C)

        # Sample source features along epipolar lines
        sampled_features = self._sample_epipolar_features(
            source_features, target_cam_name,
        )  # (B, 720, K, C) where K = N_src * n_samples

        # Apply transformer layers
        attn_weights_list = []
        for layer in self.layers:
            queries, attn_weights = layer(queries, sampled_features)
            attn_weights_list.append(attn_weights.detach())

        # Output norm and reshape
        queries = self.output_norm(queries)  # (B, 720, C)
        target_feature_map = queries.permute(0, 2, 1).reshape(B, self.d_model, tH, tW)

        return target_feature_map, attn_weights_list
