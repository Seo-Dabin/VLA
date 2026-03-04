"""Image Adaptor V1.1 model wrapper.

Replaces V1's Inverse Splat + Plucker PE with:
  - Fourier-encoded Plucker Ray PE (3-component system)
  - Epipolar-Guided Cross-Attention
  - Lightweight depth head (1x1 Conv, auxiliary)

Components:
  - EfficientNet-B4 backbone + FPN (reused from V1)
  - FourierRayPE (replaces PluckerRayPE)
  - CameraIDEmbedding (NEW)
  - EpipolarCrossAttention (replaces InverseSplat)
  - Target3DPE (NEW, precomputed)
  - MultiCameraTokenDecoder (reused, in_channels=256)
  - Depth Head: 1x1 Conv (replaces MultiCameraDepthDecoder)

No image reconstruction, no curriculum staging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EfficientNetB4Backbone
from .epipolar_attention import EpipolarCrossAttention
from .geometric_pe import CameraIDEmbedding, FourierRayPE, Target3DPE
from .token_decoder import MultiCameraTokenDecoder


class ImageAdaptorV1_1(nn.Module):
    """Image Adaptor V1.1: nuScenes images -> Physical AI visual tokens.

    Uses Epipolar Cross-Attention instead of Inverse Splat for view
    transformation, with Fourier-encoded PE for geometric awareness.

    Args:
        backbone_pretrained: Whether to use pretrained backbone weights.
        backbone_out_channels: Backbone + FPN output channels.
        fourier_L: Number of Fourier frequency bands for PE.
        epipolar_d_model: Epipolar attention feature dimension.
        epipolar_n_heads: Number of attention heads.
        epipolar_n_layers: Number of epipolar attention layers.
        epipolar_n_samples: Points per epipolar line.
        epipolar_ffn_dim: FFN hidden dimension.
        epipolar_dropout: Dropout probability.
        epipolar_depth_range: Depth range for epipolar sampling.
        target_cameras: List of Physical AI target camera names.
        target_size: Target feature map size (tH, tW).
        source_size: Source feature map size (sH, sW).
        token_d_model: Token decoder internal dimension.
        token_num_layers: Token decoder transformer layers.
        token_num_heads: Token decoder attention heads.
        num_query_tokens: Number of output visual tokens per camera.
        token_output_dim: Output token embedding dimension.
    """

    # NuScenes camera order (5 cameras)
    NUSCENES_CAM_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        backbone_pretrained: bool = True,
        backbone_out_channels: int = 256,
        fourier_L: int = 10,
        epipolar_d_model: int = 256,
        epipolar_n_heads: int = 8,
        epipolar_n_layers: int = 3,
        epipolar_n_samples: int = 32,
        epipolar_ffn_dim: int = 1024,
        epipolar_dropout: float = 0.1,
        epipolar_depth_range: Tuple[float, float] = (1.0, 60.0),
        target_cameras: List[str] = ("front_wide", "cross_left", "cross_right"),
        target_size: Tuple[int, int] = (20, 36),
        source_size: Tuple[int, int] = (20, 36),
        token_d_model: int = 512,
        token_num_layers: int = 6,
        token_num_heads: int = 8,
        num_query_tokens: int = 180,
        token_output_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.target_cameras = list(target_cameras)
        self.target_size = target_size
        self.source_size = source_size
        self.n_source = len(self.NUSCENES_CAM_ORDER)

        # Backbone: EfficientNet-B4 + FPN
        self.backbone = EfficientNetB4Backbone(
            pretrained=backbone_pretrained,
            out_channels=backbone_out_channels,
        )

        # Fourier Ray PE (for source features)
        self.fourier_ray_pe = FourierRayPE(
            out_dim=backbone_out_channels,
            fourier_L=fourier_L,
        )

        # Camera ID Embedding
        self.camera_id_embed = CameraIDEmbedding(
            n_source=self.n_source,
            n_target=len(self.target_cameras),
            dim=backbone_out_channels,
        )

        # Target 3D PE (precomputed)
        self.target_3d_pe = Target3DPE(
            target_cameras=self.target_cameras,
            target_size=target_size,
            out_dim=epipolar_d_model,
            fourier_L=fourier_L,
        )

        # Epipolar Cross-Attention (replaces InverseSplat)
        self.epipolar_attn = EpipolarCrossAttention(
            d_model=epipolar_d_model,
            n_heads=epipolar_n_heads,
            n_layers=epipolar_n_layers,
            n_samples=epipolar_n_samples,
            ffn_dim=epipolar_ffn_dim,
            dropout=epipolar_dropout,
            target_size=target_size,
            source_size=source_size,
            depth_range=epipolar_depth_range,
            target_cameras=self.target_cameras,
            source_cameras=self.NUSCENES_CAM_ORDER,
        )

        # Token decoder (reused from V1)
        self.token_decoder = MultiCameraTokenDecoder(
            in_channels=epipolar_d_model,  # 256 (was 360 in V1)
            d_model=token_d_model,
            num_layers=token_num_layers,
            num_heads=token_num_heads,
            num_query_tokens=num_query_tokens,
            output_dim=token_output_dim,
            camera_names=self.target_cameras,
        )

        # Lightweight depth head (1x1 Conv, auxiliary)
        self.depth_head = nn.Conv2d(epipolar_d_model, 1, kernel_size=1)

    def forward(
        self,
        nuscenes_images: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_extrinsics: torch.Tensor,
        capture_attention: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through Image Adaptor V1.1.

        Args:
            nuscenes_images: NuScenes 5-camera images, shape (B, 5, 3, H, W).
            source_intrinsics: NuScenes camera K matrices, shape (B, 5, 3, 3).
            source_extrinsics: NuScenes camera-to-ego transforms, shape (B, 5, 4, 4).
            capture_attention: Whether to capture attention maps.

        Returns:
            Dictionary with:
                - "feature_maps": Dict[str, Tensor] - per-camera feature maps
                - "token_preds": Dict[str, Tensor] - visual tokens
                - "depth_preds": Dict[str, Tensor] - depth maps (auxiliary)
                - "student_attention": Dict[str, list] - attention maps
                - "epipolar_attention": Dict[str, list] - epipolar attention weights
        """
        B, N, C_img, H, W = nuscenes_images.shape
        outputs: Dict[str, Any] = {}

        # 1. Backbone: (B*N, 3, H, W) -> (B*N, C, fH, fW)
        x = nuscenes_images.reshape(B * N, C_img, H, W)
        features = self.backbone(x)  # (B*N, 256, 20, 36)

        # 2. Fourier Ray PE
        intrinsics_flat = source_intrinsics.reshape(B * N, 3, 3)
        extrinsics_flat = source_extrinsics.reshape(B * N, 4, 4)
        features = self.fourier_ray_pe(features, intrinsics_flat, extrinsics_flat)

        # 3. Camera ID PE for source cameras
        cam_ids = torch.arange(N, device=features.device).repeat(B)  # (B*N,)
        cam_id_pe = self.camera_id_embed.get_source_pe(cam_ids)  # (B*N, C)
        features = features + cam_id_pe.unsqueeze(-1).unsqueeze(-1)  # broadcast to (B*N, C, fH, fW)

        # 4. Reshape: (B*N, C, fH, fW) -> (B, N, C, fH, fW)
        C_feat = features.shape[1]
        features_5cam = features.reshape(B, N, C_feat, *self.source_size)

        # 5. Epipolar Cross-Attention for each target camera
        feature_maps: Dict[str, torch.Tensor] = {}
        epipolar_attn_maps: Dict[str, list] = {}

        for cam_idx, cam_name in enumerate(self.target_cameras):
            # Get target PE
            target_pe = self.target_3d_pe.get_pe(cam_name)  # (1, C, tH, tW)
            target_cam_id_pe = self.camera_id_embed.get_target_pe(cam_idx)  # (1, C)

            fm, attn_list = self.epipolar_attn(
                features_5cam, cam_name,
                target_pe=target_pe,
                target_cam_id_pe=target_cam_id_pe,
            )
            feature_maps[cam_name] = fm  # (B, 256, 20, 36)
            epipolar_attn_maps[cam_name] = attn_list

        outputs["feature_maps"] = feature_maps
        outputs["epipolar_attention"] = epipolar_attn_maps

        # 6. Token decoder
        tokens, student_attn = self.token_decoder(
            feature_maps, capture_attention=capture_attention
        )
        outputs["token_preds"] = tokens
        outputs["student_attention"] = student_attn

        # 7. Depth head (auxiliary, at feature resolution)
        depth_preds: Dict[str, torch.Tensor] = {}
        for cam_name, fm in feature_maps.items():
            depth_preds[cam_name] = F.relu(self.depth_head(fm))  # (B, 1, 20, 36), positive depth
        outputs["depth_preds"] = depth_preds

        return outputs

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
