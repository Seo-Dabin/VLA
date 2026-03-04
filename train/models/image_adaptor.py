"""Full Image Adaptor V1 model wrapper.

Integrates all model components:
  - EfficientNet-B4 backbone + FPN neck
  - Plucker Ray positional embedding
  - Inverse Splat (Lift-Splat to target camera planes)
  - Stage 1: Depth decoder
  - Stage 2: Image reconstruction decoder
  - Stage 3: Visual token decoder

The forward pass executes only the decoders for currently active stages,
controlled by the curriculum controller.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import EfficientNetB4Backbone
from .depth_decoder import MultiCameraDepthDecoder
from .image_decoder import MultiCameraImageDecoder
from .inverse_splat import InverseSplat
from .plucker_pe import PluckerRayPE
from .token_decoder import MultiCameraTokenDecoder


class ImageAdaptorV1(nn.Module):
    """Image Adaptor V1: nuScenes images -> Physical AI visual tokens.

    Processes NuScenes 5-camera images through a shared backbone, adds
    camera-aware positional embeddings, lifts and splats to Physical AI
    camera planes, and decodes three types of outputs:
      - Stage 1: Depth maps (per target camera)
      - Stage 2: Reconstructed RGB images (per target camera)
      - Stage 3: Visual tokens (per target camera, Qwen3-VL compatible)

    Args:
        backbone_name: Backbone architecture name.
        backbone_pretrained: Whether to use pretrained backbone weights.
        backbone_out_channels: Backbone + FPN output channels.
        context_dim: Inverse Splat context dimension.
        depth_bins: Number of depth bins for Inverse Splat.
        plucker_hidden_dim: Hidden dimension for Plucker PE MLP.
        image_size: Input image size (H, W).
        feature_size: Feature map size (fH, fW).
        target_cameras: List of Physical AI target camera names.
        depth_share_weights: Whether depth decoders share weights.
        image_share_weights: Whether image decoders share weights.
        token_d_model: Token decoder internal dimension.
        token_num_layers: Token decoder transformer layers.
        token_num_heads: Token decoder attention heads.
        num_query_tokens: Number of output visual tokens per camera.
        token_output_dim: Output token embedding dimension.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b4",
        backbone_pretrained: bool = True,
        backbone_out_channels: int = 256,
        context_dim: int = 360,
        depth_bins: int = 64,
        plucker_hidden_dim: int = 128,
        image_size: Tuple[int, int] = (320, 576),
        feature_size: Tuple[int, int] = (20, 36),
        target_cameras: List[str] = ("front_wide", "cross_left", "cross_right"),
        depth_share_weights: bool = False,
        image_share_weights: bool = False,
        token_d_model: int = 512,
        token_num_layers: int = 6,
        token_num_heads: int = 8,
        num_query_tokens: int = 180,
        token_output_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.feature_size = feature_size
        self.target_cameras = list(target_cameras)

        # Backbone: EfficientNet-B4 + FPN
        self.backbone = EfficientNetB4Backbone(
            pretrained=backbone_pretrained,
            out_channels=backbone_out_channels,
        )

        # Plucker Ray positional embedding
        self.plucker_pe = PluckerRayPE(
            out_dim=backbone_out_channels,
            hidden_dim=plucker_hidden_dim,
        )

        # Inverse Splat: Lift-Splat to target camera planes
        self.inverse_splat = InverseSplat(
            in_channels=backbone_out_channels,
            context_channels=context_dim,
            depth_bins=depth_bins,
            image_size=image_size,
            feature_size=feature_size,
            target_feature_size=feature_size,
        )

        # Stage 1: Depth decoder
        self.depth_decoder = MultiCameraDepthDecoder(
            in_channels=context_dim,
            image_size=image_size,
            camera_names=target_cameras,
            share_weights=depth_share_weights,
        )

        # Stage 2: Image reconstruction decoder
        self.image_decoder = MultiCameraImageDecoder(
            in_channels=context_dim,
            image_size=image_size,
            camera_names=target_cameras,
            share_weights=image_share_weights,
        )

        # Stage 3: Token decoder
        self.token_decoder = MultiCameraTokenDecoder(
            in_channels=context_dim,
            d_model=token_d_model,
            num_layers=token_num_layers,
            num_heads=token_num_heads,
            num_query_tokens=num_query_tokens,
            output_dim=token_output_dim,
            camera_names=target_cameras,
        )

    def forward(
        self,
        nuscenes_images: torch.Tensor,
        source_intrinsics: torch.Tensor,
        source_extrinsics: torch.Tensor,
        target_cameras: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        active_stages: List[int] = (1, 2, 3),
        capture_attention: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through Image Adaptor.

        Args:
            nuscenes_images: NuScenes 5-camera images, shape (B, 5, 3, H, W).
            source_intrinsics: NuScenes camera K matrices, shape (B, 5, 3, 3).
            source_extrinsics: NuScenes camera-to-ego transforms, shape (B, 5, 4, 4).
            target_cameras: Dict mapping Physical AI camera names to
                (K_target, E_target) tuples.
            active_stages: List of active training stages (1, 2, 3).
            capture_attention: Whether to capture student attention maps.

        Returns:
            Dictionary with outputs for each active stage:
                - "feature_maps": Dict[str, Tensor] - Inverse Splat features
                - "depth_preds": Dict[str, Tensor] - depth maps (if stage 1 active)
                - "image_preds": Dict[str, Tensor] - RGB images (if stage 2 active)
                - "token_preds": Dict[str, Tensor] - visual tokens (if stage 3 active)
                - "student_attention": Dict[str, list] - attention maps (if captured)
        """
        B, N, C, H, W = nuscenes_images.shape
        outputs: Dict[str, Any] = {}

        # Reshape for backbone: (B*N, 3, H, W)
        x = nuscenes_images.reshape(B * N, C, H, W)

        # Backbone feature extraction
        features = self.backbone(x)  # (B*N, C_feat, fH, fW)

        # Add Plucker Ray PE
        intrinsics_flat = source_intrinsics.reshape(B * N, 3, 3)
        # Build 4x4 extrinsics from 3x3 R and 3 t
        extrinsics_flat = source_extrinsics.reshape(B * N, 4, 4)
        features = self.plucker_pe(features, intrinsics_flat, extrinsics_flat)

        # Reshape back: (B*N, C_feat, fH, fW) -> (B, N, C_feat, fH, fW)
        C_feat = features.shape[1]
        fH, fW = features.shape[2], features.shape[3]
        features_5cam = features.reshape(B, N, C_feat, fH, fW)

        # Inverse Splat: Lift + Splat to target camera planes
        feature_maps = self.inverse_splat(
            features_5cam,
            source_intrinsics,
            source_extrinsics,
            target_cameras,
        )  # Dict[str, (B, context_dim, tH, tW)]
        outputs["feature_maps"] = feature_maps

        # Stage 1: Depth prediction
        if 1 in active_stages:
            outputs["depth_preds"] = self.depth_decoder(feature_maps)

        # Stage 2: Image reconstruction
        if 2 in active_stages:
            outputs["image_preds"] = self.image_decoder(feature_maps)

        # Stage 3: Token generation
        if 3 in active_stages:
            tokens, attn_maps = self.token_decoder(
                feature_maps, capture_attention=capture_attention
            )
            outputs["token_preds"] = tokens
            outputs["student_attention"] = attn_maps

        return outputs

    def get_stage_parameters(self, stage: int) -> list[nn.Parameter]:
        """Get parameters specific to a training stage.

        Args:
            stage: Training stage number (1, 2, or 3).

        Returns:
            List of parameters for the specified stage.
        """
        # Shared parameters (backbone, plucker_pe, inverse_splat)
        shared_params = list(self.backbone.parameters()) + \
                       list(self.plucker_pe.parameters()) + \
                       list(self.inverse_splat.parameters())

        if stage == 1:
            return shared_params + list(self.depth_decoder.parameters())
        elif stage == 2:
            return shared_params + list(self.depth_decoder.parameters()) + \
                   list(self.image_decoder.parameters())
        elif stage == 3:
            return list(self.parameters())  # All parameters
        else:
            raise ValueError(f"Invalid stage: {stage}")

    @property
    def num_parameters(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
