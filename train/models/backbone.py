"""EfficientNet-B4 backbone with FPN neck for multi-scale feature extraction.

Uses timm's pretrained EfficientNet-B4 as the backbone encoder and adds
an FPN (Feature Pyramid Network) neck to fuse multi-scale features into
a single feature map at 1/16 resolution.

Input: (B*5, 3, 320, 576) -> Output: (B*5, C_feat, 20, 36)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FPNNeck(nn.Module):
    """Feature Pyramid Network neck for multi-scale feature fusion.

    Takes multi-scale feature maps from backbone and fuses them into
    a single feature map at the target (1/16) resolution.

    Args:
        in_channels_list: List of channel dimensions from each backbone stage.
        out_channels: Output channel dimension for all FPN levels.
    """

    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections (1x1 conv to match channel dims)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # Output convolutions (3x3 conv after fusion)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

        # Final fusion: combine all FPN levels to target resolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through FPN neck.

        Args:
            features: List of feature maps from backbone stages,
                ordered from finest to coarsest resolution.

        Returns:
            Fused feature map at 1/16 resolution, shape (B, C_out, H/16, W/16).
        """
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )

        # Output convolutions
        fpn_outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        # Resize all to target resolution (the 1/16 level)
        target_size = fpn_outs[-2].shape[2:]  # Use second-to-last as 1/16 reference
        resized = [
            F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            if f.shape[2:] != target_size else f
            for f in fpn_outs
        ]

        # Concatenate and fuse
        fused = torch.cat(resized, dim=1)
        return self.fusion_conv(fused)


class EfficientNetB4Backbone(nn.Module):
    """EfficientNet-B4 backbone with FPN neck.

    Extracts multi-scale features from input images and fuses them
    into a single feature map at 1/16 resolution.

    Args:
        pretrained: Whether to use pretrained weights.
        out_channels: Output feature dimension after FPN fusion.
        frozen_stages: Number of initial stages to freeze (-1 for none).
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_channels: int = 256,
        frozen_stages: int = -1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        # Create EfficientNet-B4 with feature extraction
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # 4 feature levels: 1/4, 1/8, 1/16, 1/32
        )

        # Get feature dimensions from backbone
        feature_info = self.backbone.feature_info.channels()
        self.feature_channels = feature_info

        # FPN neck
        self.neck = FPNNeck(
            in_channels_list=self.feature_channels,
            out_channels=out_channels,
        )

        # Freeze early stages if specified
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    def _freeze_stages(self, num_stages: int) -> None:
        """Freeze the first num_stages of the backbone.

        Args:
            num_stages: Number of stages to freeze.
        """
        for i, (name, module) in enumerate(self.backbone.named_children()):
            if i < num_stages:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features and fuse via FPN.

        Args:
            x: Input images, shape (B, 3, H, W).

        Returns:
            Feature map, shape (B, out_channels, H/16, W/16).
        """
        # Extract multi-scale features
        features = self.backbone(x)  # List of 4 feature maps
        # Fuse via FPN
        return self.neck(features)
