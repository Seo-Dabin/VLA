"""Stage 1 Depth Decoder: CNN-based depth map prediction.

Takes Inverse Splat feature maps and decodes them to per-pixel depth maps
via a series of upsampling blocks.

Input: (B, 360, 20, 36) -> Output: (B, 1, 320, 576)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    """Single upsampling block: Conv + BN + ReLU + Upsample.

    Args:
        in_channels: Input channel dimension.
        out_channels: Output channel dimension.
        scale_factor: Spatial upsampling factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Upsampled output tensor.
        """
        return self.block(x)


class DepthDecoder(nn.Module):
    """CNN decoder for depth map prediction from Inverse Splat features.

    Architecture: 4 upsampling blocks, each 2x spatial upscale.
    360 -> 256 -> 128 -> 64 -> 32, then final 1x1 conv -> 1 channel.
    Total: 16x upscale (36x20 -> 576x320).

    Args:
        in_channels: Input feature channels.
        image_size: Target output size (H, W).
    """

    def __init__(
        self,
        in_channels: int = 360,
        image_size: tuple[int, int] = (320, 576),
    ) -> None:
        super().__init__()
        self.image_size = image_size

        self.decoder = nn.Sequential(
            UpsampleBlock(in_channels, 256, scale_factor=2),  # 20x36 -> 40x72
            UpsampleBlock(256, 128, scale_factor=2),           # 40x72 -> 80x144
            UpsampleBlock(128, 64, scale_factor=2),            # 80x144 -> 160x288
            UpsampleBlock(64, 32, scale_factor=2),             # 160x288 -> 320x576
        )

        self.head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.ReLU(inplace=True),  # Depth is always positive
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode feature map to depth map.

        Args:
            x: Feature map, shape (B, C, fH, fW).

        Returns:
            Depth map, shape (B, 1, H, W).
        """
        x = self.decoder(x)

        # Ensure exact output size
        if x.shape[2:] != self.image_size:
            x = nn.functional.interpolate(
                x, size=self.image_size, mode="bilinear", align_corners=False
            )

        return self.head(x)


class MultiCameraDepthDecoder(nn.Module):
    """Depth decoder for multiple target cameras.

    Args:
        in_channels: Input feature channels.
        image_size: Target output size (H, W).
        camera_names: List of target camera names.
        share_weights: If True, all cameras share the same decoder.
    """

    def __init__(
        self,
        in_channels: int = 360,
        image_size: tuple[int, int] = (320, 576),
        camera_names: list[str] = ("front_wide", "cross_left", "cross_right"),
        share_weights: bool = False,
    ) -> None:
        super().__init__()
        self.camera_names = list(camera_names)
        self.share_weights = share_weights

        if share_weights:
            self.shared_decoder = DepthDecoder(in_channels, image_size)
        else:
            self.decoders = nn.ModuleDict({
                cam: DepthDecoder(in_channels, image_size)
                for cam in self.camera_names
            })

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode depth maps for all target cameras.

        Args:
            feature_maps: Dict mapping camera names to feature tensors (B, C, fH, fW).

        Returns:
            Dict mapping camera names to depth tensors (B, 1, H, W).
        """
        results = {}
        for cam in self.camera_names:
            if cam in feature_maps:
                if self.share_weights:
                    results[cam] = self.shared_decoder(feature_maps[cam])
                else:
                    results[cam] = self.decoders[cam](feature_maps[cam])
        return results
