"""Stage 2 Image Decoder: U-Net style image reconstruction.

Takes Inverse Splat feature maps and reconstructs RGB images.
Uses the feature map directly as the encoder representation
and applies a decoder with skip-connection-like structure.

Input: (B, 360, 20, 36) -> Output: (B, 3, 320, 576)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Decoder block: Upsample + Conv + BN + ReLU.

    Args:
        in_channels: Input channel dimension.
        out_channels: Output channel dimension.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, C_in, H, W).

        Returns:
            Upsampled and convolved tensor.
        """
        x = self.up(x)
        return self.conv(x)


class ImageDecoder(nn.Module):
    """U-Net style decoder for RGB image reconstruction.

    Architecture: 4 decoder blocks, each 2x spatial upscale.
    360 -> 256 -> 128 -> 64 -> 32, then final 1x1 conv -> 3 (RGB).
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

        # Encoder head (bottleneck processing)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder pathway
        self.dec4 = DecoderBlock(256, 256)    # 20x36 -> 40x72
        self.dec3 = DecoderBlock(256, 128)    # 40x72 -> 80x144
        self.dec2 = DecoderBlock(128, 64)     # 80x144 -> 160x288
        self.dec1 = DecoderBlock(64, 32)      # 160x288 -> 320x576

        # Final output
        self.head = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode feature map to RGB image.

        Args:
            x: Feature map, shape (B, C, fH, fW).

        Returns:
            RGB image, shape (B, 3, H, W), values in [0, 1].
        """
        x = self.bottleneck(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        # Ensure exact output size
        if x.shape[2:] != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)

        return self.head(x)


class MultiCameraImageDecoder(nn.Module):
    """Image decoder for multiple target cameras.

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
            self.shared_decoder = ImageDecoder(in_channels, image_size)
        else:
            self.decoders = nn.ModuleDict({
                cam: ImageDecoder(in_channels, image_size)
                for cam in self.camera_names
            })

    def forward(self, feature_maps: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Decode RGB images for all target cameras.

        Args:
            feature_maps: Dict mapping camera names to feature tensors (B, C, fH, fW).

        Returns:
            Dict mapping camera names to RGB tensors (B, 3, H, W).
        """
        results = {}
        for cam in self.camera_names:
            if cam in feature_maps:
                if self.share_weights:
                    results[cam] = self.shared_decoder(feature_maps[cam])
                else:
                    results[cam] = self.decoders[cam](feature_maps[cam])
        return results
