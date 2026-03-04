"""Stage 2 Image Loss: L1 + SSIM + Perceptual (VGG) loss.

Combined loss for image reconstruction quality.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SSIMLoss(nn.Module):
    """Structural Similarity (SSIM) Loss.

    SSIM measures structural similarity between two images.
    Loss = 1 - SSIM (so lower is better).

    Args:
        window_size: Size of the Gaussian averaging window.
        sigma: Standard deviation of the Gaussian window.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("window", self._create_window(window_size, sigma, 3))

    @staticmethod
    def _create_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation.

        Args:
            window_size: Size of the window.
            sigma: Standard deviation.
            channels: Number of image channels.

        Returns:
            Gaussian window, shape (channels, 1, window_size, window_size).
        """
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss (1 - SSIM).

        Args:
            pred: Predicted image, shape (B, 3, H, W), values in [0, 1].
            target: Ground truth image, shape (B, 3, H, W), values in [0, 1].

        Returns:
            Scalar SSIM loss.
        """
        # Ensure float32 for numerical stability (conv2d requires matching dtypes)
        pred = pred.float()
        target = target.float()

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(pred.device, dtype=torch.float32)
        channels = pred.shape[1]
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
        mu2 = F.conv2d(target, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss.

    Computes L1 distance between VGG16 feature maps at specified layers.

    Args:
        layers: VGG layer indices to extract features from.
    """

    # Default layers: conv1_2, conv2_2, conv3_3
    DEFAULT_LAYERS = [3, 8, 15]

    def __init__(self, layers: List[int] = None) -> None:
        super().__init__()
        if layers is None:
            layers = self.DEFAULT_LAYERS

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        max_layer = max(layers) + 1
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:max_layer])
        self.layers = layers

        # Freeze VGG
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize image to ImageNet statistics.

        Args:
            x: Image tensor in [0, 1], shape (B, 3, H, W).

        Returns:
            Normalized tensor.
        """
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            pred: Predicted image, shape (B, 3, H, W), values in [0, 1].
            target: Ground truth image, shape (B, 3, H, W), values in [0, 1].

        Returns:
            Scalar perceptual loss.
        """
        pred_norm = self._normalize(pred.float())
        target_norm = self._normalize(target.float())

        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred_norm
        x_target = target_norm

        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layers:
                loss = loss + F.l1_loss(x_pred, x_target)

        return loss


class ImageLoss(nn.Module):
    """Combined image reconstruction loss: L1 + SSIM + Perceptual.

    Args:
        l1_weight: Weight for L1 loss.
        ssim_weight: Weight for SSIM loss.
        perceptual_weight: Weight for VGG perceptual loss.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        perceptual_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = VGGPerceptualLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute image loss across all cameras.

        Args:
            pred: Dict mapping camera names to predicted RGB (B, 3, H, W).
            target: Dict mapping camera names to GT RGB (B, 3, H, W).

        Returns:
            total_loss: Scalar loss.
            metrics: Dict with individual loss components.
        """
        device = next(iter(pred.values())).device
        total_l1 = torch.tensor(0.0, device=device)
        total_ssim = torch.tensor(0.0, device=device)
        total_percept = torch.tensor(0.0, device=device)
        n_cams = 0

        for cam in pred:
            if cam in target:
                p = pred[cam]
                t = target[cam]

                total_l1 = total_l1 + self.l1_loss(p, t)
                total_ssim = total_ssim + self.ssim_loss(p, t)
                total_percept = total_percept + self.perceptual_loss(p, t)
                n_cams += 1

        if n_cams > 0:
            total_l1 = total_l1 / n_cams
            total_ssim = total_ssim / n_cams
            total_percept = total_percept / n_cams

        loss = (self.l1_weight * total_l1 +
                self.ssim_weight * total_ssim +
                self.perceptual_weight * total_percept)

        metrics = {
            "image_l1": total_l1.item(),
            "image_ssim": total_ssim.item(),
            "image_perceptual": total_percept.item(),
            "image_total": loss.item(),
        }
        return loss, metrics
