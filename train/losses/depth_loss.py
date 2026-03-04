"""Stage 1 Depth Loss: L1 + Scale-Invariant Logarithmic (SILog) loss.

SILog loss penalizes relative depth errors regardless of absolute scale,
combined with L1 for absolute accuracy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss.

    Measures variance of log-depth error, making it robust to
    global scale ambiguity in monocular depth estimation.

    L_SILog = mean(d_i^2) - lambda * mean(d_i)^2
    where d_i = log(pred_i) - log(gt_i)

    Args:
        variance_focus: Lambda parameter controlling focus on variance (0-1).
            0 = full SILog, 1 = pure mean squared log error.
    """

    def __init__(self, variance_focus: float = 0.85) -> None:
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SILog loss.

        Args:
            pred: Predicted depth, shape (B, 1, H, W).
            target: Ground truth depth, shape (B, 1, H, W) or (B, H, W).

        Returns:
            Scalar SILog loss.
        """
        if target.ndim == 3:
            target = target.unsqueeze(1)

        # Valid mask (positive depth only)
        valid = (target > 1e-3) & (pred > 1e-3)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[valid]
        target_valid = target[valid]

        d = torch.log(pred_valid) - torch.log(target_valid)
        loss = torch.mean(d ** 2) - self.variance_focus * (torch.mean(d) ** 2)
        return loss


class DepthLoss(nn.Module):
    """Combined depth loss: L1 + SILog.

    Args:
        l1_weight: Weight for L1 loss.
        silog_weight: Weight for SILog loss.
    """

    def __init__(self, l1_weight: float = 1.0, silog_weight: float = 0.5) -> None:
        super().__init__()
        self.l1_weight = l1_weight
        self.silog_weight = silog_weight
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.silog_loss = SILogLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute depth loss across all cameras.

        Args:
            pred: Dict mapping camera names to predicted depth (B, 1, H, W).
            target: Dict mapping camera names to GT depth (B, H, W).

        Returns:
            total_loss: Scalar loss.
            metrics: Dict with individual loss components.
        """
        total_l1 = torch.tensor(0.0, device=next(iter(pred.values())).device)
        total_silog = torch.tensor(0.0, device=next(iter(pred.values())).device)
        n_cams = 0

        for cam in pred:
            if cam in target:
                p = pred[cam]
                t = target[cam]

                # L1 on valid pixels
                valid = t > 1e-3
                if valid.sum() > 0:
                    if t.ndim == 3:
                        t_expanded = t.unsqueeze(1)
                        valid_expanded = valid.unsqueeze(1)
                    else:
                        t_expanded = t
                        valid_expanded = valid
                    total_l1 = total_l1 + self.l1_loss(p[valid_expanded], t_expanded[valid_expanded])

                total_silog = total_silog + self.silog_loss(p, t)
                n_cams += 1

        if n_cams > 0:
            total_l1 = total_l1 / n_cams
            total_silog = total_silog / n_cams

        loss = self.l1_weight * total_l1 + self.silog_weight * total_silog

        metrics = {
            "depth_l1": total_l1.item(),
            "depth_silog": total_silog.item(),
            "depth_total": loss.item(),
        }
        return loss, metrics
