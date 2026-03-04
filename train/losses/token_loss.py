"""Stage 3 Token Loss: MSE + Cosine Similarity + KL Attention Distillation.

Measures alignment between predicted visual tokens and ground truth
tokens from Qwen3-VL, plus attention map distillation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLoss(nn.Module):
    """Combined visual token loss with attention distillation.

    Components:
      - MSE: Mean squared error between predicted and target tokens.
      - Cosine: Cosine similarity loss (1 - cosine_similarity).
      - KL Divergence: Attention map distillation loss.

    Args:
        mse_weight: Weight for MSE loss.
        cosine_weight: Weight for cosine similarity loss.
        attention_kl_weight: Weight for attention distillation loss.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.5,
        attention_kl_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.attention_kl_weight = attention_kl_weight

        self.mse_loss = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def _compute_attention_kl(
        self,
        student_attn: List[torch.Tensor],
        teacher_attn: List[Any],
    ) -> torch.Tensor:
        """Compute KL divergence between student and teacher attention maps.

        Args:
            student_attn: List of student attention maps per layer,
                each shape (B, num_heads, Q, K).
            teacher_attn: List of teacher attention maps per layer
                (precomputed, possibly variable format).

        Returns:
            Average KL divergence across layers.
        """
        if not student_attn or not teacher_attn:
            return torch.tensor(0.0, device=student_attn[0].device if student_attn else "cpu")

        kl_total = torch.tensor(0.0, device=student_attn[0].device)
        n_layers = 0

        for s_attn, t_attn in zip(student_attn, teacher_attn):
            if isinstance(t_attn, torch.Tensor) and t_attn.numel() > 0:
                t_attn = t_attn.to(device=s_attn.device, dtype=s_attn.dtype)

                # Align shapes if needed (teacher may have different num_heads)
                if s_attn.shape != t_attn.shape:
                    # Average over heads dimension for both
                    if s_attn.ndim == 4:
                        s_avg = s_attn.mean(dim=1)  # (B, Q, K)
                    else:
                        s_avg = s_attn
                    if t_attn.ndim == 4:
                        t_avg = t_attn.mean(dim=1)
                    else:
                        t_avg = t_attn

                    # Adjust sequence lengths via interpolation
                    if s_avg.shape[-1] != t_avg.shape[-1]:
                        t_avg = F.interpolate(
                            t_avg.unsqueeze(1), size=s_avg.shape[-2:],
                            mode="bilinear", align_corners=False
                        ).squeeze(1)
                else:
                    s_avg = s_attn
                    t_avg = t_attn

                # Normalize to probability distributions
                s_prob = F.log_softmax(s_avg, dim=-1)
                t_prob = F.softmax(t_avg, dim=-1)

                kl = F.kl_div(s_prob, t_prob, reduction="batchmean")
                kl_total = kl_total + kl
                n_layers += 1

        if n_layers > 0:
            kl_total = kl_total / n_layers

        return kl_total

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        student_attention: Dict[str, List[torch.Tensor]] = None,
        teacher_attention: Dict[str, List[Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute token loss across all cameras.

        Args:
            pred: Dict mapping camera names to predicted tokens (B, Q, D).
            target: Dict mapping camera names to GT tokens (B, Q, D).
            student_attention: Dict mapping camera names to student attention maps.
            teacher_attention: Dict mapping camera names to teacher attention maps.

        Returns:
            total_loss: Scalar loss.
            metrics: Dict with individual loss components.
        """
        device = next(iter(pred.values())).device
        total_mse = torch.tensor(0.0, device=device)
        total_cosine = torch.tensor(0.0, device=device)
        total_kl = torch.tensor(0.0, device=device)
        n_cams = 0

        for cam in pred:
            if cam in target:
                p = pred[cam]
                t = target[cam]

                # Truncate/pad to match sizes, cast to float32 for stable loss
                min_tokens = min(p.shape[1], t.shape[1])
                p = p[:, :min_tokens].float()
                t = t[:, :min_tokens].float()

                total_mse = total_mse + self.mse_loss(p, t)
                cos = self.cos_sim(p, t).mean()
                total_cosine = total_cosine + (1.0 - cos)

                # Attention distillation
                if student_attention and teacher_attention:
                    s_attn = student_attention.get(cam, [])
                    t_attn = teacher_attention.get(cam, [])
                    if s_attn and t_attn:
                        total_kl = total_kl + self._compute_attention_kl(s_attn, t_attn)

                n_cams += 1

        if n_cams > 0:
            total_mse = total_mse / n_cams
            total_cosine = total_cosine / n_cams
            total_kl = total_kl / n_cams

        loss = (self.mse_weight * total_mse +
                self.cosine_weight * total_cosine +
                self.attention_kl_weight * total_kl)

        metrics = {
            "token_mse": total_mse.item(),
            "token_cosine": total_cosine.item(),
            "token_kl": total_kl.item(),
            "token_total": loss.item(),
        }
        return loss, metrics
