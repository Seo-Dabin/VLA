"""Training script for Image Adaptor V1.1 (Epipolar Cross-Attention + Fourier PE).

Simplified from V1: no curriculum, no image reconstruction.
Both token loss and depth auxiliary loss are active from epoch 0.
Depth and ViT label models are loaded simultaneously (~2.1GB total on 24GB 4090).

Usage:
    # Single GPU test
    python -m train.train_v1_1

    # 4-GPU DDP training
    torchrun --nproc_per_node=4 -m train.train_v1_1

    # Override config
    torchrun --nproc_per_node=4 -m train.train_v1_1 training.batch_size=4
"""

from __future__ import annotations

import gc
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .camera_models import NUSCENES_CAMERAS, PHYSICALAI_CAMERAS
from .dataset import ImageAdaptorDataset, collate_fn, PHYSICALAI_TARGET_CAMERAS
from .geometric_transform import GeometricTransform
from .losses.depth_loss import DepthLoss
from .losses.token_loss import TokenLoss
from .models.image_adaptor_v1_1 import ImageAdaptorV1_1


# ============================================================
# DDP Helpers
# ============================================================
def setup_ddp() -> Tuple[int, int, torch.device]:
    """Initialize DDP if launched via torchrun.

    Returns:
        rank: Process rank.
        world_size: Total number of processes.
        device: Assigned CUDA device.
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        return 0, 1, torch.device("cuda:0")


def cleanup_ddp() -> None:
    """Cleanup DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is rank 0.

    Returns:
        True if main process (rank 0).
    """
    return not dist.is_initialized() or dist.get_rank() == 0


# ============================================================
# Online Label Provider (V1.1: both models loaded simultaneously)
# ============================================================
class OnlineLabelProviderV1_1:
    """Label provider that loads both depth and ViT models simultaneously.

    Unlike V1's staged loading, V1.1 keeps both models loaded since
    both losses are active from epoch 0. Combined VRAM: ~2.1GB
    (100MB depth + 2GB ViT) fits easily on 24GB 4090.

    Args:
        device: CUDA device for label models.
        depth_model_name: HuggingFace model name for depth estimation.
        vlm_name: HuggingFace model name for visual token extraction.
        image_size: Target image size (H, W) for depth output.
    """

    def __init__(
        self,
        device: torch.device,
        depth_model_name: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
        vlm_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        image_size: Tuple[int, int] = (320, 576),
    ) -> None:
        self.device = device
        self.depth_model_name = depth_model_name
        self.vlm_name = vlm_name
        self.image_size = image_size

        self._depth_model: Optional[nn.Module] = None
        self._depth_processor: Optional[Any] = None
        self._visual_encoder: Optional[nn.Module] = None
        self._vlm_processor: Optional[Any] = None

        self._loaded = False

    def ensure_loaded(self) -> None:
        """Load both depth and ViT models if not already loaded."""
        if self._loaded:
            return

        # Load depth model
        if is_main_process():
            print(f"[LabelProvider] Loading depth model: {self.depth_model_name}")

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._depth_processor = AutoImageProcessor.from_pretrained(self.depth_model_name)
        self._depth_model = AutoModelForDepthEstimation.from_pretrained(
            self.depth_model_name, torch_dtype=torch.float32
        )
        self._depth_model.to(self.device)
        self._depth_model.eval()
        for p in self._depth_model.parameters():
            p.requires_grad = False

        # Load ViT
        if is_main_process():
            print(f"[LabelProvider] Loading ViT from: {self.vlm_name}")

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.vlm_name, torch_dtype=torch.bfloat16
        )
        self._vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_name,
            min_pixels=163840,
            max_pixels=196608,
        )

        if hasattr(qwen_model, "visual"):
            self._visual_encoder = qwen_model.visual.to(self.device)
        else:
            self._visual_encoder = qwen_model.model.visual.to(self.device)
        self._visual_encoder.eval()
        for p in self._visual_encoder.parameters():
            p.requires_grad = False

        del qwen_model
        gc.collect()
        torch.cuda.empty_cache()

        self._loaded = True
        if is_main_process():
            mem_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            print(f"[LabelProvider] Both models loaded ({mem_mb:.0f} MB GPU)")

    @torch.no_grad()
    def generate_depth_labels(
        self,
        images: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Generate depth maps for a batch of Physical AI images.

        Args:
            images: Dict mapping camera names to (B, 3, H, W) tensors in [0, 1].

        Returns:
            Dict mapping camera names to (B, 1, H, W) depth tensors.
        """
        self.ensure_loaded()
        H, W = self.image_size
        results: Dict[str, torch.Tensor] = {}

        for cam_name, img_batch in images.items():
            B = img_batch.shape[0]
            depths = []
            for i in range(B):
                img_np = (img_batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                from PIL import Image
                pil_img = Image.fromarray(img_np)

                inputs = self._depth_processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._depth_model(**inputs)
                depth = outputs.predicted_depth  # (1, h, w)

                depth = F.interpolate(
                    depth.unsqueeze(0), size=(H, W),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)  # (1, H, W)

                depths.append(depth)

            results[cam_name] = torch.stack(depths).to(img_batch.device)  # (B, 1, H, W)

        return results

    @torch.no_grad()
    def generate_token_labels(
        self,
        images: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """Generate visual tokens and attention maps for a batch of images.

        Args:
            images: Dict mapping camera names to (B, 3, H, W) tensors in [0, 1].

        Returns:
            tokens: Dict mapping camera names to (B, N_tokens, D) tensors.
            attention_maps: Dict mapping camera names to list of attn tensors.
        """
        self.ensure_loaded()
        all_tokens: Dict[str, torch.Tensor] = {}
        all_attn: Dict[str, List[torch.Tensor]] = {}

        for cam_name, img_batch in images.items():
            B = img_batch.shape[0]
            cam_tokens = []
            cam_attn: List[torch.Tensor] = []

            for i in range(B):
                img_np = (img_batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                from PIL import Image
                pil_img = Image.fromarray(img_np)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": "Describe."},
                        ],
                    }
                ]
                text = self._vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = self._vlm_processor(
                    text=[text], images=[pil_img],
                    return_tensors="pt", padding=True,
                )

                pixel_values = model_inputs["pixel_values"].to(self.device)
                image_grid_thw = model_inputs["image_grid_thw"].to(self.device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vision_output = self._visual_encoder(pixel_values, grid_thw=image_grid_thw)

                if hasattr(vision_output, "last_hidden_state"):
                    hidden = vision_output.last_hidden_state.detach()
                elif isinstance(vision_output, tuple):
                    hidden = vision_output[0].detach()
                else:
                    hidden = vision_output.detach()

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = self._visual_encoder.merger(hidden).detach()

                if tokens.ndim == 3:
                    tokens = tokens.squeeze(0)

                cam_tokens.append(tokens.float().cpu())
                cam_attn.append(torch.empty(0))

            all_tokens[cam_name] = torch.stack(cam_tokens).to(img_batch.device)
            all_attn[cam_name] = cam_attn

        return all_tokens, all_attn


# ============================================================
# Camera Parameter Helpers
# ============================================================
def build_camera_params(
    image_size: Tuple[int, int],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build source camera intrinsics and extrinsics tensors.

    V1.1 does not need target_cameras dict (handled by EpipolarCrossAttention
    with precomputed buffers).

    Args:
        image_size: Target image size (H, W).
        device: Computation device.
        batch_size: Batch size.

    Returns:
        source_intrinsics: (B, 5, 3, 3) NuScenes camera K matrices.
        source_extrinsics: (B, 5, 4, 4) NuScenes cam-to-ego transforms.
    """
    H, W = image_size
    nuscenes_cam_names = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    ]

    K_list = []
    E_list = []
    for cam_name in nuscenes_cam_names:
        cam = NUSCENES_CAMERAS[cam_name]
        scale_x = W / cam.width
        scale_y = H / cam.height
        K = torch.tensor([
            [cam.fx * scale_x, 0, cam.cx * scale_x],
            [0, cam.fy * scale_y, cam.cy * scale_y],
            [0, 0, 1],
        ], dtype=torch.float32)
        K_list.append(K)

        R = cam.rotation_matrix.float()
        t = cam.translation.float()
        E = torch.eye(4, dtype=torch.float32)
        E[:3, :3] = R
        E[:3, 3] = t
        E_list.append(E)

    source_intrinsics = torch.stack(K_list).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
    source_extrinsics = torch.stack(E_list).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

    return source_intrinsics, source_extrinsics


# ============================================================
# TensorBoard Visualization
# ============================================================
def log_tb_images(
    writer: SummaryWriter,
    epoch: int,
    batch: Dict[str, Any],
    outputs: Dict[str, Any],
    nuscenes_images: torch.Tensor,
    depth_labels: Optional[Dict[str, torch.Tensor]] = None,
    token_labels: Optional[Dict[str, torch.Tensor]] = None,
    tag_prefix: str = "viz",
    log_inputs: bool = True,
) -> None:
    """Log images and visualizations to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number.
        batch: Data batch dictionary.
        outputs: Model output dictionary.
        nuscenes_images: NuScenes transformed images (B, 5, 3, H, W).
        depth_labels: Online-generated depth labels (optional).
        token_labels: Online-generated token labels (optional).
        tag_prefix: Tag prefix for TensorBoard.
        log_inputs: Whether to log input images.
    """
    if log_inputs:
        for cam_name in PHYSICALAI_TARGET_CAMERAS:
            if cam_name in batch["physicalai_images"]:
                img = batch["physicalai_images"][cam_name][0]
                writer.add_image(f"{tag_prefix}/input_physicalai/{cam_name}", img, epoch)

        cam_names_ns = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                       "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        for cam_idx in range(min(nuscenes_images.shape[1], 5)):
            img = nuscenes_images[0, cam_idx]
            writer.add_image(f"{tag_prefix}/input_nuscenes/{cam_names_ns[cam_idx]}", img, epoch)

    # Depth predictions vs GT (at feature resolution 20x36)
    if "depth_preds" in outputs:
        for cam_name, depth_pred in outputs["depth_preds"].items():
            depth_vis = depth_pred[0, 0]
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
            writer.add_image(f"{tag_prefix}/depth_pred/{cam_name}", depth_vis.unsqueeze(0), epoch)

            if log_inputs and depth_labels and cam_name in depth_labels:
                depth_gt = depth_labels[cam_name][0, 0]
                depth_gt_vis = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min() + 1e-8)
                writer.add_image(f"{tag_prefix}/depth_gt/{cam_name}", depth_gt_vis.unsqueeze(0), epoch)

    # Feature map visualization
    if "feature_maps" in outputs:
        for cam_name, feat in outputs["feature_maps"].items():
            _log_feature_map(writer, epoch, feat[0], f"{tag_prefix}/feature_map/{cam_name}")

    # Epipolar attention visualization
    if "epipolar_attention" in outputs:
        for cam_name, attn_list in outputs["epipolar_attention"].items():
            if attn_list:
                _log_epipolar_attention(
                    writer, epoch, attn_list[-1],  # last layer
                    f"{tag_prefix}/epipolar_attn/{cam_name}",
                )

    # Token t-SNE
    if "token_preds" in outputs and token_labels:
        for cam_name in PHYSICALAI_TARGET_CAMERAS:
            if cam_name in outputs["token_preds"] and cam_name in token_labels:
                pred_t = outputs["token_preds"][cam_name][0].detach().float().cpu().numpy()
                gt_t = token_labels[cam_name][0].detach().float().cpu().numpy()
                _log_tsne(writer, epoch, pred_t, gt_t, f"{tag_prefix}/tsne/{cam_name}")


def _log_feature_map(
    writer: SummaryWriter,
    epoch: int,
    feature_map: torch.Tensor,
    tag: str,
) -> None:
    """Log feature map visualizations to TensorBoard.

    Generates three views: PCA RGB, channel mean, channel std.

    Args:
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number.
        feature_map: Feature map tensor, shape (C, H, W).
        tag: TensorBoard tag prefix.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        feat = feature_map.detach().float().cpu()
        C, H, W = feat.shape

        # Channel mean heatmap
        mean_map = feat.mean(dim=0)
        mean_norm = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-8)
        fig_mean, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(mean_norm.numpy(), cmap="viridis", aspect="auto")
        ax.set_title(f"Feature Mean (epoch {epoch})")
        ax.axis("off")
        fig_mean.colorbar(im, ax=ax, fraction=0.046)
        fig_mean.tight_layout()
        writer.add_figure(f"{tag}/mean", fig_mean, epoch)
        plt.close(fig_mean)

        # Channel std heatmap
        std_map = feat.std(dim=0)
        std_norm = (std_map - std_map.min()) / (std_map.max() - std_map.min() + 1e-8)
        fig_std, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(std_norm.numpy(), cmap="inferno", aspect="auto")
        ax.set_title(f"Feature Std (epoch {epoch})")
        ax.axis("off")
        fig_std.colorbar(im, ax=ax, fraction=0.046)
        fig_std.tight_layout()
        writer.add_figure(f"{tag}/std", fig_std, epoch)
        plt.close(fig_std)

        # PCA RGB
        flat = feat.reshape(C, -1).T
        flat_centered = flat - flat.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(flat_centered, full_matrices=False)
        pca_3 = U[:, :3] * S[:3]
        for i in range(3):
            col = pca_3[:, i]
            pca_3[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        pca_img = pca_3.reshape(H, W, 3).numpy()

        fig_pca, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.imshow(pca_img, aspect="auto")
        ax.set_title(f"Feature PCA RGB (epoch {epoch})")
        ax.axis("off")
        fig_pca.tight_layout()
        writer.add_figure(f"{tag}/pca_rgb", fig_pca, epoch)
        plt.close(fig_pca)
    except Exception:
        pass


def _log_epipolar_attention(
    writer: SummaryWriter,
    epoch: int,
    attn_weights: torch.Tensor,
    tag: str,
) -> None:
    """Log epipolar attention pattern visualization.

    Shows attention distribution over epipolar samples for representative
    target pixels (center, top-left, bottom-right).

    Args:
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number.
        attn_weights: Attention weights, shape (B, n_heads, Q, K).
        tag: TensorBoard tag.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Average over batch and heads: (Q, K)
        attn = attn_weights[0].mean(dim=0).detach().float().cpu()  # (Q, K)
        Q, K = attn.shape

        # Select representative pixels
        tH, tW = 20, 36
        center = (tH // 2) * tW + tW // 2
        top_left = 0
        bottom_right = Q - 1
        pixels = {"center": center, "top_left": top_left, "bottom_right": bottom_right}

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (name, idx) in zip(axes, pixels.items()):
            ax.bar(range(K), attn[idx].numpy(), width=1.0)
            ax.set_title(f"Pixel {name} (idx={idx})")
            ax.set_xlabel("Epipolar sample index")
            ax.set_ylabel("Attention weight")
        fig.suptitle(f"Epipolar Attention Pattern (epoch {epoch})")
        fig.tight_layout()
        writer.add_figure(tag, fig, epoch)
        plt.close(fig)
    except Exception:
        pass


def _log_tsne(
    writer: SummaryWriter,
    epoch: int,
    pred_tokens: np.ndarray,
    gt_tokens: np.ndarray,
    tag: str,
) -> None:
    """Log t-SNE visualization of predicted vs GT tokens.

    Args:
        writer: TensorBoard SummaryWriter.
        epoch: Current epoch number.
        pred_tokens: Predicted tokens (N, D).
        gt_tokens: Ground truth tokens (N, D).
        tag: TensorBoard tag.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        n_pred = min(pred_tokens.shape[0], 100)
        n_gt = min(gt_tokens.shape[0], 100)

        combined = np.concatenate([pred_tokens[:n_pred], gt_tokens[:n_gt]], axis=0)
        if combined.shape[0] < 5:
            return

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, combined.shape[0] - 1))
        embedded = tsne.fit_transform(combined)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(embedded[:n_pred, 0], embedded[:n_pred, 1], c="blue", alpha=0.5, label="Pred", s=10)
        ax.scatter(embedded[n_pred:, 0], embedded[n_pred:, 1], c="red", alpha=0.5, label="GT", s=10)
        ax.legend()
        ax.set_title(f"Token t-SNE (epoch {epoch})")
        writer.add_figure(tag, fig, epoch)
        plt.close(fig)
    except Exception:
        pass


# ============================================================
# Training / Validation Steps
# ============================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    depth_criterion: DepthLoss,
    token_criterion: TokenLoss,
    geometric_transform: GeometricTransform,
    label_provider: OnlineLabelProviderV1_1,
    device: torch.device,
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    writer: Optional[SummaryWriter],
    epoch: int,
    global_step: int,
    cfg: DictConfig,
) -> Tuple[float, int, Dict[str, float]]:
    """Run one V1.1 training epoch.

    Args:
        model: Image Adaptor V1.1 model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        depth_criterion: Depth loss function.
        token_criterion: Token loss function.
        geometric_transform: Geometric transform module.
        label_provider: Online label provider (both models loaded).
        device: Computation device.
        image_size: Image size (H, W).
        feature_size: Feature map size (fH, fW).
        writer: TensorBoard writer (None for non-main processes).
        epoch: Current epoch number.
        global_step: Global step counter.
        cfg: Hydra config.

    Returns:
        avg_loss: Average epoch loss.
        global_step: Updated global step.
        metrics: Aggregated loss metrics.
    """
    model.train()
    total_loss = 0.0
    total_metrics: Dict[str, float] = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}",
                disable=not is_main_process())

    for batch in pbar:
        physicalai_images = {
            cam: batch["physicalai_images"][cam].to(device) for cam in PHYSICALAI_TARGET_CAMERAS
        }
        B = physicalai_images["front_wide"].shape[0]

        # Online geometric transform
        nuscenes_images = geometric_transform(physicalai_images)

        # Build source camera parameters (no target cameras needed in V1.1)
        source_K, source_E = build_camera_params(image_size, device, B)

        # Generate online labels (both depth and tokens)
        depth_labels = label_provider.generate_depth_labels(physicalai_images)
        visual_tokens, attention_maps = label_provider.generate_token_labels(physicalai_images)

        # Forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                nuscenes_images, source_K, source_E,
                capture_attention=True,
            )

        # Token loss (main)
        token_loss, token_metrics = token_criterion(
            outputs["token_preds"], visual_tokens,
            student_attention=outputs.get("student_attention"),
            teacher_attention=attention_maps,
        )

        # Depth loss (auxiliary, at feature resolution)
        # Downsample depth labels to feature resolution
        fH, fW = feature_size
        depth_labels_downsampled: Dict[str, torch.Tensor] = {}
        for cam_name, depth in depth_labels.items():
            depth_labels_downsampled[cam_name] = F.interpolate(
                depth, size=(fH, fW), mode="bilinear", align_corners=False,
            )

        depth_loss, depth_metrics = depth_criterion(
            outputs["depth_preds"], depth_labels_downsampled
        )

        # Combined loss with weights from config
        depth_weight = cfg.loss_weights.depth_l1 + cfg.loss_weights.depth_silog
        loss = token_loss + depth_weight * depth_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip_norm)
        optimizer.step()

        step_metrics = {}
        step_metrics.update(token_metrics)
        step_metrics.update(depth_metrics)
        step_metrics["total_loss"] = loss.item()

        total_loss += loss.item()
        for k, v in step_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1
        global_step += 1

        if writer and global_step % 10 == 0:
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            for k, v in step_metrics.items():
                writer.add_scalar(f"train/{k}_step", v, global_step)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            tok=f"{token_metrics.get('token_total', 0):.3f}",
            dep=f"{depth_metrics.get('depth_total', 0):.4f}",
        )

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, global_step, avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    depth_criterion: DepthLoss,
    token_criterion: TokenLoss,
    geometric_transform: GeometricTransform,
    label_provider: OnlineLabelProviderV1_1,
    device: torch.device,
    image_size: Tuple[int, int],
    feature_size: Tuple[int, int],
    cfg: DictConfig,
) -> Tuple[float, Dict[str, float]]:
    """Run V1.1 validation.

    Args:
        model: Image Adaptor V1.1 model.
        dataloader: Validation data loader.
        depth_criterion: Depth loss function.
        token_criterion: Token loss function.
        geometric_transform: Geometric transform module.
        label_provider: Online label provider.
        device: Computation device.
        image_size: Image size (H, W).
        feature_size: Feature map size (fH, fW).
        cfg: Hydra config.

    Returns:
        avg_loss: Average validation loss.
        metrics: Aggregated validation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_metrics: Dict[str, float] = {}
    num_batches = 0

    for batch in tqdm(dataloader, desc="Val", disable=not is_main_process()):
        physicalai_images = {
            cam: batch["physicalai_images"][cam].to(device) for cam in PHYSICALAI_TARGET_CAMERAS
        }
        B = physicalai_images["front_wide"].shape[0]
        nuscenes_images = geometric_transform(physicalai_images)
        source_K, source_E = build_camera_params(image_size, device, B)

        # Generate labels
        depth_labels = label_provider.generate_depth_labels(physicalai_images)
        visual_tokens, _ = label_provider.generate_token_labels(physicalai_images)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(nuscenes_images, source_K, source_E, capture_attention=False)

        # Token loss
        token_loss, token_metrics = token_criterion(outputs["token_preds"], visual_tokens)

        # Depth loss (downsampled)
        fH, fW = feature_size
        depth_labels_down = {
            cam: F.interpolate(d, size=(fH, fW), mode="bilinear", align_corners=False)
            for cam, d in depth_labels.items()
        }
        depth_loss, depth_metrics = depth_criterion(outputs["depth_preds"], depth_labels_down)

        depth_weight = cfg.loss_weights.depth_l1 + cfg.loss_weights.depth_silog
        loss = token_loss + depth_weight * depth_loss

        step_metrics = {}
        step_metrics.update(token_metrics)
        step_metrics.update(depth_metrics)

        total_loss += loss.item()
        for k, v in step_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, avg_metrics


# ============================================================
# Main
# ============================================================
@hydra.main(config_path="../config", config_name="v1_1", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main V1.1 training entry point.

    Args:
        cfg: Hydra configuration.
    """
    rank, world_size, device = setup_ddp()

    if is_main_process():
        print("=" * 70)
        print("Image Adaptor V1.1 Training (Epipolar Cross-Attention + Fourier PE)")
        print(f"  GPUs: {world_size}, per-GPU batch: {cfg.training.batch_size}")
        print(f"  Effective batch size: {cfg.training.batch_size * world_size}")
        print("=" * 70)
        print(OmegaConf.to_yaml(cfg))

    seed = cfg.training.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    image_size = tuple(cfg.data.image_size)
    feature_size = (image_size[0] // 16, image_size[1] // 16)  # (20, 36)

    # Output directory
    if is_main_process():
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        checkpoint_dir = os.path.join(cfg.checkpoint.checkpoint_dir, timestamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(checkpoint_dir, "config.yaml"))
        print(f"Output directory: {checkpoint_dir}")
    else:
        checkpoint_dir = None

    if dist.is_initialized():
        obj_list = [checkpoint_dir] if rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        checkpoint_dir = obj_list[0]

    # Dataset
    train_dataset = ImageAdaptorDataset(
        data_root=cfg.data.physical_ai_root,
        split="train",
        image_size=image_size,
        max_clips=cfg.data.get("max_clips", None),
    )
    val_dataset = ImageAdaptorDataset(
        data_root=cfg.data.physical_ai_root,
        split="val",
        image_size=image_size,
        max_clips=cfg.data.get("max_clips", None),
    )

    fixed_sample_indices = train_dataset.get_fixed_samples(n_samples=5)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if is_main_process():
        print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Geometric transform
    geometric_transform = GeometricTransform(target_size=image_size).to(device)

    # Online label provider (both models loaded simultaneously)
    label_provider = OnlineLabelProviderV1_1(
        device=device,
        depth_model_name=cfg.label_models.depth_model,
        vlm_name=cfg.label_models.vlm_model,
        image_size=image_size,
    )

    # Model
    target_cams = list(cfg.model.target_cameras)
    ea_cfg = cfg.model.epipolar_attention
    td_cfg = cfg.model.token_decoder

    model = ImageAdaptorV1_1(
        backbone_pretrained=cfg.model.backbone_pretrained,
        backbone_out_channels=cfg.model.backbone_out_channels,
        fourier_L=cfg.model.pe.fourier_L,
        epipolar_d_model=ea_cfg.d_model,
        epipolar_n_heads=ea_cfg.n_heads,
        epipolar_n_layers=ea_cfg.n_layers,
        epipolar_n_samples=ea_cfg.n_samples,
        epipolar_ffn_dim=ea_cfg.ffn_dim,
        epipolar_dropout=ea_cfg.dropout,
        epipolar_depth_range=tuple(ea_cfg.depth_range),
        target_cameras=target_cams,
        target_size=feature_size,
        source_size=feature_size,
        token_d_model=td_cfg.d_model,
        token_num_layers=td_cfg.num_layers,
        token_num_heads=td_cfg.num_heads,
        num_query_tokens=td_cfg.num_query_tokens,
        token_output_dim=td_cfg.output_dim,
    ).to(device)

    if is_main_process():
        print(f"Model parameters: {model.num_parameters:,}")
        print(f"Trainable parameters: {model.num_trainable_parameters:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    raw_model = model.module if world_size > 1 else model

    # Loss functions
    depth_criterion = DepthLoss(
        l1_weight=cfg.loss_weights.depth_l1,
        silog_weight=cfg.loss_weights.depth_silog,
    )
    token_criterion = TokenLoss(
        mse_weight=cfg.loss_weights.token_mse,
        cosine_weight=cfg.loss_weights.token_cosine,
        attention_kl_weight=cfg.loss_weights.token_attention_kl,
    )

    # Optimizer (all parameters from epoch 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )

    # Learning rate scheduler with warmup
    warmup_epochs = cfg.training.warmup_epochs
    total_epochs = cfg.training.total_epochs

    def lr_lambda(epoch: int) -> float:
        """Warmup + cosine decay schedule.

        Args:
            epoch: Current epoch.

        Returns:
            Learning rate multiplier.
        """
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    resume_path = cfg.checkpoint.get("resume_from", None)

    if resume_path and os.path.isfile(resume_path):
        if is_main_process():
            print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            if is_main_process():
                print("  Warning: Could not restore optimizer state")
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except (ValueError, KeyError):
            if is_main_process():
                print("  Warning: Could not restore scheduler state")
        if is_main_process():
            print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        del ckpt
        torch.cuda.empty_cache()

    # Pre-load label models
    label_provider.ensure_loaded()

    # TensorBoard
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tb_logs"))

    if is_main_process():
        print("\n" + "=" * 70)
        print(f"Starting Training (epoch {start_epoch + 1} / {total_epochs})")
        print("=" * 70)

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss, global_step, train_metrics = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            depth_criterion=depth_criterion, token_criterion=token_criterion,
            geometric_transform=geometric_transform, label_provider=label_provider,
            device=device, image_size=image_size, feature_size=feature_size,
            writer=writer, epoch=epoch, global_step=global_step, cfg=cfg,
        )

        avg_val_loss, val_metrics = validate(
            model=model, dataloader=val_loader,
            depth_criterion=depth_criterion, token_criterion=token_criterion,
            geometric_transform=geometric_transform, label_provider=label_provider,
            device=device, image_size=image_size, feature_size=feature_size, cfg=cfg,
        )

        if world_size > 1:
            loss_tensor = torch.tensor([avg_val_loss], device=device)
            dist.all_reduce(loss_tensor)
            avg_val_loss = loss_tensor.item() / world_size

        # TensorBoard
        if writer:
            writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
            writer.add_scalar("val/loss_epoch", avg_val_loss, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # Image logging
        if writer and (epoch == 0 or (epoch + 1) % cfg.checkpoint.log_image_every_n_epochs == 0):
            fixed_batch = _get_fixed_batch(train_dataset, fixed_sample_indices, device)
            if fixed_batch is not None:
                B_fix = fixed_batch["physicalai_images"]["front_wide"].shape[0]
                ns_imgs = geometric_transform(fixed_batch["physicalai_images"])
                src_K, src_E = build_camera_params(image_size, device, B_fix)

                # Generate labels for viz
                viz_depth = label_provider.generate_depth_labels(fixed_batch["physicalai_images"])
                viz_tokens, _ = label_provider.generate_token_labels(fixed_batch["physicalai_images"])

                # Downsample depth for viz comparison
                fH, fW = feature_size
                viz_depth_down = {
                    cam: F.interpolate(d, size=(fH, fW), mode="bilinear", align_corners=False)
                    for cam, d in viz_depth.items()
                }

                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    viz_out = (raw_model if world_size == 1 else model)(
                        ns_imgs, src_K, src_E,
                    )
                model.train()

                log_tb_images(
                    writer, epoch, fixed_batch, viz_out, ns_imgs,
                    depth_labels=viz_depth_down, token_labels=viz_tokens,
                    log_inputs=(epoch == 0),
                )

                del fixed_batch, ns_imgs, src_K, src_E
                del viz_depth, viz_depth_down, viz_tokens, viz_out
                torch.cuda.empty_cache()

        if is_main_process():
            print(
                f"Epoch {epoch + 1}: train={avg_train_loss:.4f}, "
                f"val={avg_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        scheduler.step()

        # Checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if is_main_process() and (epoch + 1) % cfg.checkpoint.save_every_n_epochs == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "config": OmegaConf.to_container(cfg),
            }
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(ckpt, best_path)
                print(f"Saved best model: {best_path}")

        if dist.is_initialized():
            dist.barrier()

    if writer:
        writer.close()
    if is_main_process():
        print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")
        print(f"Checkpoints: {checkpoint_dir}")
    cleanup_ddp()


def _get_fixed_batch(
    dataset: ImageAdaptorDataset,
    indices: List[int],
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Get a fixed batch from dataset for visualization.

    Args:
        dataset: Training dataset.
        indices: Fixed sample indices.
        device: Computation device.

    Returns:
        Collated batch on device, or None if failed.
    """
    try:
        samples = [dataset[idx] for idx in indices]
        batch = collate_fn(samples)
        batch["physicalai_images"] = {
            cam: batch["physicalai_images"][cam].to(device)
            for cam in PHYSICALAI_TARGET_CAMERAS
        }
        return batch
    except Exception as e:
        print(f"Warning: Failed to load fixed batch: {e}")
        return None


if __name__ == "__main__":
    main()
