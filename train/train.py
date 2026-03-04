"""Main training script for Image Adaptor V1 (fully online).

All labels (depth maps, visual tokens) are generated online during training.
VRAM is managed by loading only the required label provider per stage:
  - Stage 1-2: Depth-Anything-V2-Small on GPU (~100MB)
  - Stage 3: Unload depth model, load Qwen3-VL-2B ViT (~2GB)

Usage:
    # Single GPU test
    python -m train.train

    # 4-GPU DDP training
    torchrun --nproc_per_node=4 -m train.train

    # Override config
    torchrun --nproc_per_node=4 -m train.train training.batch_size=4 training.lr=2e-4
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
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .camera_models import NUSCENES_CAMERAS, PHYSICALAI_CAMERAS
from .curriculum import CurriculumController
from .dataset import ImageAdaptorDataset, collate_fn, PHYSICALAI_TARGET_CAMERAS
from .geometric_transform import GeometricTransform
from .losses.depth_loss import DepthLoss
from .losses.image_loss import ImageLoss
from .losses.token_loss import TokenLoss
from .models.image_adaptor import ImageAdaptorV1


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
# Online Label Provider (VRAM-managed)
# ============================================================
class OnlineLabelProvider:
    """Manages label-generating models with per-stage VRAM lifecycle.

    Loads/unloads depth model and ViT based on current training stage
    so they never coexist on GPU simultaneously.

    Stage 1-2: Depth-Anything-V2-Small loaded (~100MB VRAM)
    Stage 3:   Depth model freed, Qwen3-VL-2B ViT loaded (~2GB VRAM)

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

        # Model references (None = not loaded)
        self._depth_model: Optional[nn.Module] = None
        self._depth_processor: Optional[Any] = None
        self._visual_encoder: Optional[nn.Module] = None
        self._vlm_processor: Optional[Any] = None

        self._current_mode: Optional[str] = None  # "depth" or "vit"

    def ensure_depth_model(self) -> None:
        """Load depth model to GPU if not already loaded."""
        if self._current_mode == "depth":
            return

        # Unload ViT first if loaded
        if self._current_mode == "vit":
            self._unload_vit()

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

        self._current_mode = "depth"
        if is_main_process():
            mem_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            print(f"[LabelProvider] Depth model loaded ({mem_mb:.0f} MB GPU)")

    def ensure_vit_model(self) -> None:
        """Load Qwen3-VL-2B ViT to GPU if not already loaded."""
        if self._current_mode == "vit":
            return

        # Unload depth model first if loaded
        if self._current_mode == "depth":
            self._unload_depth()

        if is_main_process():
            print(f"[LabelProvider] Loading ViT from: {self.vlm_name}")

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        # Load full model on CPU, extract visual encoder, delete the rest
        qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.vlm_name, torch_dtype=torch.bfloat16
        )
        self._vlm_processor = AutoProcessor.from_pretrained(
            self.vlm_name,
            min_pixels=163840,  # from helper.py
            max_pixels=196608,
        )

        self._visual_encoder = qwen_model.visual.to(self.device)
        self._visual_encoder.eval()
        for p in self._visual_encoder.parameters():
            p.requires_grad = False

        del qwen_model
        gc.collect()
        torch.cuda.empty_cache()

        self._current_mode = "vit"
        if is_main_process():
            mem_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            print(f"[LabelProvider] ViT loaded ({mem_mb:.0f} MB GPU)")

    def _unload_depth(self) -> None:
        """Free depth model from GPU."""
        if is_main_process():
            print("[LabelProvider] Unloading depth model")
        del self._depth_model
        del self._depth_processor
        self._depth_model = None
        self._depth_processor = None
        self._current_mode = None
        gc.collect()
        torch.cuda.empty_cache()

    def _unload_vit(self) -> None:
        """Free ViT from GPU."""
        if is_main_process():
            print("[LabelProvider] Unloading ViT")
        del self._visual_encoder
        del self._vlm_processor
        self._visual_encoder = None
        self._vlm_processor = None
        self._current_mode = None
        gc.collect()
        torch.cuda.empty_cache()

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
        self.ensure_depth_model()
        H, W = self.image_size
        results: Dict[str, torch.Tensor] = {}

        for cam_name, img_batch in images.items():
            B = img_batch.shape[0]
            depths = []
            for i in range(B):
                # Convert to PIL for processor
                img_np = (img_batch[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                from PIL import Image
                pil_img = Image.fromarray(img_np)

                inputs = self._depth_processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self._depth_model(**inputs)
                depth = outputs.predicted_depth  # (1, h, w)

                # Resize to target size
                depth = torch.nn.functional.interpolate(
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
            attention_maps: Dict mapping camera names to list of attn tensors per sample.
        """
        self.ensure_vit_model()
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

                # Build minimal message for Qwen3-VL processor
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

                if isinstance(vision_output, tuple):
                    tokens = vision_output[0].detach()
                else:
                    tokens = vision_output.detach()

                if tokens.ndim == 3:
                    tokens = tokens.squeeze(0)  # (N_tokens, D)

                cam_tokens.append(tokens.float().cpu())
                # Placeholder for attention maps (capture if needed later)
                cam_attn.append(torch.empty(0))

            # Stack: (B, N_tokens, D)
            all_tokens[cam_name] = torch.stack(cam_tokens).to(img_batch.device)
            all_attn[cam_name] = cam_attn

        return all_tokens, all_attn

    def prepare_for_stage(self, stage: int) -> None:
        """Pre-load the appropriate model for a training stage.

        Args:
            stage: Curriculum stage (1, 2, or 3).
        """
        if stage in (1, 2):
            self.ensure_depth_model()
        elif stage == 3:
            self.ensure_vit_model()


# ============================================================
# Camera Parameter Helpers
# ============================================================
def build_camera_params(
    image_size: Tuple[int, int],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """Build camera intrinsics and extrinsics tensors.

    Args:
        image_size: Target image size (H, W).
        device: Computation device.
        batch_size: Batch size.

    Returns:
        source_intrinsics: (B, 5, 3, 3) NuScenes camera K matrices.
        source_extrinsics: (B, 5, 4, 4) NuScenes cam-to-ego transforms.
        target_cameras: Dict mapping PhysicalAI cam names to (K, E) tuples.
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

    target_cameras = {}
    for cam_name in PHYSICALAI_TARGET_CAMERAS:
        cam = PHYSICALAI_CAMERAS[cam_name]
        K_tgt = torch.tensor([
            [cam.forward_poly[1].item(), 0, cam.cx],
            [0, cam.forward_poly[1].item(), cam.cy],
            [0, 0, 1],
        ], dtype=torch.float32)
        scale_x_tgt = W / cam.width
        scale_y_tgt = H / cam.height
        K_tgt[0, 0] *= scale_x_tgt
        K_tgt[0, 2] *= scale_x_tgt
        K_tgt[1, 1] *= scale_y_tgt
        K_tgt[1, 2] *= scale_y_tgt

        R_tgt = cam.rotation_matrix.float()
        t_tgt = cam.translation.float()
        E_tgt = torch.eye(4, dtype=torch.float32)
        E_tgt[:3, :3] = R_tgt.T
        E_tgt[:3, 3] = -R_tgt.T @ t_tgt

        K_tgt = K_tgt.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        E_tgt = E_tgt.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        target_cameras[cam_name] = (K_tgt, E_tgt)

    return source_intrinsics, source_extrinsics, target_cameras


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

    # Depth predictions vs GT
    if "depth_preds" in outputs:
        for cam_name, depth_pred in outputs["depth_preds"].items():
            depth_vis = depth_pred[0, 0]
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
            writer.add_image(f"{tag_prefix}/depth_pred/{cam_name}", depth_vis.unsqueeze(0), epoch)

            if log_inputs and depth_labels and cam_name in depth_labels:
                depth_gt = depth_labels[cam_name][0, 0]
                depth_gt_vis = (depth_gt - depth_gt.min()) / (depth_gt.max() - depth_gt.min() + 1e-8)
                writer.add_image(f"{tag_prefix}/depth_gt/{cam_name}", depth_gt_vis.unsqueeze(0), epoch)

    # Image reconstructions
    if "image_preds" in outputs:
        for cam_name, img_pred in outputs["image_preds"].items():
            writer.add_image(f"{tag_prefix}/image_pred/{cam_name}", img_pred[0], epoch)

    # Token t-SNE
    if "token_preds" in outputs and token_labels:
        for cam_name in PHYSICALAI_TARGET_CAMERAS:
            if cam_name in outputs["token_preds"] and cam_name in token_labels:
                pred_t = outputs["token_preds"][cam_name][0].detach().cpu().numpy()
                gt_t = token_labels[cam_name][0].detach().cpu().numpy()
                _log_tsne(writer, epoch, pred_t, gt_t, f"{tag_prefix}/tsne/{cam_name}")


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
    image_criterion: ImageLoss,
    token_criterion: TokenLoss,
    curriculum: CurriculumController,
    geometric_transform: GeometricTransform,
    label_provider: OnlineLabelProvider,
    device: torch.device,
    image_size: Tuple[int, int],
    writer: Optional[SummaryWriter],
    epoch: int,
    global_step: int,
    cfg: DictConfig,
) -> Tuple[float, int, Dict[str, float]]:
    """Run one training epoch with online label generation.

    Args:
        model: Image Adaptor model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        depth_criterion: Depth loss function.
        image_criterion: Image loss function.
        token_criterion: Token loss function.
        curriculum: Curriculum controller.
        geometric_transform: Geometric transform module.
        label_provider: Online label provider.
        device: Computation device.
        image_size: Image size (H, W).
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
    active_stages = curriculum.active_stages

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Stage {curriculum.current_stage}]",
                disable=not is_main_process())

    for batch in pbar:
        physicalai_images = {
            cam: batch["physicalai_images"][cam].to(device) for cam in PHYSICALAI_TARGET_CAMERAS
        }
        B = physicalai_images["front_wide"].shape[0]

        # Online geometric transform
        nuscenes_images = geometric_transform(physicalai_images)

        # Build camera parameters
        source_K, source_E, target_cams = build_camera_params(image_size, device, B)

        # Generate online labels for active stages
        depth_labels = None
        visual_tokens = None
        attention_maps = None

        if 1 in active_stages:
            depth_labels = label_provider.generate_depth_labels(physicalai_images)

        if 3 in active_stages:
            visual_tokens, attention_maps = label_provider.generate_token_labels(physicalai_images)

        # Forward pass
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                nuscenes_images, source_K, source_E, target_cams,
                active_stages=active_stages,
                capture_attention=(3 in active_stages),
            )

        # Compute losses
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        step_metrics = {}

        if 1 in active_stages and "depth_preds" in outputs and depth_labels is not None:
            d_loss, d_m = depth_criterion(outputs["depth_preds"], depth_labels)
            loss = loss + d_loss
            step_metrics.update(d_m)

        if 2 in active_stages and "image_preds" in outputs:
            i_loss, i_m = image_criterion(outputs["image_preds"], physicalai_images)
            loss = loss + i_loss
            step_metrics.update(i_m)

        if 3 in active_stages and "token_preds" in outputs and visual_tokens is not None:
            t_loss, t_m = token_criterion(
                outputs["token_preds"], visual_tokens,
                student_attention=outputs.get("student_attention"),
                teacher_attention=attention_maps,
            )
            loss = loss + t_loss
            step_metrics.update(t_m)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        for k, v in step_metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        num_batches += 1
        global_step += 1

        if writer and global_step % 10 == 0:
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            for k, v in step_metrics.items():
                writer.add_scalar(f"train/{k}_step", v, global_step)

        pbar.set_postfix(loss=f"{loss.item():.4f}", stage=curriculum.current_stage)

    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_loss, global_step, avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    depth_criterion: DepthLoss,
    image_criterion: ImageLoss,
    token_criterion: TokenLoss,
    curriculum: CurriculumController,
    geometric_transform: GeometricTransform,
    label_provider: OnlineLabelProvider,
    device: torch.device,
    image_size: Tuple[int, int],
) -> Tuple[float, Dict[str, float]]:
    """Run validation with online label generation.

    Args:
        model: Image Adaptor model.
        dataloader: Validation data loader.
        depth_criterion: Depth loss function.
        image_criterion: Image loss function.
        token_criterion: Token loss function.
        curriculum: Curriculum controller.
        geometric_transform: Geometric transform module.
        label_provider: Online label provider.
        device: Computation device.
        image_size: Image size (H, W).

    Returns:
        avg_loss: Average validation loss.
        metrics: Aggregated validation metrics.
    """
    model.eval()
    total_loss = 0.0
    total_metrics: Dict[str, float] = {}
    num_batches = 0
    active_stages = curriculum.active_stages

    for batch in tqdm(dataloader, desc="Val", disable=not is_main_process()):
        physicalai_images = {
            cam: batch["physicalai_images"][cam].to(device) for cam in PHYSICALAI_TARGET_CAMERAS
        }
        B = physicalai_images["front_wide"].shape[0]
        nuscenes_images = geometric_transform(physicalai_images)
        source_K, source_E, target_cams = build_camera_params(image_size, device, B)

        # Online labels
        depth_labels = None
        visual_tokens = None

        if 1 in active_stages:
            depth_labels = label_provider.generate_depth_labels(physicalai_images)
        if 3 in active_stages:
            visual_tokens, _ = label_provider.generate_token_labels(physicalai_images)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                nuscenes_images, source_K, source_E, target_cams,
                active_stages=active_stages, capture_attention=False,
            )

        loss = torch.tensor(0.0, device=device)
        step_metrics = {}

        if 1 in active_stages and "depth_preds" in outputs and depth_labels is not None:
            d_loss, d_m = depth_criterion(outputs["depth_preds"], depth_labels)
            loss = loss + d_loss
            step_metrics.update(d_m)

        if 2 in active_stages and "image_preds" in outputs:
            i_loss, i_m = image_criterion(outputs["image_preds"], physicalai_images)
            loss = loss + i_loss
            step_metrics.update(i_m)

        if 3 in active_stages and "token_preds" in outputs and visual_tokens is not None:
            t_loss, t_m = token_criterion(outputs["token_preds"], visual_tokens)
            loss = loss + t_loss
            step_metrics.update(t_m)

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
@hydra.main(config_path="../config", config_name="v1", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training entry point (fully online, no precomputation).

    Args:
        cfg: Hydra configuration.
    """
    rank, world_size, device = setup_ddp()

    if is_main_process():
        print("=" * 70)
        print("Image Adaptor V1 Training (Online Mode)")
        print(f"  GPUs: {world_size}, per-GPU batch: {cfg.training.batch_size}")
        print(f"  Effective batch size: {cfg.training.batch_size * world_size}")
        print("=" * 70)
        print(OmegaConf.to_yaml(cfg))

    seed = cfg.training.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    image_size = tuple(cfg.data.image_size)

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

    # Dataset (images only, no labels)
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

    # Online label provider
    label_provider = OnlineLabelProvider(
        device=device,
        depth_model_name=cfg.label_models.depth_model,
        vlm_name=cfg.label_models.vlm_model,
        image_size=image_size,
    )

    # Model
    target_cams = list(cfg.model.target_cameras)
    model = ImageAdaptorV1(
        backbone_pretrained=cfg.model.backbone_pretrained,
        backbone_out_channels=256,
        context_dim=cfg.model.context_dim,
        depth_bins=cfg.model.depth_bins,
        plucker_hidden_dim=cfg.model.plucker_hidden_dim,
        image_size=image_size,
        target_cameras=target_cams,
        depth_share_weights=cfg.model.depth_decoder.share_weights,
        image_share_weights=cfg.model.image_decoder.share_weights,
        token_d_model=cfg.model.token_decoder.d_model,
        token_num_layers=cfg.model.token_decoder.num_layers,
        token_num_heads=cfg.model.token_decoder.num_heads,
        num_query_tokens=cfg.model.token_decoder.num_query_tokens,
        token_output_dim=cfg.model.token_decoder.output_dim,
    ).to(device)

    if is_main_process():
        print(f"Model parameters: {model.num_parameters:,}")

    if world_size > 1:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    raw_model = model.module if world_size > 1 else model

    # Loss functions
    depth_criterion = DepthLoss(
        l1_weight=cfg.loss_weights.depth_l1,
        silog_weight=cfg.loss_weights.depth_silog,
    )
    image_criterion = ImageLoss(
        l1_weight=cfg.loss_weights.image_l1,
        ssim_weight=cfg.loss_weights.image_ssim,
        perceptual_weight=cfg.loss_weights.image_perceptual,
    ).to(device)
    token_criterion = TokenLoss(
        mse_weight=cfg.loss_weights.token_mse,
        cosine_weight=cfg.loss_weights.token_cosine,
        attention_kl_weight=cfg.loss_weights.token_attention_kl,
    )

    # Curriculum
    curriculum = CurriculumController(
        patience_epochs=cfg.curriculum.patience_epochs,
        min_improvement=cfg.curriculum.min_improvement,
        max_epochs_per_stage=cfg.curriculum.max_epochs_per_stage,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    resume_path = cfg.checkpoint.get("resume_from", None)

    if resume_path and os.path.isfile(resume_path):
        if is_main_process():
            print(f"Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if "curriculum_state" in ckpt:
            curriculum.load_state_dict(ckpt["curriculum_state"])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if is_main_process():
            print(f"  Resumed at epoch {start_epoch}, stage {curriculum.current_stage}, "
                  f"best_val_loss={best_val_loss:.4f}")
        del ckpt
        torch.cuda.empty_cache()

    # Pre-load label model for current stage
    label_provider.prepare_for_stage(curriculum.current_stage)

    # Optimizer (create for current curriculum stage)
    optimizer = _create_optimizer(raw_model, curriculum.current_stage, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.total_epochs - start_epoch, eta_min=1e-6
    )

    # Restore optimizer/scheduler state if resuming (after creation with correct params)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError):
            if is_main_process():
                print("  Warning: Could not restore optimizer state, using fresh optimizer")
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except (ValueError, KeyError):
            if is_main_process():
                print("  Warning: Could not restore scheduler state, using fresh scheduler")
        del ckpt
        torch.cuda.empty_cache()

    # TensorBoard
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tb_logs"))

    if is_main_process():
        print("\n" + "=" * 70)
        print(f"Starting Training (epoch {start_epoch + 1} / {cfg.training.total_epochs})")
        print("=" * 70)

    for epoch in range(start_epoch, cfg.training.total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss, global_step, train_metrics = train_one_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            depth_criterion=depth_criterion, image_criterion=image_criterion,
            token_criterion=token_criterion, curriculum=curriculum,
            geometric_transform=geometric_transform, label_provider=label_provider,
            device=device, image_size=image_size, writer=writer,
            epoch=epoch, global_step=global_step, cfg=cfg,
        )

        avg_val_loss, val_metrics = validate(
            model=model, dataloader=val_loader,
            depth_criterion=depth_criterion, image_criterion=image_criterion,
            token_criterion=token_criterion, curriculum=curriculum,
            geometric_transform=geometric_transform, label_provider=label_provider,
            device=device, image_size=image_size,
        )

        if world_size > 1:
            loss_tensor = torch.tensor([avg_val_loss], device=device)
            dist.all_reduce(loss_tensor)
            avg_val_loss = loss_tensor.item() / world_size

        # TensorBoard
        if writer:
            writer.add_scalar("train/loss_epoch", avg_train_loss, epoch)
            writer.add_scalar("val/loss_epoch", avg_val_loss, epoch)
            writer.add_scalar("curriculum/stage", curriculum.current_stage, epoch)
            writer.add_scalar("curriculum/lr", optimizer.param_groups[0]["lr"], epoch)
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
                src_K, src_E, tgt_c = build_camera_params(image_size, device, B_fix)

                # Generate labels for viz
                viz_depth = None
                viz_tokens = None
                if 1 in curriculum.active_stages:
                    viz_depth = label_provider.generate_depth_labels(fixed_batch["physicalai_images"])
                if 3 in curriculum.active_stages:
                    viz_tokens, _ = label_provider.generate_token_labels(fixed_batch["physicalai_images"])

                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    viz_out = (raw_model if world_size == 1 else model)(
                        ns_imgs, src_K, src_E, tgt_c,
                        active_stages=curriculum.active_stages,
                    )
                model.train()

                log_tb_images(
                    writer, epoch, fixed_batch, viz_out, ns_imgs,
                    depth_labels=viz_depth, token_labels=viz_tokens,
                    log_inputs=(epoch == 0),
                )

                # Free visualization tensors to prevent VRAM accumulation
                del fixed_batch, ns_imgs, src_K, src_E, tgt_c
                del viz_depth, viz_tokens, viz_out
                torch.cuda.empty_cache()

        if is_main_process():
            print(
                f"Epoch {epoch + 1}: train={avg_train_loss:.4f}, "
                f"val={avg_val_loss:.4f}, stage={curriculum.current_stage}"
            )

        # Curriculum transition
        primary_key = {1: "depth_total", 2: "image_total", 3: "token_total"}
        stage_val_loss = val_metrics.get(primary_key.get(curriculum.current_stage, ""), avg_val_loss)
        advanced, new_stage = curriculum.update(stage_val_loss)

        if advanced:
            if is_main_process():
                print(f"\n*** Stage advanced to {new_stage}! ***")

            # Switch label provider model (VRAM management)
            label_provider.prepare_for_stage(new_stage)

            # Recreate optimizer for new parameters
            optimizer = _create_optimizer(raw_model, new_stage, cfg)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.total_epochs - epoch, eta_min=1e-6
            )

            if is_main_process():
                print(f"*** Optimizer recreated, label model switched. ***\n")

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
                "curriculum_state": curriculum.state_dict(),
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


def _create_optimizer(
    model: ImageAdaptorV1,
    stage: int,
    cfg: DictConfig,
) -> torch.optim.Optimizer:
    """Create optimizer for current curriculum stage.

    Args:
        model: Image Adaptor model (unwrapped).
        stage: Current curriculum stage.
        cfg: Hydra config.

    Returns:
        Configured AdamW optimizer.
    """
    params = model.get_stage_parameters(stage)
    seen = set()
    unique_params = []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            unique_params.append(p)

    return torch.optim.AdamW(
        unique_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )


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
