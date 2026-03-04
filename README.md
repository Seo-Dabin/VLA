# Image Adaptor V1

**nuScenes camera images to Physical AI visual tokens (Alpamayo-R1-10B compatible)**

Domain adaptation module that transforms nuScenes multi-camera images into visual tokens compatible with Physical AI's vision-language-action model, bridging the gap between autonomous driving datasets and embodied AI systems.

---

## Architecture

```
PhysicalAI Images (3-cam)
        |
  Geometric Transform (f-theta -> pinhole remap)
        |
NuScenes Images (5-cam)
        |
  EfficientNet-B4 + FPN Backbone
        |
  Plucker Ray Positional Embedding
        |
  Inverse Splat (Lift-Splat to target camera planes)
        |
  Feature Maps (3 target planes: front_wide, cross_left, cross_right)
       /            |              \
  Depth Decoder  Image Decoder  Token Decoder
  (Stage 1)      (Stage 2)      (Stage 3)
```

### Key Components

| Module | Description |
|--------|-------------|
| **Geometric Transform** | Rotation-only remap from f-theta (Physical AI) to pinhole (nuScenes) via `grid_sample` |
| **EfficientNet-B4 + FPN** | Multi-scale feature extraction with pretrained backbone (timm) |
| **Plucker Ray PE** | Camera-aware 6D positional embedding for multi-view feature fusion |
| **Inverse Splat** | LSS-based Lift-Splat projecting to target camera planes (not BEV) |
| **Depth Decoder** | CNN decoder, absolute metric depth (Stage 1) |
| **Image Decoder** | U-Net decoder for RGB reconstruction (Stage 2) |
| **Token Decoder** | Transformer cross-attention decoder, 180 tokens x 2048-dim per camera (Stage 3) |

---

## 3-Stage Curriculum Training

Training follows a loss-convergence-based curriculum that progressively strengthens feature map representations:

| Stage | Task | Loss | Label Source |
|-------|------|------|--------------|
| **1** | Depth Prediction | L1 + SILog | Depth-Anything-V2-Small (online) |
| **2** | Image Reconstruction | L1 + SSIM + VGG Perceptual | Ground truth images |
| **3** | Visual Token Generation | MSE + Cosine + KL Attention Distillation | Qwen3-VL-2B ViT (online) |

Stage transitions are triggered when validation loss converges (patience window with minimum improvement threshold). All labels are generated online during training -- no precomputation required.

---

## Project Structure

```
image_adaptor/
├── config/
│   └── v1.yaml                    # Hydra training config
├── train/
│   ├── train.py                   # Main training script (DDP, bf16)
│   ├── dataset.py                 # Physical AI dataset loader
│   ├── curriculum.py              # Loss-convergence stage controller
│   ├── camera_models.py           # Pinhole + f-theta camera models (PyTorch)
│   ├── geometric_transform.py     # Physical AI -> nuScenes view transform
│   ├── models/
│   │   ├── backbone.py            # EfficientNet-B4 + FPN neck
│   │   ├── plucker_pe.py          # Plucker Ray positional embedding
│   │   ├── inverse_splat.py       # Lift-Splat to target camera planes
│   │   ├── depth_decoder.py       # Stage 1: CNN depth decoder
│   │   ├── image_decoder.py       # Stage 2: U-Net image decoder
│   │   ├── token_decoder.py       # Stage 3: Transformer token decoder
│   │   └── image_adaptor.py       # Full model wrapper
│   ├── losses/
│   │   ├── depth_loss.py          # L1 + SILog
│   │   ├── image_loss.py          # L1 + SSIM + VGG Perceptual
│   │   └── token_loss.py          # MSE + Cosine + KL Distillation
│   └── user_guide.md              # Detailed usage guide
├── sketch/                        # Design sketches & diagrams
└── checkpoints/                   # Auto-generated: TB logs, configs, models
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n image_adaptor_v1 python=3.12 -y
conda activate image_adaptor_v1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm hydra-core omegaconf tensorboard einops opencv-python scipy
pip install transformers accelerate qwen-vl-utils
pip install scikit-learn matplotlib Pillow
```

### 2. Training

```bash
cd /path/to/image_adaptor

# 4-GPU DDP (recommended)
torchrun --nproc_per_node=4 -m train.train

# Single GPU
python -m train.train training.batch_size=1

# Config overrides
torchrun --nproc_per_node=4 -m train.train \
    training.batch_size=3 \
    training.total_epochs=1000 \
    data.max_clips=30000
```

### 3. Resume from Checkpoint

```bash
torchrun --nproc_per_node=4 -m train.train \
    checkpoint.resume_from=/path/to/checkpoints/YYMMDD_HHMM/epoch_025.pt
```

### 4. TensorBoard

```bash
tensorboard --logdir checkpoints --port 6006
```

---

## Training Outputs

```
checkpoints/
  YYMMDD_HHMM/
    config.yaml        # Config snapshot
    tb_logs/            # TensorBoard logs
    epoch_005.pt        # Periodic checkpoints
    best_model.pt       # Best val loss model
```

### TensorBoard Metrics

- `train/loss_epoch`, `val/loss_epoch` -- Total loss
- `train/depth_total`, `train/image_total`, `train/token_total` -- Per-stage losses
- `curriculum/stage` -- Current training stage
- `viz/depth_pred/`, `viz/depth_gt/` -- Depth map comparison
- `viz/image_pred/` -- Image reconstruction results
- `viz/tsne/` -- Token t-SNE visualization (pred vs GT)

---

## Hardware Requirements

- **GPU**: 4x NVIDIA RTX 4090 (24GB each), or equivalent
- **Recommended batch size**: 3 per GPU (optimized for 24GB VRAM)
- **VRAM usage**: ~12GB per GPU during training
- **Mixed precision**: bf16 (automatic via `torch.autocast`)

---

## References

- [Lift-Splat-Shoot (LSS)](https://arxiv.org/abs/2008.10295) -- Core view transform architecture
- [Depth Anything V2](https://arxiv.org/abs/2406.09414) -- Online depth label generation
- [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) -- Visual token target (ViT encoder)
- [EfficientNet](https://arxiv.org/abs/1905.11946) -- Backbone architecture
