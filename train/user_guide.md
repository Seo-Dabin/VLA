# Image Adaptor V1 - User Guide

## Overview

Image Adaptor V1은 nuScenes 카메라 이미지로부터 Physical AI의 visual token (Alpamayo-R1-10B 호환)을 생성하는 모델을 학습합니다.

모든 label (depth map, visual token)은 학습 중 online으로 생성됩니다. 사전 계산(precomputation)은 필요하지 않습니다.

### Architecture
- **Backbone**: EfficientNet-B4 (pretrained, timm) + FPN neck
- **Positional Embedding**: Plücker Ray PE (camera-aware 6D coordinates)
- **View Transform**: Inverse Splat (Lift-Splat to PhysicalAI camera planes)
- **Decoders**: 3-stage curriculum (Depth → Image → Token)

### Curriculum Training
1. **Stage 1**: Depth prediction (L1 + SILog loss) — Depth-Anything-V2-Small로 online label 생성
2. **Stage 2**: Image reconstruction (L1 + SSIM + Perceptual loss)
3. **Stage 3**: Visual token generation (MSE + Cosine + KL attention distillation) — Qwen3-VL-2B ViT로 online label 생성

Stage 전환은 validation loss 수렴 기반 (patience window 방식).

### VRAM Management
- Stage 1-2: Depth-Anything-V2-Small (~100MB VRAM)
- Stage 3: Depth model 해제 → Qwen3-VL-2B ViT 로딩 (~2GB VRAM)
- 동시에 두 모델이 VRAM에 존재하지 않음

---

## 1. Environment Setup

```bash
# conda 환경 생성
conda create -n image_adaptor_v1 python=3.12 -y
conda activate image_adaptor_v1

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install timm hydra-core omegaconf tensorboard einops opencv-python scipy

# Label models (online generation)
pip install transformers accelerate

# VLM processor
pip install qwen-vl-utils

# Visualization
pip install scikit-learn matplotlib Pillow
```

---

## 2. Training

사전 계산 단계 없이 바로 학습을 시작할 수 있습니다.

### Single GPU (테스트)
```bash
cd /mnt/mydisk/VLA_domain_adaptation/image_adaptor
python -m train.train training.batch_size=1
```

### 4-GPU DDP Training
```bash
cd /mnt/mydisk/VLA_domain_adaptation/image_adaptor
torchrun --nproc_per_node=4 -m train.train
```

### Config Override 예시
```bash
# Batch size 변경
torchrun --nproc_per_node=4 -m train.train training.batch_size=4

# Learning rate 변경
torchrun --nproc_per_node=4 -m train.train training.lr=2e-4

# 클립 수 제한 (디버깅용)
torchrun --nproc_per_node=4 -m train.train data.max_clips=10

# 빠른 테스트
python -m train.train training.total_epochs=2 training.batch_size=1 data.max_clips=5
```

---

## 3. TensorBoard Monitoring

```bash
# 학습 중 or 학습 후
tensorboard --logdir /mnt/mydisk/VLA_domain_adaptation/image_adaptor/checkpoints --port 6006
```

### 확인 가능한 항목
- **train/loss_epoch**, **val/loss_epoch**: 전체 loss 추이
- **train/depth_total**, **val/depth_total**: Stage 1 depth loss
- **train/image_total**, **val/image_total**: Stage 2 image loss
- **train/token_total**, **val/token_total**: Stage 3 token loss
- **curriculum/stage**: 현재 curriculum stage
- **curriculum/lr**: Learning rate
- **viz/input_physicalai/**: Physical AI 원본 이미지 (epoch 0)
- **viz/input_nuscenes/**: NuScenes 변환 이미지 (epoch 0)
- **viz/depth_pred/**, **viz/depth_gt/**: Depth map 비교
- **viz/image_pred/**: 이미지 복원 결과
- **viz/tsne/**: Token t-SNE 비교 (pred vs GT)

---

## 4. Checkpoint Structure

```
checkpoints/
  YYMMDD_HHMM/
    config.yaml           # 실행에 사용된 config 복사본
    tb_logs/              # TensorBoard logs
    epoch_005.pt          # Periodic checkpoint
    epoch_010.pt
    ...
    best_model.pt         # Best validation loss checkpoint
```

### Checkpoint 내용
```python
ckpt = {
    "epoch": epoch,
    "global_step": global_step,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "curriculum_state": ...,
    "best_val_loss": ...,
    "config": ...,
}
```

---

## 5. Config Reference

| Config Key | Default | Description |
|---|---|---|
| `data.physical_ai_root` | /mnt/mydisk/alpamayo/data/physical_ai | Physical AI 데이터 경로 |
| `data.image_size` | [320, 576] | 입력 이미지 크기 [H, W] |
| `data.max_clips` | null | 사용할 최대 클립 수 (null=전체, 정수=디버깅) |
| `label_models.depth_model` | depth-anything/...Small-hf | Depth 모델 (HuggingFace) |
| `label_models.vlm_model` | Qwen/Qwen3-VL-2B-Instruct | VLM 모델 (HuggingFace) |
| `model.depth_bins` | 64 | Inverse Splat depth bins |
| `model.context_dim` | 360 | Feature map channel dimension |
| `model.token_decoder.num_query_tokens` | 180 | 카메라당 출력 토큰 수 |
| `model.token_decoder.output_dim` | 2048 | 토큰 임베딩 차원 |
| `curriculum.patience_epochs` | 5 | Stage 전환 patience |
| `curriculum.min_improvement` | 0.01 | 최소 개선률 (1%) |
| `curriculum.max_epochs_per_stage` | 30 | Stage당 최대 epoch |
| `training.batch_size` | 2 | GPU당 batch size |
| `training.lr` | 1e-4 | Base learning rate |
| `checkpoint.save_every_n_epochs` | 5 | Checkpoint 저장 간격 |
| `checkpoint.log_image_every_n_epochs` | 2 | 이미지 로깅 간격 |

---
---

# Image Adaptor V1.1 - User Guide

## Overview

V1.1은 V1의 Inverse Splat을 **Epipolar Cross-Attention**으로, Plucker MLP를 **Fourier PE**로 교체합니다.
Image reconstruction (Stage 2)이 제거되고, depth는 1x1 Conv auxiliary loss로 대체됩니다.
Curriculum 없이 single-stage 학습 (모든 loss가 epoch 0부터 활성).

### Architecture Changes (V1 -> V1.1)
- **PE**: Plucker MLP (6D→256) → **Fourier-encoded Plucker** (6D→126D→256) + **Camera ID Embedding**
- **View Transform**: Inverse Splat (LSS) → **Epipolar Cross-Attention** (3 layers, 8 heads)
- **Depth Decoder**: Multi-scale decoder → **1x1 Conv** (auxiliary, feature resolution 20x36)
- **Image Decoder**: Removed
- **Curriculum**: 3-stage → **None** (single-stage)
- **Parameters**: ~45M → ~38M (7M smaller)

### VRAM Management
- V1.1은 depth model과 ViT를 동시에 로딩 (~100MB + ~2GB = ~2.1GB)
- 24GB 4090에서 문제없이 동작

---

## 1. Environment Setup

```bash
# conda 환경 생성 (V1과 별도)
conda create -n image_adaptor_v1_1 python=3.12 -y
conda activate image_adaptor_v1_1

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install timm hydra-core omegaconf tensorboard einops opencv-python scipy

# Label models (online generation)
pip install transformers accelerate

# VLM processor
pip install qwen-vl-utils

# Visualization
pip install scikit-learn matplotlib Pillow
```

---

## 2. Training

### Single GPU (테스트)
```bash
cd /mnt/mydisk/VLA_domain_adaptation/image_adaptor
python -m train.train_v1_1 training.batch_size=1
```

### 4-GPU DDP Training
```bash
cd /mnt/mydisk/VLA_domain_adaptation/image_adaptor
torchrun --nproc_per_node=4 -m train.train_v1_1
```

### Config Override 예시
```bash
# 빠른 테스트 (소수 클립, 적은 epoch)
python -m train.train_v1_1 training.total_epochs=2 training.batch_size=1 data.max_clips=5

# Epipolar attention 설정 변경
torchrun --nproc_per_node=4 -m train.train_v1_1 model.epipolar_attention.n_layers=4

# Depth range 변경
torchrun --nproc_per_node=4 -m train.train_v1_1 model.epipolar_attention.depth_range="[0.5,80.0]"
```

---

## 3. TensorBoard Monitoring

```bash
tensorboard --logdir /mnt/mydisk/VLA_domain_adaptation/image_adaptor/checkpoints --port 6006
```

### V1.1 전용 확인 항목
- **train/loss_epoch**, **val/loss_epoch**: 전체 loss 추이
- **train/token_total**, **val/token_total**: Token loss
- **train/depth_total**, **val/depth_total**: Auxiliary depth loss
- **train/lr**: Learning rate (warmup + cosine decay)
- **viz/input_physicalai/**: Physical AI 원본 이미지 (epoch 0)
- **viz/input_nuscenes/**: NuScenes 변환 이미지 (epoch 0)
- **viz/depth_pred/**, **viz/depth_gt/**: Depth map 비교 (20x36 feature resolution)
- **viz/epipolar_attn/**: Epipolar attention 패턴 시각화 (NEW)
- **viz/feature_map/**: Feature map PCA/mean/std (NEW)
- **viz/tsne/**: Token t-SNE 비교 (pred vs GT)

---

## 4. Checkpoint Structure

V1과 동일한 구조 (curriculum_state 없음):

```
checkpoints/
  YYMMDD_HHMM/
    config.yaml           # 실행에 사용된 config 복사본
    tb_logs/              # TensorBoard logs
    epoch_005.pt          # Periodic checkpoint
    best_model.pt         # Best validation loss
```

---

## 5. V1.1 Config Reference

| Config Key | Default | Description |
|---|---|---|
| `model.pe.fourier_L` | 10 | Fourier encoding frequency bands |
| `model.epipolar_attention.d_model` | 256 | Attention feature dimension |
| `model.epipolar_attention.n_heads` | 8 | Number of attention heads |
| `model.epipolar_attention.n_layers` | 3 | Number of transformer layers |
| `model.epipolar_attention.n_samples` | 32 | Points per epipolar line |
| `model.epipolar_attention.depth_range` | [1.0, 60.0] | Depth sampling range (meters) |
| `model.epipolar_attention.ffn_dim` | 1024 | FFN hidden dimension |
| `model.epipolar_attention.dropout` | 0.1 | Dropout probability |
| `model.token_decoder.in_channels` | 256 | Feature input channels (was 360 in V1) |
| `loss_weights.depth_l1` | 0.1 | Depth L1 weight (auxiliary) |
| `loss_weights.depth_silog` | 0.05 | Depth SILog weight (auxiliary) |
| `training.warmup_epochs` | 5 | LR warmup epochs |

## 6. File Structure (V1.1 New Files)

```
train/models/
  geometric_pe.py         # FourierRayPE + Target3DPE + CameraIDEmbedding
  epipolar_attention.py   # EpipolarCrossAttention + precomputation
  image_adaptor_v1_1.py   # V1.1 model wrapper
train/
  train_v1_1.py           # V1.1 training script
config/
  v1_1.yaml               # V1.1 config
sketch/V1.1/
  v1.1 sketch.md          # Architecture description
  understanding.md        # Design understanding
```
