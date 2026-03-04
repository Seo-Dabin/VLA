# V1 Architecture Understanding

## 목표
nuScenes 카메라 이미지만으로 Physical AI의 visual token (Alpamayo-R1-10B 호환)을 생성하는 Image Adaptor 학습.
Depth estimation, Image reconstruction은 feature map 표현력 향상을 위한 보조 task이며, inference 시에는 Visual Token Decoder만 사용.

---

## 데이터 흐름

### 입력
- **Physical AI 원본 이미지**: 1920×1080, 5개 카메라 (Cross_left, Cross_right, Front_wide, Front_tele, +1)
  - 이 중 **3개만 사용**: Front_wide, Cross_left, Cross_right
  - Front_tele 제외 이유: 다른 이미지와 특성이 다름
- **nuScenes 카메라**: 6개 중 **5개 사용** (CAM_BACK 제외, Alpamayo가 커버하지 않는 영역)
  - CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK_LEFT, CAM_BACK_RIGHT
  - 원본 해상도: 1600×900

### Geometric Transformation (online)
- Physical AI 이미지 → nuScenes 5-camera 뷰로 변환
- 양쪽 카메라 모델, intrinsic, extrinsic 파라미터 활용
- 연산 부담이 적어 offline 전처리 불필요, 실시간 처리

### Resize
- nuScenes 뷰 이미지: 1600×900 → **576×320** (Qwen3-VL processor dynamic resolution과 의도적으로 일치)

### 시간축
- Alpamayo는 4-step 입력 구성 (5 images × 4 steps = 20 images)
- 시간축은 필수가 아님, 단일 프레임으로도 동작 가능

---

## 모델 구조

### Inverse Splat (LSS 변형) — 🔥 Learnable
모든 stage에서 **backbone 공유**

| 구성요소 | 설정 |
|---------|------|
| Image Backbone | EfficientNet |
| Depth Bins | 64 |
| Context Feature 차원 | 360 |
| Positional Embedding | Plücker Ray PE |
| 입력 | nuScenes 5-cam 이미지 (576×320×3) |
| 출력 | 3개 feature map plane (Front wide, Cross left, Cross right) |

- nuScenes 뷰 이미지를 입력받아 Physical AI 카메라 plane으로 feature 투영
- Plücker Ray PE로 카메라 기하 정보 주입 (6D: direction + moment)

### 카메라 파라미터 활용

Inverse Splat의 Lift-Splat 과정에서 **양쪽 데이터셋의 카메라 파라미터가 모두 활용**됨.
projection 자체는 고정된 기하 연산이며 learnable parameter가 아님. 학습되는 부분은 depth distribution과 context vector뿐.

```
nuScenes 이미지 (2D)
      ↓ Lift (nuScenes 카메라 파라미터)
  3D 공간 frustum features
      ↓ Splat (Physical AI 카메라 파라미터)
Physical AI camera plane feature maps (2D)
```

| 단계 | nuScenes 파라미터 | Physical AI 파라미터 | 설명 |
|------|:---:|:---:|------|
| Plücker Ray PE | O | - | nuScenes 각 카메라의 K, [R\|t]로 ray direction + moment 계산 |
| Lift (2D→3D) | O | - | K_nu로 unproject, [R\|t]_nu로 월드 좌표 변환 |
| Splat (3D→2D) | - | O | [R\|t]_pa로 target 카메라 좌표 변환, K_pa로 target plane 투영 |

---

## 학습 전략: Curriculum Training

Stage 1 loss로 시작 → 점진적으로 Stage 2, Stage 3 loss 추가.
Inverse Splat backbone은 전 stage에서 공유하며 지속 학습.

### Stage 1: Depth Estimation
- **목적**: feature map이 기하학적 구조를 인코딩하도록 유도
- **Label 생성**: Depth-Anything-V2-Metric-Outdoor-Small (❄️ frozen)
  - Physical AI 원본 이미지 (front_wide, cross_left, cross_right)에 적용
- **출력 해상도**: 576×320
- **Loss**: L1 + SILog (Scale-Invariant Logarithmic)
- **Decoder**: depth-specific decoder (각 plane별)

### Stage 2: Image Reconstruction
- **목적**: feature map이 시각적 디테일을 보존하도록 유도
- **GT**: Physical AI 원본 이미지 (front_wide, cross_left, cross_right), 576×320으로 resize
- **Decoder**: U-Net
- **출력 해상도**: 576×320
- **Loss**: L1 + SSIM + Perceptual Loss

### Stage 3: Visual Token Decoder
- **목적**: feature map에서 Alpamayo 호환 visual token 직접 생성
- **Label 생성**: Qwen3-VL-2B-Instruct processor (❄️ frozen)
  - Physical AI 원본 이미지 → ViT Encoder → PatchMerger → **180개 visual token (각 2048-dim)**
  - Front_wide: 180 tokens, Cross_left: 180 tokens, Cross_right: 180 tokens → 총 540 tokens
- **Decoder**: Transformer Decoder (student)
- **Teacher**: Qwen3-VL-2B-Instruct ViT Encoder의 Transformer Blocks
- **Loss**:
  - Token Reconstruction: MSE + Cosine Similarity
  - Attention Distillation: KL Divergence (teacher attention map → student attention map 모방)

---

## Inference Pipeline

```
nuScenes 5-cam images (576×320)
        ↓
  Plücker Ray PE + EfficientNet backbone
        ↓
  Inverse Splat (depth bins=64, context=360)
        ↓
  3 Feature Map Planes (Front wide, Cross left, Cross right)
        ↓
  Visual Token Decoder (Transformer Decoder) only
        ↓
  540 visual tokens (3 × 180 × 2048)
        → Alpamayo-R1-10B LLM에 입력
```

- Depth Decoder, Image Reconstruction Decoder는 **사용하지 않음**
- Inference 대상 GPU: RTX 3080 (10GB VRAM)

---

## Frozen Models (Label 생성용, 학습 시에만 사용)

| 모델 | 용도 | 입력 |
|------|------|------|
| Depth-Anything-V2-Metric-Outdoor-Small | Stage 1 depth label | Physical AI 원본 이미지 (3장) |
| Qwen3-VL-2B-Instruct (ViT + PatchMerger) | Stage 3 visual token label + attention map | Physical AI 원본 이미지 (3장) |

---

## 핵심 설계 판단 요약

1. **CAM_BACK 제외**: Alpamayo가 커버하지 않는 후방 영역
2. **Front_tele 제외**: 다른 카메라와 특성이 달라 별도 처리 필요
3. **576×320 해상도**: Qwen3-VL dynamic resolution 결과와 의도적 일치
4. **Curriculum training**: 기하 → 시각 → 토큰 순서로 feature map 표현력 점진적 강화
5. **공유 backbone**: Inverse Splat이 모든 task의 공통 feature extractor 역할
6. **Inference 경량화**: 보조 decoder 제거, Visual Token Decoder만 사용
