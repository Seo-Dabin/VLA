# V1 Image Adaptor - Understanding

## Core Concept
nuScenes 카메라 이미지만으로 Physical AI의 visual token (Alpamayo-R1-10B / Qwen3-VL-2B-Instruct 호환)을 생성하는 Image Adaptor를 학습한다.

## Pipeline
1. Physical AI 원본 이미지 (3카메라: front_wide, cross_left, cross_right)
2. → Geometric transformation으로 nuScenes 5-camera 뷰 생성 (rotation-only remap + grid_sample)
3. → EfficientNet-B4 backbone + FPN으로 multi-scale feature 추출
4. → Plücker Ray PE로 camera-aware positional embedding 추가
5. → Inverse Splat으로 3D lift 후 Physical AI 카메라 plane에 feature map 생성
6. → 3단계 curriculum training으로 표현력 점진 강화:
   - Stage 1: Depth decoder (CNN, L1 + SILog loss)
   - Stage 2: Image decoder (U-Net, L1 + SSIM + Perceptual loss)
   - Stage 3: Token decoder (Transformer, MSE + Cosine + KL attention distillation)

## Key Design Decisions
- **Backbone**: EfficientNet-B4 (timm pretrained) - 효율/성능 밸런스
- **Curriculum**: Loss 수렴 기반 stage 전환 (patience window)
- **Labels**: Precompute (depth map + visual tokens를 사전 생성하여 학습 효율화)
- **View Transform**: Rotation-only remap (parallax 없는 단순 기하 변환)
- **Inverse Splat**: LSS 기반이나 BEV 대신 target camera plane으로 splat

## Camera System
- Source: NuScenes pinhole 5-cam (CAM_FRONT, CAM_FRONT_LEFT, etc.)
- Target: Physical AI f-theta 3-cam (front_wide, cross_left, cross_right)
- PRIMARY_SOURCE 매핑: CAM_FRONT←front_wide, CAM_FRONT_LEFT←cross_left, etc.

## Data Flow
```
PhysicalAI images (3cam) --[geometric_transform]--> NuScenes images (5cam)
                                                        |
                                                   [backbone]
                                                        |
                                                   [plucker_pe]
                                                        |
                                                   [inverse_splat]
                                                        |
                                              Feature Maps (3 target planes)
                                               /        |         \
                                          [depth]   [image]    [token]
                                          decoder    decoder    decoder
```
