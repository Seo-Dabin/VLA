# V1.1 Understanding

## Core Insight

V1.1 replaces the explicit 3D volume construction (Inverse Splat) with an
attention-based implicit approach (Epipolar Cross-Attention). Instead of
discretizing depth into bins and creating a 3D feature volume, V1.1 lets
the network learn to attend to geometrically relevant source features
along precomputed epipolar lines.

## Key Design Decisions

### Why Epipolar Cross-Attention?
- Epipolar geometry constrains where corresponding points can appear
  between two camera views. A 3D point visible in the target camera must
  lie on a specific line (epipolar line) in each source camera.
- By sampling along these lines, we reduce the attention search space
  from all source pixels (3600) to only geometrically valid locations
  (160 per query = 5 cameras x 32 depth samples).
- The network learns soft depth selection via attention weights, avoiding
  hard depth discretization artifacts.

### Why Fourier PE?
- Standard Plucker MLP (6D -> 256D) can only represent low-frequency
  spatial patterns due to the spectral bias of MLPs.
- Fourier encoding: gamma(x) = [x, sin(2^0*pi*x), cos(2^0*pi*x), ...,
  sin(2^{L-1}*pi*x), cos(2^{L-1}*pi*x)] expands 6D to 126D before the
  MLP, enabling high-frequency geometric detail capture.

### Why Remove Image Reconstruction?
- Image reconstruction (Stage 2) was a proxy task to regularize features.
- With epipolar attention providing strong geometric guidance, the proxy
  task is no longer necessary.
- Removing it saves ~8M parameters and simplifies training.

### Why Lightweight Depth?
- Depth is still useful as auxiliary supervision to encourage geometrically
  meaningful feature representations.
- A 1x1 Conv (256 -> 1) at feature resolution (20x36) is sufficient for
  this auxiliary role, replacing the full multi-scale DepthDecoder.

## Data Flow Understanding

1. Physical AI 3-cam images are geometrically warped to NuScenes 5-cam
   views using precomputed rotation-only LUTs (GeometricTransform).

2. All 5 NuScenes views pass through shared EfficientNet-B4 + FPN backbone,
   producing (B*5, 256, 20, 36) feature maps at 1/16 resolution.

3. FourierRayPE adds camera-aware positional information using Fourier-
   encoded Plucker coordinates. CameraID adds per-camera learnable bias.

4. For each of 3 target cameras:
   - Initialize learnable target queries (720, 256) + Target3DPE + CamID
   - Sample source features along precomputed epipolar lines using grid_sample
   - 3 layers of cross-attention + FFN
   - Reshape (B, 720, 256) -> (B, 256, 20, 36)

5. Each target feature map feeds into:
   - TokenDecoder: cross-attention with learnable queries -> 180 tokens x 2048D
   - DepthHead: 1x1 Conv -> (B, 1, 20, 36) depth at feature resolution

## Camera Geometry

- Source: 5 NuScenes pinhole cameras (1600x900 original, processed at 320x576)
- Target: 3 Physical AI f-theta cameras (1920x1080 original)
- Feature maps: 20x36 at 1/16 scale
- Extrinsics: camera-to-ego transforms (rotation + translation)
- All geometry is precomputed at init and stored as buffers
