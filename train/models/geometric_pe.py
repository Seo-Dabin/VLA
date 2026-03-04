"""3-component geometric positional embedding system for V1.1.

Components:
  1. FourierRayPE: Fourier-encoded Plucker coordinates for source camera features
  2. Target3DPE: Precomputed PE for target camera ray directions (f-theta)
  3. CameraIDEmbedding: Learnable per-camera embeddings (5 source + 3 target)

Fourier encoding enables high-frequency geometric detail capture that
standard MLP-on-raw-coordinates cannot represent due to spectral bias.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..camera_models import FThetaCameraConfig, PHYSICALAI_CAMERAS


class FourierRayPE(nn.Module):
    """Fourier-encoded Plucker Ray positional embedding.

    Applies sinusoidal Fourier encoding to 6D Plucker coordinates before
    projecting through an MLP, enabling high-frequency geometric detail capture.

    Pipeline:
      pixel grid -> K_inv unproject -> world ray (R, t) -> Plucker 6D (d, m)
      -> Fourier encode: 6D -> 6*(2L+1) dims
      -> MLP: fourier_dim -> out_dim -> out_dim
      -> add to features

    Args:
        out_dim: Output dimension (should match backbone feature dim).
        fourier_L: Number of Fourier frequency bands.
    """

    def __init__(self, out_dim: int = 256, fourier_L: int = 10) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.fourier_L = fourier_L

        # Fourier encode: 6 * (2*L + 1) input dims
        fourier_dim = 6 * (2 * fourier_L + 1)

        # MLP: fourier_dim -> out_dim -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """Add Fourier-encoded Plucker ray PE to features.

        Args:
            features: Backbone features, shape (B*N, C, fH, fW).
            intrinsics: Camera intrinsic matrices, shape (B*N, 3, 3).
            extrinsics: Camera extrinsic matrices [R|t] (cam-to-world),
                shape (B*N, 4, 4).

        Returns:
            Features with Fourier PE added, shape (B*N, C, fH, fW).
        """
        BN, C, fH, fW = features.shape
        device = features.device
        dtype = features.dtype

        # Compute Plucker coordinates at feature map resolution
        plucker = self._compute_plucker_coords(
            intrinsics, extrinsics, fH, fW, device, dtype
        )  # (BN, fH, fW, 6)

        # Fourier encode
        encoded = self._fourier_encode(plucker)  # (BN, fH, fW, fourier_dim)

        # Project through MLP
        pe = self.mlp(encoded)  # (BN, fH, fW, C)
        pe = pe.permute(0, 3, 1, 2)  # (BN, C, fH, fW)

        return features + pe

    def _compute_plucker_coords(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        fH: int,
        fW: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute 6D Plucker coordinates for each feature pixel.

        Reuses logic from plucker_pe.py L77-137.

        Args:
            intrinsics: Camera K matrices, shape (BN, 3, 3).
            extrinsics: Camera-to-world transforms, shape (BN, 4, 4).
            fH: Feature height.
            fW: Feature width.
            device: Computation device.
            dtype: Computation dtype.

        Returns:
            Plucker coordinates, shape (BN, fH, fW, 6).
        """
        BN = intrinsics.shape[0]

        # Create pixel grid at feature map resolution (scaled to image coords)
        # Feature map is 1/16 of original image
        scale = 16.0
        v_coords, u_coords = torch.meshgrid(
            torch.arange(fH, device=device, dtype=dtype) * scale + scale / 2,
            torch.arange(fW, device=device, dtype=dtype) * scale + scale / 2,
            indexing="ij",
        )  # (fH, fW)

        # Unproject to camera frame rays
        ones = torch.ones_like(u_coords)
        pixels = torch.stack([u_coords, v_coords, ones], dim=-1)  # (fH, fW, 3)
        pixels = pixels.unsqueeze(0).expand(BN, -1, -1, -1)  # (BN, fH, fW, 3)

        # K_inv @ pixel -> ray in camera frame
        K_inv = torch.inverse(intrinsics.float())  # (BN, 3, 3)
        rays_cam = torch.einsum("bij,bhwj->bhwi", K_inv, pixels)  # (BN, fH, fW, 3)
        rays_cam = F.normalize(rays_cam, dim=-1)

        # Extract rotation and translation from extrinsics
        R = extrinsics[:, :3, :3].float()  # (BN, 3, 3) cam-to-world rotation
        t = extrinsics[:, :3, 3].float()  # (BN, 3) camera origin in world

        # Transform ray directions to world frame
        d = torch.einsum("bij,bhwj->bhwi", R, rays_cam)  # (BN, fH, fW, 3)
        d = F.normalize(d, dim=-1)

        # Camera origin in world frame
        o = t.unsqueeze(1).unsqueeze(1).expand(-1, fH, fW, -1)  # (BN, fH, fW, 3)

        # Moment vector: m = o x d
        m = torch.cross(o, d, dim=-1)  # (BN, fH, fW, 3)

        # Concatenate: (d, m) -> 6D Plucker coordinates
        plucker = torch.cat([d, m], dim=-1).to(dtype)  # (BN, fH, fW, 6)

        return plucker

    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier positional encoding.

        gamma(x) = [x, sin(2^0*pi*x), cos(2^0*pi*x), ...,
                     sin(2^{L-1}*pi*x), cos(2^{L-1}*pi*x)]

        Args:
            x: Input tensor, shape (..., D) where D=6 for Plucker coords.

        Returns:
            Fourier-encoded tensor, shape (..., D*(2L+1)).
        """
        encodings = [x]
        for l in range(self.fourier_L):
            freq = (2.0 ** l) * math.pi
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1)


class Target3DPE(nn.Module):
    """Precomputed PE for target camera pixels using 3D ray directions.

    Computes Plucker coordinates for each target pixel using the f-theta
    camera model, Fourier-encodes them, and projects through an MLP.
    The result is precomputed at init and registered as a buffer.

    Args:
        target_cameras: List of Physical AI target camera names.
        target_size: Target feature map size (tH, tW).
        out_dim: Output PE dimension.
        fourier_L: Number of Fourier frequency bands.
        original_size: Original Physical AI image size (H, W).
    """

    def __init__(
        self,
        target_cameras: List[str],
        target_size: Tuple[int, int] = (20, 36),
        out_dim: int = 256,
        fourier_L: int = 10,
        original_size: Tuple[int, int] = (1080, 1920),
    ) -> None:
        super().__init__()
        self.target_cameras = list(target_cameras)
        self.target_size = target_size
        self.out_dim = out_dim
        self.fourier_L = fourier_L
        self.original_size = original_size

        # Same Fourier+MLP architecture as FourierRayPE
        fourier_dim = 6 * (2 * fourier_L + 1)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

        # Precompute PE for each target camera and register as buffers
        for cam_name in self.target_cameras:
            plucker = self._compute_target_plucker(cam_name)
            self.register_buffer(f"plucker_{cam_name}", plucker)

    def _compute_target_plucker(self, cam_name: str) -> torch.Tensor:
        """Compute Plucker coordinates for a target camera's feature pixels.

        Args:
            cam_name: Physical AI camera name.

        Returns:
            Plucker coordinates, shape (1, tH, tW, 6).
        """
        cam = PHYSICALAI_CAMERAS[cam_name]
        tH, tW = self.target_size
        oH, oW = self.original_size

        # Create pixel grid at feature resolution, scaled to original PA resolution
        v_indices = torch.arange(tH, dtype=torch.float64)
        u_indices = torch.arange(tW, dtype=torch.float64)
        v_grid, u_grid = torch.meshgrid(v_indices, u_indices, indexing="ij")

        u_pa = (u_grid + 0.5) * (oW / tW)  # scale to original PA resolution
        v_pa = (v_grid + 0.5) * (oH / tH)

        # Unproject to 3D ray using f-theta model
        rays_cam = cam.pixel2ray(u_pa, v_pa)  # (tH, tW, 3)

        # Transform to world frame
        R = cam.rotation_matrix  # (3, 3), camera-to-ego
        rays_world = torch.einsum("ij,hwj->hwi", R, rays_cam)  # (tH, tW, 3)
        rays_world = F.normalize(rays_world.float(), dim=-1)

        # Camera origin in world frame
        origin = cam.translation.float()  # (3,)
        origin_expanded = origin.unsqueeze(0).unsqueeze(0).expand(tH, tW, -1)

        # Plucker coordinates: (d, m) where m = origin x d
        d = rays_world
        m = torch.cross(origin_expanded, d, dim=-1)

        plucker = torch.cat([d, m], dim=-1)  # (tH, tW, 6)
        return plucker.unsqueeze(0).float()  # (1, tH, tW, 6)

    def get_pe(self, cam_name: str) -> torch.Tensor:
        """Get precomputed PE for a target camera, projected through MLP.

        Args:
            cam_name: Physical AI camera name.

        Returns:
            PE tensor, shape (1, out_dim, tH, tW).
        """
        plucker = getattr(self, f"plucker_{cam_name}")  # (1, tH, tW, 6)
        encoded = self._fourier_encode(plucker)  # (1, tH, tW, fourier_dim)
        pe = self.mlp(encoded)  # (1, tH, tW, out_dim)
        pe = pe.permute(0, 3, 1, 2)  # (1, out_dim, tH, tW)
        return pe

    def _fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier positional encoding.

        Args:
            x: Input tensor, shape (..., D).

        Returns:
            Fourier-encoded tensor, shape (..., D*(2L+1)).
        """
        encodings = [x]
        for l in range(self.fourier_L):
            freq = (2.0 ** l) * math.pi
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        return torch.cat(encodings, dim=-1)


class CameraIDEmbedding(nn.Module):
    """Learnable per-camera embedding for source and target cameras.

    Provides a simple learnable bias per camera view to help the network
    disambiguate between different viewpoints.

    Args:
        n_source: Number of source cameras (NuScenes).
        n_target: Number of target cameras (Physical AI).
        dim: Embedding dimension.
    """

    def __init__(
        self,
        n_source: int = 5,
        n_target: int = 3,
        dim: int = 256,
    ) -> None:
        super().__init__()
        self.source_embed = nn.Embedding(n_source, dim)
        self.target_embed = nn.Embedding(n_target, dim)

        # Small init
        nn.init.normal_(self.source_embed.weight, std=0.02)
        nn.init.normal_(self.target_embed.weight, std=0.02)

    def get_source_pe(self, cam_indices: torch.Tensor) -> torch.Tensor:
        """Get source camera PE.

        Args:
            cam_indices: Camera index tensor, shape (BN,).

        Returns:
            Embeddings, shape (BN, dim).
        """
        return self.source_embed(cam_indices)

    def get_target_pe(self, cam_index: int) -> torch.Tensor:
        """Get target camera PE.

        Args:
            cam_index: Single camera index.

        Returns:
            Embedding, shape (1, dim).
        """
        idx = torch.tensor([cam_index], device=self.target_embed.weight.device)
        return self.target_embed(idx)
