"""PyTorch camera models for pinhole and f-theta projections.

Provides GPU-accelerated, autograd-compatible camera models used for
geometric transformations between Physical AI (f-theta) and NuScenes (pinhole)
camera systems.

Reference: /mnt/mydisk/alpamayo/src/alpamayo_r1/transform_physicalai_to_nuscenes.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PinholeCameraConfig:
    """Pinhole camera model (used by NuScenes), PyTorch version.

    Supports GPU tensor operations and autograd for differentiable rendering.

    Attributes:
        name: Camera identifier.
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        translation: Camera position in ego frame (3,).
        rotation_quat: Camera rotation quaternion (xyzw) from camera to ego frame.
    """

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    translation: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    rotation_quat: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    )

    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        """Return 3x3 intrinsic matrix K."""
        K = torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        return K

    @property
    def rotation_matrix(self) -> torch.Tensor:
        """Return 3x3 rotation matrix from camera to ego frame."""
        return _quat_to_rotation_matrix(self.rotation_quat)

    def pixel2ray(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates to unit ray directions in camera frame.

        Args:
            u: Pixel x-coordinates, any shape.
            v: Pixel y-coordinates, same shape as u.

        Returns:
            Unit ray directions, shape (*u.shape, 3).
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = torch.ones_like(x)
        rays = torch.stack([x, y, z], dim=-1)
        rays = F.normalize(rays, dim=-1)
        return rays

    def project(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project 3D points to pixel coordinates.

        Args:
            points: 3D points in camera frame, shape (..., 3).

        Returns:
            u: Pixel x-coordinates, shape (...).
            v: Pixel y-coordinates, shape (...).
            valid: Boolean mask for valid projections, shape (...).
        """
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        valid = z > 0.1

        u = torch.where(valid, self.fx * x / z.clamp(min=1e-8) + self.cx, torch.full_like(x, -1.0))
        v = torch.where(valid, self.fy * y / z.clamp(min=1e-8) + self.cy, torch.full_like(y, -1.0))

        in_bounds = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        valid = valid & in_bounds
        return u, v, valid

    def unproject(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Unproject pixels to 3D points in camera frame.

        Args:
            u: Pixel x-coordinates.
            v: Pixel y-coordinates.
            depth: Depth values (z-distance from camera).

        Returns:
            3D points in camera frame, shape (..., 3).
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return torch.stack([x, y, z], dim=-1)


@dataclass
class FThetaCameraConfig:
    """F-theta (fisheye) camera model (used by Physical AI), PyTorch version.

    Supports GPU tensor operations and autograd for differentiable rendering.

    Attributes:
        name: Camera identifier.
        width: Image width in pixels.
        height: Image height in pixels.
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        forward_poly: Forward polynomial coefficients [k0, k1, k2, k3, k4].
        backward_poly: Backward polynomial coefficients [j0, j1, j2, j3, j4].
        translation: Camera position in ego frame (3,).
        rotation_quat: Camera rotation quaternion (xyzw) from camera to ego frame.
    """

    name: str
    width: int
    height: int
    cx: float
    cy: float
    forward_poly: torch.Tensor
    backward_poly: torch.Tensor
    translation: torch.Tensor = field(default_factory=lambda: torch.zeros(3, dtype=torch.float64))
    rotation_quat: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    )

    @property
    def rotation_matrix(self) -> torch.Tensor:
        """Return 3x3 rotation matrix from camera to ego frame."""
        return _quat_to_rotation_matrix(self.rotation_quat)

    def _forward_poly_eval(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate forward polynomial: theta -> r_img.

        Args:
            theta: Angle from optical axis (radians).

        Returns:
            Image radius in pixels.
        """
        coeffs = self.forward_poly
        result = coeffs[0]
        theta_power = theta.clone()
        for i in range(1, len(coeffs)):
            result = result + coeffs[i] * theta_power
            theta_power = theta_power * theta
        return result

    def _backward_poly_eval(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate backward polynomial: r_img -> theta.

        Args:
            r: Image radius in pixels.

        Returns:
            Angle from optical axis (radians).
        """
        coeffs = self.backward_poly
        result = coeffs[0]
        r_power = r.clone()
        for i in range(1, len(coeffs)):
            result = result + coeffs[i] * r_power
            r_power = r_power * r
        return result

    def pixel2ray(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convert pixel coordinates to unit ray directions using f-theta model.

        Args:
            u: Pixel x-coordinates, any shape.
            v: Pixel y-coordinates, same shape as u.

        Returns:
            Unit ray directions, shape (*u.shape, 3).
        """
        px = u - self.cx
        py = v - self.cy
        r = torch.sqrt(px**2 + py**2)
        r_safe = r.clamp(min=1e-8)

        theta = self._backward_poly_eval(r)

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        rx = sin_theta * px / r_safe
        ry = sin_theta * py / r_safe
        rz = cos_theta

        at_center = r < 1e-8
        rx = torch.where(at_center, torch.zeros_like(rx), rx)
        ry = torch.where(at_center, torch.zeros_like(ry), ry)
        rz = torch.where(at_center, torch.ones_like(rz), rz)

        rays = torch.stack([rx, ry, rz], dim=-1)
        return rays

    def project(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project 3D points to pixel coordinates using f-theta model.

        Args:
            points: 3D points in camera frame, shape (..., 3).

        Returns:
            u: Pixel x-coordinates, shape (...).
            v: Pixel y-coordinates, shape (...).
            valid: Boolean mask for valid projections, shape (...).
        """
        rx = points[..., 0]
        ry = points[..., 1]
        rz = points[..., 2]

        r_norm = torch.sqrt(rx**2 + ry**2 + rz**2)
        r_norm_safe = r_norm.clamp(min=1e-8)

        cos_theta = (rz / r_norm_safe).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        r_img = self._forward_poly_eval(theta)

        rp_norm = torch.sqrt(rx**2 + ry**2)
        rp_norm_safe = rp_norm.clamp(min=1e-8)

        u = self.cx + r_img * rx / rp_norm_safe
        v = self.cy + r_img * ry / rp_norm_safe

        along_axis = rp_norm < 1e-8
        u = torch.where(along_axis, torch.full_like(u, self.cx), u)
        v = torch.where(along_axis, torch.full_like(v, self.cy), v)

        valid = (rz > 0.1) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, valid

    def unproject(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Unproject f-theta pixels to 3D points in camera frame.

        Args:
            u: Pixel x-coordinates.
            v: Pixel y-coordinates.
            depth: Depth values (distance along ray).

        Returns:
            3D points in camera frame, shape (..., 3).
        """
        rays = self.pixel2ray(u, v)
        return rays * depth.unsqueeze(-1)


def _quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to 3x3 rotation matrix.

    Args:
        q: Quaternion in (x, y, z, w) convention, shape (4,).

    Returns:
        Rotation matrix, shape (3, 3).
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)
    return R


# ============================================================
# Physical AI Camera Configurations (from calibration data)
# ============================================================
PHYSICALAI_CAMERAS: Dict[str, FThetaCameraConfig] = {
    "front_wide": FThetaCameraConfig(
        name="front_wide",
        width=1920,
        height=1080,
        cx=957.8529,
        cy=537.0255,
        forward_poly=torch.tensor([0.0, 925.540742, -6.855157, -12.767613, 0.691263], dtype=torch.float64),
        backward_poly=torch.tensor(
            [0.0, 1.080349e-03, 9.471309e-09, 1.536442e-11, 1.943676e-15], dtype=torch.float64
        ),
        translation=torch.tensor([1.696904, -0.010188, 1.435701], dtype=torch.float64),
        rotation_quat=torch.tensor([-0.499099, 0.504745, -0.499512, 0.496609], dtype=torch.float64),
    ),
    "cross_left": FThetaCameraConfig(
        name="cross_left",
        width=1920,
        height=1080,
        cx=965.7998,
        cy=546.7784,
        forward_poly=torch.tensor([0.0, 921.978679, -1.871639, -11.355496, -1.683752], dtype=torch.float64),
        backward_poly=torch.tensor(
            [0.0, 1.084380e-03, 4.305328e-09, 1.106543e-11, 6.792817e-15], dtype=torch.float64
        ),
        translation=torch.tensor([2.472564, 0.937820, 0.917814], dtype=torch.float64),
        rotation_quat=torch.tensor([0.697599, -0.145912, 0.147076, -0.685882], dtype=torch.float64),
    ),
    "cross_right": FThetaCameraConfig(
        name="cross_right",
        width=1920,
        height=1080,
        cx=960.7358,
        cy=539.0115,
        forward_poly=torch.tensor([0.0, 917.283211, 1.413971, -11.845786, -1.279825], dtype=torch.float64),
        backward_poly=torch.tensor(
            [0.0, 1.089975e-03, -2.286967e-10, 1.276346e-11, 5.447379e-15], dtype=torch.float64
        ),
        translation=torch.tensor([2.478202, -0.954542, 0.928658], dtype=torch.float64),
        rotation_quat=torch.tensor([0.141248, -0.683914, 0.701406, -0.142620], dtype=torch.float64),
    ),
}

# NuScenes camera configurations (typical calibration from sample 47c321b4)
NUSCENES_CAMERAS: Dict[str, PinholeCameraConfig] = {
    "CAM_FRONT": PinholeCameraConfig(
        name="CAM_FRONT",
        width=1600,
        height=900,
        fx=1252.8131,
        fy=1252.8131,
        cx=826.5881,
        cy=469.9847,
        translation=torch.tensor([1.7220, 0.0048, 1.4949], dtype=torch.float64),
        rotation_quat=torch.tensor([-0.4973, 0.4984, -0.4965, 0.5077], dtype=torch.float64),
    ),
    "CAM_FRONT_LEFT": PinholeCameraConfig(
        name="CAM_FRONT_LEFT",
        width=1600,
        height=900,
        fx=1257.8625,
        fy=1257.8625,
        cx=827.2411,
        cy=450.9155,
        translation=torch.tensor([1.5753, 0.5005, 1.5070], dtype=torch.float64),
        rotation_quat=torch.tensor([-0.6688, 0.2102, -0.2111, 0.6812], dtype=torch.float64),
    ),
    "CAM_FRONT_RIGHT": PinholeCameraConfig(
        name="CAM_FRONT_RIGHT",
        width=1600,
        height=900,
        fx=1256.7485,
        fy=1256.7485,
        cx=817.7888,
        cy=451.9542,
        translation=torch.tensor([1.5808, -0.4991, 1.5175], dtype=torch.float64),
        rotation_quat=torch.tensor([-0.1915, 0.6786, -0.6794, 0.2034], dtype=torch.float64),
    ),
    "CAM_BACK_LEFT": PinholeCameraConfig(
        name="CAM_BACK_LEFT",
        width=1600,
        height=900,
        fx=1254.9861,
        fy=1254.9861,
        cx=829.5769,
        cy=467.1681,
        translation=torch.tensor([1.0485, 0.4831, 1.5621], dtype=torch.float64),
        rotation_quat=torch.tensor([-0.6907, -0.1121, 0.1162, 0.7049], dtype=torch.float64),
    ),
    "CAM_BACK_RIGHT": PinholeCameraConfig(
        name="CAM_BACK_RIGHT",
        width=1600,
        height=900,
        fx=1249.9629,
        fy=1249.9629,
        cx=825.3768,
        cy=462.5482,
        translation=torch.tensor([1.0595, -0.4672, 1.5505], dtype=torch.float64),
        rotation_quat=torch.tensor([-0.1380, -0.6893, 0.6976, 0.1382], dtype=torch.float64),
    ),
}
