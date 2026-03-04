"""Dataset for Image Adaptor V1 training (online mode).

Each sample provides only Physical AI original images (3 cameras).
Depth labels and visual token labels are generated online during training
by the label provider models (Depth-Anything-V2, Qwen3-VL ViT).

NuScenes-style transformed images are also generated online via geometric transform.

Reference: /mnt/mydisk/alpamayo/src/alpamayo_r1/dataset_physicalai_paired.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Physical AI cameras used for Image Adaptor training
PHYSICALAI_TARGET_CAMERAS = ["front_wide", "cross_left", "cross_right"]

# Target image size (H, W)
DEFAULT_IMAGE_SIZE = (320, 576)


class ImageAdaptorDataset(Dataset):
    """Online dataset for Image Adaptor V1 training.

    Loads only Physical AI camera images. All labels (depth, tokens)
    are generated online by external models during training.

    Directory layout expected:
        data_root/
            {clip_id}/
                {timestamp}/
                    front_wide.jpg, cross_left.jpg, cross_right.jpg
            manifest.json

    Args:
        data_root: Root directory of Physical AI dataset.
        split: "train" or "val" (80/20 split).
        image_size: Target image size (H, W).
        max_clips: Maximum number of clips to use (None for all).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        max_clips: int | None = None,
    ) -> None:
        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        # Load manifest
        manifest_path = os.path.join(data_root, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Build sample list: (clip_id, timestamp)
        all_samples: List[Tuple[str, int]] = []
        clip_ids = list(manifest["clips"].keys())
        if max_clips is not None:
            clip_ids = clip_ids[:max_clips]

        for clip_id in clip_ids:
            clip_info = manifest["clips"][clip_id]
            for ts in clip_info["timestamps"]:
                # Verify images exist
                sample_dir = os.path.join(data_root, clip_id, str(ts))
                has_images = all(
                    os.path.exists(os.path.join(sample_dir, f"{cam}.jpg"))
                    for cam in PHYSICALAI_TARGET_CAMERAS
                )
                if has_images:
                    all_samples.append((clip_id, ts))

        # Deterministic train/val split (80/20)
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_samples))
        split_idx = int(len(all_samples) * 0.8)

        if split == "train":
            self.samples = [all_samples[i] for i in indices[:split_idx]]
        else:
            self.samples = [all_samples[i] for i in indices[split_idx:]]

        # Build clip_id -> sample indices mapping (for fixed sample selection)
        self.clip_to_indices: Dict[str, List[int]] = {}
        for idx, (clip_id, _) in enumerate(self.samples):
            if clip_id not in self.clip_to_indices:
                self.clip_to_indices[clip_id] = []
            self.clip_to_indices[clip_id].append(idx)

        n_clips = len(self.clip_to_indices)
        print(f"ImageAdaptorDataset ({split}): {len(self.samples)} samples from {n_clips} clips")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample (images only).

        Args:
            idx: Sample index.

        Returns:
            Dictionary with:
                - physicalai_images: Dict[str, Tensor] camera name -> (3, H, W) float [0,1].
                - clip_id: str
                - timestamp: int
        """
        clip_id, timestamp = self.samples[idx]
        sample_dir = os.path.join(self.data_root, clip_id, str(timestamp))

        H, W = self.image_size

        # Load Physical AI original images (3 cameras)
        physicalai_images: Dict[str, torch.Tensor] = {}
        for cam_name in PHYSICALAI_TARGET_CAMERAS:
            img_path = os.path.join(sample_dir, f"{cam_name}.jpg")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            physicalai_images[cam_name] = img_tensor

        return {
            "physicalai_images": physicalai_images,
            "clip_id": clip_id,
            "timestamp": timestamp,
        }

    def get_fixed_samples(self, n_samples: int = 5, seed: int = 42) -> List[int]:
        """Select fixed samples from different clips for visualization.

        Args:
            n_samples: Number of fixed samples to select.
            seed: Random seed for deterministic selection.

        Returns:
            List of sample indices.
        """
        rng = np.random.RandomState(seed)
        clip_ids = list(self.clip_to_indices.keys())
        rng.shuffle(clip_ids)

        selected = []
        for clip_id in clip_ids[:n_samples]:
            indices = self.clip_to_indices[clip_id]
            selected.append(indices[0])

        while len(selected) < n_samples and len(selected) < len(self.samples):
            idx = rng.randint(0, len(self.samples))
            if idx not in selected:
                selected.append(idx)

        return selected


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for ImageAdaptorDataset.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Collated batch with stacked tensors.
    """
    cam_names = PHYSICALAI_TARGET_CAMERAS

    result: Dict[str, Any] = {
        "physicalai_images": {
            cam: torch.stack([item["physicalai_images"][cam] for item in batch])
            for cam in cam_names
        },
        "clip_id": [item["clip_id"] for item in batch],
        "timestamp": [item["timestamp"] for item in batch],
    }
    return result
