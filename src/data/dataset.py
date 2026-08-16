"""Paired blurry/sharp image dataset.

Deliberately dataset-agnostic: this class only assumes two directories of
images whose filenames match after stripping a configurable set of stem
suffixes (GoPro ships blurry files as e.g. `foo_blurred.png` next to
`foo_sharp.png`; a different dataset with exact-matching filenames just
needs an empty suffix list). Swapping in a different image-pair dataset
later (e.g. scanned documents) means pointing `data.*_subdir` at new
directories in the config -- this module and its interface do not change.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import Config


class CustomToTensor:
    """PIL Image / ndarray (H, W, C) in [0, 255] -> FloatTensor (C, H, W) in [0, 1]."""

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)
        img_np = np.array(pic, dtype=np.uint8)
        if img_np.ndim < 3:
            img_np = np.expand_dims(img_np, axis=-1)
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1)))
        return img_tensor.float().div(255)


def _strip_suffixes(stem: str, suffixes: list[str]) -> str:
    for suf in suffixes:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


class PairedImageDataset(Dataset):
    """Loads matched (blurry, sharp) image pairs from two directories."""

    def __init__(
        self,
        blurry_dir: str | Path,
        sharp_dir: str | Path,
        train: bool = True,
        target_size: tuple[int, int] = (512, 512),
        blurry_stem_suffixes: list[str] | None = None,
        sharp_stem_suffixes: list[str] | None = None,
        gaussian_blur_augmentation: bool = True,
    ):
        self.blurry_dir = str(blurry_dir)
        self.sharp_dir = str(sharp_dir)
        self.train = train
        self.target_size = tuple(target_size)
        self.blurry_stem_suffixes = blurry_stem_suffixes or []
        self.sharp_stem_suffixes = sharp_stem_suffixes or []

        blurry_paths = natsorted(glob.glob(os.path.join(self.blurry_dir, "*.*")))
        sharp_paths = natsorted(glob.glob(os.path.join(self.sharp_dir, "*.*")))
        if not blurry_paths or not sharp_paths:
            raise RuntimeError(
                f"No images found in {self.blurry_dir} or {self.sharp_dir}. "
                "Check dataset paths in the config."
            )

        sharp_by_stem = {
            _strip_suffixes(Path(p).stem, self.sharp_stem_suffixes): p for p in sharp_paths
        }

        self.image_pairs: list[tuple[str, str]] = []
        for blurry_path in blurry_paths:
            stem = _strip_suffixes(Path(blurry_path).stem, self.blurry_stem_suffixes)
            sharp_path = sharp_by_stem.get(stem)
            if sharp_path:
                self.image_pairs.append((blurry_path, sharp_path))

        if not self.image_pairs:
            raise RuntimeError(
                "No matching blurry/sharp pairs found. Check filename conventions "
                "and blurry_stem_suffixes / sharp_stem_suffixes in the config."
            )

        self.base_transforms = transforms.Compose(
            [transforms.Resize(self.target_size, transforms.InterpolationMode.BICUBIC), CustomToTensor()]
        )
        self.gaussian_blur_augmentation = gaussian_blur_augmentation
        if gaussian_blur_augmentation:
            self.train_additional_transforms = transforms.Compose(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]
            )

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int):
        blurry_path, sharp_path = self.image_pairs[idx]
        blurry_image = Image.open(blurry_path).convert("RGB")
        sharp_image = Image.open(sharp_path).convert("RGB")

        if self.train:
            if blurry_image.size[0] < self.target_size[1] or blurry_image.size[1] < self.target_size[0]:
                blurry_image = transforms.functional.resize(blurry_image, self.target_size, transforms.InterpolationMode.BICUBIC)
                sharp_image = transforms.functional.resize(sharp_image, self.target_size, transforms.InterpolationMode.BICUBIC)

            i, j, h, w = transforms.RandomCrop.get_params(blurry_image, output_size=self.target_size)
            blurry_image = transforms.functional.crop(blurry_image, i, j, h, w)
            sharp_image = transforms.functional.crop(sharp_image, i, j, h, w)

            if torch.rand(1) < 0.5:
                blurry_image = transforms.functional.hflip(blurry_image)
                sharp_image = transforms.functional.hflip(sharp_image)

            blurry_tensor = self.base_transforms(blurry_image)
            sharp_tensor = self.base_transforms(sharp_image)
            if self.gaussian_blur_augmentation:
                blurry_tensor = self.train_additional_transforms(blurry_tensor)
        else:
            blurry_tensor = self.base_transforms(blurry_image)
            sharp_tensor = self.base_transforms(sharp_image)

        return blurry_tensor, sharp_tensor, blurry_path


def build_dataset(config: Config, split: str) -> PairedImageDataset:
    """split: 'train' or 'test'."""
    dataset_root = config.resolve(config.paths.dataset_root)
    if split == "train":
        blurry_dir = dataset_root / config.data.train_blurry_subdir
        sharp_dir = dataset_root / config.data.train_sharp_subdir
        train_flag = True
    elif split == "test":
        blurry_dir = dataset_root / config.data.test_blurry_subdir
        sharp_dir = dataset_root / config.data.test_sharp_subdir
        train_flag = False
    else:
        raise ValueError(f"Unknown split: {split}")

    return PairedImageDataset(
        blurry_dir=blurry_dir,
        sharp_dir=sharp_dir,
        train=train_flag,
        target_size=tuple(config.data.image_size),
        blurry_stem_suffixes=config.data.blurry_stem_suffixes,
        sharp_stem_suffixes=config.data.sharp_stem_suffixes,
        gaussian_blur_augmentation=config.data.gaussian_blur_augmentation,
    )


def build_dataloader(config: Config, split: str, shuffle: bool | None = None) -> DataLoader:
    dataset = build_dataset(config, split)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
