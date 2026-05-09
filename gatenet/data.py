from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(images_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    out.sort()
    return out


def _read_yolo_seg_polygons(label_path: Path) -> list[list[tuple[float, float]]]:
    """
    Returns list of polygons, each polygon is list of (x_norm, y_norm).
    """
    if not label_path.exists():
        return []
    polys: list[list[tuple[float, float]]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return polys
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        nums = parts[1:]
        if len(nums) % 2 != 0:
            nums = nums[:-1]
        pts: list[tuple[float, float]] = []
        for i in range(0, len(nums), 2):
            try:
                x = float(nums[i])
                y = float(nums[i + 1])
            except ValueError:
                pts = []
                break
            pts.append((x, y))
        if len(pts) >= 3:
            polys.append(pts)
    return polys


def polygons_to_mask(polys: Iterable[list[tuple[float, float]]], w: int, h: int) -> Image.Image:
    """
    Make a binary (mode 'L') mask from normalized polygons.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        xy = [(max(0.0, min(1.0, x)) * (w - 1), max(0.0, min(1.0, y)) * (h - 1)) for (x, y) in poly]
        if len(xy) >= 3:
            draw.polygon(xy, outline=255, fill=255)
    return mask


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def resize_pair(img: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    img = img.resize((size, size), resample=Image.BILINEAR)
    mask = mask.resize((size, size), resample=Image.NEAREST)
    return img, mask


@dataclass(frozen=True)
class YoloSegFolders:
    images_dir: Path
    labels_dir: Path

    @staticmethod
    def from_root(root: Path) -> "YoloSegFolders":
        return YoloSegFolders(images_dir=root / "images", labels_dir=root / "labels")


class YoloSegDataset(Dataset):
    """
    Dataset for YOLO segmentation polygon labels.
    Produces (image_tensor, mask_tensor) where mask is binary (1 channel).
    """

    def __init__(self, root: Path, img_size: int = 384, augment: bool = False, seed: int = 0) -> None:
        super().__init__()
        self.root = Path(root)
        self.folders = YoloSegFolders.from_root(self.root)
        self.img_paths = _list_images(self.folders.images_dir)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.rng = random.Random(seed)

        if not self.img_paths:
            raise FileNotFoundError(f"No images found in: {self.folders.images_dir}")

    def __len__(self) -> int:
        return len(self.img_paths)

    def _augment(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        # Lightweight, deterministic-ish aug to match paper spirit without extra deps.
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.rng.random() < 0.1:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        lbl_path = self.folders.labels_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        polys = _read_yolo_seg_polygons(lbl_path)
        mask = polygons_to_mask(polys, w=w, h=h)

        img, mask = resize_pair(img, mask, self.img_size)
        if self.augment:
            img, mask = self._augment(img, mask)

        x = pil_to_tensor(img)  # (3,H,W)
        y = pil_to_tensor(mask)  # (1,H,W), values 0..1
        y = (y >= 0.5).float()
        return x, y

