from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

from gatenet.data import (
    IMAGE_SUFFIXES,
    YoloSegObject,
    _apply_brightness_gradient,
    _apply_gaussian_noise,
    _apply_hsv,
    _apply_motion_blur,
    _apply_rolling_shutter,
    _poly_to_xy,
    _read_yolo_seg_objects,
)


@dataclass(frozen=True)
class InstanceAsset:
    image_path: Path
    label_path: Path
    class_id: int
    polygon: tuple[tuple[float, float], ...]


def _list_images(images_dir: Path) -> list[Path]:
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    paths.sort()
    return paths


def _draw_objects_mask(objects: list[YoloSegObject], w: int, h: int, dilate: int = 9) -> np.ndarray:
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for obj in objects:
        xy = _poly_to_xy(obj.polygon, w, h)
        if len(xy) >= 3:
            draw.polygon(xy, outline=255, fill=255)
    arr = np.asarray(mask, dtype=np.uint8)
    if dilate > 1 and arr.any():
        arr = cv2.dilate(arr, np.ones((int(dilate), int(dilate)), np.uint8), iterations=1)
    return arr


def _object_mask(obj: YoloSegObject, w: int, h: int) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    xy = _poly_to_xy(obj.polygon, w, h)
    if len(xy) >= 3:
        ImageDraw.Draw(mask).polygon(xy, outline=255, fill=255)
    return mask


def _crop_asset(asset: InstanceAsset, rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(asset.image_path).convert("RGB")
    w, h = img.size
    obj = YoloSegObject(class_id=asset.class_id, polygon=list(asset.polygon))
    mask = _object_mask(obj, w=w, h=h)
    mask_arr = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(mask_arr > 0)
    if ys.size == 0:
        raise ValueError(f"empty asset mask: {asset.image_path}")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    pad = int(round(max(bw, bh) * rng.uniform(0.03, 0.12)))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    crop_img = np.asarray(img.crop((x1, y1, x2 + 1, y2 + 1)), dtype=np.uint8)
    crop_mask = mask_arr[y1 : y2 + 1, x1 : x2 + 1]
    polygon_abs = np.asarray(_poly_to_xy(asset.polygon, w, h), dtype=np.float32)
    polygon_local = polygon_abs - np.asarray([x1, y1], dtype=np.float32)
    return crop_img, crop_mask, polygon_local


def _apply_hsv_np(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return np.asarray(_apply_hsv(img, rng), dtype=np.uint8)


def _random_destination_quad(
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
    rng: random.Random,
) -> np.ndarray:
    aspect = src_h / max(1.0, float(src_w))
    gate_w = rng.uniform(0.10, 0.42) * out_w
    gate_h = gate_w * aspect * rng.uniform(0.75, 1.25)
    gate_w = min(gate_w, out_w * 0.82)
    gate_h = min(gate_h, out_h * 0.82)

    cx = rng.uniform(-0.05 * out_w, 1.05 * out_w)
    cy = rng.uniform(0.08 * out_h, 0.92 * out_h)
    angle = math.radians(rng.uniform(-24.0, 24.0))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rect = np.asarray(
        [
            [-gate_w * 0.5, -gate_h * 0.5],
            [gate_w * 0.5, -gate_h * 0.5],
            [gate_w * 0.5, gate_h * 0.5],
            [-gate_w * 0.5, gate_h * 0.5],
        ],
        dtype=np.float32,
    )
    rot = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    quad = rect @ rot.T + np.asarray([cx, cy], dtype=np.float32)

    jitter = rng.uniform(0.02, 0.18)
    quad[:, 0] += np.asarray([rng.uniform(-jitter, jitter) * gate_w for _ in range(4)], dtype=np.float32)
    quad[:, 1] += np.asarray([rng.uniform(-jitter, jitter) * gate_h for _ in range(4)], dtype=np.float32)
    return quad.astype(np.float32)


def _clip_polygon_norm(points: np.ndarray, w: int, h: int) -> list[tuple[float, float]]:
    pts = points.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
    # Drop repeated adjacent points after clipping.
    out: list[tuple[float, float]] = []
    last: tuple[int, int] | None = None
    for x, y in pts:
        key = (int(round(float(x))), int(round(float(y))))
        if key == last:
            continue
        out.append((float(x) / max(1.0, w - 1), float(y) / max(1.0, h - 1)))
        last = key
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) < 1e-6 and abs(out[0][1] - out[-1][1]) < 1e-6:
        out.pop()
    return out if len(out) >= 3 else []


def _paste_warped_asset(
    canvas: np.ndarray,
    occupied: np.ndarray,
    asset: InstanceAsset,
    rng: random.Random,
    max_tries: int = 12,
) -> tuple[np.ndarray, list[tuple[float, float]]] | None:
    out_h, out_w = canvas.shape[:2]
    crop_img, crop_mask, polygon_local = _crop_asset(asset, rng)
    crop_h, crop_w = crop_mask.shape
    if crop_h < 4 or crop_w < 4:
        return None

    crop_img = _apply_hsv_np(crop_img, rng)
    src = np.asarray([[0, 0], [crop_w - 1, 0], [crop_w - 1, crop_h - 1], [0, crop_h - 1]], dtype=np.float32)

    for _ in range(max_tries):
        dst = _random_destination_quad(crop_w, crop_h, out_w, out_h, rng)
        m = cv2.getPerspectiveTransform(src, dst)
        warped_mask = cv2.warpPerspective(crop_mask, m, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)
        mask_bool = warped_mask > 0
        area = int(mask_bool.sum())
        if area < 40:
            continue
        overlap = int(np.logical_and(mask_bool, occupied > 0).sum()) / max(1, area)
        if overlap > 0.18:
            continue

        warped_img = cv2.warpPerspective(crop_img, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        alpha = (warped_mask.astype(np.float32) / 255.0)[..., None]
        edge_soften = rng.choice([0, 1, 1, 2])
        if edge_soften > 0:
            alpha_2d = cv2.GaussianBlur(warped_mask.astype(np.float32) / 255.0, (0, 0), sigmaX=float(edge_soften))
            alpha = np.clip(alpha_2d, 0.0, 1.0)[..., None]
        canvas[:] = np.clip(canvas.astype(np.float32) * (1.0 - alpha) + warped_img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        occupied[mask_bool] = 255

        ones = np.ones((polygon_local.shape[0], 1), dtype=np.float32)
        pts_h = np.concatenate([polygon_local, ones], axis=1) @ m.T
        pts = pts_h[:, :2] / np.maximum(pts_h[:, 2:3], 1e-6)
        poly_norm = _clip_polygon_norm(pts, out_w, out_h)
        return warped_mask, poly_norm

    return None


def _degrade_image(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.65:
        img = _apply_brightness_gradient(img, rng)
    if rng.random() < 0.55:
        k = rng.randrange(5, 16, 2)
        img = _apply_motion_blur(img, k, rng.uniform(0.0, math.pi))
    if rng.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))
    if rng.random() < 0.65:
        img = _apply_gaussian_noise(img, rng)
    if rng.random() < 0.35:
        img = _apply_rolling_shutter(img, rng)
    return img


def _load_background(bg_path: Path, out_w: int, out_h: int, rng: random.Random) -> Image.Image:
    img = Image.open(bg_path).convert("RGB")
    w, h = img.size
    scale = max(float(out_w) / max(1, w), float(out_h) / max(1, h))
    # Random overscale/crop gives a little parallax-like variety while keeping
    # all synthetic images at the same output size.
    scale *= rng.uniform(1.0, 1.22)
    rw = max(out_w, int(round(w * scale)))
    rh = max(out_h, int(round(h * scale)))
    img = img.resize((rw, rh), Image.BILINEAR)
    left = rng.randint(0, max(0, rw - out_w))
    top = rng.randint(0, max(0, rh - out_h))
    return img.crop((left, top, left + out_w, top + out_h))


def _write_yolo_label(path: Path, rows: list[tuple[int, list[tuple[float, float]]]]) -> None:
    lines: list[str] = []
    for class_id, poly in rows:
        if len(poly) < 3:
            continue
        coords = " ".join(f"{v:.8f}" for xy in poly for v in xy)
        lines.append(f"{int(class_id)} {coords}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _copy_split(src_split: Path, dst_split: Path, prefix: str = "") -> int:
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    (dst_split / "labels").mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in _list_images(src_split / "images"):
        label_path = src_split / "labels" / f"{img_path.stem}.txt"
        stem = f"{prefix}{img_path.stem}"
        shutil.copy2(img_path, dst_split / "images" / f"{stem}{img_path.suffix.lower()}")
        if label_path.exists():
            shutil.copy2(label_path, dst_split / "labels" / f"{stem}.txt")
        else:
            (dst_split / "labels" / f"{stem}.txt").write_text("", encoding="utf-8")
        count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MonoRace-style synthetic gate segmentation data.")
    ap.add_argument("--data", type=Path, required=True, help="Input split root containing train/ and test/")
    ap.add_argument("--out", type=Path, required=True, help="Output split root for mixed real+synthetic data")
    ap.add_argument("--background-dir", type=Path, required=True, help="Directory of unrelated background images")
    ap.add_argument("--synthetic-count", type=int, default=None, help="Number of synthetic train images. Default: 7x real train count.")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-gates", type=int, default=2)
    ap.add_argument("--max-gates", type=int, default=5)
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--out-height", type=int, default=720)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    train_in = args.data / "train"
    test_in = args.data / "test"
    train_out = args.out / "train"
    test_out = args.out / "test"

    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} already exists and is not empty. Use --overwrite to regenerate.")

    real_count = _copy_split(train_in, train_out, prefix="")
    test_count = _copy_split(test_in, test_out, prefix="")
    synthetic_count = int(args.synthetic_count) if args.synthetic_count is not None else real_count * 7

    train_images = _list_images(train_in / "images")
    assets: list[InstanceAsset] = []
    for img_path in train_images:
        label_path = train_in / "labels" / f"{img_path.stem}.txt"
        for obj in _read_yolo_seg_objects(label_path):
            if len(obj.polygon) >= 3:
                assets.append(InstanceAsset(img_path, label_path, int(obj.class_id), tuple(obj.polygon)))
    if not assets:
        raise SystemExit("No labeled gate instances found for synthesis.")
    background_images = _list_images(args.background_dir)
    if not background_images:
        raise SystemExit(f"No background images found in {args.background_dir}")

    for i in tqdm(range(synthetic_count), desc="synthesizing"):
        bg = _load_background(rng.choice(background_images), out_w=int(args.out_width), out_h=int(args.out_height), rng=rng)
        bg = _apply_hsv(bg, rng)
        canvas = np.asarray(bg, dtype=np.uint8).copy()
        occupied = np.zeros(canvas.shape[:2], dtype=np.uint8)

        rows: list[tuple[int, list[tuple[float, float]]]] = []
        min_gates = max(1, int(args.min_gates))
        max_gates = max(min_gates, int(args.max_gates))
        n_gates = rng.randint(min_gates, max_gates)
        attempts = 0
        while len(rows) < n_gates and attempts < n_gates * 8:
            attempts += 1
            asset = rng.choice(assets)
            pasted = _paste_warped_asset(canvas, occupied, asset, rng)
            if pasted is None:
                continue
            _, poly = pasted
            if len(poly) >= 3:
                rows.append((asset.class_id, poly))

        # Rare failed layout: keep trying so each synthetic image has multiple
        # labeled gate instances when space allows.
        attempts = 0
        while len(rows) < min_gates and attempts < min_gates * 12:
            attempts += 1
            for _ in range(8):
                asset = rng.choice(assets)
                pasted = _paste_warped_asset(canvas, occupied, asset, rng, max_tries=4)
                if pasted is not None and len(pasted[1]) >= 3:
                    rows.append((asset.class_id, pasted[1]))
                    break

        out_img = _degrade_image(Image.fromarray(canvas, mode="RGB"), rng)
        stem = f"synth_{i:06d}"
        out_img.save(train_out / "images" / f"{stem}.jpg", quality=int(args.jpeg_quality), subsampling=1)
        _write_yolo_label(train_out / "labels" / f"{stem}.txt", rows)

    meta = {
        "source": str(args.data),
        "real_train": real_count,
        "test": test_count,
        "synthetic_train": synthetic_count,
        "background_dir": str(args.background_dir),
        "background_count": len(background_images),
        "synthetic_to_real_ratio": synthetic_count / max(1, real_count),
        "seed": int(args.seed),
        "min_gates": int(args.min_gates),
        "max_gates": int(args.max_gates),
        "notes": [
            "Foreground gates are cropped from real labels, HSV-jittered separately, perspective warped, and composited onto unrelated background images.",
            "Each synthetic image contains multiple gate instances when layout succeeds.",
            "Background HSV, brightness gradients, motion blur, Gaussian blur/noise, and rolling-shutter degradation are applied offline.",
            "The training dataloader applies the online affine/perspective/HSV/lens/noise/blur augmentations unless --no-augment is used.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "synthesis_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
