from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - keep data loading usable without scipy.
    ndi = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloSegObject:
    class_id: int
    polygon: list[tuple[float, float]]


def _list_images(images_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    out.sort()
    return out


def _read_yolo_seg_objects(label_path: Path) -> list[YoloSegObject]:
    """
    Returns YOLO segmentation objects with original class ids and normalized polygons.
    """
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    objects: list[YoloSegObject] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
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
            objects.append(YoloSegObject(class_id=class_id, polygon=pts))
    return objects


def _read_yolo_seg_polygons(label_path: Path) -> list[list[tuple[float, float]]]:
    """
    Backward-compatible polygon reader that ignores class ids.
    """
    return [obj.polygon for obj in _read_yolo_seg_objects(label_path)]


def scan_yolo_class_ids(labels_dir: Path) -> list[int]:
    """
    Scan a YOLO-seg labels directory and return sorted original class ids.
    """
    class_ids: set[int] = set()
    if not labels_dir.exists():
        return []
    for label_path in labels_dir.glob("*.txt"):
        for obj in _read_yolo_seg_objects(label_path):
            class_ids.add(obj.class_id)
    return sorted(class_ids)


def _poly_to_xy(poly: Iterable[tuple[float, float]], w: int, h: int) -> list[tuple[float, float]]:
    return [(max(0.0, min(1.0, x)) * (w - 1), max(0.0, min(1.0, y)) * (h - 1)) for (x, y) in poly]


def polygons_to_mask(polys: Iterable[list[tuple[float, float]]], w: int, h: int) -> Image.Image:
    """
    Make a binary (mode 'L') mask from normalized polygons.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        xy = _poly_to_xy(poly, w, h)
        if len(xy) >= 3:
            draw.polygon(xy, outline=255, fill=255)
    return mask


def _draw_center_seed(instance_mask: Image.Image, radius_frac: float) -> Image.Image:
    """
    Draw a compact seed target inside one instance. For hollow/thin masks, fall back
    to the foreground pixel closest to the mask centroid.
    """
    inst = np.asarray(instance_mask, dtype=np.uint8) > 0
    ys, xs = np.nonzero(inst)
    seed = np.zeros(inst.shape, dtype=np.uint8)
    if ys.size == 0:
        return Image.fromarray(seed, mode="L")

    cx = float(xs.mean())
    cy = float(ys.mean())
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    radius = int(round(max(2.0, min(14.0, min(width, height) * float(radius_frac)))))

    tmp = Image.new("L", instance_mask.size, 0)
    draw = ImageDraw.Draw(tmp)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    proposed = (np.asarray(tmp, dtype=np.uint8) > 0) & inst

    if not proposed.any():
        d2 = (xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2
        j = int(np.argmin(d2))
        cx = float(xs[j])
        cy = float(ys[j])
        radius = max(1, radius // 2)
        tmp = Image.new("L", instance_mask.size, 0)
        draw = ImageDraw.Draw(tmp)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
        proposed = (np.asarray(tmp, dtype=np.uint8) > 0) & inst

    seed[proposed] = 255
    return Image.fromarray(seed, mode="L")


def polygons_to_instance_target(
    polys: Iterable[list[tuple[float, float]]],
    w: int,
    h: int,
    boundary_width: int = 3,
) -> tuple[Image.Image, Image.Image]:
    """
    Make legacy foreground and instance-boundary masks from normalized polygons.
    """
    foreground = Image.new("L", (w, h), 0)
    boundary = Image.new("L", (w, h), 0)
    fg_draw = ImageDraw.Draw(foreground)
    bd_draw = ImageDraw.Draw(boundary)
    line_width = max(1, int(boundary_width))

    for poly in polys:
        xy = _poly_to_xy(poly, w, h)
        if len(xy) < 3:
            continue
        fg_draw.polygon(xy, outline=255, fill=255)
        closed = xy + [xy[0]]
        bd_draw.line(closed, fill=255, width=line_width, joint="curve")

    fg = np.asarray(foreground, dtype=np.uint8)
    bd = np.asarray(boundary, dtype=np.uint8)
    bd = np.where(fg > 0, bd, 0).astype(np.uint8)
    return foreground, Image.fromarray(bd, mode="L")


def objects_to_instance_targets(
    objects: Iterable[YoloSegObject],
    w: int,
    h: int,
    class_ids: Sequence[int],
    boundary_width: int = 3,
    center_radius: float = 0.12,
) -> tuple[list[Image.Image], Image.Image, Image.Image]:
    """
    Make multi-class instance targets:
      class masks: one binary foreground channel per original class id
      boundary: per-instance polygon boundaries
      center: compact per-instance seed regions used by post-processing
    """
    ids = list(class_ids) or [0]
    class_to_channel = {cid: i for i, cid in enumerate(ids)}

    class_masks = [Image.new("L", (w, h), 0) for _ in ids]
    class_draws = [ImageDraw.Draw(mask) for mask in class_masks]
    boundary = Image.new("L", (w, h), 0)
    center = Image.new("L", (w, h), 0)
    bd_draw = ImageDraw.Draw(boundary)
    line_width = max(1, int(boundary_width))

    for obj in objects:
        channel = class_to_channel.get(obj.class_id)
        if channel is None:
            continue
        xy = _poly_to_xy(obj.polygon, w, h)
        if len(xy) < 3:
            continue

        class_draws[channel].polygon(xy, outline=255, fill=255)

        instance_mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(instance_mask).polygon(xy, outline=255, fill=255)
        center_seed = _draw_center_seed(instance_mask, radius_frac=center_radius)
        center = Image.fromarray(np.maximum(np.asarray(center, dtype=np.uint8), np.asarray(center_seed, dtype=np.uint8)), mode="L")

        closed = xy + [xy[0]]
        bd_draw.line(closed, fill=255, width=line_width, joint="curve")

    fg = np.maximum.reduce([np.asarray(mask, dtype=np.uint8) for mask in class_masks]) if class_masks else np.zeros((h, w), dtype=np.uint8)
    bd = np.asarray(boundary, dtype=np.uint8)
    ctr = np.asarray(center, dtype=np.uint8)
    bd = np.where(fg > 0, bd, 0).astype(np.uint8)
    ctr = np.where(fg > 0, ctr, 0).astype(np.uint8)
    return class_masks, Image.fromarray(bd, mode="L"), Image.fromarray(ctr, mode="L")


def objects_to_offset_instance_rasters(
    objects: Iterable[YoloSegObject],
    w: int,
    h: int,
    class_ids: Sequence[int],
    boundary_width: int = 3,
) -> tuple[list[Image.Image], Image.Image, Image.Image]:
    """
    Raster targets for the offset instance head.

    The instance id map is intentionally generated before augmentation, then
    transformed with nearest-neighbor sampling. Offsets are recomputed from the
    transformed id map so flips, affine transforms, and perspective warps keep
    the regression target geometrically valid.
    """
    ids = list(class_ids) or [0]
    class_to_channel = {cid: i for i, cid in enumerate(ids)}

    class_masks = [Image.new("L", (w, h), 0) for _ in ids]
    class_draws = [ImageDraw.Draw(mask) for mask in class_masks]
    boundary = Image.new("L", (w, h), 0)
    bd_draw = ImageDraw.Draw(boundary)
    instance_ids = np.zeros((h, w), dtype=np.uint8)
    line_width = max(1, int(boundary_width))

    next_instance_id = 1
    for obj in objects:
        channel = class_to_channel.get(obj.class_id)
        if channel is None:
            continue
        xy = _poly_to_xy(obj.polygon, w, h)
        if len(xy) < 3:
            continue

        class_draws[channel].polygon(xy, outline=255, fill=255)

        instance_mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(instance_mask).polygon(xy, outline=255, fill=255)
        mask_np = np.asarray(instance_mask, dtype=np.uint8) > 0
        if mask_np.any():
            if next_instance_id > 255:
                # PIL "L" target transforms preserve ids cheaply; practical
                # gate images have far fewer instances, so extra ids are ignored.
                continue
            instance_ids[mask_np] = next_instance_id
            next_instance_id += 1

        closed = xy + [xy[0]]
        bd_draw.line(closed, fill=255, width=line_width, joint="curve")

    fg = np.maximum.reduce([np.asarray(mask, dtype=np.uint8) for mask in class_masks]) if class_masks else np.zeros((h, w), dtype=np.uint8)
    bd = np.asarray(boundary, dtype=np.uint8)
    bd = np.where(fg > 0, bd, 0).astype(np.uint8)
    instance_ids = np.where(fg > 0, instance_ids, 0).astype(np.uint8)
    return class_masks, Image.fromarray(bd, mode="L"), Image.fromarray(instance_ids, mode="L")


def instance_id_to_offset_targets(
    instance_ids: Image.Image | np.ndarray,
    anchor: str = "centroid",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return normalized offsets from each foreground pixel to its instance anchor.
    Offsets are in image-size units, so the regression target remains in [-1, 1].

    For hollow gate frames, the mask centroid can be unstable or close to a
    neighboring nested gate. The "hole" anchor uses the largest enclosed
    opening when available, falling back to the mask centroid for partial gates.
    """
    ids = np.asarray(instance_ids, dtype=np.int32)
    anchor_kind = str(anchor).strip().lower()
    if anchor_kind not in {"centroid", "hole"}:
        raise ValueError("offset anchor must be 'centroid' or 'hole'")

    h, w = ids.shape
    offset_x = np.zeros((h, w), dtype=np.float32)
    offset_y = np.zeros((h, w), dtype=np.float32)
    norm_x = float(max(1, w - 1))
    norm_y = float(max(1, h - 1))

    for instance_id in sorted(int(v) for v in np.unique(ids) if v > 0):
        ys, xs = np.nonzero(ids == instance_id)
        if ys.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        if anchor_kind == "hole":
            filled = _fill_binary_holes(ids == instance_id)
            holes = filled & ~(ids == instance_id)
            if holes.any():
                hole_labels, n_labels = _component_labels_np(holes)
                if n_labels > 0:
                    counts = np.bincount(hole_labels.ravel())
                    if counts.size > 1:
                        hole_id = int(np.argmax(counts[1:]) + 1)
                        hys, hxs = np.nonzero(hole_labels == hole_id)
                        if hys.size > 0:
                            cx = float(hxs.mean())
                            cy = float(hys.mean())
        offset_x[ys, xs] = (cx - xs.astype(np.float32)) / norm_x
        offset_y[ys, xs] = (cy - ys.astype(np.float32)) / norm_y
    return offset_x, offset_y


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if ndi is not None:
        return ndi.binary_fill_holes(mask_bool)

    h, w = mask_bool.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for x in range(w):
        if not mask_bool[0, x]:
            exterior[0, x] = True
            q.append((0, x))
        if not mask_bool[h - 1, x]:
            exterior[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if not mask_bool[y, 0]:
            exterior[y, 0] = True
            q.append((y, 0))
        if not mask_bool[y, w - 1]:
            exterior[y, w - 1] = True
            q.append((y, w - 1))

    while q:
        cy, cx = q.popleft()
        for ny in (cy - 1, cy, cy + 1):
            if ny < 0 or ny >= h:
                continue
            for nx in (cx - 1, cx, cx + 1):
                if nx < 0 or nx >= w or (ny == cy and nx == cx):
                    continue
                if not mask_bool[ny, nx] and not exterior[ny, nx]:
                    exterior[ny, nx] = True
                    q.append((ny, nx))

    return mask_bool | (~mask_bool & ~exterior)


def _component_labels_np(mask: np.ndarray) -> tuple[np.ndarray, int]:
    mask_bool = mask.astype(bool)
    if ndi is not None:
        labels, n_labels = ndi.label(mask_bool, structure=np.ones((3, 3), dtype=bool))
        return labels.astype(np.int32, copy=False), int(n_labels)

    h, w = mask_bool.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_id = 0
    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue
            current_id += 1
            labels[y, x] = current_id
            q: deque[tuple[int, int]] = deque([(y, x)])
            while q:
                cy, cx = q.popleft()
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w or (ny == cy and nx == cx):
                            continue
                        if mask_bool[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_id
                            q.append((ny, nx))
    return labels, current_id


def instance_id_to_hole_seed_target(
    instance_ids: Image.Image | np.ndarray,
    radius_frac: float = 0.08,
) -> np.ndarray:
    """
    Return a compact seed inside the largest enclosed hole of each instance.

    Gate frames are hollow objects; this channel gives each instance a stable
    shape anchor without adding a heavy box/keypoint branch.
    """
    ids = np.asarray(instance_ids, dtype=np.int32)
    h, w = ids.shape
    out = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]

    for instance_id in sorted(int(v) for v in np.unique(ids) if v > 0):
        mask = ids == instance_id
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue

        filled = _fill_binary_holes(mask)
        holes = filled & ~mask
        if not holes.any():
            continue

        hole_labels, n_labels = _component_labels_np(holes)
        if n_labels <= 0:
            continue
        counts = np.bincount(hole_labels.ravel())
        if counts.size <= 1:
            continue
        hole_id = int(np.argmax(counts[1:]) + 1)
        hole = hole_labels == hole_id
        hys, hxs = np.nonzero(hole)
        if hys.size == 0:
            continue

        cx = float(hxs.mean())
        cy = float(hys.mean())
        box_w = int(xs.max() - xs.min() + 1)
        box_h = int(ys.max() - ys.min() + 1)
        radius = max(1, int(round(min(box_w, box_h) * float(radius_frac))))
        seed = ((xx - cx) ** 2 + (yy - cy) ** 2 <= float(radius * radius)) & hole
        if not seed.any():
            seed[int(round(cy)), int(round(cx))] = True
        out[seed] = 1.0

    return out


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


def resize_triplet(
    img: Image.Image,
    foreground: Image.Image,
    boundary: Image.Image,
    size: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    img = img.resize((size, size), resample=Image.BILINEAR)
    foreground = foreground.resize((size, size), resample=Image.NEAREST)
    boundary = boundary.resize((size, size), resample=Image.NEAREST)
    return img, foreground, boundary


def resize_targets(img: Image.Image, targets: Sequence[Image.Image], size: int) -> tuple[Image.Image, list[Image.Image]]:
    img = img.resize((size, size), resample=Image.BILINEAR)
    resized = [target.resize((size, size), resample=Image.NEAREST) for target in targets]
    return img, resized


def _affine_inverse_coeffs(w: int, h: int, angle_deg: float, scale: float, tx: float, ty: float) -> tuple[float, float, float, float, float, float]:
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    a = scale * cos_t
    b = -scale * sin_t
    d = scale * sin_t
    e = scale * cos_t
    c = cx + tx - a * cx - b * cy
    f = cy + ty - d * cx - e * cy
    m = np.array([[a, b, c], [d, e, f], [0.0, 0.0, 1.0]], dtype=np.float64)
    inv = np.linalg.inv(m)
    return (float(inv[0, 0]), float(inv[0, 1]), float(inv[0, 2]), float(inv[1, 0]), float(inv[1, 1]), float(inv[1, 2]))


def _perspective_coeffs(src: Sequence[tuple[float, float]], dst: Sequence[tuple[float, float]]) -> tuple[float, ...]:
    matrix: list[list[float]] = []
    rhs: list[float] = []
    for (x_dst, y_dst), (x_src, y_src) in zip(dst, src, strict=False):
        matrix.append([x_dst, y_dst, 1.0, 0.0, 0.0, 0.0, -x_src * x_dst, -x_src * y_dst])
        matrix.append([0.0, 0.0, 0.0, x_dst, y_dst, 1.0, -y_src * x_dst, -y_src * y_dst])
        rhs.extend([x_src, y_src])
    coeffs = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return tuple(float(v) for v in coeffs.tolist())


def _apply_transform_to_all(
    img: Image.Image,
    targets: Sequence[Image.Image],
    method: int,
    data: tuple[float, ...],
) -> tuple[Image.Image, list[Image.Image]]:
    size = img.size
    img = img.transform(size, method, data, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    out = [target.transform(size, method, data, resample=Image.NEAREST, fillcolor=0) for target in targets]
    return img, out


def _apply_hsv(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img.convert("HSV"), dtype=np.float32)
    hue_shift = rng.uniform(-12.0, 12.0)
    sat_scale = rng.uniform(0.65, 1.45)
    val_scale = rng.uniform(0.65, 1.35)
    arr[..., 0] = (arr[..., 0] + hue_shift) % 255.0
    arr[..., 1] = np.clip(arr[..., 1] * sat_scale, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * val_scale, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")


def _apply_brightness_gradient(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx / max(1, w - 1)) * 2.0 - 1.0
    yy = (yy / max(1, h - 1)) * 2.0 - 1.0
    angle = rng.uniform(0.0, 2.0 * math.pi)
    direction = math.cos(angle) * xx + math.sin(angle) * yy
    strength = rng.uniform(-0.28, 0.28)
    gain = 1.0 + strength * direction
    arr = np.clip(arr * gain[..., None], 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _horizontal_box_blur(arr: np.ndarray, size: int) -> np.ndarray:
    pad = max(1, int(size) // 2)
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=1, dtype=np.float32)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0), (0, 0)), mode="constant")
    w = arr.shape[1]
    return (cumsum[:, size : size + w] - cumsum[:, :w]) / float(size)


def _apply_motion_blur(img: Image.Image, size: int, angle: float) -> Image.Image:
    """
    Directional box blur with arbitrary odd kernel size. Rotate the image into
    blur coordinates, average along rows, then rotate back.
    """
    angle_deg = math.degrees(angle)
    rotated = img.rotate(angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    blurred = _horizontal_box_blur(np.asarray(rotated, dtype=np.float32), size=max(3, int(size)))
    out = Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")
    return out.rotate(-angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def _apply_gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    sigma = rng.uniform(2.0, 16.0)
    noise = np.random.default_rng(rng.randrange(0, 2**32)).normal(0.0, sigma, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _shift_row(row: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return row
    out = np.empty_like(row)
    if shift > 0:
        out[:shift] = row[:1]
        out[shift:] = row[:-shift]
    else:
        s = -shift
        out[-s:] = row[-1:]
        out[:-s] = row[s:]
    return out


def _apply_rolling_shutter(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.uint8)
    h = arr.shape[0]
    amplitude = rng.uniform(-5.0, 5.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    frequency = rng.uniform(0.75, 1.75)
    out = np.empty_like(arr)
    for y in range(h):
        t = y / max(1, h - 1)
        shift = int(round(amplitude * (2.0 * t - 1.0) + 1.5 * math.sin(phase + frequency * 2.0 * math.pi * t)))
        out[y] = _shift_row(arr[y], shift)
    return Image.fromarray(out, mode="RGB").filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.45)))


def _bilinear_sample_rgb(arr: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    src_x = np.clip(src_x, 0.0, w - 1.0)
    src_y = np.clip(src_y, 0.0, h - 1.0)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (src_x - x0)[..., None]
    wy = (src_y - y0)[..., None]
    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _nearest_sample_l(arr: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    x = np.clip(np.rint(src_x).astype(np.int32), 0, w - 1)
    y = np.clip(np.rint(src_y).astype(np.int32), 0, h - 1)
    return arr[y, x]


def _apply_lens_distortion(
    img: Image.Image,
    targets: Sequence[Image.Image],
    rng: random.Random,
) -> tuple[Image.Image, list[Image.Image]]:
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    xn = (xx - cx) / max(1.0, cx)
    yn = (yy - cy) / max(1.0, cy)
    r2 = xn * xn + yn * yn
    k = rng.uniform(-0.08, 0.08)
    factor = 1.0 + k * r2
    src_x = cx + xn * factor * max(1.0, cx)
    src_y = cy + yn * factor * max(1.0, cy)

    img_out = np.clip(_bilinear_sample_rgb(arr, src_x, src_y), 0, 255).astype(np.uint8)
    target_out = []
    for target in targets:
        m = np.asarray(target, dtype=np.uint8)
        target_out.append(Image.fromarray(_nearest_sample_l(m, src_x, src_y).astype(np.uint8), mode="L"))
    return Image.fromarray(img_out, mode="RGB"), target_out


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
    Produces (image_tensor, target_tensor) where target has:
      channels [0:num_classes): per-class foreground masks
      offset head:
        channel num_classes: instance boundary mask
        channels num_classes + 1:num_classes + 3: x/y offsets to instance anchor
      offset_hole head:
        channel num_classes: instance boundary mask
        channel num_classes + 1: compact seed inside each gate's enclosed hole
        channels num_classes + 2:num_classes + 4: x/y offsets to instance anchor
      center head (legacy):
        channel num_classes: instance boundary mask
        channel num_classes + 1: instance center seed mask
    """

    def __init__(
        self,
        root: Path,
        img_size: int = 384,
        augment: bool = False,
        seed: int = 0,
        boundary_width: int = 3,
        class_ids: Sequence[int] | None = None,
        center_radius: float = 0.12,
        hole_radius: float = 0.08,
        instance_head: str = "offset",
        offset_anchor: str = "centroid",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.folders = YoloSegFolders.from_root(self.root)
        self.img_paths = _list_images(self.folders.images_dir)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.rng = random.Random(seed)
        self.boundary_width = int(boundary_width)
        self.class_ids = list(class_ids) if class_ids is not None else scan_yolo_class_ids(self.folders.labels_dir)
        if not self.class_ids:
            self.class_ids = [0]
        self.center_radius = float(center_radius)
        self.hole_radius = float(hole_radius)
        self.instance_head = str(instance_head).strip().lower()
        self.offset_anchor = str(offset_anchor).strip().lower()
        if self.instance_head not in {"offset", "offset_hole", "center"}:
            raise ValueError("instance_head must be 'offset', 'offset_hole', or 'center'")
        if self.offset_anchor not in {"centroid", "hole"}:
            raise ValueError("offset_anchor must be 'centroid' or 'hole'")
        if self.instance_head == "offset":
            self.output_channels = [f"class_{cid}" for cid in self.class_ids] + ["boundary", "offset_x", "offset_y"]
        elif self.instance_head == "offset_hole":
            self.output_channels = [f"class_{cid}" for cid in self.class_ids] + ["boundary", "hole", "offset_x", "offset_y"]
        else:
            self.output_channels = [f"class_{cid}" for cid in self.class_ids] + ["boundary", "center"]

        if not self.img_paths:
            raise FileNotFoundError(f"No images found in: {self.folders.images_dir}")

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    @property
    def out_channels(self) -> int:
        if self.instance_head == "offset_hole":
            return self.num_classes + 4
        return self.num_classes + (3 if self.instance_head == "offset" else 2)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _augment(
        self,
        img: Image.Image,
        targets: Sequence[Image.Image],
    ) -> tuple[Image.Image, list[Image.Image]]:
        targets = list(targets)
        w, h = img.size

        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            targets = [target.transpose(Image.FLIP_LEFT_RIGHT) for target in targets]

        if self.rng.random() < 0.85:
            coeffs = _affine_inverse_coeffs(
                w=w,
                h=h,
                angle_deg=self.rng.uniform(-14.0, 14.0),
                scale=self.rng.uniform(0.82, 1.18),
                tx=self.rng.uniform(-0.08, 0.08) * w,
                ty=self.rng.uniform(-0.08, 0.08) * h,
            )
            img, targets = _apply_transform_to_all(img, targets, Image.AFFINE, coeffs)

        if self.rng.random() < 0.35:
            shift = 0.055 * min(w, h)
            src = [(0.0, 0.0), (float(w - 1), 0.0), (float(w - 1), float(h - 1)), (0.0, float(h - 1))]
            dst = [(x + self.rng.uniform(-shift, shift), y + self.rng.uniform(-shift, shift)) for x, y in src]
            coeffs = _perspective_coeffs(src=src, dst=dst)
            img, targets = _apply_transform_to_all(img, targets, Image.PERSPECTIVE, coeffs)

        if self.rng.random() < 0.25:
            img, targets = _apply_lens_distortion(img, targets, self.rng)

        if self.rng.random() < 0.9:
            img = _apply_hsv(img, self.rng)
        if self.rng.random() < 0.45:
            img = _apply_brightness_gradient(img, self.rng)
        if self.rng.random() < 0.35:
            k = self.rng.randrange(5, 16, 2)
            img = _apply_motion_blur(img, k, self.rng.uniform(0.0, math.pi))
        if self.rng.random() < 0.35:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.25, 1.4)))
        if self.rng.random() < 0.55:
            img = _apply_gaussian_noise(img, self.rng)
        if self.rng.random() < 0.25:
            img = _apply_rolling_shutter(img, self.rng)

        return img, targets

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_paths[idx]
        lbl_path = self.folders.labels_dir / (img_path.stem + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        objects = _read_yolo_seg_objects(lbl_path)
        if self.instance_head == "center":
            class_masks, boundary, center = objects_to_instance_targets(
                objects,
                w=w,
                h=h,
                class_ids=self.class_ids,
                boundary_width=self.boundary_width,
                center_radius=self.center_radius,
            )
            target_maps: list[Image.Image] = [*class_masks, boundary, center]
            img, target_maps = resize_targets(img, target_maps, self.img_size)
            if self.augment:
                img, target_maps = self._augment(img, target_maps)

            x = pil_to_tensor(img)
            ys = [(pil_to_tensor(mask) >= 0.5).float() for mask in target_maps]
            y = torch.cat(ys, dim=0)
            return x, y

        class_masks, boundary, instance_ids = objects_to_offset_instance_rasters(
            objects,
            w=w,
            h=h,
            class_ids=self.class_ids,
            boundary_width=self.boundary_width,
        )
        target_maps = [*class_masks, boundary, instance_ids]
        img, target_maps = resize_targets(img, target_maps, self.img_size)
        if self.augment:
            img, target_maps = self._augment(img, target_maps)

        x = pil_to_tensor(img)
        mask_tensors = [(pil_to_tensor(mask) >= 0.5).float() for mask in target_maps[:-1]]
        offset_x, offset_y = instance_id_to_offset_targets(target_maps[-1], anchor=self.offset_anchor)
        offset_tensor = torch.from_numpy(np.stack([offset_x, offset_y], axis=0)).float()
        if self.instance_head == "offset_hole":
            hole_seed = instance_id_to_hole_seed_target(target_maps[-1], radius_frac=self.hole_radius)
            hole_tensor = torch.from_numpy(hole_seed[None, ...]).float()
            y = torch.cat([*mask_tensors, hole_tensor, offset_tensor], dim=0)
            return x, y

        y = torch.cat([*mask_tensors, offset_tensor], dim=0)
        return x, y
