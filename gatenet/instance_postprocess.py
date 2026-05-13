from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def _component_labels(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    h, w = mask_bool.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_id = 0

    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue

            pixels: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(y, x)])
            labels[y, x] = -1

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w or (ny == cy and nx == cx):
                            continue
                        if mask_bool[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = -1
                            q.append((ny, nx))

            if len(pixels) < int(min_area):
                for py, px in pixels:
                    labels[py, px] = 0
                continue

            current_id += 1
            for py, px in pixels:
                labels[py, px] = current_id

    return labels


def _grow_labels_into_foreground(labels: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    grown = labels.copy()
    foreground_bool = foreground.astype(bool)
    h, w = grown.shape
    q: deque[tuple[int, int]] = deque()

    ys, xs = np.nonzero(grown > 0)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        q.append((y, x))

    while q:
        cy, cx = q.popleft()
        label_id = grown[cy, cx]
        for ny in (cy - 1, cy, cy + 1):
            if ny < 0 or ny >= h:
                continue
            for nx in (cx - 1, cx, cx + 1):
                if nx < 0 or nx >= w or (ny == cy and nx == cx):
                    continue
                if foreground_bool[ny, nx] and grown[ny, nx] == 0:
                    grown[ny, nx] = label_id
                    q.append((ny, nx))

    return grown


def instance_stats(label_map: np.ndarray) -> list[dict]:
    stats: list[dict] = []
    for label_id in sorted(int(v) for v in np.unique(label_map) if v > 0):
        ys, xs = np.nonzero(label_map == label_id)
        if ys.size == 0:
            continue
        stats.append(
            {
                "id": label_id,
                "area": int(ys.size),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            }
        )
    return stats


def postprocess_instances(
    foreground_prob: np.ndarray,
    boundary_prob: np.ndarray,
    threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    min_area: int = 20,
) -> tuple[np.ndarray, list[dict]]:
    """
    Convert foreground and boundary probabilities into an instance id map.

    Seeds are connected components from foreground pixels with boundary removed.
    Boundary pixels are then grown back into the nearest seed component.
    """
    foreground = np.asarray(foreground_prob) >= float(threshold)
    boundary = np.asarray(boundary_prob) >= float(boundary_threshold)
    seeds = foreground & ~boundary

    labels = _component_labels(seeds, min_area=min_area)
    if labels.max() == 0:
        labels = _component_labels(foreground, min_area=min_area)
    else:
        labels = _grow_labels_into_foreground(labels, foreground)

    return labels, instance_stats(labels)


def label_map_to_u16(label_map: np.ndarray) -> Image.Image:
    arr = np.asarray(label_map, dtype=np.uint16)
    return Image.fromarray(arr, mode="I;16")


def label_map_to_color(label_map: np.ndarray) -> Image.Image:
    labels = np.asarray(label_map, dtype=np.int32)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        color = np.array(
            [
                (37 * label_id + 53) % 255,
                (97 * label_id + 101) % 255,
                (17 * label_id + 193) % 255,
            ],
            dtype=np.uint8,
        )
        rgb[labels == label_id] = color
    return Image.fromarray(rgb, mode="RGB")


def overlay_instances_on_image(img_rgb: Image.Image, label_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = img_rgb.convert("RGB")
    color = np.asarray(label_map_to_color(label_map), dtype=np.uint8)
    base = np.asarray(img, dtype=np.uint8)
    mask = np.asarray(label_map) > 0
    out = base.copy()
    out[mask] = (base[mask].astype(np.float32) * (1.0 - alpha) + color[mask].astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
