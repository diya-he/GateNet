from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - keep inference usable without scipy.
    ndi = None


_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def _component_labels(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if ndi is not None:
        raw_labels, num_labels = ndi.label(mask_bool, structure=_CONNECTIVITY_8)
        if num_labels == 0:
            return np.zeros(mask_bool.shape, dtype=np.int32)

        counts = np.bincount(raw_labels.ravel())
        remap = np.zeros(num_labels + 1, dtype=np.int32)
        next_id = 1
        for label_id in range(1, num_labels + 1):
            if int(counts[label_id]) >= int(min_area):
                remap[label_id] = next_id
                next_id += 1
        return remap[raw_labels].astype(np.int32, copy=False)

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
    grown[~foreground_bool] = 0

    if ndi is not None:
        if not foreground_bool.any() or grown.max() <= 0:
            return grown

        fg_components = _component_labels(foreground_bool, min_area=1)
        for comp_id in sorted(int(v) for v in np.unique(fg_components) if v > 0):
            comp = fg_components == comp_id
            seeds = comp & (grown > 0)
            missing = comp & (grown == 0)
            if not seeds.any() or not missing.any():
                continue

            _, nearest = ndi.distance_transform_edt(~seeds, return_indices=True)
            nearest_y, nearest_x = nearest
            grown[missing] = grown[nearest_y[missing], nearest_x[missing]]
        return grown

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


def _filter_and_remap_labels(label_map: np.ndarray, min_area: int) -> np.ndarray:
    out = np.zeros(label_map.shape, dtype=np.int32)
    next_id = 1
    for label_id in sorted(int(v) for v in np.unique(label_map) if v > 0):
        mask = label_map == label_id
        if int(mask.sum()) < int(min_area):
            continue
        out[mask] = next_id
        next_id += 1
    return out


def _label_count(label_map: np.ndarray) -> int:
    return int(sum(1 for v in np.unique(label_map) if v > 0))


def _label_centroids(label_map: np.ndarray) -> dict[int, tuple[float, float]]:
    centers: dict[int, tuple[float, float]] = {}
    for label_id in sorted(int(v) for v in np.unique(label_map) if v > 0):
        ys, xs = np.nonzero(label_map == label_id)
        if ys.size:
            centers[label_id] = (float(xs.mean()), float(ys.mean()))
    return centers


def _label_areas(label_map: np.ndarray) -> dict[int, int]:
    return {int(label_id): int((label_map == label_id).sum()) for label_id in np.unique(label_map) if label_id > 0}


def _merge_labels_to_count(label_map: np.ndarray, target_count: int) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32).copy()
    target = max(1, int(target_count))
    count = _label_count(labels)
    if count <= target:
        return _filter_and_remap_labels(labels, min_area=1)

    areas = _label_areas(labels)
    centers = _label_centroids(labels)
    keep = {
        label_id
        for label_id, _ in sorted(
            areas.items(),
            key=lambda item: (-item[1], item[0]),
        )[:target]
    }
    if not keep:
        return np.zeros(labels.shape, dtype=np.int32)

    for label_id in sorted(areas):
        if label_id in keep:
            continue
        cx, cy = centers[label_id]
        nearest_id = min(
            keep,
            key=lambda keep_id: (centers[keep_id][0] - cx) ** 2 + (centers[keep_id][1] - cy) ** 2,
        )
        labels[labels == label_id] = nearest_id

    return _filter_and_remap_labels(labels, min_area=1)


def _merge_tiny_labels(label_map: np.ndarray, tiny_area: int) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32).copy()
    if _label_count(labels) <= 1:
        return labels

    areas = _label_areas(labels)
    centers = _label_centroids(labels)
    threshold = max(1, int(tiny_area))
    if not areas or not centers:
        return labels

    for label_id, area in sorted(areas.items(), key=lambda item: item[1]):
        if area >= threshold:
            continue
        if not (labels == label_id).any():
            continue
        candidates = [other_id for other_id, other_area in areas.items() if other_id != label_id and other_area >= area]
        if not candidates:
            continue
        cx, cy = centers[label_id]
        nearest_id = min(
            candidates,
            key=lambda other_id: (centers[other_id][0] - cx) ** 2 + (centers[other_id][1] - cy) ** 2,
        )
        labels[labels == label_id] = nearest_id
        areas[nearest_id] = areas.get(nearest_id, 0) + area
        areas[label_id] = 0

    return _filter_and_remap_labels(labels, min_area=1)


def _holes_in_component(component: np.ndarray, min_hole_area: int) -> list[tuple[float, float, int]]:
    if ndi is None:
        return []

    comp = component.astype(bool)
    if not comp.any():
        return []

    holes = ndi.binary_fill_holes(comp) & ~comp
    hole_labels, num_holes = ndi.label(holes, structure=_CONNECTIVITY_8)
    out: list[tuple[float, float, int]] = []
    min_area = max(1, int(min_hole_area))
    for hole_id in range(1, int(num_holes) + 1):
        ys, xs = np.nonzero(hole_labels == hole_id)
        area = int(ys.size)
        if area < min_area:
            continue
        out.append((float(xs.mean()), float(ys.mean()), area))
    return out


def _split_components_by_enclosed_holes(
    label_map: np.ndarray,
    foreground: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """
    Split a merged hollow-frame component when the geometry says it contains
    more gate openings than instance labels.

    A valid full gate is approximately one rectangular frame around one
    enclosed hole. This function is intentionally guarded at the image level:
    it only acts when the number of significant enclosed holes is larger than
    the current instance count, which keeps already-correct images from being
    over-split by small mask defects.
    """
    if ndi is None:
        return label_map

    labels = np.asarray(label_map, dtype=np.int32).copy()
    fg = np.asarray(foreground).astype(bool)
    if _label_count(labels) <= 0 or not fg.any():
        return labels

    comp_labels, num_components = ndi.label(fg, structure=_CONNECTIVITY_8)
    components: list[tuple[int, np.ndarray, list[tuple[float, float, int]], list[int]]] = []
    total_holes = 0
    min_hole_floor = max(1, int(min_area))

    for comp_id in range(1, int(num_components) + 1):
        comp = comp_labels == comp_id
        comp_area = int(comp.sum())
        if comp_area < min_area:
            continue

        min_hole_area = max(min_hole_floor, int(round(comp_area * 0.005)))
        holes = _holes_in_component(comp, min_hole_area=min_hole_area)
        local_labels = sorted(int(v) for v in np.unique(labels[comp]) if v > 0)
        total_holes += len(holes)
        components.append((comp_id, comp, holes, local_labels))

    if total_holes <= _label_count(labels):
        return labels

    next_id = int(labels.max()) + 1
    for _, comp, holes, local_labels in components:
        if len(holes) <= len(local_labels) or len(holes) <= 1:
            continue

        ys, xs = np.nonzero(comp)
        if ys.size < int(min_area):
            continue

        centers = np.asarray([(cx, cy) for cx, cy, _ in holes], dtype=np.float32)
        assigned = np.zeros(xs.shape[0], dtype=np.int32)
        chunk = 32768
        for start in range(0, xs.shape[0], chunk):
            end = min(start + chunk, xs.shape[0])
            dx = xs[start:end, None].astype(np.float32) - centers[None, :, 0]
            dy = ys[start:end, None].astype(np.float32) - centers[None, :, 1]
            assigned[start:end] = np.argmin(dx * dx + dy * dy, axis=1).astype(np.int32)

        labels[comp] = 0
        for hole_index in range(len(holes)):
            part = assigned == hole_index
            if int(part.sum()) < int(min_area):
                continue
            labels[ys[part], xs[part]] = next_id
            next_id += 1

    labels[~fg] = 0
    return _filter_and_remap_labels(labels, min_area=max(1, int(min_area)))


def _dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    size = max(1, int(kernel_size))
    if size <= 1:
        return mask.astype(bool)
    if ndi is None:
        pad = size // 2
        padded = np.pad(mask.astype(bool), pad, mode="constant", constant_values=False)
        out = np.zeros(mask.shape, dtype=bool)
        for dy in range(size):
            for dx in range(size):
                out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        return out
    return ndi.binary_dilation(mask.astype(bool), structure=np.ones((size, size), dtype=bool))


def _boundary_pair_count(boundary: np.ndarray, min_component_area: int = 120) -> int:
    boundary_components = _component_labels(boundary.astype(bool), min_area=max(1, int(min_component_area)))
    component_count = _label_count(boundary_components)
    if component_count <= 0:
        return 0
    return max(1, int(round(component_count / 2.0)))


def _recover_nested_labels(
    labels: np.ndarray,
    foreground: np.ndarray,
    boundary: np.ndarray,
    offset_labels: np.ndarray | None,
    min_area: int,
    boundary_component_min_area: int = 120,
) -> tuple[np.ndarray, bool]:
    """
    Recover one likely nested/adjacent gate when conservative boundary fallback
    merges it into a larger instance.

    A single gate frame often produces two boundary components (inner/outer
    edges). We only split when both the boundary-pair estimate and multi-scale
    thick-boundary seeds suggest more instances, which avoids splitting large
    single frames that only have multiple visible edges.
    """
    base_count = _label_count(labels)
    if base_count <= 0 or base_count > 2:
        return labels, False

    pair_count = _boundary_pair_count(boundary, min_component_area=boundary_component_min_area)
    if pair_count <= base_count:
        return labels, False

    fg = foreground.astype(bool)
    bd = boundary.astype(bool)
    target_count = base_count + 1
    candidates: list[tuple[int, int, np.ndarray]] = []
    for kernel_size in (7, 9, 11, 13, 15, 19):
        thick_boundary = _dilate_mask(bd, kernel_size)
        seed_labels = _component_labels(fg & ~thick_boundary, min_area=min_area)
        seed_count = _label_count(seed_labels)
        if seed_count > base_count:
            candidates.append((abs(seed_count - target_count), kernel_size, seed_labels))

    if not candidates:
        return labels, False

    exact_candidates = [item for item in candidates if _label_count(item[2]) == target_count]
    if exact_candidates:
        _, _, recovered = sorted(exact_candidates, key=lambda item: item[1])[0]
        return recovered, True

    if offset_labels is not None and _label_count(offset_labels) >= target_count:
        return _merge_labels_to_count(offset_labels, target_count=target_count), True

    _, _, recovered = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return _merge_labels_to_count(recovered, target_count=target_count), True


def _fill_unlabeled_foreground(
    label_map: np.ndarray,
    foreground: np.ndarray,
    min_new_instance_area: int,
) -> np.ndarray:
    """
    Ensure every foreground pixel belongs to an instance.

    Offset voting can intentionally drop tiny vote clusters. Those pixels should
    not leave holes in the instance map: if they touch an existing instance they
    are grown into it; very small disconnected leftovers are assigned to the
    nearest existing instance instead of becoming noisy extra counts.
    """
    fg = foreground.astype(bool)
    labels = np.asarray(label_map, dtype=np.int32).copy()
    labels[~fg] = 0

    if not fg.any():
        return np.zeros(labels.shape, dtype=np.int32)

    if labels.max() > 0:
        labels = _grow_labels_into_foreground(labels, fg)

    remaining = fg & (labels == 0)
    if not remaining.any():
        return labels

    leftovers = _component_labels(remaining, min_area=1)
    centers = _label_centroids(labels)
    next_id = int(labels.max()) + 1
    min_area = max(1, int(min_new_instance_area))

    for comp_id in sorted(int(v) for v in np.unique(leftovers) if v > 0):
        comp = leftovers == comp_id
        area = int(comp.sum())
        if area >= min_area or not centers:
            labels[comp] = next_id
            ys, xs = np.nonzero(comp)
            centers[next_id] = (float(xs.mean()), float(ys.mean()))
            next_id += 1
            continue

        ys, xs = np.nonzero(comp)
        cx = float(xs.mean())
        cy = float(ys.mean())
        nearest_id = min(
            centers,
            key=lambda label_id: (centers[label_id][0] - cx) ** 2 + (centers[label_id][1] - cy) ** 2,
        )
        labels[comp] = nearest_id

    return labels


def _normalize_offset_array(offset_xy: np.ndarray) -> np.ndarray | None:
    offsets = np.asarray(offset_xy, dtype=np.float32)
    if offsets.ndim != 3:
        return None
    if offsets.shape[0] == 2:
        return offsets
    if offsets.shape[-1] == 2:
        return np.moveaxis(offsets, -1, 0)
    return None


def _offset_vote_labels(
    foreground: np.ndarray,
    offset_xy: np.ndarray,
    min_area: int,
    seed_min_area: int,
    vote_bin: int = 8,
    vote_min_count: int | None = None,
) -> np.ndarray:
    offsets = _normalize_offset_array(offset_xy)
    if offsets is None:
        return np.zeros(foreground.shape, dtype=np.int32)

    fg = foreground.astype(bool)
    h, w = fg.shape
    if offsets.shape[-2:] != (h, w):
        return np.zeros(fg.shape, dtype=np.int32)

    ys, xs = np.nonzero(fg)
    if ys.size < int(min_area):
        return np.zeros(fg.shape, dtype=np.int32)

    norm_x = float(max(1, w - 1))
    norm_y = float(max(1, h - 1))
    vote_x = np.clip(xs.astype(np.float32) + offsets[0, ys, xs] * norm_x, 0.0, norm_x)
    vote_y = np.clip(ys.astype(np.float32) + offsets[1, ys, xs] * norm_y, 0.0, norm_y)

    bin_size = max(2, int(vote_bin))
    hist_h = int(np.ceil(h / bin_size))
    hist_w = int(np.ceil(w / bin_size))
    bx = np.clip((vote_x / bin_size).astype(np.int32), 0, hist_w - 1)
    by = np.clip((vote_y / bin_size).astype(np.int32), 0, hist_h - 1)
    hist = np.zeros((hist_h, hist_w), dtype=np.int32)
    np.add.at(hist, (by, bx), 1)

    min_votes = int(vote_min_count) if vote_min_count is not None else max(int(seed_min_area), 3)
    peak_labels = _component_labels(hist >= min_votes, min_area=1)

    centers: list[tuple[float, float]] = []
    for peak_id in sorted(int(v) for v in np.unique(peak_labels) if v > 0):
        py, px = np.nonzero(peak_labels == peak_id)
        weights = hist[py, px].astype(np.float32)
        total = float(weights.sum())
        if total < float(min_votes):
            continue
        cx = float(((px.astype(np.float32) + 0.5) * bin_size * weights).sum() / total)
        cy = float(((py.astype(np.float32) + 0.5) * bin_size * weights).sum() / total)
        centers.append((min(cx, norm_x), min(cy, norm_y)))

    if not centers:
        return np.zeros(fg.shape, dtype=np.int32)

    center_arr = np.asarray(centers, dtype=np.float32)
    label_ids = np.zeros(ys.shape[0], dtype=np.int32)
    chunk = 32768
    for start in range(0, ys.shape[0], chunk):
        end = min(start + chunk, ys.shape[0])
        dx = vote_x[start:end, None] - center_arr[None, :, 0]
        dy = vote_y[start:end, None] - center_arr[None, :, 1]
        label_ids[start:end] = np.argmin(dx * dx + dy * dy, axis=1).astype(np.int32) + 1

    labels = np.zeros(fg.shape, dtype=np.int32)
    labels[ys, xs] = label_ids
    return _filter_and_remap_labels(labels, min_area=min_area)


def _boundary_seed_labels(
    foreground: np.ndarray,
    boundary: np.ndarray,
    min_area: int,
) -> np.ndarray:
    seeds = foreground.astype(bool) & ~boundary.astype(bool)
    labels = _component_labels(seeds, min_area=min_area)
    if labels.max() > 0:
        labels = _grow_labels_into_foreground(labels, foreground)
    return labels


def _labels_from_seed_centers(
    foreground: np.ndarray,
    centers: list[tuple[float, float]],
    min_area: int,
) -> np.ndarray:
    fg = foreground.astype(bool)
    if not centers or not fg.any():
        return np.zeros(fg.shape, dtype=np.int32)

    ys, xs = np.nonzero(fg)
    center_arr = np.asarray(centers, dtype=np.float32)
    label_ids = np.zeros(ys.shape[0], dtype=np.int32)
    chunk = 32768
    for start in range(0, ys.shape[0], chunk):
        end = min(start + chunk, ys.shape[0])
        dx = xs[start:end, None].astype(np.float32) - center_arr[None, :, 0]
        dy = ys[start:end, None].astype(np.float32) - center_arr[None, :, 1]
        label_ids[start:end] = np.argmin(dx * dx + dy * dy, axis=1).astype(np.int32) + 1

    labels = np.zeros(fg.shape, dtype=np.int32)
    labels[ys, xs] = label_ids
    return _filter_and_remap_labels(labels, min_area=min_area)


def _hole_seed_labels(
    foreground: np.ndarray,
    hole_prob: np.ndarray,
    threshold: float,
    min_area: int,
    seed_min_area: int,
) -> np.ndarray:
    holes = np.asarray(hole_prob, dtype=np.float32) >= float(threshold)
    seed_labels = _component_labels(holes, min_area=max(1, int(seed_min_area)))
    centers = list(_label_centroids(seed_labels).values())
    return _labels_from_seed_centers(foreground, centers, min_area=min_area)


def instance_stats(
    label_map: np.ndarray,
    class_probs: np.ndarray | None = None,
    class_ids: list[int] | None = None,
) -> list[dict]:
    stats: list[dict] = []
    probs = None if class_probs is None else np.asarray(class_probs, dtype=np.float32)
    ids = class_ids
    if probs is not None:
        if probs.ndim == 2:
            probs = probs[None, ...]
        if ids is None:
            ids = list(range(probs.shape[0]))

    for label_id in sorted(int(v) for v in np.unique(label_map) if v > 0):
        ys, xs = np.nonzero(label_map == label_id)
        if ys.size == 0:
            continue
        row = {
            "id": label_id,
            "area": int(ys.size),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        }
        if probs is not None and ids is not None and probs.shape[0] > 0:
            scores = probs[:, ys, xs].mean(axis=1)
            class_index = int(np.argmax(scores))
            row["class_index"] = class_index
            row["class_id"] = int(ids[class_index]) if class_index < len(ids) else class_index
            row["class_score"] = float(scores[class_index])
        stats.append(row)
    return stats


def postprocess_instances(
    foreground_prob: np.ndarray,
    boundary_prob: np.ndarray,
    center_prob: np.ndarray | None = None,
    hole_prob: np.ndarray | None = None,
    offset_xy: np.ndarray | None = None,
    class_probs: np.ndarray | None = None,
    class_ids: list[int] | None = None,
    threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    center_threshold: float = 0.45,
    hole_threshold: float = 0.45,
    min_area: int = 20,
    seed_min_area: int = 3,
    vote_bin: int = 8,
    vote_min_count: int | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Convert foreground and boundary probabilities into an instance id map.

    Offset-head checkpoints use foreground pixels to vote for their instance
    centers, which is better for hollow gate frames. Center-head checkpoints use
    compact seeds as instance origins. Legacy two-channel checkpoints fall back
    to foreground-minus-boundary seeds.
    """
    if class_probs is not None:
        probs = np.asarray(class_probs, dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[None, ...]
        foreground_prob = probs.max(axis=0)

    foreground = np.asarray(foreground_prob) >= float(threshold)
    boundary = np.asarray(boundary_prob) >= float(boundary_threshold)

    labels = np.zeros(foreground.shape, dtype=np.int32)
    hole_labels = np.zeros(foreground.shape, dtype=np.int32)
    if hole_prob is not None:
        hole_labels = _hole_seed_labels(
            foreground,
            hole_prob,
            threshold=hole_threshold,
            min_area=min_area,
            seed_min_area=seed_min_area,
        )

    if offset_xy is not None:
        offset_labels = _offset_vote_labels(
            foreground,
            offset_xy,
            min_area=min_area,
            seed_min_area=seed_min_area,
            vote_bin=vote_bin,
            vote_min_count=vote_min_count,
        )
        labels = offset_labels

        boundary_labels = _boundary_seed_labels(foreground, boundary, min_area=min_area)
        offset_count = _label_count(offset_labels)
        boundary_count = _label_count(boundary_labels)
        if boundary_count > 0 and (offset_count == 0 or offset_count > boundary_count + 1):
            labels = boundary_labels

        hole_count = _label_count(hole_labels)
        labels_count = _label_count(labels)
        if hole_count > 0:
            if labels_count == 0:
                labels = hole_labels
            elif labels_count <= 2 and hole_count == labels_count + 1:
                labels = _merge_labels_to_count(hole_labels, target_count=labels_count + 1)

        labels, _ = _recover_nested_labels(
            labels,
            foreground,
            boundary,
            offset_labels=offset_labels,
            min_area=min_area,
        )

    if labels.max() == 0 and center_prob is not None:
        centers = (np.asarray(center_prob) >= float(center_threshold)) & foreground
        labels = _component_labels(centers, min_area=seed_min_area)
        if labels.max() > 0:
            labels = _grow_labels_into_foreground(labels, foreground & ~boundary)
            labels = _grow_labels_into_foreground(labels, foreground)

    if labels.max() == 0 and hole_labels.max() > 0:
        labels = hole_labels

    if labels.max() == 0:
        labels = _boundary_seed_labels(foreground, boundary, min_area=min_area)
        if labels.max() == 0:
            labels = _component_labels(foreground, min_area=min_area)

    labels = _fill_unlabeled_foreground(labels, foreground, min_new_instance_area=min_area)
    labels = _split_components_by_enclosed_holes(labels, foreground, min_area=min_area)
    labels = _merge_tiny_labels(labels, tiny_area=max(int(min_area) + 10, 30))
    return labels, instance_stats(labels, class_probs=class_probs, class_ids=class_ids)


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
