from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from gatenet.data import YoloSegDataset
from gatenet.instance_postprocess import label_map_to_color, label_map_to_u16, overlay_instances_on_image, postprocess_instances
from gatenet.losses import foreground_from_channels, iou_binary
from gatenet.model import GateNet


def auto_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def resolve_output_layout(meta: dict, out_channels: int) -> str:
    layout = str(meta.get("output_layout") or "")
    if layout in {"class_masks_boundary_offset", "class_masks_boundary_hole_offset", "class_masks_boundary_center"}:
        return layout
    if int(meta.get("regression_channels", 0)) == 2:
        if out_channels >= 5 and str(meta.get("instance_head") or "") == "offset_hole":
            return "class_masks_boundary_hole_offset"
        return "class_masks_boundary_offset"
    return "legacy" if out_channels <= 2 else "class_masks_boundary_center"


def class_channel_count(out_channels: int, output_layout: str) -> int:
    if out_channels <= 2:
        return 1
    if output_layout == "class_masks_boundary_hole_offset":
        return max(1, out_channels - 4)
    if output_layout == "class_masks_boundary_offset":
        return max(1, out_channels - 3)
    return max(1, out_channels - 2)


def resolve_class_ids(meta: dict, out_channels: int, output_layout: str) -> list[int]:
    if out_channels <= 2:
        return list(meta.get("class_ids") or [0])
    n_cls = class_channel_count(out_channels, output_layout)
    ids = list(meta.get("class_ids") or [])
    if len(ids) != n_cls:
        ids = list(range(n_cls))
    return [int(v) for v in ids]


def load_model(ckpt_path: Path, device: torch.device) -> tuple[GateNet, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta") or {}
    f = int(meta.get("f", 4))
    conv_kind = str(meta.get("conv_kind", "standard"))
    up_kind = str(meta.get("up_kind", "transpose"))
    state_dict = ckpt["state_dict"]
    out_channels = int(meta.get("out_channels") or state_dict["outc4.conv.weight"].shape[0])
    output_layout = resolve_output_layout(meta, out_channels)
    regression_channels = 2 if output_layout in {"class_masks_boundary_offset", "class_masks_boundary_hole_offset"} else 0
    model = GateNet(
        in_channels=3,
        f=f,
        out_channels=out_channels,
        regression_channels=regression_channels,
        conv_kind=conv_kind,
        up_kind=up_kind,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    meta = {
        **meta,
        "out_channels": out_channels,
        "regression_channels": regression_channels,
        "output_layout": output_layout,
        "conv_kind": conv_kind,
        "up_kind": up_kind,
        "class_ids": resolve_class_ids(meta, out_channels, output_layout),
    }
    return model, meta


def split_output(
    pred: torch.Tensor,
    class_ids: list[int],
    output_layout: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray | None]:
    arr = pred[0].detach().float().cpu().numpy()
    if arr.shape[0] <= 2:
        fg_prob = arr[0]
        bd_prob = arr[1]
        return fg_prob, bd_prob, None, None, arr[0:1], None

    n_cls = class_channel_count(arr.shape[0], output_layout)
    class_probs = arr[:n_cls]
    fg_prob = class_probs.max(axis=0)
    bd_prob = arr[n_cls]
    if output_layout == "class_masks_boundary_hole_offset":
        hole_prob = arr[n_cls + 1]
        offset_xy = arr[n_cls + 2 : n_cls + 4]
        return fg_prob, bd_prob, None, offset_xy, class_probs, hole_prob
    if output_layout == "class_masks_boundary_offset":
        offset_xy = arr[n_cls + 1 : n_cls + 3]
        return fg_prob, bd_prob, None, offset_xy, class_probs, None
    center_prob = arr[n_cls + 1]
    return fg_prob, bd_prob, center_prob, None, class_probs, None


def instance_class_counts(instances: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for inst in instances:
        if "class_id" not in inst:
            continue
        key = str(inst["class_id"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_yolo_instances(label_path: Path) -> int:
    """
    Count instances in YOLO-seg label file (one line per instance).
    """
    if not label_path.exists():
        return 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    n = 0
    for line in text.splitlines():
        parts = line.strip().split()
        # cls + at least 3 points (x,y)*3 => 1 + 6 = 7 tokens minimum
        if len(parts) >= 7:
            n += 1
    return n


def count_connected_components(mask01: torch.Tensor, min_area: int = 20) -> int:
    """
    Count 8-connected components in a binary mask.
    mask01: (H,W) tensor, values 0/1 (or 0..1).
    """
    m = (mask01.detach().float().cpu().numpy() >= 0.5).astype(np.uint8)
    h, w = m.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    n_comp = 0

    for y in range(h):
        for x in range(w):
            if m[y, x] == 0 or visited[y, x] == 1:
                continue
            stack = [(y, x)]
            visited[y, x] = 1
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in (cy - 1, cy, cy + 1):
                    if ny < 0 or ny >= h:
                        continue
                    for nx in (cx - 1, cx, cx + 1):
                        if nx < 0 or nx >= w:
                            continue
                        if visited[ny, nx] == 0 and m[ny, nx] == 1:
                            visited[ny, nx] = 1
                            stack.append((ny, nx))
            if area >= int(min_area):
                n_comp += 1
    return n_comp


def read_yolo_polys(label_path: Path) -> list[list[tuple[float, float]]]:
    """
    Parse YOLO segmentation polygons from label file.
    Returns list of polygons, each polygon is list of (x_norm, y_norm).
    """
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    polys: list[list[tuple[float, float]]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        nums = parts[1:]
        if len(nums) % 2 != 0:
            nums = nums[:-1]
        pts: list[tuple[float, float]] = []
        ok = True
        for i in range(0, len(nums), 2):
            try:
                x = float(nums[i])
                y = float(nums[i + 1])
            except ValueError:
                ok = False
                break
            pts.append((x, y))
        if ok and len(pts) >= 3:
            polys.append(pts)
    return polys


def polys_to_mask_pil(polys: list[list[tuple[float, float]]], w: int, h: int) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        xy = [(max(0.0, min(1.0, x)) * (w - 1), max(0.0, min(1.0, y)) * (h - 1)) for (x, y) in poly]
        if len(xy) >= 3:
            draw.polygon(xy, outline=255, fill=255)
    return mask


def tensor_mask_to_pil(mask01: torch.Tensor) -> Image.Image:
    """
    mask01: (H,W) float/uint8, values 0..1 or 0/1
    """
    m = mask01.detach().float().cpu().numpy()
    m = (m * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def overlay_mask_on_image(img_rgb: Image.Image, mask01: torch.Tensor, color=(255, 0, 0), alpha: float = 0.45) -> Image.Image:
    img = img_rgb.convert("RGB")
    m = mask01.detach().float().cpu().numpy()
    m = (m >= 0.5).astype(np.uint8)
    overlay = np.array(img, dtype=np.uint8)
    col = np.array(color, dtype=np.uint8)[None, None, :]
    overlay[m == 1] = (overlay[m == 1].astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference on YOLO-seg test set, save masks and speed stats.")
    ap.add_argument("--data", type=Path, required=True, help="Split root containing test/ (with images/, labels/)")
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path (.pt) produced by training")
    ap.add_argument("--out", type=Path, default=Path("runs/infer_test"), help="Output directory for images/results")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--boundary-threshold", type=float, default=0.5)
    ap.add_argument("--center-threshold", type=float, default=0.45)
    ap.add_argument("--hole-threshold", type=float, default=0.45)
    ap.add_argument("--vote-bin", type=int, default=8, help="Offset-vote histogram bin size in pixels")
    ap.add_argument("--vote-min-count", type=int, default=None, help="Min votes for an offset center bin")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations (not counted in speed)")
    ap.add_argument("--min-area", type=int, default=20, help="Min seed area to keep as an instance")
    ap.add_argument("--seed-min-area", type=int, default=3, help="Min center-seed area to keep")
    ap.add_argument(
        "--save-orig-size",
        action="store_true",
        help="Also save predicted/GT masks (and overlay) in original image size",
    )
    ap.add_argument("--save-overlay", action="store_true", help="Save RGB overlay images")
    ap.add_argument("--save-gt", action="store_true", help="Save GT masks as PNG too")
    args = ap.parse_args()

    device = auto_device(args.device)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_foreground").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_boundary").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_center").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_hole").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_offset_x").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_offset_y").mkdir(parents=True, exist_ok=True)
    (out_dir / "instance_maps").mkdir(parents=True, exist_ok=True)
    (out_dir / "instance_color").mkdir(parents=True, exist_ok=True)
    if args.save_orig_size:
        (out_dir / "pred_foreground_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "pred_boundary_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "pred_center_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "pred_hole_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "pred_offset_x_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "pred_offset_y_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "instance_maps_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "instance_color_orig").mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
        if args.save_orig_size:
            (out_dir / "overlay_orig").mkdir(parents=True, exist_ok=True)
    if args.save_gt:
        (out_dir / "gt_masks").mkdir(parents=True, exist_ok=True)
        if args.save_orig_size:
            (out_dir / "gt_masks_orig").mkdir(parents=True, exist_ok=True)

    model, meta = load_model(args.ckpt, device)
    class_ids = list(meta.get("class_ids") or [0])
    output_layout = str(meta.get("output_layout") or "legacy")
    if output_layout == "class_masks_boundary_hole_offset":
        instance_head = "offset_hole"
    elif output_layout == "class_masks_boundary_offset":
        instance_head = "offset"
    else:
        instance_head = "center"

    test_root = args.data / "test"
    ds = YoloSegDataset(test_root, img_size=args.img_size, augment=False, seed=0, class_ids=class_ids, instance_head=instance_head)

    # Warmup
    for i in range(min(args.warmup, len(ds))):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(device)
        _ = model(x)[-1]
    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    ious: list[float] = []
    per_image: list[dict] = []

    for i in range(len(ds)):
        img_path = ds.img_paths[i]
        orig_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_img.size

        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(x)[-1]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        times_ms.append(dt_ms)

        iou_val = iou_binary(pred, y, thresh=args.threshold, output_layout=output_layout)
        ious.append(iou_val)

        fg_prob, bd_prob, center_prob, offset_xy, class_probs, hole_prob = split_output(pred, class_ids=class_ids, output_layout=output_layout)
        pred_bin = (fg_prob >= args.threshold).astype(np.uint8) * 255
        boundary_bin = (bd_prob >= args.boundary_threshold).astype(np.uint8) * 255
        pred_png = Image.fromarray(pred_bin, mode="L")
        boundary_png = Image.fromarray(boundary_bin, mode="L")
        pred_png.save(out_dir / "pred_foreground" / f"{img_path.stem}.png")
        boundary_png.save(out_dir / "pred_boundary" / f"{img_path.stem}.png")
        center_png = None
        if center_prob is not None:
            center_bin = (center_prob >= args.center_threshold).astype(np.uint8) * 255
            center_png = Image.fromarray(center_bin, mode="L")
            center_png.save(out_dir / "pred_center" / f"{img_path.stem}.png")
        hole_png = None
        if hole_prob is not None:
            hole_bin = (hole_prob >= args.hole_threshold).astype(np.uint8) * 255
            hole_png = Image.fromarray(hole_bin, mode="L")
            hole_png.save(out_dir / "pred_hole" / f"{img_path.stem}.png")
        offset_pngs = None
        if offset_xy is not None:
            offset_x_png = Image.fromarray(np.clip((offset_xy[0] * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8), mode="L")
            offset_y_png = Image.fromarray(np.clip((offset_xy[1] * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8), mode="L")
            offset_x_png.save(out_dir / "pred_offset_x" / f"{img_path.stem}.png")
            offset_y_png.save(out_dir / "pred_offset_y" / f"{img_path.stem}.png")
            offset_pngs = (offset_x_png, offset_y_png)

        gt_label_path = test_root / "labels" / f"{img_path.stem}.txt"
        gt_instances = count_yolo_instances(gt_label_path)
        label_map, instances = postprocess_instances(
            fg_prob,
            bd_prob,
            center_prob=center_prob,
            hole_prob=hole_prob,
            offset_xy=offset_xy,
            class_probs=class_probs,
            class_ids=class_ids,
            threshold=args.threshold,
            boundary_threshold=args.boundary_threshold,
            center_threshold=args.center_threshold,
            hole_threshold=args.hole_threshold,
            min_area=args.min_area,
            seed_min_area=args.seed_min_area,
            vote_bin=args.vote_bin,
            vote_min_count=args.vote_min_count,
        )
        pred_instances = len(instances)
        label_map_to_u16(label_map).save(out_dir / "instance_maps" / f"{img_path.stem}.png")
        label_map_to_color(label_map).save(out_dir / "instance_color" / f"{img_path.stem}.png")

        if args.save_gt:
            gt_png = tensor_mask_to_pil(foreground_from_channels(y, output_layout=output_layout)[0, 0])
            gt_png.save(out_dir / "gt_masks" / f"{img_path.stem}.png")

        if args.save_overlay:
            img_rgb = Image.open(img_path).convert("RGB").resize((args.img_size, args.img_size), Image.BILINEAR)
            ov = overlay_instances_on_image(img_rgb, label_map)
            ov.save(out_dir / "overlay" / f"{img_path.stem}.png")

        iou_orig = None
        if args.save_orig_size:
            boundary_orig = boundary_png.resize((orig_w, orig_h), resample=Image.NEAREST)
            boundary_orig.save(out_dir / "pred_boundary_orig" / f"{img_path.stem}.png")
            if center_png is not None:
                center_orig = center_png.resize((orig_w, orig_h), resample=Image.NEAREST)
                center_orig.save(out_dir / "pred_center_orig" / f"{img_path.stem}.png")
            if hole_png is not None:
                hole_orig = hole_png.resize((orig_w, orig_h), resample=Image.NEAREST)
                hole_orig.save(out_dir / "pred_hole_orig" / f"{img_path.stem}.png")
            if offset_pngs is not None:
                offset_pngs[0].resize((orig_w, orig_h), resample=Image.NEAREST).save(out_dir / "pred_offset_x_orig" / f"{img_path.stem}.png")
                offset_pngs[1].resize((orig_w, orig_h), resample=Image.NEAREST).save(out_dir / "pred_offset_y_orig" / f"{img_path.stem}.png")
            label_orig = label_map_to_u16(label_map).resize((orig_w, orig_h), resample=Image.NEAREST)
            label_orig.save(out_dir / "instance_maps_orig" / f"{img_path.stem}.png")
            label_orig_np = np.asarray(label_orig, dtype=np.uint16).astype(np.int32)
            label_map_to_color(label_orig_np).save(out_dir / "instance_color_orig" / f"{img_path.stem}.png")
            pred_orig_arr = np.where(label_orig_np > 0, 255, 0).astype(np.uint8)
            pred_orig = Image.fromarray(pred_orig_arr, mode="L")
            pred_orig.save(out_dir / "pred_foreground_orig" / f"{img_path.stem}.png")

            # GT mask in original size from polygons (avoids resize artifacts)
            polys = read_yolo_polys(gt_label_path)
            gt_orig = polys_to_mask_pil(polys, w=orig_w, h=orig_h)
            if args.save_gt:
                gt_orig.save(out_dir / "gt_masks_orig" / f"{img_path.stem}.png")

            if args.save_overlay:
                ov_orig = overlay_instances_on_image(orig_img, label_orig_np)
                ov_orig.save(out_dir / "overlay_orig" / f"{img_path.stem}.png")

            # Optional IoU computed on original size (pred is resized back; gt is exact polygon render)
            pred_arr = pred_orig_arr >= 128
            gt_arr = (np.asarray(gt_orig, dtype=np.uint8) >= 128)
            inter = np.logical_and(pred_arr, gt_arr).sum()
            union = np.logical_or(pred_arr, gt_arr).sum()
            iou_orig = float((inter + 1e-6) / (union + 1e-6))

        per_image.append(
            {
                "image": img_path.name,
                "orig_size": [int(orig_w), int(orig_h)],
                "latency_ms": float(dt_ms),
                "iou_384": float(iou_val),
                "iou_orig": iou_orig,
                "pred_instances": int(pred_instances),
                "gt_instances": int(gt_instances),
                "class_counts": instance_class_counts(instances),
                "instances": instances,
            }
        )

    mean_ms = float(np.mean(times_ms)) if times_ms else 0.0
    p50_ms = float(np.percentile(times_ms, 50)) if times_ms else 0.0
    p90_ms = float(np.percentile(times_ms, 90)) if times_ms else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

    results = {
        "ckpt": str(args.ckpt),
        "device": str(device),
        "img_size": args.img_size,
        "threshold": args.threshold,
        "boundary_threshold": args.boundary_threshold,
        "center_threshold": args.center_threshold,
        "hole_threshold": args.hole_threshold,
        "vote_bin": args.vote_bin,
        "vote_min_count": args.vote_min_count,
        "output_layout": output_layout,
        "class_ids": class_ids,
        "count": len(ds),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_ms_per_image": mean_ms,
        "p50_ms_per_image": p50_ms,
        "p90_ms_per_image": p90_ms,
        "fps": fps,
        "min_area": args.min_area,
        "seed_min_area": args.seed_min_area,
        "save_orig_size": bool(args.save_orig_size),
        "per_image": per_image,
        "ckpt_meta": meta,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
