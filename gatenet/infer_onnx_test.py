from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import onnxruntime as ort

from gatenet.instance_postprocess import label_map_to_color, label_map_to_u16, overlay_instances_on_image, postprocess_instances


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    imgs.sort()
    return imgs


def read_yolo_polys(label_path: Path) -> list[list[tuple[float, float]]]:
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


def count_yolo_instances(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0
    n = 0
    for line in text.splitlines():
        if len(line.strip().split()) >= 7:
            n += 1
    return n


def polys_to_mask_pil(polys: list[list[tuple[float, float]]], w: int, h: int) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        xy = [(max(0.0, min(1.0, x)) * (w - 1), max(0.0, min(1.0, y)) * (h - 1)) for (x, y) in poly]
        if len(xy) >= 3:
            draw.polygon(xy, outline=255, fill=255)
    return mask


def overlay_mask_on_image(img_rgb: Image.Image, mask_u8: np.ndarray, color=(255, 0, 0), alpha: float = 0.45) -> Image.Image:
    img = img_rgb.convert("RGB")
    overlay = np.array(img, dtype=np.uint8)
    m = (mask_u8 >= 128)
    col = np.array(color, dtype=np.uint8)[None, None, :]
    overlay[m] = (overlay[m].astype(np.float32) * (1 - alpha) + col.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(overlay, mode="RGB")


def iou_from_u8(pred_u8: np.ndarray, gt_u8: np.ndarray) -> float:
    p = (pred_u8 >= 128)
    g = (gt_u8 >= 128)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def count_connected_components(mask_u8: np.ndarray, min_area: int = 20) -> int:
    m = (mask_u8 >= 128).astype(np.uint8)
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


def preprocess(img_path: Path, img_size: int) -> tuple[np.ndarray, tuple[int, int], Image.Image]:
    orig = Image.open(img_path).convert("RGB")
    ow, oh = orig.size
    resized = orig.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # 1x3xHxW
    return arr, (ow, oh), orig


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNXRuntime instance inference on YOLO-seg test split.")
    ap.add_argument("--data", type=Path, required=True, help="Split root containing test/images and test/labels")
    ap.add_argument("--onnx", type=Path, required=True, help="ONNX path exported by gatenet.export_onnx")
    ap.add_argument("--out", type=Path, default=Path("runs/onnx_infer_test"))
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--boundary-threshold", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=20)
    ap.add_argument("--save-overlay", action="store_true")
    ap.add_argument("--save-gt", action="store_true")
    ap.add_argument("--save-orig-size", action="store_true")
    ap.add_argument("--providers", type=str, default="auto", help="auto|cpu|cuda")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_foreground").mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_boundary").mkdir(parents=True, exist_ok=True)
    (out_dir / "instance_maps").mkdir(parents=True, exist_ok=True)
    (out_dir / "instance_color").mkdir(parents=True, exist_ok=True)
    if args.save_orig_size:
        (out_dir / "pred_foreground_orig").mkdir(parents=True, exist_ok=True)
        (out_dir / "pred_boundary_orig").mkdir(parents=True, exist_ok=True)
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

    if args.providers == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif args.providers == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ort.get_available_providers()

    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    test_images = args.data / "test" / "images"
    test_labels = args.data / "test" / "labels"
    imgs = list_images(test_images)
    if not imgs:
        raise SystemExit(f"No images found in {test_images}")

    # Warmup
    for i in range(min(args.warmup, len(imgs))):
        x, _, _ = preprocess(imgs[i], args.img_size)
        _ = sess.run([out_name], {inp_name: x})[0]

    times_ms: list[float] = []
    ious_384: list[float] = []
    ious_orig: list[float] = []
    per_image: list[dict] = []

    for img_path in imgs:
        x, (ow, oh), orig_img = preprocess(img_path, args.img_size)

        t0 = time.perf_counter()
        y4 = sess.run([out_name], {inp_name: x})[0]  # 1x2xHxW, sigmoid already
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        times_ms.append(float(dt_ms))

        fg_prob = y4[0, 0]
        bd_prob = y4[0, 1]
        pred_bin = (fg_prob >= float(args.threshold)).astype(np.uint8) * 255
        boundary_bin = (bd_prob >= float(args.boundary_threshold)).astype(np.uint8) * 255

        Image.fromarray(pred_bin, mode="L").save(out_dir / "pred_foreground" / f"{img_path.stem}.png")
        Image.fromarray(boundary_bin, mode="L").save(out_dir / "pred_boundary" / f"{img_path.stem}.png")

        lbl_path = test_labels / f"{img_path.stem}.txt"
        gt_instances = count_yolo_instances(lbl_path)
        label_map, instances = postprocess_instances(
            fg_prob,
            bd_prob,
            threshold=args.threshold,
            boundary_threshold=args.boundary_threshold,
            min_area=args.min_area,
        )
        pred_instances = len(instances)
        label_map_to_u16(label_map).save(out_dir / "instance_maps" / f"{img_path.stem}.png")
        label_map_to_color(label_map).save(out_dir / "instance_color" / f"{img_path.stem}.png")

        # GT 384 (for iou_384)
        polys = read_yolo_polys(lbl_path)
        gt_384 = polys_to_mask_pil(polys, w=args.img_size, h=args.img_size)
        gt_384_u8 = np.asarray(gt_384, dtype=np.uint8)
        iou384 = iou_from_u8(pred_bin, gt_384_u8)
        ious_384.append(iou384)

        if args.save_gt:
            gt_384.save(out_dir / "gt_masks" / f"{img_path.stem}.png")

        if args.save_overlay:
            resized_rgb = orig_img.resize((args.img_size, args.img_size), Image.BILINEAR)
            overlay = overlay_instances_on_image(resized_rgb, label_map)
            overlay.save(out_dir / "overlay" / f"{img_path.stem}.png")

        iouorig = None
        if args.save_orig_size:
            pred_orig = Image.fromarray(pred_bin, mode="L").resize((ow, oh), Image.NEAREST)
            pred_orig.save(out_dir / "pred_foreground_orig" / f"{img_path.stem}.png")
            boundary_orig = Image.fromarray(boundary_bin, mode="L").resize((ow, oh), Image.NEAREST)
            boundary_orig.save(out_dir / "pred_boundary_orig" / f"{img_path.stem}.png")
            label_orig = label_map_to_u16(label_map).resize((ow, oh), Image.NEAREST)
            label_orig.save(out_dir / "instance_maps_orig" / f"{img_path.stem}.png")
            label_orig_np = np.asarray(label_orig, dtype=np.uint16).astype(np.int32)
            label_map_to_color(label_orig_np).save(out_dir / "instance_color_orig" / f"{img_path.stem}.png")

            gt_orig = polys_to_mask_pil(polys, w=ow, h=oh)
            gt_orig_u8 = np.asarray(gt_orig, dtype=np.uint8)
            if args.save_gt:
                gt_orig.save(out_dir / "gt_masks_orig" / f"{img_path.stem}.png")

            if args.save_overlay:
                ov_orig = overlay_instances_on_image(orig_img, label_orig_np)
                ov_orig.save(out_dir / "overlay_orig" / f"{img_path.stem}.png")

            iouorig = iou_from_u8(np.asarray(pred_orig, dtype=np.uint8), gt_orig_u8)
            ious_orig.append(iouorig)

        per_image.append(
            {
                "image": img_path.name,
                "orig_size": [int(ow), int(oh)],
                "latency_ms": float(dt_ms),
                "iou_384": float(iou384),
                "iou_orig": (float(iouorig) if iouorig is not None else None),
                "pred_instances": int(pred_instances),
                "gt_instances": int(gt_instances),
                "instances": instances,
            }
        )

    mean_ms = float(np.mean(times_ms)) if times_ms else 0.0
    p50_ms = float(np.percentile(times_ms, 50)) if times_ms else 0.0
    p90_ms = float(np.percentile(times_ms, 90)) if times_ms else 0.0
    fps = (1000.0 / mean_ms) if mean_ms > 0 else 0.0

    results = {
        "onnx": str(args.onnx),
        "providers": sess.get_providers(),
        "img_size": args.img_size,
        "threshold": args.threshold,
        "boundary_threshold": args.boundary_threshold,
        "count": len(imgs),
        "mean_iou_384": float(np.mean(ious_384)) if ious_384 else None,
        "mean_iou_orig": float(np.mean(ious_orig)) if ious_orig else None,
        "mean_ms_per_image": mean_ms,
        "p50_ms_per_image": p50_ms,
        "p90_ms_per_image": p90_ms,
        "fps": fps,
        "min_area": args.min_area,
        "save_orig_size": bool(args.save_orig_size),
        "per_image": per_image,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

