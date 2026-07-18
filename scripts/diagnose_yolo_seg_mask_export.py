from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def use_local_ultralytics(root: Path = Path("ultralytics")) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    local_root = (repo_root / root).resolve()
    package_init = local_root / "ultralytics" / "__init__.py"
    if not package_init.exists():
        raise FileNotFoundError(f"Local Ultralytics package not found: {package_init}")
    sys.path.insert(0, str(local_root))
    os.environ["PYTHONPATH"] = f"{local_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    return local_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare YOLO .pt masks with TRT85 TopK ONNX postprocess masks.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument(
        "--pt",
        type=Path,
        default=Path("runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best.pt"),
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        default=Path("runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best_trt85_topk30.onnx"),
    )
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--cuda-visible-devices", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("runs/yolo26/mask_export_diagnose"))
    ap.add_argument("--ultralytics-root", type=Path, default=Path("ultralytics"))
    ap.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    ap.add_argument("--retina-masks", action="store_true")
    ap.add_argument("--pt-rect", action="store_true", help="Use Ultralytics rectangular inference for the PT path.")
    ap.add_argument("--skip-onnx-nms", action="store_true", help="Match YOLO26 end2end PT output by skipping extra NMS.")
    return ap.parse_args()


def preprocess_image(image_path: Path, imgsz: int, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.array(Image.open(image_path).convert("RGB"))
    if mode == "resize":
        net_image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    else:
        from ultralytics.data.augment import LetterBox

        # Fixed-shape ONNX engines need a full square input. This matches
        # model.predict(..., rect=False) rather than Ultralytics' auto-rect path.
        net_image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    x = net_image.astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    return image, net_image, x


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = np.maximum(0.0, boxes[i, 2] - boxes[i, 0]) * np.maximum(0.0, boxes[i, 3] - boxes[i, 1])
        area_r = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-7)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def crop_masks(mask_logits: np.ndarray, boxes_384: np.ndarray, imgsz: int) -> np.ndarray:
    n, h, w = mask_logits.shape
    boxes = boxes_384.astype(np.float32).copy()
    boxes[:, [0, 2]] *= w / float(imgsz)
    boxes[:, [1, 3]] *= h / float(imgsz)
    xs = np.arange(w, dtype=np.float32)[None, None, :]
    ys = np.arange(h, dtype=np.float32)[None, :, None]
    inside = (
        (xs >= boxes[:, 0][:, None, None])
        & (xs < boxes[:, 2][:, None, None])
        & (ys >= boxes[:, 1][:, None, None])
        & (ys < boxes[:, 3][:, None, None])
    )
    return mask_logits * inside


def postprocess_topk_onnx(
    detections: np.ndarray, proto: np.ndarray, imgsz: int, conf: float, iou: float, do_nms: bool = True
):
    det = detections[0]
    proto = proto[0].astype(np.float32)
    det = det[det[:, 4] >= conf]
    if det.size == 0:
        return [], np.zeros((0, imgsz, imgsz), dtype=np.uint8)
    boxes = det[:, :4].astype(np.float32)
    scores = det[:, 4].astype(np.float32)
    coeffs = det[:, 6:].astype(np.float32)
    if do_nms:
        keep = nms_xyxy(boxes, scores, iou)
        boxes, scores, coeffs = boxes[keep], scores[keep], coeffs[keep]
    c, mh, mw = proto.shape
    masks = coeffs @ proto.reshape(c, -1)
    masks = masks.reshape(-1, mh, mw)
    masks = crop_masks(masks, boxes, imgsz)
    masks384 = np.stack([cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR) for m in masks], axis=0)
    binary = (masks384 > 0).astype(np.uint8)
    keep = binary.max(axis=(1, 2)) > 0
    boxes, scores, binary = boxes[keep], scores[keep], binary[keep]
    return [{"box": b, "score": float(s)} for b, s in zip(boxes, scores)], binary


def overlay_masks(image_rgb: np.ndarray, masks: np.ndarray, boxes: list | None = None) -> np.ndarray:
    canvas = cv2.resize(image_rgb, (masks.shape[2], masks.shape[1]), interpolation=cv2.INTER_LINEAR) if masks.size else image_rgb.copy()
    colors = [(0, 255, 0), (0, 90, 255), (255, 0, 0), (255, 180, 0), (180, 0, 255)]
    out = canvas.astype(np.float32)
    for i, m in enumerate(masks):
        color = np.asarray(colors[i % len(colors)], dtype=np.float32)
        idx = m.astype(bool)
        out[idx] = out[idx] * 0.55 + color * 0.45
        if boxes:
            x1, y1, x2, y2 = boxes[i]["box"].astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), tuple(int(v) for v in color.tolist()), 2)
    return np.clip(out, 0, 255).astype(np.uint8)


def run_pt(args: argparse.Namespace, image_rgb: np.ndarray):
    from ultralytics import YOLO

    model = YOLO(str(args.pt))
    result = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        retina_masks=args.retina_masks,
        rect=args.pt_rect,
        verbose=False,
    )[0]
    if result.masks is None:
        return np.zeros((0, args.imgsz, args.imgsz), dtype=np.uint8)
    masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
    if masks.shape[-2:] != (args.imgsz, args.imgsz):
        masks = np.stack([cv2.resize(m, (args.imgsz, args.imgsz), interpolation=cv2.INTER_NEAREST) for m in masks], axis=0)
    return masks


def run_onnx(args: argparse.Namespace, x: np.ndarray):
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    available = ort.get_available_providers()
    if args.ort_provider == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.ort_provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is unavailable: {available}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    outputs = session.run(None, {session.get_inputs()[0].name: x})
    return outputs, session.get_providers()


def main() -> None:
    args = parse_args()
    use_local_ultralytics(args.ultralytics_root)
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_rgb, net_image_rgb, x = preprocess_image(args.image, args.imgsz, args.preprocess)
    pt_masks = run_pt(args, image_rgb)
    (detections, proto), providers = run_onnx(args, x)
    dets, onnx_masks = postprocess_topk_onnx(
        detections, proto, args.imgsz, args.conf, args.iou, do_nms=not args.skip_onnx_nms
    )

    Image.fromarray(overlay_masks(net_image_rgb, pt_masks)).save(args.out_dir / f"{args.image.stem}_pt_overlay.png")
    Image.fromarray(overlay_masks(net_image_rgb, onnx_masks, dets)).save(args.out_dir / f"{args.image.stem}_onnx_overlay.png")
    print(
        {
            "image": str(args.image),
            "preprocess": args.preprocess,
            "retina_masks": bool(args.retina_masks),
            "pt_rect": bool(args.pt_rect),
            "onnx_nms": not bool(args.skip_onnx_nms),
            "pt_masks": int(pt_masks.shape[0]),
            "onnx_masks": int(onnx_masks.shape[0]),
            "onnx_providers": providers,
            "out_dir": str(args.out_dir),
        }
    )


if __name__ == "__main__":
    main()
