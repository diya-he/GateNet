from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def use_local_ultralytics(root: Path = Path("ultralytics")) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    local_root = (repo_root / root).resolve()
    package_init = local_root / "ultralytics" / "__init__.py"
    if not package_init.exists():
        raise FileNotFoundError(f"Local Ultralytics package not found: {package_init}")
    sys.path.insert(0, str(local_root))
    os.environ["PYTHONPATH"] = f"{local_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    return local_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check numerical consistency between YOLO26-seg PT wrapper and ONNX.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--branch", choices=("auto", "one2one", "one2many"), default="one2one")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--ort-provider", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    ap.add_argument("--ultralytics-root", type=Path, default=Path("ultralytics"))
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def preprocess(image_path: Path, imgsz: int, mode: str) -> np.ndarray:
    image = np.array(Image.open(image_path).convert("RGB"))
    if mode == "resize":
        net_image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    else:
        from ultralytics.data.augment import LetterBox

        net_image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    x = net_image.astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    diff = np.abs(a - b)
    return {
        "shape_pt": list(a.shape),
        "shape_onnx": list(b.shape),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "p99_abs": float(np.percentile(diff, 99)) if diff.size else 0.0,
    }


def choose_providers(kind: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if kind == "cpu":
        return ["CPUExecutionProvider"]
    if kind == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is unavailable: {available}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available] or ["CPUExecutionProvider"]


def main() -> None:
    args = parse_args()
    use_local_ultralytics(args.ultralytics_root)

    from scripts.export_yolo26_seg_trt85_topk_onnx import TRT85TopKSegWrapper, prepare_model
    import onnxruntime as ort

    x_np = preprocess(args.image, args.imgsz, args.preprocess)
    device = torch.device(args.device)
    model = prepare_model(args.pt, device)
    wrapper = TRT85TopKSegWrapper(model, args.topk, args.branch).to(device).eval()

    with torch.no_grad():
        detections_pt, proto_pt = wrapper(torch.from_numpy(x_np).to(device))
    detections_pt = detections_pt.detach().cpu().numpy()
    proto_pt = proto_pt.detach().cpu().numpy()

    providers = choose_providers(args.ort_provider)
    session = ort.InferenceSession(str(args.onnx), providers=providers)
    detections_onnx, proto_onnx = session.run(None, {session.get_inputs()[0].name: x_np})
    high_conf = (detections_pt[0, :, 4] >= 0.25) | (detections_onnx[0, :, 4] >= 0.25)

    report = {
        "image": str(args.image),
        "pt": str(args.pt),
        "onnx": str(args.onnx),
        "imgsz": int(args.imgsz),
        "topk": int(args.topk),
        "branch": args.branch,
        "preprocess": args.preprocess,
        "ort_provider": args.ort_provider,
        "providers": session.get_providers(),
        "detections": diff_stats(detections_pt, detections_onnx),
        "detections_first5": diff_stats(detections_pt[:, :5], detections_onnx[:, :5]),
        "detections_conf_ge_025": diff_stats(detections_pt[:, high_conf], detections_onnx[:, high_conf]),
        "proto": diff_stats(proto_pt, proto_onnx),
        "first_scores_pt": detections_pt[0, : min(5, detections_pt.shape[1]), 4].tolist(),
        "first_scores_onnx": detections_onnx[0, : min(5, detections_onnx.shape[1]), 4].tolist(),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
