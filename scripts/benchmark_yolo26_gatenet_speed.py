from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def use_local_ultralytics(root: Path = Path("ultralytics")) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    local_root = (repo_root / root).resolve()
    package_init = local_root / "ultralytics" / "__init__.py"
    if not package_init.exists():
        raise FileNotFoundError(f"Local Ultralytics package not found: {package_init}")
    import sys

    sys.path.insert(0, str(local_root))
    os.environ["PYTHONPATH"] = f"{local_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    return local_root


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark YOLO26-seg and GateNet inference speed.")
    ap.add_argument("--data-yaml", type=Path, default=Path("data/ultralytics/image1_monorace_aug_yolo26/dataset.yaml"))
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--yolo-pt", type=Path, required=True)
    ap.add_argument("--yolo-onnx", type=Path, default=None)
    ap.add_argument("--gatenet-results", type=Path, default=Path("runs/gatenet_image1_offset_hole_aug_holeonly_v2/test_infer_b035_h015_holesplit/results.json"))
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--device", type=str, default="0", help="CUDA index inside CUDA_VISIBLE_DEVICES, or cpu.")
    ap.add_argument("--cuda-visible-devices", type=str, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=0, help="0 means all images in split.")
    ap.add_argument("--half", action="store_true", help="Use FP16 for PyTorch YOLO benchmark.")
    ap.add_argument("--out", type=Path, default=Path("runs/yolo26/speed_compare_yolo26_gatenet_384.json"))
    ap.add_argument("--ultralytics-root", type=Path, default=Path("ultralytics"))
    return ap.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {path}")
    return data


def resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    cfg = load_yaml(data_yaml)
    root = Path(cfg.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    rel = cfg.get(split)
    if rel is None and split == "test":
        rel = cfg.get("val")
    if rel is None:
        raise KeyError(f"Split '{split}' is missing in {data_yaml}")

    img_dir = Path(rel)
    if not img_dir.is_absolute():
        img_dir = root / img_dir
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return images


def preprocess_image(path: Path, imgsz: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


def summarize_ms(values: list[float]) -> dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "fps": 0.0}
    ordered = sorted(values)
    p50 = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    mean_ms = statistics.mean(ordered)
    return {
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p90_ms": p90,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def torch_device_name(device: str) -> str:
    if device.lower() == "cpu":
        return "cpu"
    return f"cuda:{device}" if device.isdigit() else device


def benchmark_yolo_pt(yolo_pt: Path, images: list[Path], imgsz: int, device: str, warmup: int, half: bool) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    dev = torch.device(torch_device_name(device) if torch.cuda.is_available() and device.lower() != "cpu" else "cpu")
    model = YOLO(str(yolo_pt))
    net = model.model.to(dev).eval()
    if hasattr(net, "fuse"):
        net = net.fuse().eval()
    if half and dev.type == "cuda":
        net = net.half()

    params = sum(p.numel() for p in net.parameters())
    first = torch.from_numpy(preprocess_image(images[0], imgsz)).to(dev)
    if half and dev.type == "cuda":
        first = first.half()

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = net(first)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        times: list[float] = []
        for path in images:
            x = torch.from_numpy(preprocess_image(path, imgsz)).to(dev)
            if half and dev.type == "cuda":
                x = x.half()
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _ = net(x)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "backend": "pytorch_raw_forward",
        "device": str(dev),
        "half": bool(half and dev.type == "cuda"),
        "params": int(params),
        **summarize_ms(times),
    }


def benchmark_yolo_onnx(yolo_onnx: Path, images: list[Path], imgsz: int, warmup: int) -> dict[str, Any]:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(yolo_onnx), providers=providers)
    input_name = session.get_inputs()[0].name
    first = preprocess_image(images[0], imgsz)

    for _ in range(max(0, warmup)):
        _ = session.run(None, {input_name: first})

    times: list[float] = []
    for path in images:
        x = preprocess_image(path, imgsz)
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "backend": "onnxruntime_raw_forward",
        "providers": session.get_providers(),
        **summarize_ms(times),
    }


def read_gatenet_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "backend": "gatenet_existing_result",
        "source": str(path),
        "img_size": data.get("img_size"),
        "mean_ms": data.get("mean_ms_per_image"),
        "p50_ms": data.get("p50_ms_per_image"),
        "p90_ms": data.get("p90_ms_per_image"),
        "fps": data.get("fps"),
        "mean_iou": data.get("mean_iou"),
    }


def read_yolo_results_csv(run_dir: Path) -> dict[str, Any] | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return None
    last = rows[-1]
    return {
        "epochs_logged": len(rows),
        "last_epoch": last.get("epoch"),
        "last_box_map50": last.get("metrics/mAP50(B)"),
        "last_mask_map50": last.get("metrics/mAP50(M)"),
        "last_mask_map50_95": last.get("metrics/mAP50-95(M)"),
    }


def capture_benchmark(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # Keep train/export scripts from failing after successful training.
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    args = parse_args()
    local_root = use_local_ultralytics(args.ultralytics_root)
    print(f"Using local Ultralytics: {local_root}")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    images = resolve_split_images(args.data_yaml, args.split)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]

    yolo_pt = args.yolo_pt.resolve()
    yolo_onnx = args.yolo_onnx.resolve() if args.yolo_onnx else None
    run_dir = yolo_pt.parent.parent

    summary: dict[str, Any] = {
        "data_yaml": str(args.data_yaml),
        "split": args.split,
        "num_images": len(images),
        "imgsz": args.imgsz,
        "yolo_pt": str(yolo_pt),
        "yolo_onnx": str(yolo_onnx) if yolo_onnx else None,
        "yolo_train": read_yolo_results_csv(run_dir),
        "gatenet": read_gatenet_result(args.gatenet_results),
    }

    if yolo_onnx and yolo_onnx.exists():
        summary["yolo_onnxruntime"] = capture_benchmark(benchmark_yolo_onnx, yolo_onnx, images, args.imgsz, args.warmup)
    summary["yolo_pytorch"] = capture_benchmark(
        benchmark_yolo_pt, yolo_pt, images, args.imgsz, args.device, args.warmup, args.half
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
