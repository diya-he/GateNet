from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark YOLO and GateNet ONNX with CUDAExecutionProvider.")
    ap.add_argument("--data-root", type=Path, default=Path("data/ultralytics/image1_monorace_aug_yolo26"))
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--yolo-onnx", type=Path, required=True)
    ap.add_argument("--gatenet-onnx", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--cuda-device", type=int, default=0, help="CUDA EP device id after CUDA_VISIBLE_DEVICES filtering.")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("runs/yolo26/cuda_onnx_speed_compare.json"))
    return ap.parse_args()


def list_images(data_root: Path, split: str) -> list[Path]:
    img_dir = data_root / split / "images"
    images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return images


def preprocess(path: Path, imgsz: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(arr)


def summarize(times_ms: list[float]) -> dict[str, float]:
    ordered = sorted(float(v) for v in times_ms)
    mean_ms = statistics.mean(ordered)
    return {
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9)))],
        "p99_ms": ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.99)))],
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def make_session(path: Path, cuda_device: int) -> ort.InferenceSession:
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls(directory="")
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDAExecutionProvider is unavailable. Providers: {available}")

    providers: list[Any] = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": int(cuda_device),
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(str(path), providers=providers)
    actual = sess.get_providers()
    if not actual or actual[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"Session did not bind CUDAExecutionProvider first: {actual}")
    return sess


def benchmark_one(name: str, onnx_path: Path, inputs: list[np.ndarray], cuda_device: int, warmup: int, repeat: int) -> dict[str, Any]:
    sess = make_session(onnx_path, cuda_device)
    input_name = sess.get_inputs()[0].name
    output_shapes = []
    for output in sess.get_outputs():
        output_shapes.append([str(v) for v in output.shape])

    for i in range(max(0, warmup)):
        _ = sess.run(None, {input_name: inputs[i % len(inputs)]})

    times_ms: list[float] = []
    for _ in range(max(1, repeat)):
        for x in inputs:
            t0 = time.perf_counter()
            _ = sess.run(None, {input_name: x})
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "name": name,
        "onnx": str(onnx_path),
        "providers": sess.get_providers(),
        "input_name": input_name,
        "output_shapes": output_shapes,
        "runs": len(times_ms),
        **summarize(times_ms),
    }


def main() -> None:
    args = parse_args()
    images = list_images(args.data_root, args.split)
    inputs = [preprocess(p, args.imgsz) for p in images]

    result = {
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "data_root": str(args.data_root),
        "split": args.split,
        "num_images": len(images),
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "cuda_device": args.cuda_device,
        "yolo26": benchmark_one("yolo26-seg", args.yolo_onnx, inputs, args.cuda_device, args.warmup, args.repeat),
        "gatenet": benchmark_one("gatenet", args.gatenet_onnx, inputs, args.cuda_device, args.warmup, args.repeat),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
