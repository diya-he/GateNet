from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


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
    ap = argparse.ArgumentParser(description="Train a local-Ultralytics YOLO26-seg model, export ONNX, and benchmark speed.")
    ap.add_argument("--data", type=Path, default=Path("data/ultralytics/image1_monorace_aug_yolo26/dataset.yaml"))
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("ultralytics/ultralytics/cfg/models/26/yolo26-gate-lite-seg.yaml"),
    )
    ap.add_argument("--init-weights", type=Path, default=None, help="Optional checkpoint to partially initialize a YAML model.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cuda-visible-devices", type=str, default="1")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", type=str, default="runs/yolo26")
    ap.add_argument("--name", type=str, default="image1_monorace_aug_yolo26_gate_lite_seg_384")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opset", type=int, default=None)
    ap.add_argument("--export-half", action="store_true", help="Export FP16 ONNX when supported.")
    ap.add_argument("--force", action="store_true", help="Delete the target run even if it already looks complete.")
    ap.add_argument("--keep-incomplete", action="store_true", help="Fail instead of deleting an incomplete target run.")
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--ultralytics-root", type=Path, default=Path("ultralytics"))
    ap.add_argument("--gatenet-results", type=Path, default=Path("runs/gatenet_image1_offset_hole_aug_holeonly_v2/test_infer_b035_h015_holesplit/results.json"))
    return ap.parse_args()


def candidate_run_dirs(project: str, name: str) -> list[Path]:
    p = Path(project)
    dirs: list[Path] = []
    if p.is_absolute():
        dirs.append(p / name)
    else:
        dirs.append(Path("runs/segment") / p / name)
        dirs.append(p / name)
    return dirs


def target_run_dir(project: str, name: str) -> Path:
    candidates = candidate_run_dirs(project, name)
    for d in candidates:
        if d.exists():
            return d
    return candidates[0]


def results_status(run_dir: Path) -> tuple[int, int | None]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0, None
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return 0, None
    try:
        last_epoch = int(float(rows[-1].get("epoch", "")))
    except ValueError:
        last_epoch = None
    return len(rows), last_epoch


def is_complete(run_dir: Path, epochs: int) -> bool:
    rows, last_epoch = results_status(run_dir)
    return rows >= epochs or (last_epoch is not None and last_epoch >= epochs)


def clean_target_if_needed(run_dir: Path, epochs: int, force: bool, keep_incomplete: bool) -> bool:
    if not run_dir.exists():
        return False
    rows, last_epoch = results_status(run_dir)
    if force:
        print(f"Deleting target run because --force is set: {run_dir}")
        shutil.rmtree(run_dir)
        return False
    if is_complete(run_dir, epochs):
        print(f"Target run already complete: {run_dir} (rows={rows}, last_epoch={last_epoch})")
        return True
    if keep_incomplete:
        raise RuntimeError(f"Incomplete run exists: {run_dir} (rows={rows}, last_epoch={last_epoch})")
    print(f"Deleting incomplete target run: {run_dir} (rows={rows}, last_epoch={last_epoch})")
    shutil.rmtree(run_dir)
    return False


def train_yolo(args: argparse.Namespace, run_dir: Path) -> Path:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if args.init_weights is not None:
        model.load(str(args.init_weights))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=0,
        seed=args.seed,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", run_dir))
    if not is_complete(save_dir, args.epochs):
        rows, last_epoch = results_status(save_dir)
        raise RuntimeError(f"YOLO training did not reach {args.epochs} epochs: {save_dir} rows={rows} last_epoch={last_epoch}")
    return save_dir


def export_onnx(best_pt: Path, args: argparse.Namespace) -> Path:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        device=args.device,
        simplify=True,
        opset=args.opset,
        half=args.export_half,
        end2end=False,
        nms=False,
    )
    return Path(exported)


def val_speed(best_pt: Path, args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    from ultralytics import YOLO

    model = YOLO(str(best_pt))
    model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=1,
        device=args.device,
        split="test",
        project=args.project,
        name=f"{args.name}_val_speed",
        exist_ok=True,
    )


def run_benchmark(best_pt: Path, onnx_path: Path | None, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/benchmark_yolo26_gatenet_speed.py",
        "--data-yaml",
        str(args.data),
        "--split",
        "test",
        "--yolo-pt",
        str(best_pt),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--gatenet-results",
        str(args.gatenet_results),
        "--ultralytics-root",
        str(args.ultralytics_root),
        "--out",
        str(Path("runs/yolo26") / f"speed_compare_{args.name}.json"),
    ]
    if onnx_path:
        cmd.extend(["--yolo-onnx", str(onnx_path)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    local_root = use_local_ultralytics(args.ultralytics_root)
    print(f"Using local Ultralytics: {local_root}")
    run_dir = target_run_dir(args.project, args.name)
    already_complete = clean_target_if_needed(run_dir, args.epochs, args.force, args.keep_incomplete)

    if already_complete:
        save_dir = run_dir
    else:
        save_dir = train_yolo(args, run_dir)

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(best_pt)

    if not args.no_val:
        val_speed(best_pt, args)

    onnx_path: Path | None = None
    if not args.no_export:
        onnx_path = export_onnx(best_pt, args)

    if not args.no_benchmark:
        run_benchmark(best_pt, onnx_path, args)

    print(f"YOLO run: {save_dir}")
    print(f"Best checkpoint: {best_pt}")
    if onnx_path:
        print(f"ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
