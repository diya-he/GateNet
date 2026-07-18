from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from gatenet.data import YoloSegDataset, scan_yolo_class_ids
from gatenet.losses import gatenet_multiscale_loss, iou_binary
from gatenet.model import GateNet


def output_layout_from_instance_head(instance_head: str) -> str:
    if instance_head == "offset_hole":
        return "class_masks_boundary_hole_offset"
    if instance_head == "offset":
        return "class_masks_boundary_offset"
    return "class_masks_boundary_center"


def auto_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
    for pg in optim.param_groups:
        pg["lr"] = lr


def lr_schedule(base_lr: float, epoch: int) -> float:
    """
    Paper schedule: base_lr * (sqrt(0.1))^k at epochs 10,33,66,90
    """
    drops = 0
    for e in (10, 33, 66, 90):
        if epoch >= e:
            drops += 1
    return base_lr * (math.sqrt(0.1) ** drops)


def parse_class_ids_arg(value: str, train_root: Path, test_root: Path) -> list[int]:
    if value.strip().lower() == "auto":
        ids = set(scan_yolo_class_ids(train_root / "labels"))
        ids.update(scan_yolo_class_ids(test_root / "labels"))
        return sorted(ids) or [0]
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return sorted(dict.fromkeys(ids)) or [0]


def loss_kwargs_from_args(args: argparse.Namespace) -> dict:
    output_layout = output_layout_from_instance_head(args.instance_head)
    return {
        "output_layout": output_layout,
        "boundary_weight": float(args.boundary_loss_weight),
        "center_weight": float(args.center_loss_weight),
        "offset_weight": float(args.offset_loss_weight),
        "hole_weight": float(args.hole_loss_weight),
        "boundary_pos_weight": float(args.boundary_pos_weight),
        "center_pos_weight": float(args.center_pos_weight),
        "hole_pos_weight": float(args.hole_pos_weight),
    }


def make_worker_init_fn(seed: int):
    def _init_worker(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id) + 1
        random.seed(worker_seed)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and hasattr(worker_info.dataset, "rng"):
            worker_info.dataset.rng.seed(worker_seed)

    return _init_worker


@torch.no_grad()
def evaluate(model: GateNet, loader: DataLoader, device: torch.device, loss_kwargs: dict) -> dict:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = gatenet_multiscale_loss(preds, y, **loss_kwargs)
        total_loss += float(loss.item())
        # highest-resolution output is y4 (after full upsampling)
        total_iou += iou_binary(preds[-1], y, output_layout=loss_kwargs.get("output_layout", "auto"))
        n += 1
    if n == 0:
        return {"loss": None, "iou": None}
    return {"loss": total_loss / n, "iou": total_iou / n}


def apply_global_l1_pruning_inplace(model: torch.nn.Module, amount: float) -> None:
    """
    Global unstructured L1 pruning over Conv2d/Linear weights.
    This makes the saved checkpoint sparse (many exact zeros), without changing architecture.
    """
    if amount <= 0:
        return
    if amount >= 1:
        raise ValueError("--save-prune-amount must be in [0, 1).")

    parameters_to_prune: list[tuple[torch.nn.Module, str]] = []
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((m, "weight"))

    if not parameters_to_prune:
        return

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in parameters_to_prune:
        prune.remove(mod, name)


@torch.no_grad()
def pruning_stats(model: torch.nn.Module) -> dict:
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            w = m.weight
            if w is None:
                continue
            total += w.numel()
            zeros += int((w == 0).sum().item())
    sparsity = (zeros / total) if total > 0 else 0.0
    return {"pruned_total": total, "pruned_zeros": zeros, "pruned_sparsity": sparsity}


def save_checkpoint(
    out_dir: Path,
    name: str,
    model: GateNet,
    meta: dict,
    prune_amount: float | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_model = model
    if prune_amount is not None and prune_amount > 0:
        save_model = GateNet(
            in_channels=3,
            f=int(meta["f"]),
            out_channels=int(meta.get("out_channels", 2)),
            regression_channels=int(meta.get("regression_channels", 0)),
            conv_kind=str(meta.get("conv_kind", "standard")),
            up_kind=str(meta.get("up_kind", "transpose")),
        )
        save_model.load_state_dict(model.state_dict(), strict=True)
        apply_global_l1_pruning_inplace(save_model, float(prune_amount))

    state = save_model.state_dict()
    payload = {
        "model": "GateNetInstance",
        "state_dict": state,
        "meta": meta,
    }
    path = out_dir / name
    torch.save(payload, path)
    return path


def train_hole_rows_only(model: GateNet, hole_channel: int) -> dict:
    """
    Freeze the existing GateNet trunk and output rows, leaving only the newly
    inserted hole channel in each multi-scale output head trainable.
    """
    hole_channel = int(hole_channel)
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_tensors = 0
    for module in (model.outc0, model.outc1, model.outc2, model.outc3, model.outc4):
        conv = module.conv
        for param in (conv.weight, conv.bias):
            if param is None:
                continue
            if hole_channel < 0 or hole_channel >= param.shape[0]:
                raise ValueError(f"Invalid hole channel {hole_channel} for output shape {tuple(param.shape)}")
            mask = torch.zeros_like(param)
            mask[hole_channel : hole_channel + 1] = 1
            param.requires_grad_(True)
            param.register_hook(lambda grad, mask=mask: grad * mask.to(device=grad.device, dtype=grad.dtype))
            trainable_tensors += 1

    return {
        "hole_only_head": True,
        "hole_only_channel": hole_channel,
        "hole_only_trainable_tensors": trainable_tensors,
    }


def train_output_rows_only(model: GateNet, row_channels: list[int], mode_name: str) -> dict:
    """
    Freeze the trunk and train only selected rows in the multi-scale output
    heads. This lets us retarget offset regression without changing foreground
    or boundary rows learned by a stable checkpoint.
    """
    rows = sorted(dict.fromkeys(int(ch) for ch in row_channels))
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_tensors = 0
    for module in (model.outc0, model.outc1, model.outc2, model.outc3, model.outc4):
        conv = module.conv
        for param in (conv.weight, conv.bias):
            if param is None:
                continue
            mask = torch.zeros_like(param)
            for row in rows:
                if row < 0 or row >= param.shape[0]:
                    raise ValueError(f"Invalid output row {row} for output shape {tuple(param.shape)}")
                mask[row : row + 1] = 1
            param.requires_grad_(True)
            param.register_hook(lambda grad, mask=mask: grad * mask.to(device=grad.device, dtype=grad.dtype))
            trainable_tensors += 1

    return {
        f"{mode_name}_head": True,
        f"{mode_name}_channels": rows,
        f"{mode_name}_trainable_tensors": trainable_tensors,
    }


def train_output_heads_only(model: GateNet) -> dict:
    """
    Freeze the encoder/decoder trunk and train only the multi-scale output heads.

    This is a middle ground between full fine-tuning, which can damage the
    foreground mask on a shifted synthetic mix, and training only the inserted
    hole row, which is often too weak to learn a new geometric cue.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_tensors = 0
    for module in (model.outc0, model.outc1, model.outc2, model.outc3, model.outc4):
        for param in module.parameters():
            param.requires_grad_(True)
            trainable_tensors += 1

    return {
        "head_only": True,
        "head_only_trainable_tensors": trainable_tensors,
    }


def set_batchnorm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def init_model_from_checkpoint(model: GateNet, ckpt_path: Path, output_layout: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src_state = ckpt["state_dict"]
    src_meta = ckpt.get("meta") or {}
    src_layout = str(src_meta.get("output_layout") or "")
    dst_state = model.state_dict()
    copied_exact = 0
    copied_remap = 0

    for key, src in src_state.items():
        if key not in dst_state:
            continue
        dst = dst_state[key]
        if tuple(dst.shape) == tuple(src.shape):
            dst_state[key] = src
            copied_exact += 1
            continue

        is_out_head = key.startswith("outc") and ".conv." in key and src.ndim in {1, 4} and dst.ndim == src.ndim
        can_remap_offset_hole = (
            is_out_head
            and output_layout == "class_masks_boundary_hole_offset"
            and src_layout == "class_masks_boundary_offset"
            and int(src_meta.get("regression_channels", 0)) == 2
            and src.shape[0] + 1 == dst.shape[0]
        )
        if not can_remap_offset_hole:
            continue

        n_cls = int(src_meta.get("num_classes") or max(1, src.shape[0] - 3))
        remapped = dst.clone()
        # old: [class..., boundary, offset_x, offset_y]
        # new: [class..., boundary, hole, offset_x, offset_y]
        remapped[: n_cls + 1] = src[: n_cls + 1]
        remapped[n_cls + 2 : n_cls + 4] = src[n_cls + 1 : n_cls + 3]
        dst_state[key] = remapped
        copied_remap += 1

    model.load_state_dict(dst_state, strict=True)
    return {
        "init_from": str(ckpt_path),
        "init_source_layout": src_layout,
        "init_copied_exact": copied_exact,
        "init_copied_remap": copied_remap,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GateNet instance segmentation on YOLO-seg dataset.")
    ap.add_argument("--data", type=Path, required=True, help="Split root containing train/ and test/")
    ap.add_argument("--out", type=Path, default=Path("runs/gatenet"), help="Output directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--f", type=int, default=4, help="Channel scale factor f (paper uses 4)")
    ap.add_argument("--conv-kind", choices=("standard", "separable"), default="standard")
    ap.add_argument("--up-kind", choices=("transpose", "bilinear"), default="transpose")
    ap.add_argument("--boundary-width", type=int, default=3, help="Boundary target width in pixels before resize")
    ap.add_argument("--center-radius", type=float, default=0.12, help="Center seed radius as a fraction of instance bbox size")
    ap.add_argument("--hole-radius", type=float, default=0.08, help="Hole seed radius as a fraction of instance bbox size")
    ap.add_argument(
        "--offset-anchor",
        choices=("centroid", "hole"),
        default="centroid",
        help="Offset target anchor. 'hole' uses the largest enclosed gate opening when available.",
    )
    ap.add_argument(
        "--instance-head",
        choices=("offset", "offset_hole", "center"),
        default="offset",
        help="offset_hole adds a light hollow-frame seed prior; offset keeps the current head; center keeps the legacy seed head",
    )
    ap.add_argument(
        "--class-ids",
        type=str,
        default="auto",
        help="Comma-separated original YOLO class ids to train, or auto to scan labels",
    )
    ap.add_argument("--boundary-loss-weight", type=float, default=0.75)
    ap.add_argument("--center-loss-weight", type=float, default=0.75)
    ap.add_argument("--offset-loss-weight", type=float, default=2.0)
    ap.add_argument("--hole-loss-weight", type=float, default=0.75)
    ap.add_argument("--boundary-pos-weight", type=float, default=4.0)
    ap.add_argument("--center-pos-weight", type=float, default=8.0)
    ap.add_argument("--hole-pos-weight", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--augment", dest="augment", action="store_true", help="Enable light online geometric augmentation for training")
    ap.add_argument("--no-augment", dest="augment", action="store_false", help="Disable online augmentation for training")
    ap.set_defaults(augment=True)
    ap.add_argument("--init-from", type=Path, default=None, help="Initialize matching weights from an existing checkpoint")
    ap.add_argument(
        "--hole-only-head",
        action="store_true",
        help="Freeze trunk and existing output rows, training only the inserted hole channel for offset_hole",
    )
    ap.add_argument(
        "--head-only",
        action="store_true",
        help="Freeze trunk and train only the multi-scale output heads",
    )
    ap.add_argument(
        "--offset-only-head",
        action="store_true",
        help="Freeze trunk and existing output rows, training only offset_x/offset_y rows",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--save-prune-amount",
        type=float,
        default=0.0,
        help="Apply global L1 unstructured pruning to Conv/Linear weights when saving (e.g. 0.3)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = auto_device(args.device)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = args.data / "train"
    test_root = args.data / "test"
    class_ids = parse_class_ids_arg(args.class_ids, train_root=train_root, test_root=test_root)
    output_layout = output_layout_from_instance_head(args.instance_head)
    regression_channels = 2 if args.instance_head in {"offset", "offset_hole"} else 0
    if args.instance_head == "offset_hole":
        out_channels = len(class_ids) + 4
        output_channels = [f"class_{cid}" for cid in class_ids] + ["boundary", "hole", "offset_x", "offset_y"]
    elif args.instance_head == "offset":
        out_channels = len(class_ids) + 3
        output_channels = [f"class_{cid}" for cid in class_ids] + ["boundary", "offset_x", "offset_y"]
    else:
        out_channels = len(class_ids) + 2
        output_channels = [f"class_{cid}" for cid in class_ids] + ["boundary", "center"]
    loss_kwargs = loss_kwargs_from_args(args)

    ds_train = YoloSegDataset(
        train_root,
        img_size=args.img_size,
        augment=bool(args.augment),
        seed=args.seed,
        boundary_width=args.boundary_width,
        class_ids=class_ids,
        center_radius=args.center_radius,
        hole_radius=args.hole_radius,
        instance_head=args.instance_head,
        offset_anchor=args.offset_anchor,
    )
    ds_test = YoloSegDataset(
        test_root,
        img_size=args.img_size,
        augment=False,
        seed=args.seed,
        boundary_width=args.boundary_width,
        class_ids=class_ids,
        center_radius=args.center_radius,
        hole_radius=args.hole_radius,
        instance_head=args.instance_head,
        offset_anchor=args.offset_anchor,
    )

    worker_init_fn = make_worker_init_fn(args.seed) if args.num_workers > 0 else None
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": worker_init_fn,
    }
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = GateNet(
        in_channels=3,
        f=args.f,
        out_channels=out_channels,
        regression_channels=regression_channels,
        conv_kind=args.conv_kind,
        up_kind=args.up_kind,
    ).to(device)
    init_meta = init_model_from_checkpoint(model, args.init_from, output_layout=output_layout) if args.init_from else {}
    row_only_modes = int(bool(args.hole_only_head)) + int(bool(args.head_only)) + int(bool(args.offset_only_head))
    if row_only_modes > 1:
        raise ValueError("--hole-only-head, --head-only, and --offset-only-head are mutually exclusive")
    head_only_meta = {}
    hole_only_meta = {}
    offset_only_meta = {}
    if args.head_only:
        head_only_meta = train_output_heads_only(model)
    if args.hole_only_head:
        if output_layout != "class_masks_boundary_hole_offset":
            raise ValueError("--hole-only-head requires --instance-head offset_hole")
        hole_only_meta = train_hole_rows_only(model, hole_channel=len(class_ids) + 1)
    if args.offset_only_head:
        if output_layout == "class_masks_boundary_hole_offset":
            offset_rows = [len(class_ids) + 2, len(class_ids) + 3]
        elif output_layout == "class_masks_boundary_offset":
            offset_rows = [len(class_ids) + 1, len(class_ids) + 2]
        else:
            raise ValueError("--offset-only-head requires --instance-head offset or offset_hole")
        offset_only_meta = train_output_rows_only(model, offset_rows, mode_name="offset_only")
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=float(args.weight_decay),
    )

    best_iou = -1.0
    best_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        lr = lr_schedule(args.lr, epoch - 1)
        set_lr(optim, lr)

        model.train()
        if args.hole_only_head or args.head_only or args.offset_only_head:
            set_batchnorm_eval(model)
        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{args.epochs} lr={lr:.2e}", leave=False)
        running = 0.0
        steps = 0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            preds = model(x)
            loss = gatenet_multiscale_loss(preds, y, **loss_kwargs)
            loss.backward()
            optim.step()
            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=running / max(1, steps))

        train_loss = running / max(1, steps)
        metrics = evaluate(model, dl_test, device, loss_kwargs=loss_kwargs)

        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "test_loss": metrics["loss"], "test_iou": metrics["iou"]}
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        meta = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "test_loss": metrics["loss"],
            "test_iou": metrics["iou"],
            "img_size": args.img_size,
            "f": args.f,
            "conv_kind": args.conv_kind,
            "up_kind": args.up_kind,
            "out_channels": out_channels,
            "regression_channels": regression_channels,
            "class_ids": class_ids,
            "num_classes": len(class_ids),
            "instance_head": args.instance_head,
            "output_layout": output_layout,
            "output_channels": output_channels,
            "weight_decay": float(args.weight_decay),
            "train_augment": bool(args.augment),
            "boundary_width": args.boundary_width,
            "center_radius": args.center_radius,
            "hole_radius": args.hole_radius,
            "offset_anchor": args.offset_anchor,
            **init_meta,
            **head_only_meta,
            **hole_only_meta,
            **offset_only_meta,
            **loss_kwargs,
        }

        if args.save_prune_amount and args.save_prune_amount > 0:
            meta = {**meta, **{"save_prune_amount": float(args.save_prune_amount)}}
            tmp = GateNet(
                in_channels=3,
                f=args.f,
                out_channels=out_channels,
                regression_channels=regression_channels,
                conv_kind=args.conv_kind,
                up_kind=args.up_kind,
            )
            tmp.load_state_dict(model.state_dict(), strict=True)
            apply_global_l1_pruning_inplace(tmp, float(args.save_prune_amount))
            meta = {**meta, **pruning_stats(tmp)}

        # always save last (pruned, if enabled)
        save_checkpoint(out_dir, "last_pruned.pt" if args.save_prune_amount > 0 else "last.pt", model, meta=meta, prune_amount=args.save_prune_amount)

        is_best = False
        if metrics["iou"] is not None:
            current_iou = float(metrics["iou"])
            current_loss = float(metrics["loss"]) if metrics["loss"] is not None else float("inf")
            if current_iou > best_iou:
                is_best = True
            elif abs(current_iou - best_iou) <= 1e-9 and current_loss < best_loss:
                is_best = True
            if is_best:
                best_iou = current_iou
                best_loss = current_loss

        if is_best:
            save_checkpoint(out_dir, "best_pruned.pt" if args.save_prune_amount > 0 else "best.pt", model, meta=meta, prune_amount=args.save_prune_amount)

        print(f"epoch {epoch}: train_loss={train_loss:.4f} test_loss={metrics['loss']:.4f} test_iou={metrics['iou']:.4f} best_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
