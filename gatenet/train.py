from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from gatenet.data import YoloSegDataset
from gatenet.losses import gatenet_multiscale_loss, iou_binary
from gatenet.model import GateNet


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


@torch.no_grad()
def evaluate(model: GateNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = gatenet_multiscale_loss(preds, y)
        total_loss += float(loss.item())
        # highest-resolution output is y4 (after full upsampling)
        total_iou += iou_binary(preds[-1], y)
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
        save_model = GateNet(in_channels=3, f=int(meta["f"]))
        save_model.load_state_dict(model.state_dict(), strict=True)
        apply_global_l1_pruning_inplace(save_model, float(prune_amount))

    state = save_model.state_dict()
    payload = {
        "model": "GateNet",
        "state_dict": state,
        "meta": meta,
    }
    path = out_dir / name
    torch.save(payload, path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train GateNet (MonoRace) on YOLO-seg dataset.")
    ap.add_argument("--data", type=Path, required=True, help="Split root containing train/ and test/")
    ap.add_argument("--out", type=Path, default=Path("runs/gatenet"), help="Output directory")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--f", type=int, default=4, help="Channel scale factor f (paper uses 4)")
    ap.add_argument("--lr", type=float, default=1e-3)
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
    ds_train = YoloSegDataset(train_root, img_size=args.img_size, augment=True, seed=args.seed)
    ds_test = YoloSegDataset(test_root, img_size=args.img_size, augment=False, seed=args.seed)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GateNet(in_channels=3, f=args.f).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_iou = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        lr = lr_schedule(args.lr, epoch - 1)
        set_lr(optim, lr)

        model.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{args.epochs} lr={lr:.2e}", leave=False)
        running = 0.0
        steps = 0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            preds = model(x)
            loss = gatenet_multiscale_loss(preds, y)
            loss.backward()
            optim.step()
            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=running / max(1, steps))

        train_loss = running / max(1, steps)
        metrics = evaluate(model, dl_test, device)

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
        }

        if args.save_prune_amount and args.save_prune_amount > 0:
            meta = {**meta, **{"save_prune_amount": float(args.save_prune_amount)}}
            tmp = GateNet(in_channels=3, f=args.f)
            tmp.load_state_dict(model.state_dict(), strict=True)
            apply_global_l1_pruning_inplace(tmp, float(args.save_prune_amount))
            meta = {**meta, **pruning_stats(tmp)}

        # always save last (pruned, if enabled)
        save_checkpoint(out_dir, "last_pruned.pt" if args.save_prune_amount > 0 else "last.pt", model, meta=meta, prune_amount=args.save_prune_amount)

        if metrics["iou"] is not None and metrics["iou"] > best_iou:
            best_iou = float(metrics["iou"])
            save_checkpoint(out_dir, "best_pruned.pt" if args.save_prune_amount > 0 else "best.pt", model, meta=meta, prune_amount=args.save_prune_amount)

        print(f"epoch {epoch}: train_loss={train_loss:.4f} test_loss={metrics['loss']:.4f} test_iou={metrics['iou']:.4f} best_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()

