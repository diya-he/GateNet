from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from gatenet.model import GateNet


class GateNetY4(nn.Module):
    """
    Export-friendly wrapper: only output the highest-resolution map (y4).
    """

    def __init__(self, base: GateNet) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)[-1]


def load_ckpt(ckpt_path: Path, device: torch.device) -> tuple[GateNet, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta") or {}
    f = int(meta.get("f", 4))
    model = GateNet(in_channels=3, f=f).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Export GateNet checkpoint to ONNX (y4 output only).")
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint .pt (best_pruned.pt / best.pt)")
    ap.add_argument("--out", type=Path, required=True, help="Output onnx path, e.g. runs/model.onnx")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic batch/H/W axes")
    ap.add_argument("--half", action="store_true", help="Export fp16 weights (best with CUDA EP)")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, meta = load_ckpt(args.ckpt, device=device)
    wrapper = GateNetY4(model)
    wrapper.eval()

    if args.half:
        wrapper = wrapper.half()

    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    if args.half:
        dummy = dummy.half()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "y4": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["y4"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(
        f"Exported ONNX -> {args.out}\n"
        f"meta.f={meta.get('f', 4)} img_size={args.img_size} opset={args.opset} dynamic={args.dynamic} half={args.half}"
    )


if __name__ == "__main__":
    main()

