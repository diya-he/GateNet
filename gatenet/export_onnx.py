from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from gatenet.model import GateNet


class GateNetY4(nn.Module):
    """
    Export-friendly wrapper: only output the highest-resolution instance map (y4).
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
    state_dict = ckpt["state_dict"]
    out_channels = int(meta.get("out_channels") or state_dict["outc4.conv.weight"].shape[0])
    if out_channels != 2:
        raise ValueError(
            f"{ckpt_path} is a {out_channels}-channel semantic checkpoint. "
            "Retrain with the instance model before exporting instance ONNX."
        )
    model = GateNet(in_channels=3, f=f, out_channels=out_channels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Export GateNet instance checkpoint to ONNX (foreground+boundary y4 only).")
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
            "instance_y4": {0: "batch", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        input_names=["input"],
        output_names=["instance_y4"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(
        f"Exported ONNX -> {args.out}\n"
        f"meta.f={meta.get('f', 4)} out_channels={meta.get('out_channels', 2)} "
        f"img_size={args.img_size} opset={args.opset} dynamic={args.dynamic} half={args.half}"
    )


if __name__ == "__main__":
    main()

