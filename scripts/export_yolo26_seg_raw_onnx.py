from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO


class RawSegmentWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred, proto = self.model(x)
        return pred, proto


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export YOLO26-seg raw predictions without TopK/end2end postprocess.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--dynamic", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    yolo = YOLO(str(args.weights))
    model = yolo.model.eval()
    head = model.model[-1]

    # TensorRT 8.5 fails on the end2end TopK postprocess graph. Export decoded raw
    # predictions and run threshold/NMS/mask decoding outside TensorRT.
    head.end2end = False
    head.export = True
    head.format = "onnx"
    if hasattr(head, "dynamic"):
        head.dynamic = bool(args.dynamic)

    wrapper = RawSegmentWrapper(model).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, dtype=torch.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "raw_pred": {0: "batch", 2: "anchors"},
            "proto": {0: "batch", 2: "mask_height", 3: "mask_width"},
        }

    with torch.no_grad():
        pred, proto = wrapper(dummy)
    print(f"raw_pred shape={tuple(pred.shape)} proto shape={tuple(proto.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        input_names=["images"],
        output_names=["raw_pred", "proto"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"Exported raw YOLO26-seg ONNX -> {args.out} opset={args.opset} dynamic={args.dynamic}")


if __name__ == "__main__":
    main()
