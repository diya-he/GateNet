from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn


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
    ap = argparse.ArgumentParser(description="Export a TRT8.5-friendly YOLO26-seg ONNX with fixed 1D TopK prefilter.")
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_seg_384/weights/best.pt"),
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--branch", choices=("auto", "one2one", "one2many"), default="auto")
    ap.add_argument("--ultralytics-root", type=Path, default=Path("ultralytics"))
    ap.add_argument("--no-simplify", action="store_true")
    return ap.parse_args()


class TRT85TopKSegWrapper(nn.Module):
    """Batch-1 segmentation wrapper that makes TopK run over a rank-1 score tensor."""

    def __init__(self, model: nn.Module, topk: int, branch: str = "auto"):
        super().__init__()
        self.model = model
        self.topk = int(topk)
        self.branch = branch

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred, proto, boxes_are_xyxy = self.forward_raw(images)
        pred = pred[0].transpose(0, 1).contiguous()  # [anchors, 4 + classes + masks], batch fixed to 1
        boxes = pred[:, 0:4]
        scores = pred[:, 4]
        mask_coeff = pred[:, 5:]

        scores, indices = torch.topk(scores, self.topk, dim=0, largest=True, sorted=True)
        boxes = torch.index_select(boxes, 0, indices)
        mask_coeff = torch.index_select(mask_coeff, 0, indices)

        if boxes_are_xyxy:
            boxes_xyxy = boxes
        else:
            xy = boxes[:, 0:2]
            half_wh = boxes[:, 2:4] * 0.5
            boxes_xyxy = torch.cat((xy - half_wh, xy + half_wh), dim=1)
        classes = torch.zeros_like(scores).unsqueeze(1)
        detections = torch.cat((boxes_xyxy, scores.unsqueeze(1), classes, mask_coeff), dim=1)
        return detections.unsqueeze(0), proto

    def forward_raw(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        y = []
        x = images
        head = self.model.model[-1]
        for m in self.model.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m is head:
                use_one2one = self.branch == "one2one" or (
                    self.branch == "auto" and getattr(head, "end2end", False) and hasattr(head, "one2one_cv2")
                )
                if use_one2one:
                    preds = head.forward_head(x, **head.one2one)
                    boxes_are_xyxy = True
                else:
                    original_end2end = bool(getattr(head, "end2end", False))
                    head.end2end = False
                    preds = head.forward_head(x, **head.one2many)
                    boxes_are_xyxy = False
                    head.end2end = original_end2end
                pred = head._inference(preds)
                proto = head.proto(x)
                return pred, proto, boxes_are_xyxy
            x = m(x)
            y.append(x if m.i in self.model.save else None)
        raise RuntimeError("Segmentation head was not reached")


def prepare_model(weights: Path, device: torch.device) -> nn.Module:
    from ultralytics import YOLO
    from ultralytics.nn.modules import Classify, Detect, RTDETRDecoder, Segment, Segment26, SemanticSegment

    model = YOLO(str(weights)).model.to(device).eval().float()
    has_end2end_head = any(isinstance(m, Detect) and getattr(m, "end2end", False) for m in model.modules())
    has_segment_head = isinstance(model.model[-1], (Segment, Segment26))
    if has_end2end_head or has_segment_head:
        print("Skipping model.fuse() to keep segmentation head branches for TopK export.")
    else:
        model = model.fuse().eval()
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (Classify, SemanticSegment)):
            m.export = True
            m.format = "onnx"
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = "onnx"
            m.xyxy = False
            m.shape = None
    return model


def simplify_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(path)
    try:
        import onnxslim

        model = onnxslim.slim(model)
    except Exception as e:
        print(f"WARNING: onnxslim failed: {e}")
    if getattr(model, "ir_version", 0) > 10:
        model.ir_version = 10
    onnx.save(model, path)


def summarize_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(path)
    ops = {node.op_type for node in model.graph.node}
    topk_axes = [attr.i for node in model.graph.node if node.op_type == "TopK" for attr in node.attribute if attr.name == "axis"]
    outputs = [
        (o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
        for o in model.graph.output
    ]
    print(f"ONNX: {path} ({path.stat().st_size / 1024 / 1024:.3f} MB)")
    print(f"TopK axes: {topk_axes}")
    print(f"Blocked ops: {{'GatherElements': {'GatherElements' in ops}, 'NonMaxSuppression': {'NonMaxSuppression' in ops}}}")
    print(f"Outputs: {outputs}")


def main() -> None:
    args = parse_args()
    local_root = use_local_ultralytics(args.ultralytics_root)
    print(f"Using local Ultralytics: {local_root}")

    device = torch.device(args.device)
    out = args.out or args.weights.with_name(f"{args.weights.stem}_trt85_topk{args.topk}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    model = prepare_model(args.weights, device)
    wrapper = TRT85TopKSegWrapper(model, args.topk, args.branch).to(device).eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        detections, proto = wrapper(dummy)
    print(f"Trace outputs: detections={tuple(detections.shape)}, proto={tuple(proto.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        opset_version=args.opset,
        input_names=["images"],
        output_names=["detections", "proto"],
        do_constant_folding=True,
        dynamo=False,
    )
    if not args.no_simplify:
        simplify_onnx(out)
    summarize_onnx(out)


if __name__ == "__main__":
    main()
