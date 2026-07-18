from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Patch ONNX negative axis attributes for older TensorRT parsers.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    return ap.parse_args()


def value_ranks(model: onnx.ModelProto) -> dict[str, int]:
    ranks: dict[str, int] = {}
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for vi in values:
        t = vi.type.tensor_type
        if t.elem_type == TensorProto.UNDEFINED or not t.HasField("shape"):
            continue
        ranks[vi.name] = len(t.shape.dim)
    return ranks


def patch_axis_attrs(model: onnx.ModelProto, ranks: dict[str, int]) -> list[str]:
    changes: list[str] = []
    for idx, node in enumerate(model.graph.node):
        if not node.input:
            continue
        rank = ranks.get(node.input[0])
        if rank is None:
            continue

        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(helper.get_attribute_value(attr))
                if axis < 0:
                    new_axis = axis + rank
                    if new_axis < 0 or new_axis > rank:
                        raise ValueError(f"{node.name or idx}: cannot convert axis={axis} with rank={rank}")
                    attr.i = int(new_axis)
                    changes.append(f"{idx}: {node.op_type} {node.name}: axis {axis} -> {new_axis} rank={rank}")

            elif attr.name == "axes":
                axes = list(helper.get_attribute_value(attr))
                if any(int(a) < 0 for a in axes):
                    new_axes = []
                    for axis in axes:
                        axis = int(axis)
                        new_axis = axis + rank if axis < 0 else axis
                        if new_axis < 0 or new_axis > rank:
                            raise ValueError(f"{node.name or idx}: cannot convert axes={axes} with rank={rank}")
                        new_axes.append(new_axis)
                    del attr.ints[:]
                    attr.ints.extend(new_axes)
                    changes.append(f"{idx}: {node.op_type} {node.name}: axes {axes} -> {new_axes} rank={rank}")
    return changes


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input)

    inferred = shape_inference.infer_shapes(model)
    ranks = value_ranks(inferred)
    changes = patch_axis_attrs(model, ranks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    if args.check:
        onnx.checker.check_model(args.output)

    print(f"Saved TensorRT-axis patched ONNX: {args.output}")
    if not changes:
        print("No negative axis attributes were patched.")
    else:
        print("Patched axis attributes:")
        for line in changes:
            print(f"  {line}")


if __name__ == "__main__":
    main()
