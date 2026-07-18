from __future__ import annotations

import torch
import torch.nn.functional as F


def _infer_output_layout(x: torch.Tensor) -> str:
    if x.shape[1] <= 2:
        return "legacy"
    if x.shape[1] >= 4:
        tail = x[:, -2:]
        if bool((tail < -1e-4).any().item()) or bool((tail > 1.0 + 1e-4).any().item()):
            if x.shape[1] >= 5:
                return "class_masks_boundary_hole_offset"
            return "class_masks_boundary_offset"
    return "class_masks_boundary_center"


def _num_class_channels(x: torch.Tensor, output_layout: str = "auto") -> int:
    layout = _infer_output_layout(x) if output_layout == "auto" else output_layout
    if layout == "legacy":
        return 1
    if layout == "class_masks_boundary_offset":
        return max(1, x.shape[1] - 3)
    if layout == "class_masks_boundary_hole_offset":
        return max(1, x.shape[1] - 4)
    if layout == "class_masks_boundary_center":
        return max(1, x.shape[1] - 2)
    raise ValueError(f"Unsupported output_layout: {output_layout}")


def foreground_from_channels(x: torch.Tensor, output_layout: str = "auto") -> torch.Tensor:
    """
    Return a class-agnostic foreground view.

    Legacy checkpoints have channels [foreground, boundary].
    Center checkpoints have [class_*, boundary, center].
    Offset checkpoints have [class_*, boundary, offset_x, offset_y].
    Offset-hole checkpoints have [class_*, boundary, hole, offset_x, offset_y].
    """
    if x.shape[1] <= 2:
        return x[:, :1]
    n_cls = _num_class_channels(x, output_layout=output_layout)
    return x[:, :n_cls].amax(dim=1, keepdim=True)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/target: (B,1,H,W) in [0,1]
    """
    pred = pred.contiguous()
    target = target.contiguous()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, target)


def weighted_bce_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(eps, 1.0 - eps)
    pos = float(pos_weight) * target * torch.log(pred)
    neg = (1.0 - target) * torch.log(1.0 - pred)
    return -(pos + neg).mean()


def single_scale_instance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    output_layout: str = "auto",
    boundary_weight: float = 0.75,
    center_weight: float = 0.75,
    offset_weight: float = 2.0,
    hole_weight: float = 0.75,
    boundary_pos_weight: float = 4.0,
    center_pos_weight: float = 8.0,
    hole_pos_weight: float = 8.0,
) -> torch.Tensor:
    if pred.shape[1] <= 2:
        return dice_loss(pred, target) + 2.0 * bce_loss(pred, target)

    layout = _infer_output_layout(target) if output_layout == "auto" else output_layout

    if layout == "class_masks_boundary_offset":
        n_cls = pred.shape[1] - 3
        pred_cls = pred[:, :n_cls]
        tgt_cls = target[:, :n_cls]
        pred_boundary = pred[:, n_cls : n_cls + 1]
        tgt_boundary = target[:, n_cls : n_cls + 1]
        pred_offsets = pred[:, n_cls + 1 : n_cls + 3]
        tgt_offsets = target[:, n_cls + 1 : n_cls + 3]

        class_loss = dice_loss(pred_cls, tgt_cls) + 2.0 * bce_loss(pred_cls, tgt_cls)
        boundary_loss = dice_loss(pred_boundary, tgt_boundary) + 2.0 * weighted_bce_loss(
            pred_boundary,
            tgt_boundary,
            pos_weight=boundary_pos_weight,
        )

        fg = tgt_cls.amax(dim=1, keepdim=True)
        denom = fg.sum() * 2.0 + 1e-6
        offset_loss = (F.smooth_l1_loss(pred_offsets * fg, tgt_offsets * fg, reduction="sum") / denom)
        return class_loss + float(boundary_weight) * boundary_loss + float(offset_weight) * offset_loss

    if layout == "class_masks_boundary_hole_offset":
        n_cls = pred.shape[1] - 4
        pred_cls = pred[:, :n_cls]
        tgt_cls = target[:, :n_cls]
        pred_boundary = pred[:, n_cls : n_cls + 1]
        tgt_boundary = target[:, n_cls : n_cls + 1]
        pred_hole = pred[:, n_cls + 1 : n_cls + 2]
        tgt_hole = target[:, n_cls + 1 : n_cls + 2]
        pred_offsets = pred[:, n_cls + 2 : n_cls + 4]
        tgt_offsets = target[:, n_cls + 2 : n_cls + 4]

        class_loss = dice_loss(pred_cls, tgt_cls) + 2.0 * bce_loss(pred_cls, tgt_cls)
        boundary_loss = dice_loss(pred_boundary, tgt_boundary) + 2.0 * weighted_bce_loss(
            pred_boundary,
            tgt_boundary,
            pos_weight=boundary_pos_weight,
        )
        hole_loss = dice_loss(pred_hole, tgt_hole) + 2.0 * weighted_bce_loss(
            pred_hole,
            tgt_hole,
            pos_weight=hole_pos_weight,
        )

        fg = tgt_cls.amax(dim=1, keepdim=True)
        denom = fg.sum() * 2.0 + 1e-6
        offset_loss = (F.smooth_l1_loss(pred_offsets * fg, tgt_offsets * fg, reduction="sum") / denom)
        return (
            class_loss
            + float(boundary_weight) * boundary_loss
            + float(hole_weight) * hole_loss
            + float(offset_weight) * offset_loss
        )

    n_cls = pred.shape[1] - 2
    pred_cls = pred[:, :n_cls]
    tgt_cls = target[:, :n_cls]
    pred_boundary = pred[:, n_cls : n_cls + 1]
    tgt_boundary = target[:, n_cls : n_cls + 1]
    pred_center = pred[:, n_cls + 1 : n_cls + 2]
    tgt_center = target[:, n_cls + 1 : n_cls + 2]

    class_loss = dice_loss(pred_cls, tgt_cls) + 2.0 * bce_loss(pred_cls, tgt_cls)
    boundary_loss = dice_loss(pred_boundary, tgt_boundary) + 2.0 * weighted_bce_loss(
        pred_boundary,
        tgt_boundary,
        pos_weight=boundary_pos_weight,
    )
    center_loss = dice_loss(pred_center, tgt_center) + 2.0 * weighted_bce_loss(
        pred_center,
        tgt_center,
        pos_weight=center_pos_weight,
    )
    return class_loss + float(boundary_weight) * boundary_loss + float(center_weight) * center_loss


def gatenet_multiscale_loss(
    preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_full: torch.Tensor,
    output_layout: str = "auto",
    boundary_weight: float = 0.75,
    center_weight: float = 0.75,
    offset_weight: float = 2.0,
    hole_weight: float = 0.75,
    boundary_pos_weight: float = 4.0,
    center_pos_weight: float = 8.0,
    hole_pos_weight: float = 8.0,
) -> torch.Tensor:
    """
    Paper:
      Li = Dice(yi, y^i) + 2 * BCE(yi, y^i)
      Ltot = 4*L0 + 2*L1 + sum_{i=2..4} Li

    Here L0 corresponds to y0 (deepest, lowest resolution) and uses a downscaled target.
    For instance segmentation each output has foreground and boundary channels.
    """
    y0, y1, y2, y3, y4 = preds
    outs = [y0, y1, y2, y3, y4]

    total = 0.0
    for i, y in enumerate(outs):
        tgt = target_full
        if y.shape[-2:] != target_full.shape[-2:]:
            tgt = F.interpolate(target_full, size=y.shape[-2:], mode="nearest")
        li = single_scale_instance_loss(
            y,
            tgt,
            output_layout=output_layout,
            boundary_weight=boundary_weight,
            center_weight=center_weight,
            offset_weight=offset_weight,
            hole_weight=hole_weight,
            boundary_pos_weight=boundary_pos_weight,
            center_pos_weight=center_pos_weight,
            hole_pos_weight=hole_pos_weight,
        )
        if i == 0:
            total = total + 4.0 * li
        elif i == 1:
            total = total + 2.0 * li
        else:
            total = total + li
    return total


@torch.no_grad()
def iou_binary(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresh: float = 0.5,
    eps: float = 1e-6,
    output_layout: str = "auto",
) -> float:
    """
    pred/target: (B,C,H,W). IoU is computed on class-agnostic foreground.
    """
    p = (foreground_from_channels(pred, output_layout=output_layout) >= thresh).to(torch.uint8)
    t = (foreground_from_channels(target, output_layout=output_layout) >= 0.5).to(torch.uint8)
    inter = (p & t).sum(dim=(1, 2, 3)).float()
    union = (p | t).sum(dim=(1, 2, 3)).float()
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())
