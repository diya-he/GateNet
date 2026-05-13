from __future__ import annotations

import torch
import torch.nn.functional as F


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


def gatenet_multiscale_loss(
    preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    target_full: torch.Tensor,
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
        li = dice_loss(y, tgt) + 2.0 * bce_loss(y, tgt)
        if i == 0:
            total = total + 4.0 * li
        elif i == 1:
            total = total + 2.0 * li
        else:
            total = total + li
    return total


@torch.no_grad()
def iou_binary(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5, eps: float = 1e-6) -> float:
    """
    pred/target: (B,C,H,W). IoU is computed on channel 0 (foreground).
    """
    p = (pred[:, :1] >= thresh).to(torch.uint8)
    t = (target[:, :1] >= 0.5).to(torch.uint8)
    inter = (p & t).sum(dim=(1, 2, 3)).float()
    union = (p | t).sum(dim=(1, 2, 3)).float()
    iou = (inter + eps) / (union + eps)
    return float(iou.mean().item())

