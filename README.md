# GateNet instance segmentation

This repo contains a lightweight gate instance-segmentation training pipeline based on the
**GateNet** architecture described in `paper/monorace.pdf` (U-Net style, 5 multi-scale
outputs, Dice+BCE).

The model predicts two channels at each scale:

- `foreground`: gate/object foreground probability
- `boundary`: instance boundary probability

Post-processing converts these two maps into instance IDs by thresholding foreground,
removing predicted boundaries to create instance seeds, then growing each seed back into
the foreground mask. `results.json` reports `pred_instances` for every image.

## Dataset

Expected YOLO-seg folder layout:

```
data/xxxxxx/
  images/  # *.jpg, *.png ...
  labels/  # *.txt (YOLO segmentation polygons)
```

Each label line:

```
<class_id> <x1> <y1> <x2> <y2> ...   # normalized to [0,1]
```

Each line is treated as one instance. The trainer turns the polygons into a foreground
mask plus a boundary mask, so you can train instance separation from standard YOLO-seg
labels without changing the dataset layout.

## Split train/test

This will create:

```
data/splits/image1/
  train/images, train/labels
  test/images,  test/labels
```

```bash
python -m gatenet.split_yolo_seg --src data/image1 --out data/splits/image1 --test-ratio 0.05 --seed 42
```

## Train + eval

```bash
pip install -r requirements.txt
python -m gatenet.train \
  --data data/splits/image1 \
  --epochs 100 --batch-size 16 --img-size 384 --f 4 \
  --boundary-width 3 \
  --lr 1e-3 --device auto \
  --out runs/gatenet_image1 \
  --save-prune-amount 0.3
```

Outputs:
- `runs/.../best_pruned.pt`: best checkpoint (by IoU on test) with **pruning** applied
- `runs/.../last_pruned.pt`: last checkpoint with **pruning** applied

## Inference on test split (speed + masks)

```bash
python -m gatenet.infer_test \
  --data data/splits/image1 \
  --ckpt runs/gatenet_image1/best_pruned.pt \
  --out runs/gatenet_image1/test_infer \
  --device auto \
  --threshold 0.5 --boundary-threshold 0.5 --min-area 20 \
  --save-overlay --save-gt
```

Outputs:
- `.../pred_foreground/*.png`: predicted foreground binary masks
- `.../pred_boundary/*.png`: predicted boundary binary masks
- `.../instance_maps/*.png`: 16-bit PNG where pixel value is the instance id (`0` = background)
- `.../instance_color/*.png`: colorized instance preview
- `.../gt_masks/*.png`: GT masks (optional)
- `.../overlay/*.png`: overlay preview (optional)
- `.../results.json`: foreground IoU + latency + FPS + per-image instance counts

In `results.json`, each `per_image` item includes:

- `pred_instances`: number of predicted instances
- `gt_instances`: number of YOLO label lines
- `instances`: each predicted instance's id, area and bounding box

## Export ONNX + ONNXRuntime inference

Export (highest-resolution `y4` output only, shape `1x2xHxW`):

```bash
python -m gatenet.export_onnx \
  --ckpt runs/gatenet_image1/best_pruned.pt \
  --out runs/gatenet_image1/gatenet_instance_y4.onnx \
  --img-size 384
```

Infer with ONNXRuntime:

```bash
python -m gatenet.infer_onnx_test \
  --data data/splits/image1 \
  --onnx runs/gatenet_image1/gatenet_instance_y4.onnx \
  --out runs/gatenet_image1/test_infer_onnx \
  --providers auto \
  --threshold 0.5 --boundary-threshold 0.5 --min-area 20 \
  --save-overlay --save-gt --save-orig-size
```

The ONNX preprocessing is:

1. Open image as RGB.
2. Resize to `--img-size x --img-size`.
3. Convert to float32 `[0, 1]`.
4. Transpose from HWC to NCHW and add batch dimension.

The ONNX post-processing matches PyTorch inference and writes the same output folders,
including `instance_maps`, `instance_color`, overlays and `results.json`.

