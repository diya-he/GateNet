# GateNet instance segmentation

This repo contains a lightweight gate instance-segmentation training pipeline based on the
**GateNet** architecture described in `paper/monorace.pdf` (U-Net style, 5 multi-scale
outputs, Dice+BCE).

The backbone stays the same as MonoRace GateNet: `384x384`, `f=4`, five multi-scale
outputs, Xavier init, AdamW, and Dice+BCE supervision. The instance head is widened
slightly so the model predicts:

- `class_<id>`: one foreground probability map per original YOLO class id
- `boundary`: instance boundary probability
- `center`: compact per-instance seed probability

Post-processing converts these maps into instance IDs by growing predicted `center`
seeds through non-boundary foreground, then assigning each instance a `class_id` from
the average class probabilities. Legacy two-channel checkpoints still work.

## YOLO version

This repo includes a local Ultralytics checkout at `ultralytics/` with
`ultralytics.__version__ == 8.4.90`. YOLO comparison/training/export scripts use this
local package by default via `--ultralytics-root ultralytics`, and the included YOLO
baseline is YOLO26 segmentation (`yolo26n-seg.pt` and the custom YOLO26 gate-lite seg
configs under `ultralytics/ultralytics/cfg/models/26/`).

## YOLO26 gate-lite segmentation changes

The local Ultralytics checkout contains a customized YOLO26 segmentation baseline for
gate masks. The goal is to spend more resolution on the mask prototype branch while
making the rest of the network lighter:

- `yolo26-gate-lite-seg.yaml` keeps the YOLO26 `Segment26` head but uses a smaller
  scale (`depth=0.50`, `width=0.125`, `max_channels=512`) and fewer mask channels
  (`nm=16`, `npr=128`) than the stock YOLO26n-seg (`width=0.25`, `max_channels=1024`,
  `nm=32`, `npr=256`).
- `yolo26-gate-lite-himask-seg.yaml` uses the same light backbone/head width, but swaps
  the head to `Segment26HighRes` with `proto_scale=2`. For `384x384` input this raises
  the mask prototype map from the normal stride-4 size (`96x96`) to `192x192`.
- `Proto26HighRes` adds only a lightweight interpolation/refinement stage after the
  standard prototype branch, so the masks keep finer boundaries without paying for a
  full-size heavy YOLO26 segmentation network.
- Training uses `mask_ratio=1` in `scripts/train_yolo26_gate_lite_himask_seg.py`, so
  supervision keeps full mask detail for thin gate/ring structures.

In short: the detector/feature extractor is made narrower for speed, while the mask
prototype branch is given higher spatial resolution where segmentation quality is most
sensitive. This makes the YOLO baseline better suited for efficient gate instance
segmentation and TensorRT-oriented deployment.

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

Each line is treated as one instance. The trainer turns the polygons into per-class
foreground masks plus boundary and center-seed masks, so you can train instance
separation from standard YOLO-seg labels without changing the dataset layout.

Class ids are kept as their original YOLO ids. With `--class-ids auto`, the trainer scans
the labels and writes the class id list into the checkpoint metadata.

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
  --boundary-width 3 --center-radius 0.12 --class-ids auto \
  --lr 1e-3 --device auto \
  --out runs/gatenet_image1 \
  --save-prune-amount 0.3
```

Training-time augmentation follows the MonoRace paper's dataloader strategy: random
affine/perspective geometry, HSV perturbation, directional brightness gradients, motion
blur with 5-15 px kernels, Gaussian blur, additive Gaussian noise, lens distortion, and
rolling-shutter-like row shifts.

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
  --threshold 0.5 --boundary-threshold 0.5 --center-threshold 0.45 \
  --min-area 20 --seed-min-area 3 \
  --save-overlay --save-gt
```

Outputs:
- `.../pred_foreground/*.png`: predicted foreground binary masks
- `.../pred_boundary/*.png`: predicted boundary binary masks
- `.../pred_center/*.png`: predicted center-seed binary masks
- `.../instance_maps/*.png`: 16-bit PNG where pixel value is the instance id (`0` = background)
- `.../instance_color/*.png`: colorized instance preview
- `.../gt_masks/*.png`: GT masks (optional)
- `.../overlay/*.png`: overlay preview (optional)
- `.../results.json`: foreground IoU + latency + FPS + per-image instance counts

In `results.json`, each `per_image` item includes:

- `pred_instances`: number of predicted instances
- `gt_instances`: number of YOLO label lines
- `class_counts`: predicted instance count per original class id
- `instances`: each predicted instance's id, class id, class score, area and bounding box

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
  --threshold 0.5 --boundary-threshold 0.5 --center-threshold 0.45 \
  --min-area 20 --seed-min-area 3 --class-ids auto \
  --save-overlay --save-gt --save-orig-size
```

The ONNX preprocessing is:

1. Open image as RGB.
2. Resize to `--img-size x --img-size`.
3. Convert to float32 `[0, 1]`.
4. Transpose from HWC to NCHW and add batch dimension.

The ONNX post-processing matches PyTorch inference and writes the same output folders,
including `instance_maps`, `instance_color`, overlays and `results.json`.
