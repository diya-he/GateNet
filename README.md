# GateNet segmentation (MonoRace)

This repo contains a lightweight gate-segmentation training pipeline based on the **GateNet**
architecture described in `paper/monorace.pdf` (U-Net style, 5 multi-scale outputs, Dice+BCE).

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
  --save-overlay --save-gt
```

Outputs:
- `.../pred_masks/*.png`: predicted binary masks
- `.../gt_masks/*.png`: GT masks (optional)
- `.../overlay/*.png`: overlay preview (optional)
- `.../results.json`: IoU + latency + FPS

## Export ONNX + ONNXRuntime inference

Export (y4 output only):

```bash
python -m gatenet.export_onnx --ckpt runs/gatenet_image1/best_pruned.pt --out runs/gatenet_image1/gatenet_y4.onnx --img-size 384
```

Infer with ONNXRuntime:

```bash
python -m gatenet.infer_onnx_test \
  --data data/splits/image1 \
  --onnx runs/gatenet_image1/gatenet_y4.onnx \
  --out runs/gatenet_image1/test_infer_onnx \
  --providers auto \
  --save-overlay --save-gt --save-orig-size
```

