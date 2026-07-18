# TRT8.5 YOLO26-seg TopK30 后处理说明

本文档用于指导 AI 或工程师实现 `best_trt85_topk30.onnx` / TensorRT engine 的后处理。

## 模型变体

当前有三个 TopK30 导出版本，后处理流程相同，但 mask 维度和 proto 分辨率不同：

```text
高分辨率轻量版:
runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_himask_seg_384/weights/best_trt85_topk30.onnx
detections: [1, 30, 30] = 4 box + 1 score + 1 class + 24 mask coeff
proto:      [1, 24, 192, 192]

轻量版:
runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_seg_384/weights/best_trt85_topk30.onnx
detections: [1, 30, 22] = 4 box + 1 score + 1 class + 16 mask coeff
proto:      [1, 16, 96, 96]

原 YOLO26n-seg 版:
runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best_trt85_topk30.onnx
detections: [1, 30, 38] = 4 box + 1 score + 1 class + 32 mask coeff
proto:      [1, 32, 96, 96]
```

写后处理时不要把 mask 维度写死为 16/24/32，也不要把 proto 分辨率写死为 96。
应该从 `proto.shape` 和 `detections.shape[-1] - 6` 推断。

## 可直接复制给 AI 的任务说明

```text
请为一个 YOLO26-seg TensorRT8.5 engine 写后处理。不要按 Ultralytics 原始 raw 输出 [1,21,3024] 写。

这个模型已经在 ONNX/engine 内完成了固定 TopK 预筛选，输出只有 30 个候选：
1. detections: shape [1, 30, 6 + mask_dim]
   每个候选格式是：
   [x1, y1, x2, y2, score, class_id, mask_coeff_0 ... mask_coeff_N-1]
   坐标是 384x384 网络输入图上的 xyxy 像素坐标。
   score 已经是 sigmoid 后的置信度。
   class_id 固定为 0，因为只有门这一类。

2. proto: shape [1, mask_dim, proto_h, proto_w]
   这是 N 个 mask prototype。标准版通常是 96x96，高分辨率轻量版是 192x192。
   高分辨率轻量版 N=24，轻量版 N=16，原 YOLO26n-seg 版 N=32。

后处理流程：
1. 对 30 个候选按 score 做 confidence threshold，建议 conf_thres=0.25。
2. 对剩余候选做 class-agnostic NMS，建议 iou_thres=0.45 或 0.50。
3. NMS 后再生成 mask，不要在 NMS 前对所有候选生成大图 mask。
4. 对每个保留实例：
   mask_logits_proto = sum_i(mask_coeff_i * proto_i)
   然后把检测框从 384x384 坐标缩放到 proto_w x proto_h 坐标，对 mask_logits 做框内裁剪。
   如果需要 384x384 mask，就把裁剪后的 proto logits 双线性上采样到 384x384。
   二值 mask 使用 mask_logits > 0，等价于 sigmoid(mask_logits) > 0.5。
5. 如果前处理是直接 resize 到 384x384，则把 box 和 mask 按 original_w/384、original_h/384 映射回原图。
   如果前处理是 letterbox，则必须先去 padding，再除以 resize ratio。

性能要求：
- NMS 只处理最多 30 个候选。
- mask 只对 NMS 保留下来的实例生成。
- 不要对 3024 anchors 做 CPU NMS。
- 不要对 30 个候选全部生成 384x384 float mask 后再 NMS。
```

## 模型文件

当前推荐部署文件：

```text
runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_himask_seg_384/weights/best_trt85_topk30.onnx
```

对应导出脚本：

```text
scripts/export_yolo26_seg_trt85_topk_onnx.py
```

导出命令：

```bash
python scripts/export_yolo26_seg_trt85_topk_onnx.py \
  --weights runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_himask_seg_384/weights/best.pt \
  --imgsz 384 \
  --topk 30 \
  --opset 13 \
  --device cpu \
  --branch one2many \
  --out runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_himask_seg_384/weights/best_trt85_topk30.onnx
```

TensorRT8.5 转 engine：

```bash
trtexec \
  --onnx=runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_himask_seg_384/weights/best_trt85_topk30.onnx \
  --saveEngine=runs/segment/runs/yolo26/image1_monorace_aug_yolo26_gate_lite_himask_seg_384/weights/best_trt85_topk30.engine \
  --fp16
```

## 输出定义

### detections

Shape:

```text
[1, 30, 6 + mask_dim]
```

最后一维含义：

```text
0: x1
1: y1
2: x2
3: y2
4: score
5: class_id
6-end: mask coefficients
```

注意：

- `x1,y1,x2,y2` 是网络输入图 `384x384` 上的坐标。
- `score` 已经 sigmoid，不要再次 sigmoid。
- `class_id` 固定是 `0`。
- 这 30 个候选已经按 score 降序排列，但还没有做 NMS。

### proto

Shape:

```text
[1, mask_dim, proto_h, proto_w]
```

含义：

- `mask_dim` 与 detections 的 mask coefficients 数量对齐。
- `proto_h x proto_w` 是 prototype mask 分辨率，必须从输出 shape 读取。
- `proto` 是 logits 特征，不要先二值化。

## 推荐后处理参数

```text
input_size = 384
mask_dim = proto.shape[1]
proto_h = proto.shape[2]
proto_w = proto.shape[3]
topk = 30
conf_thres = 0.25
iou_thres = 0.45 或 0.50
mask_bin_thres = 0.0
```

`mask_bin_thres=0.0` 是因为 mask 输出是 logits。若代码显式做 sigmoid，则阈值用 `0.5`。

## PT / ONNX 一致性验收

这个 TopK30 ONNX 是为 TensorRT8.5 改写过的部署图，不要直接拿它和 Ultralytics 默认
`model.predict()` 做裸对比。默认 `.pt` 推理可能使用 rectangular letterbox，而且 YOLO26
`end2end=True` 的 PT 后处理是“TopK 后按 conf 过滤”，不会再做一次 NMS。

严格验收导出是否正确时使用：

```bash
python scripts/check_yolo26_seg_onnx_consistency.py \
  --image data/ultralytics/image1_monorace_aug_yolo26/test/images/175.jpg \
  --pt runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best.pt \
  --onnx runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best_trt85_topk30.onnx \
  --imgsz 384 \
  --topk 30 \
  --branch one2one \
  --preprocess letterbox \
  --ort-provider cpu
```

可视化对比时使用固定 square-letterbox、关闭 ONNX 侧额外 NMS：

```bash
python scripts/diagnose_yolo_seg_mask_export.py \
  --image data/ultralytics/image1_monorace_aug_yolo26/test/images/175.jpg \
  --pt runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best.pt \
  --onnx runs/segment/runs/yolo26/image1_monorace_aug_yolo26n_seg_384/weights/best_trt85_topk30.onnx \
  --imgsz 384 \
  --conf 0.25 \
  --preprocess letterbox \
  --skip-onnx-nms \
  --ort-provider cpu
```

部署到 CUDA/TensorRT 时，极少数低置信候选会因为 FP32/FP16 或 CUDA kernel 数值差异在阈值边缘抖动。
如果某个候选分数贴着 0.25，例如 0.248 到 0.251，PT/ONNX/TRT 的数量可能差 1 个。实际部署建议把
`conf_thres` 提到 `0.30`，或在业务逻辑里接受这个边界差异。

## 后处理步骤

### 1. 置信度过滤

遍历 `detections[0]` 的 30 个候选：

```text
keep = score >= conf_thres
```

如果没有候选，直接返回空实例列表。

### 2. NMS

对过滤后的 boxes 做 class-agnostic NMS。

IoU 使用标准 xyxy box IoU：

```text
inter = max(0, min(x2a,x2b)-max(x1a,x1b)) * max(0, min(y2a,y2b)-max(y1a,y1b))
iou = inter / (area_a + area_b - inter + eps)
```

因为只有一类门，不需要 class offset。

### 3. mask 生成

NMS 之后，对保留下来的 `N` 个实例生成 mask：

```text
coeffs: [N, mask_dim]
proto:  [mask_dim, proto_h, proto_w]

mask_logits = coeffs @ proto.reshape(mask_dim, proto_h*proto_w)
mask_logits = mask_logits.reshape(N, proto_h, proto_w)
```

这个顺序很重要：先 NMS，再算 mask。

### 4. 按 box 裁剪 mask

box 在 384 坐标系，prototype 在 `proto_w x proto_h` 坐标系，所以：

```text
scale_x = proto_w / 384
scale_y = proto_h / 384

box_proto = [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]
```

把 `mask_logits` 中 box 外部区域置为 0 或一个很小的负值。与 Ultralytics 一致的逻辑是：

```text
mask *= (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
```

### 5. 上采样和二值化

如果下游需要 `384x384` mask：

```text
mask_384 = bilinear_resize(mask_logits_proto, 384, 384)
binary = mask_384 > 0
```

如果要回到原图：

- 直接 resize 前处理：把 `384x384` mask resize 到原图大小。
- letterbox 前处理：先裁掉 padding 区域，再 resize 到原图大小。

## Python 参考实现

```python
import numpy as np
import cv2


def nms_xyxy(boxes, scores, iou_thres=0.45):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = np.maximum(0.0, boxes[i, 2] - boxes[i, 0]) * np.maximum(0.0, boxes[i, 3] - boxes[i, 1])
        area_j = np.maximum(0.0, boxes[order[1:], 2] - boxes[order[1:], 0]) * np.maximum(
            0.0, boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        iou = inter / (area_i + area_j - inter + 1e-7)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=np.int64)


def crop_masks_proto(mask_logits, boxes_384, input_size=384):
    n, h, w = mask_logits.shape
    boxes = boxes_384.copy()
    boxes[:, [0, 2]] *= w / float(input_size)
    boxes[:, [1, 3]] *= h / float(input_size)

    xs = np.arange(w, dtype=np.float32)[None, None, :]
    ys = np.arange(h, dtype=np.float32)[None, :, None]
    x1 = boxes[:, 0][:, None, None]
    y1 = boxes[:, 1][:, None, None]
    x2 = boxes[:, 2][:, None, None]
    y2 = boxes[:, 3][:, None, None]
    inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    return mask_logits * inside


def postprocess_topk30(detections, proto, conf_thres=0.25, iou_thres=0.45):
    det = detections[0]          # [30, 6 + mask_dim]
    proto = proto[0].astype(np.float32)  # [mask_dim, proto_h, proto_w]

    scores = det[:, 4]
    valid = scores >= conf_thres
    det = det[valid]
    if det.shape[0] == 0:
        return []

    boxes = det[:, 0:4].astype(np.float32)
    scores = det[:, 4].astype(np.float32)
    coeffs = det[:, 6:].astype(np.float32)

    keep = nms_xyxy(boxes, scores, iou_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    coeffs = coeffs[keep]

    c, mh, mw = proto.shape
    mask_logits = coeffs @ proto.reshape(c, -1)
    mask_logits = mask_logits.reshape(-1, mh, mw)
    mask_logits = crop_masks_proto(mask_logits, boxes, input_size=384)

    results = []
    for box, score, mask_proto in zip(boxes, scores, mask_logits):
        mask384_logits = cv2.resize(mask_proto, (384, 384), interpolation=cv2.INTER_LINEAR)
        mask384 = (mask384_logits > 0).astype(np.uint8)
        results.append(
            {
                "box_384_xyxy": box,
                "score": float(score),
                "class_id": 0,
                "mask_384": mask384,
            }
        )
    return results
```

## 坐标映射回原图

如果前处理是直接把原图 resize 到 `384x384`：

```python
box[:, [0, 2]] *= original_w / 384.0
box[:, [1, 3]] *= original_h / 384.0
mask_original = cv2.resize(mask384, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
```

如果前处理是 letterbox：

```python
box[:, [0, 2]] -= pad_x
box[:, [1, 3]] -= pad_y
box /= ratio
box = clip_to_image(box, original_w, original_h)

mask_no_pad = mask384[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w]
mask_original = cv2.resize(mask_no_pad, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
```

前处理和后处理必须一致。不要训练/测试用 resize，部署却按 letterbox 反算坐标。

## 常见错误

1. 对 `score` 再 sigmoid 一次。
   这个模型输出的 score 已经 sigmoid。

2. 把 `detections` 当成 raw YOLO 输出。
   当前输出已经是 TopK 后的 `[1,30,6+mask_dim]`，不是 `[1,21,3024]`。

3. NMS 前生成所有大图 mask。
   正确做法是先 conf filter 和 NMS，再生成保留实例的 mask。

4. mask 阈值用错。
   不做 sigmoid 时用 `mask_logits > 0`。做 sigmoid 时用 `prob > 0.5`。

5. 忘记 crop mask。
   不 crop 会导致不同门实例之间的 mask 泄漏，实例边界会更乱。

6. letterbox 和 resize 混用。
   如果前处理用了 letterbox，回原图必须去 padding。

## 速度目标

当前 CUDA ONNX 测试结果：

```text
best_trt85_topk30.onnx: mean 2.11 ms
输出候选数: 30
```

后处理应该只和 `30` 个候选有关，不应该再和 `3024` 个 anchors 成正比。
