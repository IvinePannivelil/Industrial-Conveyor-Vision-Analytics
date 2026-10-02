# Evaluation Methodology — Industrial Conveyor Vision Analytics

## Overview

This document describes the evaluation approach for the YOLOv8n detector used in
this project and the key findings from model reliability analysis.

> **Note on dataset availability**: The proprietary industrial dataset (~10 GB) is
> withheld from public distribution. The metrics documented here were derived from
> training runs on the original data. Evaluation scripts will gracefully display
> baseline metrics from `results.csv` when the live dataset is unavailable.

---

## 1. Standard Detection Evaluation

Run evaluation against validation or test split:

```bash
python scripts/evaluate_model.py
python scripts/evaluate_model.py --split test
python scripts/evaluate_model.py --weights runs/detect/milk_bottle_calibrated/weights/best.pt
```

### Metrics Reported

| Metric | Formulation | Description |
| :--- | :--- | :--- |
| **Precision** | `TP / (TP + FP)` | Proportion of predicted positive detections that match a ground-truth object |
| **Recall** | `TP / (TP + FN)` | Proportion of all ground-truth object instances that were successfully detected |
| **mAP@0.50** | Area under PR at IoU ≥ 0.50 | Standard detection metric at standard IoU threshold |
| **mAP@0.50:0.95** | Mean over 10 IoU thresholds (0.50 to 0.95 step 0.05) | Strict localization metric requiring tight bounding box boundaries |
| **Inference latency** | Wall-clock per frame | Preprocessing, forward pass, and NMS postprocessing time |

---

## 2. Baseline v1 Metrics (from `runs/detect/milk_bottle/results.csv`)

These metrics were produced by a frame-level random-shuffle split.
Sequence-aware validation analysis identified that these metrics may reflect
temporal frame leakage rather than genuine generalisation to unseen viewpoints.

| Metric | Final Epoch | Peak (Epoch) |
| :--- | :---: | :---: |
| mAP@0.50 | 98.31% | 98.38% (ep. 94) |
| mAP@0.50:0.95 | 63.89% | 64.01% (ep. 94) |
| Precision | 92.07% | 94.81% (ep. 92) |
| Recall | 96.56% | 97.57% (ep. 88) |

> **Interpretation**: These figures should be treated as upper-bound indicators
> of in-distribution performance. Sequence-aware re-evaluation (Experiment v2)
> is the recommended approach for honest generalisation estimates.

---

## 3. Calibrated Model Metrics (Experiment v2-Patch)

After 15-epoch negative-calibration fine-tuning on 181 mixed positive/negative images:

| Metric | Calibrated Model | vs. Baseline v1 | Note |
| :--- | :---: | :---: | :--- |
| Precision | 94.4% | +2.33 percentage points | +2.5% relative change |
| Recall | 84.2% | −12.36 percentage points | −12.8% relative change (more conservative boundary) |
| mAP@0.50 | 95.2% | −3.11 percentage points | Evaluated with 76% empty-background frames |
| mAP@0.50:0.95 | 79.1% | **+15.24 percentage points** | **+23.85% relative improvement** in tight overlap |
| Whiteboard FP | 0.0% conf (0 detections) | Resolved on test image | Peak was 74.5% in baseline |

> **Context**: The recall drop is expected — the calibrated model is more
> conservative on uncertain regions near the decision boundary. The +15.24
> percentage point gain in mAP@0.50:0.95 reflects substantially tighter bounding box
> localisation, not new class knowledge.

---

## 4. Comparative Evaluation

To reproduce the v1 vs. calibrated comparison:

```bash
python scripts/evaluate_calibration.py
```

Tests performed:
1. **False-positive suppression**: Empty whiteboard / white surface image
2. **Bottle recall retention**: Real conveyor sample images
3. **Video stream detection consistency**: 25-frame synthetic stream test

---

## 5. Deployment / Runtime Evaluation

For assessing runtime performance rather than academic detection metrics:

```bash
# Run on video file, save telemetry
python detect_live.py \
  --source <video.mp4> \
  --track --count-line \
  --csv-log results/session.csv
```

The resulting `results/session.csv` contains per-frame:
- Detection count
- Cumulative object count
- Throughput (objects/min)
- FPS and inference confidence

---

## 6. ONNX Runtime Benchmarks

ONNX export enables cross-platform deployment without PyTorch:

```bash
python export_model.py --verify
```

| Runtime | Latency | Notes |
| :--- | :--- | :--- |
| ONNX CPU (Intel x86) | ~17.8 ms/frame | Measured locally |
| PyTorch MPS (Apple Silicon) | ~35.9 ms/frame | During YOLO val runs |

> These are independent measurements on different hardware and are not directly comparable.
