# System Architecture — Industrial Conveyor Vision Analytics

## Overview

Industrial Conveyor Vision Analytics implements two conceptually distinct workflows:

1. **Runtime Industrial Vision System** — real-time detection, tracking, counting, and throughput analytics on a live camera or video feed.
2. **Model Reliability Workflow** — offline dataset quality analysis, sequence-aware validation, hard-negative experiments, and calibration fine-tuning.

---

## 1. Runtime Pipeline

```
Camera / Video Input (webcam, file, RTSP)
          │
          ▼
  YOLOv8n Detector
  ┌─────────────────────────────────┐
  │  model.track() with ByteTrack   │
  │  confidence threshold: 0.40     │
  │  IoU threshold: 0.45            │
  │  image size: 640×640            │
  └─────────────────────────────────┘
          │
          ▼
  Persistent Track ID Assignment
  ┌─────────────────────────────────┐
  │  ByteTrack two-stage matching   │
  │  Kalman filter motion priors    │
  │  track_buffer: 30 frames        │
  │  high/low confidence thresholds │
  └─────────────────────────────────┘
          │
          ▼
  Virtual Line Crossing Detection
  ┌─────────────────────────────────┐
  │  Horizontal counting line at    │
  │  configurable Y position        │
  │  Direction: up / down / both    │
  │  Trajectory history per track   │
  │  Duplicate prevention via set   │
  └─────────────────────────────────┘
          │
          ▼
  Unique Object Counting
  ┌─────────────────────────────────┐
  │  counted_track_ids: set()       │
  │  Each track ID counted once     │
  │  Jitter-resistant (ID-locked)   │
  └─────────────────────────────────┘
          │
          ▼
  Throughput Analytics
  ┌─────────────────────────────────┐
  │  Real wall-clock timestamps     │
  │  bottles_counted / elapsed_min  │
  │  Guarded: < 1s → 0.0 bpm        │
  └─────────────────────────────────┘
          │
          ▼
  Telemetry & Export
  ┌─────────────────────────────────┐
  │  Per-frame CSV logging          │
  │  Fields: timestamp, frame,      │
  │    detections, count,           │
  │    throughput, fps, conf        │
  └─────────────────────────────────┘
          │
          ▼
  OpenCV Display / Headless Output
  ┌─────────────────────────────────┐
  │  Bounding boxes + track IDs     │
  │  Counting line overlay          │
  │  HUD: count, throughput, FPS    │
  │  Detection count graph          │
  └─────────────────────────────────┘
```

### Entry Point

```bash
python detect_live.py --help
```

---

## 2. Model Reliability Workflow

```
Raw Dataset (proprietary, withheld)
          │
          ▼
  Dataset Quality Audit
  ┌─────────────────────────────────┐
  │  Bounding box distribution      │
  │  Annotation consistency check   │
  │  Cap-only vs. full-body policy  │
  └─────────────────────────────────┘
          │
          ▼
  Leakage Analysis
  ┌─────────────────────────────────┐
  │  Temporal frame adjacency       │
  │  Random split → frame leakage   │
  │  Sequence-aware split design    │
  └─────────────────────────────────┘
          │
          ▼
  Hard-Negative Collection
  ┌─────────────────────────────────┐
  │  Factory floor, rails, walls    │
  │  Conveyor gaps, machinery       │
  │  Augmented variations           │
  │  138 images with 0-byte labels  │
  └─────────────────────────────────┘
          │
          ▼
  Calibration Fine-Tuning (COMPLETED)
  ┌─────────────────────────────────┐
  │  15-epoch transfer fine-tune    │
  │  lr0=0.001 (gentle)             │
  │  43 positives + 138 negatives   │
  │  False positive: 74.5% → 0.0%  │
  └─────────────────────────────────┘
          │
          ▼
  Full Retraining Experiment (PLANNED)
  ┌─────────────────────────────────┐
  │  Requires raw Annotated/ data   │
  │  Sequence-level split           │
  │  100 epochs from yolov8n.pt     │
  │  scripts/prepare_dataset_v2.py  │
  └─────────────────────────────────┘
          │
          ▼
  Evaluation
  ┌─────────────────────────────────┐
  │  mAP@50, mAP@50-95              │
  │  Precision, Recall              │
  │  Inference latency              │
  │  Comparative v1 vs. calibrated  │
  └─────────────────────────────────┘
```

### Entry Points

```bash
# Standard evaluation
python scripts/evaluate_model.py --help

# Comparative calibration evaluation
python scripts/evaluate_calibration.py

# Pre-flight check for v2 retraining
python scripts/verify_v2_ready.py
```

---

## 3. Model Architecture

| Property | Value |
| :--- | :--- |
| Architecture | YOLOv8n (nano) |
| Parameters | ~3.0M |
| GFLOPs | 8.1 @ 640×640 |
| Backbone | Layers 0–9: Conv + C2f + SPPF |
| Neck | Layers 10–21: PAN-FPN |
| Head | Layer 22: Decoupled classification + regression |
| Loss | DFL + CIoU + BCE |
| Tracker | ByteTrack (Ultralytics built-in) |

---

## 4. File Organisation

```
detect_live.py           ← Runtime pipeline entry point
export_model.py          ← ONNX export utility
train.py                 ← Baseline training script (reference)
train_calibrated.py      ← Calibration fine-tuning script
train_v2.py              ← Full retraining script (v2, requires raw data)

scripts/
  evaluate_model.py      ← Standard model evaluation
  evaluate_calibration.py ← Comparative baseline vs. calibrated
  build_hard_negatives.py ← Hard-negative image extraction
  prepare_dataset.py     ← Baseline dataset preparation
  prepare_dataset_v2.py  ← Leakage-free v2 dataset preparation
  prepare_calibration_dataset.py ← Calibration dataset builder
  verify_v2_ready.py     ← Pre-flight retraining readiness check

configs/
  dataset.yaml           ← v1 dataset config
  dataset_calibrated.yaml ← Calibration dataset config
  dataset_v2.yaml        ← v2 leakage-free config

tests/
  test_counter.py        ← Line-crossing & counting unit tests
  test_tracker.py        ← ByteTrack tracking unit tests

docs/
  architecture.md        ← This file
  evaluation.md          ← Evaluation methodology & findings
  experiments.md         ← Experiment log (see also EXPERIMENTS.md)
```
