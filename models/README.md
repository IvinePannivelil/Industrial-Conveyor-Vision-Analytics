# Model Artifacts & Specifications

This directory documents the model architectures, checkpoints, and export formats available in this repository.

## Model Summary

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Architecture** | YOLOv8n (nano) | Decoupled anchor-free detection head |
| **Parameters** | 3,005,843 (~3.0M) | Compact footprint suitable for edge inference |
| **GFLOPs** | 8.1 GFLOPs | Computed at 640 × 640 resolution |
| **Base Resolution** | 640 × 640 | RGB 3-channel input |
| **Output Tensor** | `(1, 5, 8400)` | 8,400 anchors with `[cx, cy, w, h, conf_milk_bottle]` |
| **Backbone** | Layers 0–9 | Conv + C2f + SPPF multi-scale feature extractor |
| **Neck** | Layers 10–21 | PAN-FPN cross-scale feature aggregation network |
| **Detection Head** | Layer 22 | Decoupled classification and regression branches |
| **Loss Formulation** | DFL + CIoU + BCE | Distribution Focal Loss, Complete IoU, Binary Cross-Entropy |

## Model Checkpoints

### 1. PyTorch Checkpoint (`runs/detect/milk_bottle/weights/best.pt`)
- **Format**: PyTorch checkpoint (`.pt`)
- **Size**: ~6.2 MB
- **Use Case**: Primary weight file used for live webcam tracking (`detect_live.py`) and evaluation (`scripts/evaluate_model.py`).

### 2. Exported ONNX Model (`exported_model/milk_bottle_detector.onnx`)
- **Format**: Open Neural Network Exchange (ONNX)
- **Opset**: 11 (universal compatibility across edge runtimes)
- **Size**: ~11.8 MB
- **Optimization**: Cleaned and graph-pruned via `onnxslim`
- **Dynamic Dimensions**: Accepts variable batch sizes at runtime `[batch, 3, 640, 640]`

## Exporting & Verifying the ONNX Model

To export the PyTorch model to ONNX format:
```bash
python export_model.py
```

To run a verification check and forward pass latency benchmark:
```bash
python export_model.py --verify
```

## Benchmark & Latency Measurements

The following latency figures represent independent test runs on different environments and execution runtimes (they are not a direct hardware or backend comparison):

- **ONNX Runtime on Intel CPU**: **17.79 ms/frame** (measured with `onnxruntime` on local Intel x86_64 CPU, input resolution 640 × 640, batch size 1, graph simplified via `onnxslim`).
- **PyTorch on Apple Silicon MPS**: **~35.9 ms/frame** (measured during YOLO validation runs on Apple Silicon MPS device, input resolution 640 × 640, batch size 16 on validation split).
