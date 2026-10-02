# Milk Bottle Detector — ONNX Model

## Model Specifications
- **Architecture**: YOLOv8n (nano object detection)
- **Trained on**: 858 images (1,073 total, 80/10/10 split)
- **Classes**: `milk_bottle` (1 class)
- **Input Resolution**: 640x640 (dynamic batch dimension)
- **Opset**: 11
- **Optimization**: Graph pruned and simplified via onnxslim

## Quick Start — Python (onnxruntime)

```python
import onnxruntime as rt
import numpy as np
import cv2

# Initialize session
session = rt.InferenceSession("milk_bottle_detector.onnx")
input_name = session.get_inputs()[0].name

# Load & preprocess
img = cv2.imread("sample.jpg")
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[np.newaxis]   # Shape: (1, 3, 640, 640)

# Run inference
outputs = session.run(None, {input_name: img})
predictions = outputs[0]  # Shape: (1, 5, 8400) -> [cx, cy, w, h, conf]
```

## Install Runtime

```bash
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # CUDA GPU
```

## Benchmark Latency
- **ONNX Runtime (Intel CPU)**: **17.79 ms/frame** (measured via `python export_model.py --verify`, batch 1, 640 × 640).
*(Note: Distinct from PyTorch Apple MPS training validation latency of ~35.9 ms/frame; these are independent runtime measurements and not a direct hardware comparison).*
