#!/usr/bin/env python3
"""
export_model.py
---------------
Exports the trained milk bottle YOLOv8 model to ONNX format and verifies it.

ONNX is an open, cross-platform format compatible with:
  • Python (onnxruntime)
  • C++ production applications
  • Web deployment (onnxruntime-web)
  • Edge devices (NVIDIA Jetson, Raspberry Pi)
  • OpenCV DNN module

Usage:
  python export_model.py
  python export_model.py --verify
  python export_model.py --verify-only exported_model/milk_bottle_detector.onnx
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR     = Path(__file__).parent.resolve()
WEIGHTS_PATH = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
EXPORT_DIR   = BASE_DIR / "exported_model"
IMAGE_SIZE   = 640


def check_weights(weights_path: Path):
    if not weights_path.exists():
        print(f"❌  Weights not found at: {weights_path}")
        print("   Run  python train.py  first.")
        sys.exit(1)
    print(f"✅  Weights found: {weights_path}")


def export_onnx(weights_path: Path, imgsz: int = IMAGE_SIZE):
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  Exporting YOLOv8 → ONNX")
    print("=" * 60)
    print(f"  Source  : {weights_path}")
    print(f"  Format  : ONNX (opset 11)")
    print(f"  Img size: {imgsz}x{imgsz}")
    print(f"  Dynamic : True  (accepts variable batch size at runtime)")
    print(f"  Simplify: True  (optimises graph with onnxslim)")
    print("=" * 60)

    model = YOLO(str(weights_path))

    start = time.time()
    exported_path = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        dynamic  = True,          # variable batch size
        simplify = True,          # clean up redundant ops with onnxslim
        opset    = 11,            # broadest runtime compatibility
        half     = False,         # FP32 for universal compatibility
    )
    elapsed = time.time() - start

    return Path(exported_path), elapsed


def copy_to_export_dir(onnx_path: Path, export_dir: Path, imgsz: int = IMAGE_SIZE):
    """Copy the .onnx file and metadata into the exported_model/ folder."""
    import shutil

    export_dir.mkdir(parents=True, exist_ok=True)
    dest = export_dir / "milk_bottle_detector.onnx"
    shutil.copy2(onnx_path, dest)

    readme = export_dir / "README.md"
    readme.write_text(f"""# Industrial Conveyor Vision Analytics — ONNX Model

## Model Specifications
- **Architecture**: YOLOv8n (nano object detection)
- **Trained on**: Muralya Dairy conveyor imagery (authorized)
- **Classes**: `milk_bottle` (1 class — current industrial use case)
- **Input Resolution**: {imgsz}x{imgsz} (dynamic batch dimension)
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
img = cv2.resize(img, ({imgsz}, {imgsz}))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[np.newaxis]   # Shape: (1, 3, {imgsz}, {imgsz})

# Run inference
outputs = session.run(None, {{input_name: img}})
predictions = outputs[0]  # Shape: (1, 5, 8400) -> [cx, cy, w, h, conf]
```

## Install Runtime

```bash
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # CUDA GPU
```
""", encoding="utf-8")
    return dest


def verify_onnx_model(onnx_path: Path, imgsz: int = IMAGE_SIZE) -> bool:
    """Verify ONNX model loading, input/output tensors, and forward pass latency."""
    print("\n" + "=" * 60)
    print("  🔍  ONNX Model Verification")
    print("=" * 60)
    if not onnx_path.exists():
        print(f"❌  ONNX model not found at: {onnx_path}")
        return False

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  • Model File  : {onnx_path.resolve()}")
    print(f"  • Model Size  : {size_mb:.2f} MB")

    try:
        import onnxruntime as rt
    except ImportError:
        print("⚠️  onnxruntime is not installed. Run: pip install onnxruntime")
        return False

    try:
        session = rt.InferenceSession(str(onnx_path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print("\n  Inputs:")
        for idx, inp in enumerate(inputs):
            print(f"    [{idx}] Name: '{inp.name}' | Type: {inp.type} | Shape: {inp.shape}")

        print("\n  Outputs:")
        for idx, out in enumerate(outputs):
            print(f"    [{idx}] Name: '{out.name}' | Type: {out.type} | Shape: {out.shape}")

        # Dummy forward pass benchmark
        input_name = inputs[0].name
        dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

        # Warmup
        _ = session.run(None, {input_name: dummy_input})

        # Measure 10 runs
        times = []
        for _ in range(10):
            t0 = time.time()
            out = session.run(None, {input_name: dummy_input})
            times.append((time.time() - t0) * 1000.0)

        avg_lat = sum(times) / len(times)
        out_shape = out[0].shape
        print(f"\n  • Test Forward Pass : Successful")
        print(f"  • Output Shape      : {out_shape}")
        print(f"  • Average Latency   : {avg_lat:.2f} ms per frame (CPU benchmark)")
        print("=" * 60)
        print("✅  ONNX Model Verification: PASSED")
        return True
    except Exception as e:
        print(f"❌  ONNX Verification Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export trained YOLOv8 model to ONNX format")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH), help="Path to best.pt weights")
    parser.add_argument("--output-dir", type=str, default=str(EXPORT_DIR), help="Output directory for exported model")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Inference resolution")
    parser.add_argument("--verify", action="store_true", help="Run ONNX model verification test after export")
    parser.add_argument("--verify-only", type=str, default=None, help="Verify an existing ONNX file without re-exporting")
    args = parser.parse_args()

    if args.verify_only:
        verify_onnx_model(Path(args.verify_only), imgsz=args.imgsz)
        return

    weights_p = Path(args.weights)
    export_d = Path(args.output_dir)

    print("=" * 60)
    print("  Industrial Conveyor Vision Analytics — Model Export")
    print("=" * 60)

    check_weights(weights_p)
    onnx_path, elapsed = export_onnx(weights_p, imgsz=args.imgsz)

    dest = copy_to_export_dir(onnx_path, export_d, imgsz=args.imgsz)

    print("\n" + "=" * 60)
    print("  📦  Exported Model Package")
    print("=" * 60)
    print(f"  Folder  : {export_d}/")
    print(f"  Model   : {dest.name}")
    print(f"  README  : README.md")

    if args.verify:
        verify_onnx_model(dest, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
