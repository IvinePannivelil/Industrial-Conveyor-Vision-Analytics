#!/usr/bin/env python3
"""
train_calibrated.py
-------------------
Fine-tunes the baseline milk bottle detector weights with negative calibration
to suppress high-confidence false positives on empty surfaces.

Key Settings:
  - Base Weights: runs/detect/milk_bottle/weights/best.pt
  - Dataset: configs/dataset_calibrated.yaml (181 images, 76.2% negative coverage)
  - Epochs: 15
  - Destination: runs/detect/milk_bottle_calibrated/
  - Gentle Learning Rate: lr0=0.001 (prevents catastrophic forgetting)
"""

import sys
import time
from pathlib import Path

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ultralytics import YOLO

BASE_DIR      = Path(__file__).parent.resolve()
BASE_WEIGHTS  = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
DATASET_YAML  = BASE_DIR / "configs" / "dataset_calibrated.yaml"
PROJECT_DIR   = BASE_DIR / "runs" / "detect"
RUN_NAME      = "milk_bottle_calibrated"
EPOCHS        = 15
IMAGE_SIZE    = 640
BATCH_SIZE    = 16


def get_best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main():
    if not BASE_WEIGHTS.exists():
        print(f"❌ Base weights not found: {BASE_WEIGHTS}")
        sys.exit(1)
    if not DATASET_YAML.exists():
        print(f"❌ Config not found: {DATASET_YAML}")
        sys.exit(1)

    device = get_best_device()

    print("\n" + "=" * 65)
    print("  🚀 Starting Negative-Only Calibration Fine-Tuning")
    print("=" * 65)
    print(f"  Base Model  : {BASE_WEIGHTS.name} ({BASE_WEIGHTS.parent})")
    print(f"  Config      : {DATASET_YAML}")
    print(f"  Output Run  : {PROJECT_DIR / RUN_NAME}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Learning Rate: lr0=0.001 (fine-tuning calibration)")
    print("=" * 65 + "\n")

    model = YOLO(str(BASE_WEIGHTS))

    start_time = time.time()
    results = model.train(
        data        = str(DATASET_YAML),
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        device      = device,
        project     = str(PROJECT_DIR),
        name        = RUN_NAME,
        exist_ok    = True,
        plots       = True,
        save        = True,
        verbose     = True,
        # Fine-tuning hyperparameters
        lr0         = 0.001,     # Lower learning rate for fine-tuning
        lrf         = 0.01,
        mosaic      = 0.1,       # Low mosaic to preserve clean boundaries
        close_mosaic= 5,
        scale       = 0.1,
    )

    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\n✅  Calibration fine-tuning complete in {mins}m {secs}s")
    print(f"    Calibrated weights saved to -> {best_weights}")
    return best_weights


if __name__ == "__main__":
    main()
