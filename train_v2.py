#!/usr/bin/env python3
"""
train_v2.py
-----------
Stage 5A Retraining Pipeline for Milk Bottle Detection v2.

Key Upgrades over v1:
1. Destination Run: Isolated in runs/detect/milk_bottle_v2 (preserves v1 completely).
2. Data Source: Uses configs/dataset_v2.yaml (includes 138 validated hard-negatives and group-level leakage-free splits).
3. Augmentation Tuning:
   - Mosaic reduced to 0.2 (avoids unnatural bottle chopping while maintaining scale diversity).
   - Close mosaic 10 (mosaic disabled in final 10 epochs for clean boundary refinement).
   - Scale reduced to 0.2 (preserves realistic conveyor perspective).
   - Color jitter retained for specular reflection resilience.
"""

import os
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
DATASET_YAML  = BASE_DIR / "configs" / "dataset_v2.yaml"
MODEL_NAME    = "yolov8n.pt"
PROJECT_DIR   = BASE_DIR / "runs" / "detect"
RUN_NAME      = "milk_bottle_v2"
EPOCHS        = 100
IMAGE_SIZE    = 640
BATCH_SIZE    = 16
PATIENCE      = 20


def get_best_device() -> str:
    """Select CUDA GPU, Apple MPS, or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def check_prerequisites():
    if not DATASET_YAML.exists():
        print(f"❌  Configuration not found: {DATASET_YAML}")
        print("   Run  python scripts/prepare_dataset_v2.py  first.")
        sys.exit(1)

    dataset_v2_dir = BASE_DIR / "dataset_v2"
    if not dataset_v2_dir.exists() or not (dataset_v2_dir / "images" / "train").exists():
        print(f"⚠️  dataset_v2 directory not found at: {dataset_v2_dir}")
        print("   To execute training, place your authorized Annotated/ batches and run:")
        print("   python scripts/prepare_dataset_v2.py")
        sys.exit(1)


def train():
    check_prerequisites()
    device = get_best_device()

    print("\n" + "=" * 65)
    print("  🚀 Starting YOLOv8n Retraining — Experiment v2")
    print("=" * 65)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Config      : {DATASET_YAML}")
    print(f"  Output Run  : {PROJECT_DIR / RUN_NAME}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {EPOCHS} (patience={PATIENCE})")
    print(f"  Mosaic      : 0.2 (reduced from 1.0 to prevent partial bottle chopping)")
    print(f"  Negatives   : Integrated (penalizing false positives on empty conveyors)")
    print("=" * 65 + "\n")

    model = YOLO(MODEL_NAME)

    start_time = time.time()
    results = model.train(
        data        = str(DATASET_YAML),
        epochs      = EPOCHS,
        imgsz       = IMAGE_SIZE,
        batch       = BATCH_SIZE,
        device      = device,
        patience    = PATIENCE,
        project     = str(PROJECT_DIR),
        name        = RUN_NAME,
        exist_ok    = True,
        plots       = True,
        save        = True,
        verbose     = True,
        # Hyperparameters isolating dataset quality & alignment
        mosaic      = 0.2,       # Reduced from 1.0 to prevent severe bottle truncation
        close_mosaic= 10,        # Clean final epochs
        scale       = 0.2,       # Controlled scaling
        degrees     = 5.0,       # Slight conveyor jitter
        translate   = 0.1,
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.4,
        hsv_v       = 0.3,
    )

    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    print(f"\n✅  Retraining v2 complete in {mins}m {secs}s")
    print(f"    Saved best weights → {best_weights}")
    return best_weights


if __name__ == "__main__":
    train()
