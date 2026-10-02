#!/usr/bin/env python3
"""
evaluate_model.py
-----------------
Reproducible model evaluation script for the Industrial Conveyor Vision Analytics YOLOv8n detector.
Computes and reports:
  • Mean Precision (mp)
  • Mean Recall (mr)
  • mAP @ 0.50
  • mAP @ 0.50:0.95
  • Per-stage inference latency (preprocess, inference, postprocess)

Usage:
  python scripts/evaluate_model.py
  python scripts/evaluate_model.py --weights runs/detect/milk_bottle/weights/best.pt --split test
"""

import sys
import os
from pathlib import Path

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
DEFAULT_DATA    = BASE_DIR / "configs" / "dataset.yaml"
RESULTS_CSV     = BASE_DIR / "runs" / "detect" / "milk_bottle" / "results.csv"


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


def display_baseline_metrics():
    """Display verified historical metrics from results.csv when dataset is offline."""
    print("\n" + "=" * 65)
    print("  📋  Verified Baseline Training Metrics (from results.csv)")
    print("=" * 65)
    if not RESULTS_CSV.exists():
        print(f"⚠️  Baseline results file not found at: {RESULTS_CSV}")
        return

    try:
        import pandas as pd
        df = pd.read_csv(RESULTS_CSV)
        df.columns = [c.strip() for c in df.columns]

        final_row = df.iloc[-1]
        peak_map50 = df["metrics/mAP50(B)"].max()
        peak_map = df["metrics/mAP50-95(B)"].max()
        peak_p = df["metrics/precision(B)"].max()
        peak_r = df["metrics/recall(B)"].max()

        print(f"  • Epochs Trained        : {int(final_row.get('epoch', len(df)))}")
        print("  ─────────────────────────────────────────────────────────────")
        print(f"  • Final Epoch (100) mAP@50    : {final_row.get('metrics/mAP50(B)', 0.0):.4f} ({final_row.get('metrics/mAP50(B)', 0.0):.1%})")
        print(f"  • Final Epoch (100) mAP@50-95 : {final_row.get('metrics/mAP50-95(B)', 0.0):.4f} ({final_row.get('metrics/mAP50-95(B)', 0.0):.1%})")
        print(f"  • Final Epoch (100) Precision : {final_row.get('metrics/precision(B)', 0.0):.4f} ({final_row.get('metrics/precision(B)', 0.0):.1%})")
        print(f"  • Final Epoch (100) Recall    : {final_row.get('metrics/recall(B)', 0.0):.4f} ({final_row.get('metrics/recall(B)', 0.0):.1%})")
        print("  ─────────────────────────────────────────────────────────────")
        print(f"  • Peak Validation mAP@50      : {peak_map50:.4f} ({peak_map50:.1%})")
        print(f"  • Peak Validation mAP@50-95   : {peak_map:.4f} ({peak_map:.1%})")
        print(f"  • Peak Validation Precision   : {peak_p:.4f} ({peak_p:.1%})")
        print(f"  • Peak Validation Recall      : {peak_r:.4f} ({peak_r:.1%})")
        print("=" * 65)
    except Exception as e:
        print(f"⚠️  Could not parse baseline metrics: {e}")


def evaluate(
    weights_path: str = str(DEFAULT_WEIGHTS),
    data_yaml: str = str(DEFAULT_DATA),
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.60,
    device: str = None,
    batch: int = 16,
):
    weights_p = Path(weights_path)
    if not weights_p.exists():
        print(f"❌  Model weights not found at: {weights_p}")
        print("   Please check the path or run python train.py first.")
        sys.exit(1)

    data_p = Path(data_yaml)
    if not data_p.exists():
        print(f"❌  Dataset configuration not found at: {data_p}")
        sys.exit(1)

    sel_device = device or get_best_device()

    print("=" * 65)
    print("  Industrial Conveyor Vision Analytics — Model Evaluation")
    print("=" * 65)
    print(f"  • Model Weights : {weights_p}")
    print(f"  • Dataset Config: {data_p}")
    print(f"  • Evaluation Set: {split}")
    print(f"  • Image Size    : {imgsz}x{imgsz}")
    print(f"  • Compute Device: {sel_device}")
    print("=" * 65)

    from ultralytics import YOLO
    model = YOLO(str(weights_p))

    # Check if dataset path specified in YAML exists locally
    import yaml
    with open(data_p, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    dataset_root = Path(data_cfg.get("path", "dataset"))
    if not dataset_root.is_absolute():
        dataset_root = BASE_DIR / dataset_root

    split_folder = dataset_root / data_cfg.get(split, f"images/{split}")
    if not split_folder.exists():
        print(f"\n⚠️  Evaluation dataset split not found at:\n    {split_folder}")
        print("ℹ️  Note: The proprietary industrial dataset (~10GB) is withheld from")
        print("   public distribution. See data/README.md for dataset structure.")
        display_baseline_metrics()
        return None

    print("\nRunning Ultralytics validation engine...")
    metrics = model.val(
        data=str(data_p),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        batch=batch,
        device=sel_device,
        verbose=True,
    )

    map50 = float(metrics.box.map50)
    map_all = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    print("\n" + "=" * 65)
    print("  📊  Evaluation Results")
    print("=" * 65)
    print(f"  • Mean Precision (mp)    : {mp:.4f} ({mp:.1%})")
    print(f"  • Mean Recall (mr)       : {mr:.4f} ({mr:.1%})")
    print(f"  • mAP @ 0.50             : {map50:.4f} ({map50:.1%})")
    print(f"  • mAP @ 0.50:0.95        : {map_all:.4f} ({map_all:.1%})")
    if hasattr(metrics, "speed") and metrics.speed:
        print("  ─────────────────────────────────────────────────────────────")
        print(f"  • Inference Latency      : {metrics.speed.get('inference', 0.0):.1f} ms/image")
        print(f"  • Preprocess Latency     : {metrics.speed.get('preprocess', 0.0):.1f} ms/image")
        print(f"  • Postprocess Latency    : {metrics.speed.get('postprocess', 0.0):.1f} ms/image")
    print("=" * 65)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8n milk bottle model")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to weights file (.pt)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="Path to dataset.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image resolution (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for mAP evaluation (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.60, help="IoU threshold for NMS (default: 0.60)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cpu, cuda, mps (default: auto)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    evaluate(
        weights_path=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
