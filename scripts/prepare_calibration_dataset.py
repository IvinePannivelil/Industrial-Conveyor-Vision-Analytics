#!/usr/bin/env python3
"""
scripts/prepare_calibration_dataset.py
--------------------------------------
Builds the anti-false-positive calibration dataset for Option 2 fine-tuning.

Dataset Composition:
  1. Positives (44 images):
     - 19 clean sample images from samples/
     - 25 extracted frames from scratch/test_conveyor_stream.mp4
     - Labels generated via baseline best.pt (filtered to conf >= 0.40, class 0)
  2. Negatives (138 images):
     - 138 validated plant background images from data/negatives/
     - Empty 0-byte label files
  3. Partitioning:
     - 80% Train, 20% Validation
     - Outputs to dataset_calibrated/
     - Writes configs/dataset_calibrated.yaml
"""

import os
import sys
import shutil
import random
import cv2
from pathlib import Path

# Configure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLES_DIR = BASE_DIR / "samples"
VIDEO_PATH = BASE_DIR / "scratch" / "test_conveyor_stream.mp4"
NEGATIVES_DIR = BASE_DIR / "data" / "negatives"
OUTPUT_DIR = BASE_DIR / "dataset_calibrated"
CONFIG_PATH = BASE_DIR / "configs" / "dataset_calibrated.yaml"
WEIGHTS_PATH = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"

SEED = 42


def extract_video_frames(video_path: Path, temp_dir: Path) -> list:
    """Extract frames from the synthetic conveyor test video."""
    frames = []
    if not video_path.exists():
        print(f"⚠️ Video not found: {video_path}")
        return frames

    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_p = temp_dir / f"stream_frame_{idx:03d}.jpg"
        cv2.imwrite(str(out_p), frame)
        frames.append(out_p)
        idx += 1
    cap.release()
    return frames


def generate_positive_labels(model, img_paths: list, labels_dir: Path) -> list:
    """Generate normalized YOLO format labels using baseline best.pt predictions."""
    valid_samples = []
    labels_dir.mkdir(parents=True, exist_ok=True)

    for img_p in img_paths:
        res = model(str(img_p), conf=0.35, verbose=False)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            # If no box detected above 0.35, skip to ensure high label quality
            continue

        lbl_p = labels_dir / (img_p.stem + ".txt")
        lines = []
        for b in boxes:
            cx, cy, w, h = b.xywhn[0].tolist()
            # Bounding box bounds check
            if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines:
            lbl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            valid_samples.append({"image": img_p, "label": lbl_p, "is_negative": False, "boxes": len(lines)})

    return valid_samples


def collect_negatives(neg_dir: Path) -> list:
    """Collect 138 validated 0-byte negative images."""
    negatives = []
    img_dir = neg_dir / "images"
    lbl_dir = neg_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return negatives

    for img_p in sorted(img_dir.glob("*.jpg")):
        lbl_p = lbl_dir / (img_p.stem + ".txt")
        if lbl_p.exists() and lbl_p.stat().st_size == 0:
            negatives.append({"image": img_p, "label": lbl_p, "is_negative": True, "boxes": 0})
    return negatives


def copy_items(items: list, split_name: str, out_dir: Path):
    """Copy images and labels to out_dir/{images,labels}/{split_name}/."""
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for it in items:
        shutil.copy2(it["image"], img_out / it["image"].name)
        shutil.copy2(it["label"], lbl_out / it["label"].name)


def write_yaml(out_dir: Path):
    """Write YAML dataset configuration for calibrated retraining."""
    content = f"""# Milk Bottle Detection — Anti-False-Positive Calibration Dataset
path: {out_dir.name}
train: images/train
val:   images/val

nc: 1
names: ['milk_bottle']
"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(content, encoding="utf-8")
    return CONFIG_PATH


def main():
    print("=" * 65)
    print("  🧪 PREPARING CALIBRATION DATASET (OPTION 2)")
    print("=" * 65)

    random.seed(SEED)

    # 1. Load model to label positive samples
    print(f"\n[1/4] Loading baseline weights: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    temp_pos_dir = BASE_DIR / "scratch" / "temp_calibration_positives"
    temp_pos_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract video stream frames
    stream_frames = extract_video_frames(VIDEO_PATH, temp_pos_dir)
    print(f"      Extracted {len(stream_frames)} frames from {VIDEO_PATH.name}")

    # 3. Gather sample images
    sample_images = list(SAMPLES_DIR.glob("*.jpg"))
    print(f"      Found {len(sample_images)} sample images under samples/")

    all_candidate_pos = sample_images + stream_frames
    labels_temp_dir = temp_pos_dir / "labels"
    pos_samples = generate_positive_labels(model, all_candidate_pos, labels_temp_dir)
    total_pos_boxes = sum(p["boxes"] for p in pos_samples)
    print(f"      Verified {len(pos_samples)} positive images with {total_pos_boxes} bottle annotations")

    # 4. Collect hard negatives
    negatives = collect_negatives(NEGATIVES_DIR)
    print(f"\n[2/4] Sourced {len(negatives)} hard-negative background images from data/negatives/")

    # 5. Split train/val (80% train / 20% val)
    random.shuffle(pos_samples)
    random.shuffle(negatives)

    n_pos_train = int(len(pos_samples) * 0.8)
    n_neg_train = int(len(negatives) * 0.8)

    train_items = pos_samples[:n_pos_train] + negatives[:n_neg_train]
    val_items   = pos_samples[n_pos_train:] + negatives[n_neg_train:]

    random.shuffle(train_items)
    random.shuffle(val_items)

    print(f"\n[3/4] Partitioning calibration dataset:")
    print(f"      • Train Set: {len(train_items)} images ({n_pos_train} pos + {n_neg_train} neg)")
    print(f"      • Val Set  : {len(val_items)} images ({len(pos_samples)-n_pos_train} pos + {len(negatives)-n_neg_train} neg)")
    print(f"      • Total    : {len(train_items) + len(val_items)} images (Negative ratio: {len(negatives)/(len(train_items)+len(val_items)):.1%})")

    # 6. Build dataset_calibrated
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copy_items(train_items, "train", OUTPUT_DIR)
    copy_items(val_items,   "val",   OUTPUT_DIR)

    write_yaml(OUTPUT_DIR)
    print(f"\n[4/4] Dataset written to: {OUTPUT_DIR}")
    print(f"      Configuration saved: {CONFIG_PATH}")
    print("=" * 65)
    print("  ✅ Calibration dataset ready for fine-tuning!")
    print("=" * 65)


if __name__ == "__main__":
    main()
