#!/usr/bin/env python3
"""
prepare_dataset.py
------------------
Merges all 5 annotated batches from Label Studio (YOLO format) into a clean
dataset with train / val / test splits, validates image and label integrity,
and writes dataset.yaml.

Run:
    python scripts/prepare_dataset.py
"""

import os
import shutil
import random
import sys
from pathlib import Path

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR          = Path(__file__).resolve().parents[1]
ANNOTATED_DIR     = BASE_DIR / "Annotated"
OUTPUT_DIR        = BASE_DIR / "dataset"
CONFIG_YAML       = BASE_DIR / "configs" / "dataset.yaml"
TRAIN_RATIO       = 0.80
VAL_RATIO         = 0.10
TEST_RATIO        = 0.10
SEED              = 42
ANNOTATED_BATCHES = ["annotated1", "annotated2", "annotated3", "annotated4", "annotated5"]


def validate_label_file(label_path: Path) -> bool:
    """Validate that label file contains valid normalized YOLO annotations."""
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                print(f"  [WARN] Invalid label format in {label_path.name}: expected 5 tokens, got {len(parts)}")
                return False
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                print(f"  [WARN] Out of bounds bbox coordinates in {label_path.name}")
                return False
        return True
    except Exception as e:
        print(f"  [WARN] Error parsing {label_path.name}: {e}")
        return False


def collect_samples(annotated_dir: Path, batches: list) -> list:
    """Return list of validated {image, label} path dicts."""
    samples = []
    for batch in batches:
        img_dir = annotated_dir / batch / "images"
        lbl_dir = annotated_dir / batch / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"  [WARN] Skipping {batch}: missing images/ or labels/ subdirectories")
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            if img_path.stat().st_size == 0:
                print(f"  [WARN] Empty image file: {img_path.name} — skipping")
                continue

            stem = img_path.stem
            lbl_path = lbl_dir / (stem + ".txt")
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                if validate_label_file(lbl_path):
                    samples.append({"image": img_path, "label": lbl_path})
            else:
                print(f"  [WARN] Missing or empty label for {img_path.name} — skipping")
    return samples


def split_samples(samples: list, train_r: float, val_r: float, seed: int):
    """Shuffle and split into train / val / test lists with fixed random seed."""
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    train   = shuffled[:n_train]
    val     = shuffled[n_train : n_train + n_val]
    test    = shuffled[n_train + n_val:]
    return train, val, test


def copy_split(split_samples: list, split_name: str, out_dir: Path):
    """Copy images and labels into out_dir/{images,labels}/{split_name}/."""
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for s in split_samples:
        shutil.copy2(s["image"], img_out / s["image"].name)
        shutil.copy2(s["label"], lbl_out / s["label"].name)


def write_yaml(out_dir: Path, class_names: list):
    """Write YOLO dataset.yaml with portable paths."""
    yaml_content = f"""# Milk Bottle Detection — YOLOv8 Dataset Config
path: {out_dir.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    # Also update configs/dataset.yaml with relative path
    CONFIG_YAML.parent.mkdir(parents=True, exist_ok=True)
    rel_yaml_content = f"""# Milk Bottle Detection — YOLOv8 Dataset Config
path: dataset
train: images/train
val:   images/val
test:  images/test

nc: {len(class_names)}
names: {class_names}
"""
    CONFIG_YAML.write_text(rel_yaml_content, encoding="utf-8")
    return yaml_path


def main():
    print("=" * 60)
    print("  🥛 Milk Bottle Dataset Preparation & Validation")
    print("=" * 60)

    if not ANNOTATED_DIR.exists():
        print(f"\n⚠️  Annotated directory not found at: {ANNOTATED_DIR}")
        print("ℹ️  Note: The proprietary industrial dataset (~10GB) is withheld from")
        print("   public distribution. Please see data/README.md for dataset instructions.")
        sys.exit(0)

    # 1. Collect and validate samples
    print(f"\n[1/4] Scanning annotated batches in: {ANNOTATED_DIR}")
    samples = collect_samples(ANNOTATED_DIR, ANNOTATED_BATCHES)
    print(f"      Validated {len(samples)} image-label pairs")
    if not samples:
        print("❌  No valid samples found in Annotated/ batches.")
        sys.exit(1)

    # 2. Read class names from first batch
    classes_file = ANNOTATED_DIR / ANNOTATED_BATCHES[0] / "classes.txt"
    if classes_file.exists():
        class_names = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        class_names = ["milk_bottle"]
    print(f"      Classes: {class_names}")

    # 3. Split
    print(f"\n[2/4] Splitting ({TRAIN_RATIO*100:.0f}% train / {VAL_RATIO*100:.0f}% val / {TEST_RATIO*100:.0f}% test)")
    train, val, test = split_samples(samples, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"      Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    # 4. Copy to output directory
    print(f"\n[3/4] Copying files to: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    copy_split(train, "train", OUTPUT_DIR)
    copy_split(val,   "val",   OUTPUT_DIR)
    copy_split(test,  "test",  OUTPUT_DIR)

    # 5. Write dataset.yaml
    print(f"\n[4/4] Writing dataset configurations")
    yaml_path = write_yaml(OUTPUT_DIR, class_names)
    print(f"      Saved: {yaml_path}")
    print(f"      Saved: {CONFIG_YAML} (canonical)")

    print("\n" + "=" * 60)
    print("  ✅  Dataset Preparation Complete!")
    print("=" * 60)
    print(f"  Location : {OUTPUT_DIR}")
    print(f"  Train    : {len(train):>5} images")
    print(f"  Val      : {len(val):>5} images")
    print(f"  Test     : {len(test):>5} images")
    print(f"  Classes  : {class_names}")
    print(f"  Config   : {CONFIG_YAML} (canonical)")
    print("\nNext → run: python train.py")


if __name__ == "__main__":
    main()
