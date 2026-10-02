#!/usr/bin/env python3
"""
prepare_dataset_v2.py
---------------------
Stage 5A Dataset Correction & Retraining Preparation.

Key Upgrades over v1:
1. Leakage-Free Group Split: Splits by video batch/sequence instead of random frame shuffling.
   - Train Group: annotated1, annotated2, annotated5 (covers single-file, side-standing, and dense crates)
   - Val Group:   annotated3 (curve transition station - unseen viewpoint)
   - Test Group:  annotated4 (top-down single-file track - unseen viewpoint)
2. Canonical Annotation Enforcement: Validates bounding boxes against single-class canonical policy.
3. Hard-Negative Integration: Automatically integrates verified 0-byte negative images (data/negatives)
   to penalize false positives on empty conveyors, white walls, and steel machinery.
4. Config Generation: Outputs configs/dataset_v2.yaml for runs/detect/milk_bottle_v2.
"""

import sys
import shutil
import random
from pathlib import Path

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR        = Path(__file__).resolve().parents[1]
ANNOTATED_DIR   = BASE_DIR / "Annotated"
NEGATIVES_DIR   = BASE_DIR / "data" / "negatives"
OUTPUT_DIR      = BASE_DIR / "dataset_v2"
CONFIG_YAML     = BASE_DIR / "configs" / "dataset_v2.yaml"

TRAIN_BATCHES   = ["annotated1", "annotated2", "annotated5"]
VAL_BATCHES     = ["annotated3"]
TEST_BATCHES    = ["annotated4"]

SEED = 42


def validate_label_canonical(label_path: Path) -> list:
    """Read and validate YOLO bounding boxes under the canonical single-class policy."""
    valid_boxes = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.strip().split()
            if not parts or len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            # Coordinate bounds check
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                continue
            # Single class policy: class 0 only
            if cls_id != 0:
                cls_id = 0
            valid_boxes.append((cls_id, cx, cy, w, h))
    except Exception:
        pass
    return valid_boxes


def collect_group_samples(annotated_dir: Path, batches: list) -> list:
    """Collect validated samples strictly belonging to the specified batch group."""
    samples = []
    if not annotated_dir.exists():
        return samples

    for batch in batches:
        img_dir = annotated_dir / batch / "images"
        lbl_dir = annotated_dir / batch / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_p in sorted(img_dir.glob("*.jpg")):
            if img_p.stat().st_size == 0:
                continue
            lbl_p = lbl_dir / (img_p.stem + ".txt")
            if lbl_p.exists() and lbl_p.stat().st_size > 0:
                boxes = validate_label_canonical(lbl_p)
                if boxes:
                    samples.append({"image": img_p, "label": lbl_p, "group": batch, "boxes": len(boxes)})
    return samples


def collect_negatives(neg_dir: Path) -> list:
    """Collect verified 0-byte hard negative images."""
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


def copy_split_items(items: list, split_name: str, out_dir: Path):
    """Copy images and labels to out_dir/{images,labels}/{split_name}/."""
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for it in items:
        shutil.copy2(it["image"], img_out / it["image"].name)
        shutil.copy2(it["label"], lbl_out / it["label"].name)


def write_yaml(out_dir: Path):
    """Write YAML dataset configuration for v2."""
    content = f"""# Milk Bottle Detection v2 — Leakage-Free Dataset with Hard Negatives
path: {out_dir.name}
train: images/train
val:   images/val
test:  images/test

nc: 1
names: ['milk_bottle']
"""
    CONFIG_YAML.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_YAML.write_text(content, encoding="utf-8")
    return CONFIG_YAML


def main():
    print("=" * 65)
    print("  🥛 Stage 5A Dataset Correction & Retraining Preparation")
    print("=" * 65)

    # 1. Collect positive samples by group
    train_pos = collect_group_samples(ANNOTATED_DIR, TRAIN_BATCHES)
    val_pos   = collect_group_samples(ANNOTATED_DIR, VAL_BATCHES)
    test_pos  = collect_group_samples(ANNOTATED_DIR, TEST_BATCHES)

    # 2. Collect hard-negatives and partition strictly by source family (LEAKAGE-FREE)
    negatives = collect_negatives(NEGATIVES_DIR)
    import re
    families = {}
    for neg in negatives:
        fname = neg["image"].name
        m = re.match(r'^(neg_[a-zA-Z0-9_]+?)(_(flip|bright|dark))?\.jpg$', fname)
        base_family = m.group(1) if m else fname.replace('.jpg', '')
        families.setdefault(base_family, []).append(neg)

    random.seed(SEED)
    family_keys = sorted(list(families.keys()))
    random.shuffle(family_keys)

    n_fam = len(family_keys)
    n_train_f = int(n_fam * 0.70)
    n_val_f   = int(n_fam * 0.15)

    train_families = family_keys[:n_train_f]
    val_families   = family_keys[n_train_f : n_train_f + n_val_f]
    test_families  = family_keys[n_train_f + n_val_f:]

    train_neg = [it for k in train_families for it in families[k]]
    val_neg   = [it for k in val_families for it in families[k]]
    test_neg  = [it for k in test_families for it in families[k]]

    train_total = train_pos + train_neg
    val_total   = val_pos + val_neg
    test_total  = test_pos + test_neg

    print(f"\n[1/3] Leakage-Free Group Partitioning:")
    print(f"      Positive Video Sequences:")
    print(f"        • Train Group: {TRAIN_BATCHES} ({len(train_pos)} positive images)")
    print(f"        • Val Group  : {VAL_BATCHES}   ({len(val_pos)} positive images)")
    print(f"        • Test Group : {TEST_BATCHES}  ({len(test_pos)} positive images)")
    print(f"      Negative Source Families (Zero Cross-Split Leakage):")
    print(f"        • Train Neg  : {len(train_families)} families ({len(train_neg)} negative images)")
    print(f"        • Val Neg    : {len(val_families)} families ({len(val_neg)} negative images)")
    print(f"        • Test Neg   : {len(test_families)} families ({len(test_neg)} negative images)")
    print(f"      Total Negatives: {len(negatives)} across {len(families)} source families")

    # 3. If raw Annotated is present locally, copy and write
    if train_pos or val_pos or test_pos:
        print(f"\n[2/3] Building {OUTPUT_DIR}...")
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        copy_split_items(train_total, "train", OUTPUT_DIR)
        copy_split_items(val_total,   "val",   OUTPUT_DIR)
        copy_split_items(test_total,  "test",  OUTPUT_DIR)
        write_yaml(OUTPUT_DIR)
        print(f"      Saved canonical config: {CONFIG_YAML}")
    else:
        # Build placeholder configuration and instructions
        write_yaml(OUTPUT_DIR)
        print(f"\n[2/3] Local Raw Dataset Status:")
        print(f"      ℹ️ Note: Annotated/ directory is not present locally (proprietary data).")
        print(f"      Canonical configuration written to: {CONFIG_YAML}")
        print(f"      Hard-negatives catalogued: {len(negatives)} images under data/negatives/")

    print("\n[3/3] Retraining Specification Ready for runs/detect/milk_bottle_v2")
    print("=" * 65)


if __name__ == "__main__":
    main()
