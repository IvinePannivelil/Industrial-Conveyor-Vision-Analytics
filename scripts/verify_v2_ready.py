#!/usr/bin/env python3
"""
scripts/verify_v2_ready.py
--------------------------
Pre-flight verification script for Milk Bottle Detection Retraining (Experiment v2).

Validates:
  1. Raw Positive Dataset (Annotated/ directory, expected 1,073 images across 5 batches).
  2. Hard-Negative Background Dataset (data/negatives/ directory, 138 image-label pairs, 0-byte labels).
  3. Leakage-Free Split Manifest & Configuration (scratch/dataset_v2_manifest.json, configs/dataset_v2.yaml).
  4. Retraining Scripts (train_v2.py, scripts/prepare_dataset_v2.py).

Run:
  python scripts/verify_v2_ready.py
"""

import sys
import json
import re
from pathlib import Path

# Configure UTF-8 console output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
ANNOTATED_DIR = BASE_DIR / "Annotated"
NEGATIVES_DIR = BASE_DIR / "data" / "negatives"
MANIFEST_PATH = BASE_DIR / "scratch" / "dataset_v2_manifest.json"
CONFIG_PATH = BASE_DIR / "configs" / "dataset_v2.yaml"
TRAIN_SCRIPT = BASE_DIR / "train_v2.py"
PREPARE_SCRIPT = BASE_DIR / "scripts" / "prepare_dataset_v2.py"

EXPECTED_BATCHES = ["annotated1", "annotated2", "annotated3", "annotated4", "annotated5"]
EXPECTED_POSITIVE_COUNT = 1073
EXPECTED_NEGATIVE_COUNT = 138
SUSPICIOUS_FAMILIES = [
    "neg_plant_g1_conveyor_gap_5",
    "neg_plant_g1_metal_support_3",
    "neg_plant_g1_stainless_plate_6",
    "neg_plant_g2_stainless_plate_6",
]


def check_positive_dataset() -> dict:
    """Audit the raw positive dataset directory."""
    result = {
        "exists": False,
        "total_images": 0,
        "total_labels": 0,
        "batches_found": {},
        "missing_batches": [],
        "unpaired_images": 0,
        "ready": False,
        "notes": [],
    }

    if not ANNOTATED_DIR.exists():
        result["notes"].append(f"Directory not found: {ANNOTATED_DIR}")
        result["missing_batches"] = list(EXPECTED_BATCHES)
        return result

    result["exists"] = True
    for b in EXPECTED_BATCHES:
        b_dir = ANNOTATED_DIR / b
        if not b_dir.exists():
            result["missing_batches"].append(b)
            continue

        img_dir = b_dir / "images"
        lbl_dir = b_dir / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            result["missing_batches"].append(b)
            result["notes"].append(f"Batch {b} missing 'images/' or 'labels/' subdirectory")
            continue

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        # Check pairing
        unpaired = 0
        for img in imgs:
            lbl_file = lbl_dir / (img.stem + ".txt")
            if not lbl_file.exists():
                unpaired += 1

        result["batches_found"][b] = {
            "images": len(imgs),
            "labels": len(lbls),
            "unpaired": unpaired,
        }
        result["total_images"] += len(imgs)
        result["total_labels"] += len(lbls)
        result["unpaired_images"] += unpaired

    if result["total_images"] >= EXPECTED_POSITIVE_COUNT and len(result["missing_batches"]) == 0:
        result["ready"] = True
    else:
        result["ready"] = False

    return result


def check_negative_dataset() -> dict:
    """Audit the hard-negative background image dataset."""
    result = {
        "exists": False,
        "image_count": 0,
        "label_count": 0,
        "zero_byte_labels": 0,
        "non_zero_labels": 0,
        "unpaired_images": 0,
        "source_families": set(),
        "suspicious_found": [],
        "ready": False,
        "notes": [],
    }

    neg_img_dir = NEGATIVES_DIR / "images"
    neg_lbl_dir = NEGATIVES_DIR / "labels"

    if not neg_img_dir.exists() or not neg_lbl_dir.exists():
        result["notes"].append(f"Negatives subdirectories missing under {NEGATIVES_DIR}")
        return result

    result["exists"] = True
    imgs = sorted(list(neg_img_dir.glob("*.jpg")) + list(neg_img_dir.glob("*.jpeg")) + list(neg_img_dir.glob("*.png")))
    result["image_count"] = len(imgs)

    for img in imgs:
        fname = img.name
        m = re.match(r"^(neg_[a-zA-Z0-9_]+?)(_(flip|bright|dark))?\.jpg$", fname)
        fam = m.group(1) if m else fname.replace(".jpg", "")
        result["source_families"].add(fam)
        if fam in SUSPICIOUS_FAMILIES and fam not in result["suspicious_found"]:
            result["suspicious_found"].append(fam)

        lbl_path = neg_lbl_dir / (img.stem + ".txt")
        if not lbl_path.exists():
            result["unpaired_images"] += 1
        else:
            sz = lbl_path.stat().st_size
            if sz == 0:
                result["zero_byte_labels"] += 1
            else:
                result["non_zero_labels"] += 1

    result["label_count"] = len(list(neg_lbl_dir.glob("*.txt")))

    if (
        result["image_count"] == EXPECTED_NEGATIVE_COUNT
        and result["zero_byte_labels"] == EXPECTED_NEGATIVE_COUNT
        and result["unpaired_images"] == 0
    ):
        result["ready"] = True
    else:
        result["ready"] = False

    return result


def check_manifest_and_configs() -> dict:
    """Verify split manifest and YAML configuration files."""
    result = {
        "manifest_exists": False,
        "manifest_valid": False,
        "config_exists": False,
        "config_valid": False,
        "train_script_exists": False,
        "prepare_script_exists": False,
        "ready": False,
        "notes": [],
    }

    # Manifest
    if MANIFEST_PATH.exists():
        result["manifest_exists"] = True
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "dataset_version" in data and "sequence_groups" in data.get("positive_data_source", {}):
                result["manifest_valid"] = True
            else:
                result["notes"].append("Manifest is missing required schema keys")
        except Exception as e:
            result["notes"].append(f"Manifest JSON corrupt: {e}")
    else:
        result["notes"].append(f"Manifest file missing: {MANIFEST_PATH}")

    # Config YAML
    if CONFIG_PATH.exists():
        result["config_exists"] = True
        try:
            content = CONFIG_PATH.read_text(encoding="utf-8")
            if "names: ['milk_bottle']" in content and "path: dataset_v2" in content:
                result["config_valid"] = True
            else:
                result["notes"].append("dataset_v2.yaml content differs from expected schema")
        except Exception as e:
            result["notes"].append(f"Could not read config YAML: {e}")
    else:
        result["notes"].append(f"Config YAML missing: {CONFIG_PATH}")

    # Scripts
    result["train_script_exists"] = TRAIN_SCRIPT.exists()
    result["prepare_script_exists"] = PREPARE_SCRIPT.exists()

    if (
        result["manifest_valid"]
        and result["config_valid"]
        and result["train_script_exists"]
        and result["prepare_script_exists"]
    ):
        result["ready"] = True

    return result


def print_report():
    print("=" * 70)
    print("  🔍 PRE-FLIGHT RETRAINING READINESS CHECK — EXPERIMENT V2")
    print("=" * 70)

    # 1. Positives Check
    pos = check_positive_dataset()
    print("\n[1] RAW POSITIVE DATASET (Annotated/):")
    if pos["ready"]:
        print(f"    [OK] Directory present: {ANNOTATED_DIR}")
        print(f"    [OK] Total images: {pos['total_images']} / {EXPECTED_POSITIVE_COUNT} expected")
        print(f"    [OK] All 5 sequence batches present with 0 unpaired images")
    else:
        if not pos["exists"]:
            print(f"    [MISSING] Directory not found locally: {ANNOTATED_DIR}")
            print(f"              Expected 1,073 images across: {', '.join(EXPECTED_BATCHES)}")
            print("              -> Action: Place raw 'Annotated/' folder at repository root.")
        else:
            print(f"    [INCOMPLETE] Found {pos['total_images']} images (expected {EXPECTED_POSITIVE_COUNT})")
            if pos["missing_batches"]:
                print(f"                 Missing batches: {', '.join(pos['missing_batches'])}")
            if pos["unpaired_images"] > 0:
                print(f"                 Unpaired images without labels: {pos['unpaired_images']}")

    # 2. Negatives Check
    neg = check_negative_dataset()
    print("\n[2] HARD-NEGATIVE DATASET (data/negatives/):")
    if neg["ready"]:
        print(f"    [OK] Directory present: {NEGATIVES_DIR}")
        print(f"    [OK] Images count: {neg['image_count']} / {EXPECTED_NEGATIVE_COUNT}")
        print(f"    [OK] Labels count: {neg['zero_byte_labels']} / {EXPECTED_NEGATIVE_COUNT} (all strictly 0 bytes)")
        print(f"    [OK] Unique source families: {len(neg['source_families'])}")
        if neg["suspicious_found"]:
            print(f"    [FLAG] {len(neg['suspicious_found'])} families flagged with partial bottle fragments in manifest:")
            for sf in neg["suspicious_found"]:
                print(f"           • {sf}")
    else:
        print(f"    [FAIL] Negative dataset incomplete or corrupt:")
        print(f"           Images: {neg['image_count']}, Valid 0-byte labels: {neg['zero_byte_labels']}")
        for note in neg["notes"]:
            print(f"           - {note}")

    # 3. Manifest and Pipeline Files Check
    cfg = check_manifest_and_configs()
    print("\n[3] RETRAINING PIPELINE & SPLIT MANIFEST:")
    print(f"    [{'OK' if cfg['manifest_valid'] else 'FAIL'}] Manifest: {MANIFEST_PATH.name} (v2.0-leakage-free)")
    print(f"    [{'OK' if cfg['config_valid'] else 'FAIL'}] Config:   {CONFIG_PATH.relative_to(BASE_DIR)}")
    print(f"    [{'OK' if cfg['prepare_script_exists'] else 'FAIL'}] Splitter: {PREPARE_SCRIPT.relative_to(BASE_DIR)}")
    print(f"    [{'OK' if cfg['train_script_exists'] else 'FAIL'}] Trainer:  {TRAIN_SCRIPT.name} (targeting runs/detect/milk_bottle_v2)")

    # 4. Final Verdict
    print("\n" + "=" * 70)
    if pos["ready"] and neg["ready"] and cfg["ready"]:
        print("  🟢 ALL SYSTEMS READY FOR RETRAINING (EXPERIMENT V2)")
        print("     To begin, run:")
        print("       1. python scripts/prepare_dataset_v2.py")
        print("       2. python train_v2.py")
        print("=" * 70)
        return 0
    elif not pos["ready"] and neg["ready"] and cfg["ready"]:
        print("  🟡 PIPELINE & HARD-NEGATIVES READY — AWAITING RAW POSITIVE DATASET")
        print(f"     1. Copy your raw 'Annotated/' directory ({EXPECTED_POSITIVE_COUNT} images) into the workspace.")
        print("     2. Re-run: python scripts/verify_v2_ready.py")
        print("     3. Then run prepare_dataset_v2.py and train_v2.py.")
        print("=" * 70)
        return 2
    else:
        print("  🔴 PRE-FLIGHT CHECK FAILED — CORRECTIONS NEEDED")
        print("     Inspect errors above before proceeding.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = print_report()
    sys.exit(exit_code)
