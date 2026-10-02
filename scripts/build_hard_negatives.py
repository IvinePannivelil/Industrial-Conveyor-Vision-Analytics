#!/usr/bin/env python3
"""
build_hard_negatives.py
-----------------------
Extracts and synthesizes a dedicated collection of diverse hard-negative images
(empty conveyor surfaces, factory floor tiles, stainless steel guides, white walls,
whiteboard frames, lighting reflections) with 0-byte label files for training.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_IMG_DIR = BASE_DIR / "data" / "negatives" / "images"
OUT_LBL_DIR = BASE_DIR / "data" / "negatives" / "labels"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

neg_count = 0

def save_negative(img: np.ndarray, name: str):
    global neg_count
    if img is None or img.shape[0] < 20 or img.shape[1] < 20:
        return
    # Ensure standard 640x640 resolution
    resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    img_path = OUT_IMG_DIR / f"{name}.jpg"
    lbl_path = OUT_LBL_DIR / f"{name}.txt"
    cv2.imwrite(str(img_path), resized)
    lbl_path.write_text("", encoding="utf-8")  # Explicit 0-byte file
    neg_count += 1

# 1. Extract from factory validation grids (actual operating plant)
val_grids = [
    BASE_DIR / "runs" / "detect" / "milk_bottle" / "val_batch0_labels.jpg",
    BASE_DIR / "runs" / "detect" / "milk_bottle" / "val_batch1_labels.jpg",
    BASE_DIR / "runs" / "detect" / "milk_bottle" / "val_batch2_labels.jpg",
]

for g_idx, g_path in enumerate(val_grids):
    if not g_path.exists():
        continue
    grid = cv2.imread(str(g_path))
    if grid is None:
        continue
    gh, gw = grid.shape[:2]

    # Specific coordinate patches containing strictly machine parts, floor, conveyor rails
    regions = [
        # (ymin, xmin, ymax, xmax, desc)
        (360, 10, 470, 350, "conveyor_rollers"),
        (10, 10, 110, 350, "machine_hood"),
        (840, 10, 950, 350, "lower_rail"),
        (1340, 10, 1430, 350, "metal_support"),
        (10, 370, 100, 720, "top_ceiling_tiles"),
        (380, 370, 470, 720, "conveyor_gap"),
        (850, 370, 950, 720, "stainless_plate"),
        (10, 740, 100, 1080, "piping_wall"),
        (400, 740, 480, 1080, "steel_guide"),
        (880, 740, 960, 1080, "tile_floor"),
        (1340, 740, 1430, 1080, "drain_grate"),
    ]
    for r_idx, (ymin, xmin, ymax, xmax, desc) in enumerate(regions):
        if ymax <= gh and xmax <= gw:
            crop = grid[ymin:ymax, xmin:xmax]
            save_negative(crop, f"neg_plant_g{g_idx}_{desc}_{r_idx}")
            # Augmented variations: horizontal flip & slight brightness shift
            save_negative(cv2.flip(crop, 1), f"neg_plant_g{g_idx}_{desc}_{r_idx}_flip")
            bright = cv2.convertScaleAbs(crop, alpha=1.2, beta=15)
            save_negative(bright, f"neg_plant_g{g_idx}_{desc}_{r_idx}_bright")
            dark = cv2.convertScaleAbs(crop, alpha=0.8, beta=-15)
            save_negative(dark, f"neg_plant_g{g_idx}_{desc}_{r_idx}_dark")

# 2. Extract from live camera scene (whiteboard, white wall, room features that caused 74.5% FP)
webcam_path = BASE_DIR / "scratch" / "webcam_frame.jpg"
if webcam_path.exists():
    wb_frame = cv2.imread(str(webcam_path))
    if wb_frame is not None:
        wh, ww = wb_frame.shape[:2]
        # Crop the whiteboard and upper dark frame specifically
        # Box was around [352, 0, 615, 262]
        wb_crop1 = wb_frame[0:270, 340:630]
        save_negative(wb_crop1, "neg_room_whiteboard_frame")
        save_negative(cv2.flip(wb_crop1, 1), "neg_room_whiteboard_frame_flip")

        wb_crop2 = wb_frame[70:350, 360:640]
        save_negative(wb_crop2, "neg_room_whiteboard_surface")
        save_negative(cv2.flip(wb_crop2, 1), "neg_room_whiteboard_surface_flip")

        # White wall & door frame
        wall_crop = wb_frame[0:250, 150:350]
        save_negative(wall_crop, "neg_room_door_frame")

        # Upper ceiling and light panels
        ceiling_crop = wb_frame[0:150, 0:300]
        save_negative(ceiling_crop, "neg_room_ceiling_lighting")

print(f"✅ Total hard-negative images generated: {neg_count}")
print(f"   Images: {OUT_IMG_DIR}")
print(f"   Labels: {OUT_LBL_DIR}")
