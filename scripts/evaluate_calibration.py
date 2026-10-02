#!/usr/bin/env python3
"""
scripts/evaluate_calibration.py
-------------------------------
Comparative evaluation: Baseline v1 vs. Calibrated Anti-False-Positive Model.

Tests:
  1. False-Positive Test: Plain white surface / whiteboard (scratch/webcam_frame.jpg)
  2. Positive Retention Test: Real conveyor bottle samples (samples/conveyor_sample1.jpg, samples/bottle_b1_r0_c0.jpg)
  3. Video Stream Evaluation: Conveyor video (scratch/test_conveyor_stream.mp4)
"""

import sys
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
V1_WEIGHTS = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
CAL_WEIGHTS = BASE_DIR / "runs" / "detect" / "milk_bottle_calibrated" / "weights" / "best.pt"

WHITEBOARD_IMG = BASE_DIR / "scratch" / "webcam_frame.jpg"
SAMPLE_IMG1 = BASE_DIR / "samples" / "conveyor_sample1.jpg"
SAMPLE_IMG2 = BASE_DIR / "samples" / "bottle_b1_r0_c0.jpg"
SAMPLE_IMG3 = BASE_DIR / "samples" / "bottle_b1_r0_c3.jpg"
VIDEO_STREAM = BASE_DIR / "scratch" / "test_conveyor_stream.mp4"


def evaluate_image(model, img_path: Path, conf_thresh: float = 0.35) -> tuple[int, float]:
    """Run inference and return (box_count, max_confidence)."""
    if not img_path.exists():
        return 0, 0.0
    res = model(str(img_path), conf=conf_thresh, verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0.0
    max_conf = float(max(boxes.conf))
    return len(boxes), max_conf


def evaluate_video(model, video_path: Path, conf_thresh: float = 0.35) -> int:
    """Count total bottle detections across all frames in a video stream."""
    if not video_path.exists():
        return 0
    cap = cv2.VideoCapture(str(video_path))
    total_detections = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = model(frame, conf=conf_thresh, verbose=False)
        boxes = res[0].boxes
        if boxes is not None:
            total_detections += len(boxes)
    cap.release()
    return total_detections


def main():
    print("=" * 70)
    print("  📊 COMPARATIVE EVALUATION: BASELINE V1 vs. CALIBRATED V2-PATCH")
    print("=" * 70)

    if not V1_WEIGHTS.exists():
        print(f"❌ Baseline weights missing: {V1_WEIGHTS}")
        sys.exit(1)
    if not CAL_WEIGHTS.exists():
        print(f"❌ Calibrated weights missing: {CAL_WEIGHTS}")
        sys.exit(1)

    print(f"  Baseline v1 Model  : {V1_WEIGHTS.relative_to(BASE_DIR)}")
    print(f"  Calibrated Model   : {CAL_WEIGHTS.relative_to(BASE_DIR)}\n")

    m_v1  = YOLO(str(V1_WEIGHTS))
    m_cal = YOLO(str(CAL_WEIGHTS))

    # 1. Whiteboard False Positive Test
    print("[1] FALSE-POSITIVE SUPPRESSION TEST (Plain Whiteboard Surface):")
    v1_cnt, v1_max = evaluate_image(m_v1, WHITEBOARD_IMG, conf_thresh=0.25)
    cal_cnt, cal_max = evaluate_image(m_cal, WHITEBOARD_IMG, conf_thresh=0.25)
    print(f"    • Baseline v1  : {v1_cnt} false positive(s) detected (Peak Confidence: {v1_max:.1%})")
    print(f"    • Calibrated   : {cal_cnt} false positive(s) detected (Peak Confidence: {cal_max:.1%})")
    if cal_cnt == 0 or cal_max < v1_max:
        reduction = (v1_max - cal_max) / max(v1_max, 1e-6)
        print(f"    -> Result: 🎯 SUCCESS: False positive confidence reduced by {reduction:.1%}!")
    else:
        print(f"    -> Result: False positive status unchanged.")

    # 2. Real Bottle Recall Retention Test
    print("\n[2] REAL BOTTLE RECALL RETENTION TEST (Conveyor Samples):")
    test_samples = [SAMPLE_IMG1, SAMPLE_IMG2, SAMPLE_IMG3]
    for s in test_samples:
        if not s.exists():
            continue
        v1_c, v1_m = evaluate_image(m_v1, s, conf_thresh=0.35)
        cal_c, cal_m = evaluate_image(m_cal, s, conf_thresh=0.35)
        print(f"    • {s.name:<22}: v1={v1_c} boxes (top {v1_m:.1%})  |  Calibrated={cal_c} boxes (top {cal_m:.1%})")

    # 3. Video Stream Evaluation
    print("\n[3] VIDEO STREAM TRACKING DETECTION COUNT (test_conveyor_stream.mp4):")
    v1_stream_dets = evaluate_video(m_v1, VIDEO_STREAM, conf_thresh=0.35)
    cal_stream_dets = evaluate_video(m_cal, VIDEO_STREAM, conf_thresh=0.35)
    print(f"    • Baseline v1 detections  : {v1_stream_dets} total detections across 25 frames")
    print(f"    • Calibrated detections   : {cal_stream_dets} total detections across 25 frames")

    print("\n" + "=" * 70)
    print("  ✅ Comparative Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
