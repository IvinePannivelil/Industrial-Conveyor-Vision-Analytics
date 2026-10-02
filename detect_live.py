#!/usr/bin/env python3
"""
detect_live.py
--------------
Industrial Conveyor Vision Analytics — Real-time runtime pipeline.

Performs YOLOv8 inference with ByteTrack multi-object tracking, persistent track
ID management, virtual line-crossing counting, throughput analytics, and CSV
telemetry logging on live camera or video input.

Displays:
  • Bounding boxes + confidence scores + Track IDs
  • Bounding box center tracking dots
  • Virtual horizontal counting line
  • Live HUD: CURRENT DETECTIONS, OBJECTS COUNTED, THROUGHPUT (objects/min)
  • Status telemetry: FPS, Inference latency, Confidence, Tracking, Counting
  • Mini live-graph of detection count over time (top-right panel)

Controls:
  q  — quit
  s  — save screenshot
  p  — pause / resume
  +  — increase confidence threshold
  -  — decrease confidence threshold

CLI Options:
  --source          Camera index (0) or path to image/video file
  --weights         Path to YOLOv8 weights (.pt)
  --conf            Confidence threshold (default: 0.40)
  --device          Device to use: cpu, cuda, mps (default: auto)
  --no-window       Run without GUI display window (headless)
  --track           Enable ByteTrack persistent object tracking
  --count-line      Enable virtual line-crossing unique object counting
  --line-position   Vertical position of counting line: relative (0.1-0.9) or pixel Y (default: 0.5)
  --count-direction Crossing direction to count: both, down, up (default: both)
  --csv-log         Path to optional CSV telemetry output log file

Run:
  python detect_live.py --help
  python detect_live.py --source conveyor.mp4 --track --count-line --csv-log results/session.csv
"""

import cv2
import time
import sys
import os
import csv
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

# Configure UTF-8 output on Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration Defaults ───────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
WEIGHTS_PATH   = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
CAMERA_INDEX   = 0       # Default built-in webcam
CONF_THRESHOLD = 0.40    # Default detection confidence (adjustable with +/-)
IOU_THRESHOLD  = 0.45
GRAPH_HISTORY  = 150     # Number of frames to show in the live graph
SCREENSHOT_DIR = BASE_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ── Colours & Fonts ──────────────────────────────────────────────────────────
CLR_BOX      = (0,   200, 80)    # Green bounding box
CLR_LABEL_BG = (0,   160, 60)    # Green label background
CLR_COUNT_BG = (15,  15,  15)    # Dark HUD background
CLR_FPS      = (200, 200, 200)   # Light gray text
CLR_GRAPH_BG = (18,  18,  18)    # Graph panel background
CLR_GRAPH_LN = (0,   210, 100)   # Graph plot line
CLR_GRAPH_AX = (80,  80,  80)    # Graph axes
CLR_LINE     = (0,   215, 255)   # Gold / Yellow virtual counting line
CLR_CENTER   = (0,   255, 255)   # Cyan center tracking point
FONT         = cv2.FONT_HERSHEY_SIMPLEX

# ── Graph Panel Dimensions ───────────────────────────────────────────────────
GRAPH_W = 340
GRAPH_H = 220   # Placed top-right of the frame


def get_best_device() -> str:
    """Automatically select CUDA GPU, Apple Silicon MPS, or CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_model(weights_path=WEIGHTS_PATH):
    """Load YOLOv8 model. Falls back to yolov8n.pt if custom weights absent."""
    from ultralytics import YOLO
    weights_p = Path(weights_path)
    if weights_p.exists():
        print(f"✅  Loading custom weights: {weights_p}")
        return YOLO(str(weights_p))
    else:
        print(f"⚠️  Custom weights not found at:\n    {weights_p}")
        print("   Falling back to base YOLOv8n (not trained on your data).")
        print("   Run  python train.py  first for accurate results.\n")
        return YOLO("yolov8n.pt")


# ── Tracking & Geometry Helpers ──────────────────────────────────────────────
def calculate_center(xyxy) -> tuple[int, int]:
    """Calculate center point (cx, cy) of a bounding box."""
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def resolve_line_y(line_position: float, frame_height: int) -> int:
    """Calculate integer Y-coordinate for the horizontal counting line."""
    if 0.0 < line_position <= 1.0:
        line_y = int(line_position * frame_height)
    else:
        line_y = int(line_position)
    return max(10, min(frame_height - 10, line_y))


def check_line_crossing(prev_y: int, curr_y: int, line_y: int, direction: str = "both") -> tuple[bool, str]:
    """
    Check if an object center crossed the horizontal counting line.
    Returns:
      (crossed: bool, event_direction: str or None)
    """
    crossed_down = (prev_y < line_y <= curr_y)
    crossed_up   = (prev_y > line_y >= curr_y)

    if direction == "down" and crossed_down:
        return True, "DOWN"
    elif direction == "up" and crossed_up:
        return True, "UP"
    elif direction == "both":
        if crossed_down:
            return True, "DOWN"
        elif crossed_up:
            return True, "UP"

    return False, None


def calculate_throughput(bottles_passed: int, elapsed_seconds: float) -> float:
    """
    Calculate production throughput in bottles per minute.
    Safely handles zero or near-zero elapsed time.
    """
    if elapsed_seconds < 1.0 or bottles_passed == 0:
        return 0.0
    elapsed_minutes = elapsed_seconds / 60.0
    return bottles_passed / elapsed_minutes


def update_track_history_and_count(
    detections: list,
    track_history: dict,
    counted_track_ids: set,
    line_y: int,
    total_bottles: int,
    direction: str = "both",
) -> int:
    """
    Update track trajectory history and detect line-crossing events.
    Each unique track_id is counted at most once upon crossing the line.
    """
    for det in detections:
        track_id = det["track_id"]
        if track_id is None:
            continue

        curr_cx, curr_cy = det["center"]

        # Check for line crossing if we have prior position history for this track
        if track_id in track_history and len(track_history[track_id]) > 0:
            prev_cx, prev_cy = track_history[track_id][-1]
            crossed, event_dir = check_line_crossing(prev_cy, curr_cy, line_y, direction=direction)
            if crossed:
                if track_id not in counted_track_ids:
                    counted_track_ids.add(track_id)
                    total_bottles += 1
                    print(f"🔔  COUNT EVENT: Track ID #{track_id} | Direction: {event_dir} | Total Bottles Passed: {total_bottles}")

        # Record current position
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=30)
        track_history[track_id].append((curr_cx, curr_cy))

    # Periodic cleanup of stale tracks to prevent memory growth
    if len(track_history) > 500:
        active_ids = {d["track_id"] for d in detections if d["track_id"] is not None}
        stale_ids = [tid for tid in track_history if tid not in active_ids and tid in counted_track_ids]
        for tid in stale_ids[:100]:
            del track_history[tid]

    return total_bottles


# ── CSV Telemetry Logger ─────────────────────────────────────────────────────
class CSVTelemetryLogger:
    """Appends per-frame production telemetry to a CSV log file."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.writer = None
        self._init_csv()

    def _init_csv(self):
        try:
            path = Path(self.filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.file = open(path, mode="w", newline="", encoding="utf-8")
            self.writer = csv.writer(self.file)
            self.writer.writerow([
                "timestamp",
                "elapsed_seconds",
                "frame_number",
                "current_detections",
                "bottles_passed",
                "throughput_bottles_per_min",
                "fps",
                "confidence",
                "tracking_enabled",
                "counting_enabled",
                "count_direction",
            ])
            self.file.flush()
            print(f"📝  CSV Telemetry Logger initialized: {path.resolve()}")
        except Exception as e:
            print(f"⚠️  Failed to initialize CSV logger at '{self.filepath}': {e}")
            self.file = None
            self.writer = None

    def log_frame(
        self,
        elapsed_seconds: float,
        frame_number: int,
        current_detections: int,
        bottles_passed: int,
        throughput: float,
        fps: float,
        confidence: float,
        tracking_enabled: bool,
        counting_enabled: bool,
        count_direction: str,
    ):
        if self.writer is None:
            return
        try:
            now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.writer.writerow([
                now_iso,
                f"{elapsed_seconds:.2f}",
                frame_number,
                current_detections,
                bottles_passed,
                f"{throughput:.2f}",
                f"{fps:.1f}",
                f"{confidence:.2f}",
                tracking_enabled,
                counting_enabled,
                count_direction,
            ])
            self.file.flush()
        except Exception as e:
            print(f"⚠️  Error writing telemetry row: {e}")

    def close(self):
        if self.file and not self.file.closed:
            try:
                self.file.close()
            except Exception:
                pass


# ── Drawing & HUD Functions ──────────────────────────────────────────────────
def draw_boxes(frame: np.ndarray, boxes, class_names: list, conf_thresh: float, is_tracking: bool = False) -> tuple[int, list]:
    """
    Draw bounding boxes, labels, and center markers.
    Returns:
      (current_detections, list_of_active_detections)
    """
    count = 0
    detections = []
    if boxes is None or len(boxes) == 0:
        return count, detections

    for box in boxes:
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        if conf < conf_thresh:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = calculate_center((x1, y1, x2, y2))

        # Check if tracker assigned an ID
        track_id = int(box.id[0]) if (is_tracking and hasattr(box, 'id') and box.id is not None) else None

        if track_id is not None:
            label = f"Bottle #{track_id}  {conf:.0%}"
        else:
            label = f"{class_names[cls]}  {conf:.0%}"

        detections.append({
            "track_id": track_id,
            "conf": conf,
            "cls": cls,
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
        })

        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), CLR_BOX, 2)

        # Draw Center Point for tracked bottles
        if is_tracking:
            cv2.circle(frame, (cx, cy), 4, CLR_CENTER, -1)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), CLR_LABEL_BG, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return count, detections


def draw_counting_line(frame: np.ndarray, line_y: int, direction: str = "both"):
    """Draw a highlighted virtual counting line across the frame."""
    h, w = frame.shape[:2]
    # Draw horizontal line
    cv2.line(frame, (0, line_y), (w, line_y), CLR_LINE, 2, cv2.LINE_AA)
    # Line label badge
    dir_str = f" [{direction.upper()}]" if direction != "both" else ""
    label = f"COUNTING LINE (Y={line_y}){dir_str}"
    cv2.putText(frame, label, (15, line_y - 8), FONT, 0.45, CLR_LINE, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    current_detections: int,
    total_bottles: int,
    fps: float,
    conf: float,
    paused: bool,
    tracking_enabled: bool = False,
    counting_enabled: bool = False,
    throughput: float = 0.0,
):
    """Overlay detection count, unique bottles passed, throughput, FPS, and status telemetry."""
    h, w = frame.shape[:2]

    # Semi-transparent count badge (bottom-left)
    badge_w = 340 if counting_enabled else 270
    badge_h = 105 if counting_enabled else 85
    badge = np.zeros((badge_h, badge_w, 3), dtype=np.uint8)
    badge[:] = CLR_COUNT_BG

    if counting_enabled:
        # Top row: CURRENT DETECTIONS and BOTTLES PASSED
        cv2.putText(badge, "CURRENT DETECTIONS", (12, 20), FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(badge, str(current_detections), (12, 54), FONT, 1.1, CLR_GRAPH_LN, 2, cv2.LINE_AA)

        cv2.putText(badge, "OBJECTS COUNTED", (175, 20), FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(badge, str(total_bottles), (175, 54), FONT, 1.1, CLR_LINE, 2, cv2.LINE_AA)

        # Bottom row: THROUGHPUT
        tp_str = f"THROUGHPUT: {throughput:.1f} obj/min"
        cv2.putText(badge, tp_str, (12, 90), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Detection-only or Tracking-only
        cv2.putText(badge, "CURRENT DETECTIONS", (12, 24), FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(badge, str(current_detections), (12, 72), FONT, 2.2, CLR_GRAPH_LN, 3, cv2.LINE_AA)

    badge_alpha = 0.82
    frame[h - badge_h : h, 0 : badge_w] = cv2.addWeighted(
        frame[h - badge_h : h, 0 : badge_w], 1 - badge_alpha, badge, badge_alpha, 0
    )

    # Status line (top-left)
    track_str = "ON" if tracking_enabled else "OFF"
    count_str = "ON" if counting_enabled else "OFF"
    status = f"FPS: {fps:.1f}  |  Conf: {conf:.0%}  |  Tracking: {track_str}  |  Counting: {count_str}"
    if paused:
        status += "  |  [PAUSED]"
    cv2.putText(frame, status, (10, 24), FONT, 0.55, CLR_FPS, 1, cv2.LINE_AA)

    # Controls hint (bottom)
    cv2.putText(
        frame,
        "q=quit  s=screenshot  p=pause  +/-=confidence",
        (10, h - 8),
        FONT,
        0.42,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )


def draw_graph(frame: np.ndarray, history: deque, max_count: int):
    """Draw a mini line-graph of bottle count over time (top-right corner)."""
    h, w = frame.shape[:2]
    panel = np.full((GRAPH_H, GRAPH_W, 3), CLR_GRAPH_BG, dtype=np.uint8)

    # Title
    cv2.putText(panel, "Detection Count Over Time", (8, 18), FONT, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    # Axes
    margin = (28, 12, 20, 10)  # top, right, bottom, left
    gx0, gy0 = margin[3], margin[0]
    gx1 = GRAPH_W - margin[1]
    gy1 = GRAPH_H - margin[2]
    cv2.rectangle(panel, (gx0, gy0), (gx1, gy1), CLR_GRAPH_AX, 1)

    # Y-axis label
    cap_val = max(max_count, 1)
    cv2.putText(panel, f"{cap_val}", (0, gy0 + 8), FONT, 0.38, (150, 150, 150), 1)
    cv2.putText(panel, "0", (0, gy1),              FONT, 0.38, (150, 150, 150), 1)

    # Plot line
    pts = list(history)
    n   = len(pts)
    if n >= 2:
        gw  = gx1 - gx0
        gh  = gy1 - gy0
        xs  = [int(gx0 + i * gw / (GRAPH_HISTORY - 1)) for i in range(n)]
        ys  = [int(gy1 - (v / cap_val) * gh) for v in pts]
        for i in range(1, n):
            cv2.line(panel, (xs[i-1], ys[i-1]), (xs[i], ys[i]), CLR_GRAPH_LN, 2, cv2.LINE_AA)
        # Latest value dot
        cv2.circle(panel, (xs[-1], ys[-1]), 4, (255, 255, 100), -1)
        cv2.putText(panel, str(pts[-1]), (xs[-1] + 6, ys[-1] + 4), FONT, 0.45, (255, 255, 100), 1)

    # Composite onto frame (top-right)
    px = w - GRAPH_W
    frame[0:GRAPH_H, px:w] = cv2.addWeighted(
        frame[0:GRAPH_H, px:w], 0.25, panel, 0.75, 0
    )


# ── Single Image Detection / Tracking ─────────────────────────────────────────
def run_image_detection(
    image_path: str,
    model,
    conf_th: float,
    device: str,
    no_window: bool = False,
    save_path: str = None,
    track: bool = False,
    count_line: bool = False,
    line_position: float = 0.5,
    count_direction: str = "both",
    csv_log: str = None,
):
    """Run detection or tracking on a single image file."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌  Could not load image: {image_path}")
        return

    h, w = frame.shape[:2]
    class_names = model.names if hasattr(model, "names") else {0: "milk_bottle"}

    if track:
        results = model.track(
            source=frame,
            conf=conf_th,
            iou=IOU_THRESHOLD,
            device=device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )
    else:
        results = model.predict(
            source=frame,
            conf=conf_th,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False,
        )

    annotated = frame.copy()
    count, detections = draw_boxes(annotated, results[0].boxes, class_names, conf_th, is_tracking=track)

    line_y = resolve_line_y(line_position, h)
    total_bottles = 0
    if count_line:
        counted_track_ids = set()
        for det in detections:
            tid = det["track_id"] if det["track_id"] is not None else len(counted_track_ids) + 1
            if count_direction == "up":
                is_passed = (det["center"][1] <= line_y)
            else:  # down or both
                is_passed = (det["center"][1] >= line_y)

            if is_passed:
                counted_track_ids.add(tid)
        total_bottles = len(counted_track_ids)
        draw_counting_line(annotated, line_y, count_direction)

    draw_hud(
        annotated,
        current_detections=count,
        total_bottles=total_bottles,
        fps=0.0,
        conf=conf_th,
        paused=False,
        tracking_enabled=track,
        counting_enabled=count_line,
        throughput=0.0,
    )

    print(f"\n✅  Processed Image: {image_path}")
    print(f"🥛  CURRENT DETECTIONS: {count}")
    if count_line:
        print(f"📊  BOTTLES PASSED (relative to line Y={line_y}, dir={count_direction.upper()}): {total_bottles}")
    for idx, det in enumerate(detections):
        tid_str = f" [Track #{det['track_id']}]" if det["track_id"] is not None else ""
        c_name = class_names[det["cls"]]
        print(f"   • Bottle #{idx+1}{tid_str}: {c_name} ({det['conf']:.1%}) at {list(det['bbox'])} center={det['center']}")

    out_file = save_path or str(SCREENSHOT_DIR / f"detection_{Path(image_path).name}")
    cv2.imwrite(out_file, annotated)
    print(f"📸  Annotated output saved to: {out_file}")

    # Log to CSV if enabled
    if csv_log:
        logger = CSVTelemetryLogger(csv_log)
        logger.log_frame(
            elapsed_seconds=0.0,
            frame_number=1,
            current_detections=count,
            bottles_passed=total_bottles,
            throughput=0.0,
            fps=0.0,
            confidence=conf_th,
            tracking_enabled=track,
            counting_enabled=count_line,
            count_direction=count_direction,
        )
        logger.close()

    if not no_window:
        print("   Press any key to close the image preview window...")
        cv2.imshow("Industrial Conveyor Vision Analytics — Image Preview", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── Live Stream / Video Detection ────────────────────────────────────────────
def run_detection(
    source="0",
    weights_path=WEIGHTS_PATH,
    conf_threshold=CONF_THRESHOLD,
    device=None,
    no_window=False,
    track=False,
    count_line=False,
    line_position=0.5,
    count_direction="both",
    csv_log=None,
):
    sel_device = device or get_best_device()
    print(f"⚡  Inference Device: {sel_device}")

    # If count-line is requested, ensure tracking is active
    if count_line and not track:
        track = True

    model = load_model(weights_path)
    class_names = model.names if hasattr(model, "names") else {0: "milk_bottle"}

    # Check if source is a static image
    source_str = str(source)
    if not source_str.isdigit() and Path(source_str).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        run_image_detection(
            image_path=source_str,
            model=model,
            conf_th=conf_threshold,
            device=sel_device,
            no_window=no_window,
            track=track,
            count_line=count_line,
            line_position=line_position,
            count_direction=count_direction,
            csv_log=csv_log,
        )
        return

    # Video or webcam source
    cam_source = int(source_str) if source_str.isdigit() else source_str
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"❌  Cannot open video/camera source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize CSV logger if path is provided
    csv_logger = CSVTelemetryLogger(csv_log) if csv_log else None

    history           = deque([0] * GRAPH_HISTORY, maxlen=GRAPH_HISTORY)
    conf_th           = conf_threshold
    paused            = False
    session_start     = time.time()
    prev_time         = session_start
    fps               = 0.0
    last_frame        = None
    max_count         = 1
    total_bottles     = 0
    counted_track_ids = set()
    track_history     = {}
    frame_count       = 0
    throughput        = 0.0

    print("\n📷  Input stream opened. Starting live detection…")
    print(f"   Mode: Tracking={'ON' if track else 'OFF'}, Counting={'ON' if count_line else 'OFF'}, Direction={count_direction.upper()}")
    print("   Controls: q=quit  s=screenshot  p=pause  +/-=confidence\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("🏁  End of video stream or failed to grab frame.")
                    break
                last_frame = frame.copy()
                frame_count += 1
                h, w = frame.shape[:2]
                line_y = resolve_line_y(line_position, h)

                # ── Inference (Tracking vs Detection) ──────────────────────────
                if track:
                    results = model.track(
                        source=frame,
                        conf=conf_th,
                        iou=IOU_THRESHOLD,
                        device=sel_device,
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False,
                        stream=False,
                    )
                else:
                    results = model.predict(
                        source=frame,
                        conf=conf_th,
                        iou=IOU_THRESHOLD,
                        device=sel_device,
                        verbose=False,
                        stream=False,
                    )

                # Draw boxes, labels, and centers
                current_detections, detections = draw_boxes(
                    frame, results[0].boxes, class_names, conf_th, is_tracking=track
                )

                # Line crossing calculation
                if count_line and track:
                    total_bottles = update_track_history_and_count(
                        detections=detections,
                        track_history=track_history,
                        counted_track_ids=counted_track_ids,
                        line_y=line_y,
                        total_bottles=total_bottles,
                        direction=count_direction,
                    )
                    draw_counting_line(frame, line_y, count_direction)

                history.append(current_detections)
                max_count = max(max_count, current_detections, 1)

                # ── Time, FPS & Throughput Calculation ─────────────────────────
                now             = time.time()
                elapsed_seconds = now - session_start
                fps             = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
                prev_time       = now
                throughput      = calculate_throughput(total_bottles, elapsed_seconds)

                # ── CSV Telemetry Logging ──────────────────────────────────────
                if csv_logger:
                    csv_logger.log_frame(
                        elapsed_seconds=elapsed_seconds,
                        frame_number=frame_count,
                        current_detections=current_detections,
                        bottles_passed=total_bottles,
                        throughput=throughput,
                        fps=fps,
                        confidence=conf_th,
                        tracking_enabled=track,
                        counting_enabled=count_line,
                        count_direction=count_direction,
                    )
            else:
                frame = last_frame.copy() if last_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

            draw_hud(
                frame,
                current_detections=history[-1],
                total_bottles=total_bottles,
                fps=fps,
                conf=conf_th,
                paused=paused,
                tracking_enabled=track,
                counting_enabled=count_line,
                throughput=throughput,
            )
            draw_graph(frame, history, max_count)

            if not no_window:
                cv2.imshow("Industrial Conveyor Vision Analytics — Press q to quit", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = SCREENSHOT_DIR / f"detection_{ts}.jpg"
                    cv2.imwrite(str(path), frame)
                    print(f"📸  Screenshot saved → {path}")
                elif key == ord("p"):
                    paused = not paused
                    print("⏸  Paused" if paused else "▶  Resumed")
                elif key == ord("+") or key == ord("="):
                    conf_th = min(conf_th + 0.05, 0.95)
                    print(f"🔼  Confidence threshold: {conf_th:.0%}")
                elif key == ord("-"):
                    conf_th = max(conf_th - 0.05, 0.05)
                    print(f"🔽  Confidence threshold: {conf_th:.0%}")

    finally:
        cap.release()
        if not no_window:
            cv2.destroyAllWindows()
        if csv_logger:
            csv_logger.close()

    total_elapsed = time.time() - session_start
    avg_fps = frame_count / max(total_elapsed, 1e-6)
    final_throughput = calculate_throughput(total_bottles, total_elapsed)

    print("\n" + "=" * 60)
    print("  📊  Session Summary")
    print("=" * 60)
    print(f"   • Total Frames Processed: {frame_count}")
    print(f"   • Processing Time: {total_elapsed:.2f} seconds")
    print(f"   • Average FPS: {avg_fps:.1f}")
    if count_line:
        print(f"   • Total Unique Objects Counted: {total_bottles}")
        print(f"   • Final Throughput: {final_throughput:.1f} obj/min")
        print(f"   • Unique Track IDs Counted: {sorted(list(counted_track_ids))}")
    else:
        print(f"   • Current Detections (Final Frame): {history[-1] if len(history) > 0 else 0}")
    print(f"   • Tracking: {'ON' if track else 'OFF'}")
    print(f"   • Counting: {'ON' if count_line else 'OFF'}")
    print(f"   • Count Direction: {count_direction.upper()}")
    print(f"   • CSV Log: {csv_log if csv_log else 'Disabled'}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Industrial Conveyor Vision Analytics — "
            "Real-time object detection, tracking, counting, and throughput analytics "
            "using YOLOv8 and ByteTrack."
        )
    )
    parser.add_argument("--source", type=str, default=str(CAMERA_INDEX), help="Camera index (0) or path to image/video file")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH), help="Path to YOLOv8 weights (.pt)")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--device", type=str, default=None, help="Device to use: cpu, cuda, mps (default: auto)")
    parser.add_argument("--no-window", action="store_true", help="Run without GUI display window (headless)")
    parser.add_argument("--track", action="store_true", help="Enable ByteTrack persistent object tracking")
    parser.add_argument("--count-line", action="store_true", help="Enable virtual line-crossing unique bottle counting")
    parser.add_argument("--line-position", type=float, default=0.5, help="Vertical position of counting line: relative (0.1-0.9) or pixel Y (default: 0.5)")
    parser.add_argument("--count-direction", type=str, default="both", choices=["both", "down", "up"], help="Crossing direction to count: both, down, up (default: both)")
    parser.add_argument("--csv-log", type=str, default=None, help="Path to optional CSV telemetry output log file (e.g. results/session.csv)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Industrial Conveyor Vision Analytics")
    print("  Real-time detection, tracking, counting & throughput")
    print("=" * 60)
    run_detection(
        source=args.source,
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
        no_window=args.no_window,
        track=args.track,
        count_line=args.count_line,
        line_position=args.line_position,
        count_direction=args.count_direction,
        csv_log=args.csv_log,
    )


if __name__ == "__main__":
    main()
