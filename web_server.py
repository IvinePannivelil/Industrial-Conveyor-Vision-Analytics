#!/usr/bin/env python3
"""
web_server.py
-------------
Industrial Conveyor Vision Analytics — Interactive Web Application Suite.
Provides a browser-based dashboard running YOLOv8 detection, ByteTrack tracking,
virtual line crossing counting, throughput analytics, model diagnostics, and
architecture documentation.

Telemetry Architecture
----------------------
Two separate outputs are maintained:

1. frame_telemetry (in-memory deque, also written to results/live_run.csv if used):
   Per-frame operational stats: FPS, detections, cumulative count, confidence, etc.

2. count_events (results/count_events.csv):
   One row per valid line-crossing event only. Enables time-window queries such as
   "how many objects crossed between 10:15 and 10:45?". Never deleted by session
   reset — reset only clears the in-session counter displayed in the dashboard.
"""

import csv
import cv2
import time
import math
import random
import threading
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Generator

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, render_template

from detect_live import check_line_crossing

# ── Paths & Setup ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent.resolve()
WEIGHTS_PATH   = BASE_DIR / "runs" / "detect" / "milk_bottle" / "weights" / "best.pt"
STATIC_DIR     = BASE_DIR / "web" / "static"
TEMPLATES_DIR  = BASE_DIR / "web" / "templates"
COUNT_EVENTS_CSV = BASE_DIR / "results" / "count_events.csv"

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    template_folder=str(TEMPLATES_DIR),
)


# ── Count Event Logger ────────────────────────────────────────────────────────
class CountEventLogger:
    """
    Append-safe CSV logger for individual line-crossing events.

    Each row represents one valid, deduplicated crossing event.
    This file is NEVER cleared by session reset — it accumulates across
    the lifetime of a local deployment session. Historical data can be
    queried via the /api/count_history endpoint.

    CSV columns:
        timestamp       — HH:MM:SS of the crossing event
        date            — YYYY-MM-DD (for multi-day sessions)
        track_id        — ByteTrack track ID (integer) or "SIM-XXXX" for simulator
        direction       — DOWN | UP
        confidence      — Detection confidence at crossing (0.00–1.00)
        cumulative_count — session-cumulative count at time of this event
        source_mode     — "simulator" | "live_camera"
        line_position   — Fractional line Y position (0.0–1.0)
        frame_number    — Frame index at crossing
    """

    CSV_FIELDS = [
        "timestamp",
        "date",
        "track_id",
        "direction",
        "confidence",
        "cumulative_count",
        "source_mode",
        "line_position",
        "frame_number",
    ]

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the CSV file with headers if it does not yet exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def log_event(
        self,
        track_id: Any,
        direction: str,
        confidence: float,
        cumulative_count: int,
        source_mode: str,
        line_position: float,
        frame_number: int,
    ) -> None:
        """Append one crossing event row to the CSV. Thread-safe."""
        now = datetime.now()
        row = {
            "timestamp": now.strftime("%H:%M:%S"),
            "date": now.strftime("%Y-%m-%d"),
            "track_id": track_id,
            "direction": direction,
            "confidence": f"{confidence:.4f}",
            "cumulative_count": cumulative_count,
            "source_mode": source_mode,
            "line_position": f"{line_position:.3f}",
            "frame_number": frame_number,
        }
        with self._lock:
            try:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                    writer.writerow(row)
            except Exception as e:
                print(f"[CountEventLogger] Failed to write event row: {e}")

    def read_events(self) -> List[Dict[str, str]]:
        """Read all stored events. Returns list of dicts. Thread-safe."""
        with self._lock:
            if not self.csv_path.exists():
                return []
            try:
                with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    return [dict(row) for row in reader]
            except Exception as e:
                print(f"[CountEventLogger] Failed to read events: {e}")
                return []

    def query_window(
        self,
        from_time: str,
        to_time: str,
        date: Optional[str] = None,
        direction_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Filter events by time window (HH:MM or HH:MM:SS strings).
        date: 'YYYY-MM-DD' to restrict to a single day (defaults to today).
        direction_filter: 'DOWN', 'UP', or None for all.
        source_filter: 'simulator', 'live_camera', or None for all.
        """
        events = self.read_events()
        target_date = date or datetime.now().strftime("%Y-%m-%d")

        def parse_t(t: str) -> str:
            # Normalize to HH:MM:SS for comparison
            parts = t.split(":")
            if len(parts) == 2:
                return f"{parts[0]:0>2}:{parts[1]:0>2}:00"
            return t

        from_t = parse_t(from_time)
        to_t = parse_t(to_time)

        filtered = []
        for ev in events:
            ev_date = ev.get("date", "")
            ev_ts = ev.get("timestamp", "")
            if ev_date != target_date:
                continue
            if not (from_t <= ev_ts <= to_t):
                continue
            if direction_filter and ev.get("direction") != direction_filter.upper():
                continue
            if source_filter and ev.get("source_mode") != source_filter:
                continue
            filtered.append(ev)
        return filtered


# ── Production History Analytics Engine ───────────────────────────────────────
class ProductionHistoryEngine:
    """
    Aggregates raw count_events.csv crossing records into operational
    production-history analytics.

    Answers: "How many objects passed during period X?" and
    "How was production distributed over time?"

    Source of truth: results/count_events.csv (via CountEventLogger.read_events)
    No database required.
    """

    # Bucket-size rules (range_minutes → bucket_minutes)
    BUCKET_RULES: List[Tuple[float, int]] = [
        (30,   1),
        (120,  5),
        (360,  15),
        (1440, 30),
    ]
    DEFAULT_BUCKET = 60  # > 24 hours

    @staticmethod
    def _parse_datetime(dt_str: str, is_end: bool = False) -> Optional[datetime]:
        """
        Parse ISO-like datetime string or time string from query params.
        Accepts: 'YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DDTHH:MM',
                 'YYYY-MM-DD HH:MM:SS', 'YYYY-MM-DD HH:MM',
                 'YYYY-MM-DD', 'HH:MM:SS', 'HH:MM'.
        """
        if not dt_str:
            return None
        s = dt_str.strip()
        # Date only (YYYY-MM-DD): end of day if is_end, else start of day
        if len(s) == 10 and s.count("-") == 2:
            try:
                base = datetime.strptime(s, "%Y-%m-%d")
                return base.replace(hour=23, minute=59, second=59) if is_end else base
            except ValueError:
                pass

        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

        # Time-only (HH:MM:SS or HH:MM): bind to today's date
        today_date = datetime.now().date()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                t = datetime.strptime(s, fmt).time()
                return datetime.combine(today_date, t)
            except ValueError:
                continue

        return None

    @staticmethod
    def _event_to_datetime(ev: Dict[str, str]) -> Optional[datetime]:
        """Convert a CSV event row to a datetime object."""
        date_str = ev.get("date", "")
        ts_str = ev.get("timestamp", "")
        if not date_str or not ts_str:
            return None
        try:
            return datetime.strptime(f"{date_str} {ts_str}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def choose_bucket_size(self, range_minutes: float) -> int:
        """Return bucket size in minutes based on selected range."""
        for threshold, bucket in self.BUCKET_RULES:
            if range_minutes <= threshold:
                return bucket
        return self.DEFAULT_BUCKET

    def aggregate(
        self,
        events: List[Dict[str, str]],
        from_dt: datetime,
        to_dt: datetime,
        direction_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Filter and aggregate events into production-history summary.

        Returns:
            total_objects, down_count, up_count, average_rate,
            bucket_size_minutes, buckets, has_mixed_source,
            sources_present
        """
        # 1. Filter by datetime range + direction + source
        filtered: List[Dict[str, str]] = []
        for ev in events:
            ev_dt = self._event_to_datetime(ev)
            if ev_dt is None:
                continue
            if not (from_dt <= ev_dt <= to_dt):
                continue
            if direction_filter:
                if ev.get("direction", "").upper() != direction_filter.upper():
                    continue
            if source_filter:
                if ev.get("source_mode", "") != source_filter:
                    continue
            filtered.append(ev)

        # 2. Summary counts
        total = len(filtered)
        down_count = sum(1 for e in filtered if e.get("direction", "").upper() == "DOWN")
        up_count = sum(1 for e in filtered if e.get("direction", "").upper() == "UP")

        # 3. Average rate
        range_minutes = max((to_dt - from_dt).total_seconds() / 60.0, 0.001)
        average_rate = round(total / range_minutes, 2) if range_minutes > 0 else 0.0

        # 4. Source presence
        sources_present = list({e.get("source_mode", "unknown") for e in filtered})
        has_mixed_source = len(sources_present) > 1

        # 5. Time bucket aggregation
        bucket_size = self.choose_bucket_size(range_minutes)
        buckets = self._build_buckets(filtered, from_dt, to_dt, bucket_size)

        return {
            "total_objects": total,
            "down_count": down_count,
            "up_count": up_count,
            "average_rate": average_rate,
            "range_minutes": round(range_minutes, 1),
            "bucket_size_minutes": bucket_size,
            "buckets": buckets,
            "has_mixed_source": has_mixed_source,
            "sources_present": sources_present,
        }

    def _build_buckets(
        self,
        events: List[Dict[str, str]],
        from_dt: datetime,
        to_dt: datetime,
        bucket_minutes: int,
    ) -> List[Dict[str, Any]]:
        """Partition the time range into equal buckets and count events per bucket."""
        buckets: List[Dict[str, Any]] = []
        if bucket_minutes <= 0:
            return buckets

        bucket_delta = timedelta(minutes=bucket_minutes)
        cursor = from_dt

        while cursor < to_dt:
            bucket_end = cursor + bucket_delta
            count = sum(
                1 for ev in events
                if (ev_dt := self._event_to_datetime(ev)) and (
                    (cursor <= ev_dt < bucket_end) or (ev_dt == to_dt and cursor <= ev_dt <= bucket_end)
                )
            )
            buckets.append({
                "start": cursor.strftime("%Y-%m-%dT%H:%M:%S"),
                "end": bucket_end.strftime("%Y-%m-%dT%H:%M:%S"),
                "label": cursor.strftime("%H:%M"),
                "count": count,
            })
            cursor = bucket_end

        return buckets


production_history_engine = ProductionHistoryEngine()


# ── Global State & Locks ──────────────────────────────────────────────────────
class GlobalAnalyticsState:
    """Thread-safe state container for vision analytics pipeline."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.conf_thresh: float = 0.40
        self.line_position: float = 0.50  # 0.1 to 0.9 of frame height
        self.direction: str = "both"       # "down", "up", "both"
        self.feed_mode: str = "simulation" # "simulation", "camera"
        self.camera_index: int = 0
        self.tracking_enabled: bool = True
        self.counting_enabled: bool = True
        self.paused: bool = False

        # Metrics & Telemetry
        self.current_detections: int = 0
        self.total_counted: int = 0        # SESSION count — reset on Reset button
        self.active_tracks: int = 0        # Distinct track IDs visible this frame
        self.throughput_bpm: float = 0.0   # kept in telemetry/charts; removed from top KPI
        self.fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self.start_time: float = time.time()
        self.frame_number: int = 0

        # Track management
        self.track_history: Dict[int, deque] = {}
        self.counted_track_ids: set = set()
        self.events_log: deque = deque(maxlen=40)
        self.telemetry_history: deque = deque(maxlen=60)  # Last 60 data points

    def reset_counts(self) -> None:
        """
        Reset current SESSION counter only.
        Does NOT delete results/count_events.csv — historical events are preserved.
        """
        with self.lock:
            self.total_counted = 0
            self.start_time = time.time()
            self.counted_track_ids.clear()
            self.track_history.clear()
            self.active_tracks = 0
            self.events_log.clear()
            self.events_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "event": "Session counters reset by operator — historical CSV data preserved",
                "type": "info",
            })


state = GlobalAnalyticsState()
count_event_logger = CountEventLogger(COUNT_EVENTS_CSV)


# ── Synthetic Conveyor Belt Simulator ─────────────────────────────────────────
class SimulatedBottle:
    """Represents a virtual bottle moving on the conveyor belt."""

    def __init__(self, x: float, y: float, speed: float, width: int = 64, height: int = 140) -> None:
        self.x = x
        self.y = y
        self.speed = speed
        self.width = width
        self.height = height
        self.track_id: int = random.randint(1000, 9999)
        self.cap_color = random.choice([
            (210, 80, 20),   # Industrial Blue (BGR)
            (30, 30, 200),   # Industrial Red (BGR)
            (20, 180, 50),   # Green (BGR)
        ])
        self.fill_level = random.uniform(0.85, 0.96)


class ConveyorSimulator:
    """Generates realistic industrial conveyor video frames with moving bottles."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.w = width
        self.h = height
        self.bottles: List[SimulatedBottle] = []
        self.last_spawn_time = time.time()
        self.belt_offset: float = 0.0
        self.next_id: int = 101

    def update_and_render(self, speed_mult: float = 1.0) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Update physics and render frame."""
        now = time.time()
        # Create base industrial conveyor background
        frame = np.full((self.h, self.w, 3), 42, dtype=np.uint8)

        # Draw stainless steel conveyor bed (center region)
        bed_left = 140
        bed_right = self.w - 140
        bed_w = bed_right - bed_left
        frame[:, bed_left:bed_right] = (62, 66, 72)

        # Draw moving conveyor belt slats / rollers
        self.belt_offset = (self.belt_offset + 3.2 * speed_mult) % 40
        for y_line in range(int(self.belt_offset) - 40, self.h + 40, 40):
            if 0 <= y_line < self.h:
                cv2.line(frame, (bed_left, y_line), (bed_right, y_line), (50, 53, 58), 2)
                cv2.line(frame, (bed_left, y_line + 1), (bed_right, y_line + 1), (74, 78, 85), 1)

        # Draw left and right guide rails with metallic gradient
        cv2.rectangle(frame, (bed_left - 24, 0), (bed_left, self.h), (110, 115, 125), -1)
        cv2.rectangle(frame, (bed_left - 18, 0), (bed_left - 6, self.h), (160, 168, 180), -1)
        cv2.rectangle(frame, (bed_right, 0), (bed_right + 24, self.h), (110, 115, 125), -1)
        cv2.rectangle(frame, (bed_right + 6, 0), (bed_right + 18, self.h), (160, 168, 180), -1)

        # Outer factory floor texture
        frame[:, : bed_left - 24] = (30, 32, 35)
        frame[:, bed_right + 24 :] = (30, 32, 35)

        # Spawn new bottles at top of conveyor belt
        if now - self.last_spawn_time > random.uniform(1.1 / speed_mult, 2.2 / speed_mult):
            x_pos = random.uniform(bed_left + 70, bed_right - 130)
            speed = random.uniform(3.4, 4.2) * speed_mult
            b = SimulatedBottle(x_pos, -150.0, speed)
            b.track_id = self.next_id
            self.next_id += 1
            self.bottles.append(b)
            self.last_spawn_time = now

        # Update bottle positions & render
        detections: List[Dict[str, Any]] = []
        surviving_bottles: List[SimulatedBottle] = []

        for b in self.bottles:
            b.y += b.speed
            bx = int(b.x)
            by = int(b.y)
            bw = b.width
            bh = b.height

            # Discard bottles that have scrolled off frame
            if by > self.h + 20:
                continue
            surviving_bottles.append(b)

            # Only draw/detect if partially on screen
            if by + bh < 0:
                continue

            # --- Bottle rendering ---
            # 1. Body (rounded rectangle look)
            body_x1 = bx
            body_y1 = max(0, by + int(bh * 0.1))
            body_x2 = bx + bw
            body_y2 = min(self.h, by + int(bh * 0.92))
            fill_y2 = body_y2 - int((body_y2 - body_y1) * (1 - b.fill_level))
            cv2.rectangle(frame, (body_x1, fill_y2), (body_x2, body_y2), (215, 230, 245), -1)
            cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), (185, 200, 220), -1)
            cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), (150, 165, 185), 2)

            # 2. Label area highlight
            label_y1 = body_y1 + int((body_y2 - body_y1) * 0.3)
            label_y2 = body_y1 + int((body_y2 - body_y1) * 0.65)
            cv2.rectangle(frame, (body_x1 + 4, label_y1), (body_x2 - 4, label_y2), (235, 242, 252), -1)

            # 3. Fill level indicator
            cv2.rectangle(frame, (body_x1, fill_y2), (body_x2, body_y2), (40, 140, 200), -1)
            cv2.rectangle(frame, (body_x1, fill_y2), (body_x2, body_y2), (30, 115, 175), 1)

            # 4. Shoulder
            neck_w = int(bw * 0.38)
            neck_x = bx + (bw - neck_w) // 2
            neck_top = max(0, by + int(bh * 0.08))
            shoulder_pts = np.array([
                [body_x1, body_y1],
                [body_x2, body_y1],
                [neck_x + neck_w, neck_top + 10],
                [neck_x, neck_top + 10]
            ], np.int32)
            cv2.fillPoly(frame, [shoulder_pts], (230, 234, 238))
            cv2.polylines(frame, [shoulder_pts], True, (185, 192, 202), 2)

            cv2.rectangle(frame, (neck_x, neck_top), (neck_x + neck_w, neck_top + 12), (235, 238, 242), -1)

            # 5. Colored Cap
            cap_h = int(bh * 0.08)
            cv2.rectangle(frame, (neck_x - 2, by), (neck_x + neck_w + 2, by + cap_h), b.cap_color, -1)
            cv2.rectangle(frame, (neck_x - 2, by), (neck_x + neck_w + 2, by + cap_h), (20, 20, 20), 1)

            # Detection metadata
            cx = bx + bw // 2
            cy = by + bh // 2
            detections.append({
                "track_id": b.track_id,
                "conf": random.uniform(0.89, 0.96),
                "cls": 0,
                "bbox": (bx, max(0, by), bx + bw, min(self.h, by + bh)),
                "center": (cx, cy),
            })

        self.bottles = surviving_bottles
        return frame, detections


# ── YOLO Model Pipeline ───────────────────────────────────────────────────────
class VisionPipelineManager:
    """Manages model inference and fallback behavior."""

    def __init__(self) -> None:
        self.model = None
        self.device = "cpu"
        self.model_loaded = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from ultralytics import YOLO
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            if WEIGHTS_PATH.exists():
                print(f"[VisionEngine] Loading YOLOv8 weights from {WEIGHTS_PATH} on {self.device}...")
                self.model = YOLO(str(WEIGHTS_PATH))
                self.model_loaded = True
            else:
                print(f"[VisionEngine] Baseline model weights are not included because redistribution permission is not confirmed. Simulator mode is active.")
                self.model = None
                self.model_loaded = False
        except Exception as e:
            print(f"[VisionEngine] Could not load YOLOv8: {e}. Falling back to simulator mode.")
            self.model = None
            self.model_loaded = False

    def infer_camera(
        self,
        frame: np.ndarray,
        conf_thresh: float,
        tracking_enabled: bool = True,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run YOLO inference or ByteTrack tracking on camera frame."""
        t0 = time.time()
        detections: List[Dict[str, Any]] = []
        if self.model is not None:
            try:
                if tracking_enabled:
                    results = self.model.track(
                        source=frame,
                        conf=conf_thresh,
                        device=self.device,
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = self.model.predict(
                        source=frame,
                        conf=conf_thresh,
                        device=self.device,
                        verbose=False,
                    )
                latency_ms = (time.time() - t0) * 1000.0
                if len(results) > 0 and results[0].boxes is not None:
                    for i, box in enumerate(results[0].boxes):
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        track_id = (
                            int(box.id[0])
                            if (tracking_enabled and hasattr(box, "id") and box.id is not None)
                            else None
                        )
                        detections.append({
                            "track_id": track_id,
                            "conf": conf,
                            "cls": cls_id,
                            "bbox": (x1, y1, x2, y2),
                            "center": (cx, cy),
                        })
                return detections, latency_ms
            except Exception as e:
                print(f"[VisionEngine] Inference error: {e}")
        latency_ms = (time.time() - t0) * 1000.0
        return detections, latency_ms


pipeline_manager = VisionPipelineManager()
simulator = ConveyorSimulator()


# ── Frame Processing & Video Streaming ─────────────────────────────────────────
def process_and_draw_frame(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """Overlay tracking, HUD, counting line, and analytics on frame."""
    h, w = frame.shape[:2]
    line_y = int(state.line_position * h)
    curr_time = time.time()
    elapsed = curr_time - state.start_time

    # Determine source mode for event logging
    source_mode = "simulator" if state.feed_mode == "simulation" else "live_camera"

    # Update counts and track trajectories
    active_in_frame = 0
    active_track_ids_this_frame: set = set()

    with state.lock:
        conf_thresh = state.conf_thresh
        counting_enabled = state.counting_enabled
        tracking_enabled = state.tracking_enabled
        direction = state.direction

        for det in detections:
            if det["conf"] < conf_thresh:
                continue
            active_in_frame += 1
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["center"]
            tid = det["track_id"]

            if tid is not None:
                active_track_ids_this_frame.add(tid)

            # Line crossing check
            if counting_enabled and tid is not None:
                if tid in state.track_history and len(state.track_history[tid]) > 0:
                    prev_cx, prev_cy = state.track_history[tid][-1]
                    crossed, cross_dir = check_line_crossing(
                        prev_cy, cy, line_y, direction=direction
                    )

                    if crossed and tid not in state.counted_track_ids:
                        state.counted_track_ids.add(tid)
                        state.total_counted += 1
                        state.events_log.appendleft({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "event": f"Bottle #{tid} crossed {cross_dir} (conf: {det['conf']:.0%})",
                            "type": "count",
                            "track_id": tid,
                            "direction": cross_dir,
                            "total": state.total_counted,
                            "source_mode": source_mode,
                        })
                        # Log to persistent CSV — one row per crossing event
                        count_event_logger.log_event(
                            track_id=tid,
                            direction=cross_dir,
                            confidence=det["conf"],
                            cumulative_count=state.total_counted,
                            source_mode=source_mode,
                            line_position=state.line_position,
                            frame_number=state.frame_number,
                        )

                # Maintain history deque
                if tid not in state.track_history:
                    state.track_history[tid] = deque(maxlen=24)
                state.track_history[tid].append((cx, cy))

            # Draw visual elements
            # 1. Bounding Box (Industrial Cyan / Emerald Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 115), 2)

            # 2. Tracking Trajectory Trail
            if tracking_enabled and tid is not None and tid in state.track_history:
                pts = list(state.track_history[tid])
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    thickness = int(1 + alpha * 2)
                    cv2.line(frame, pts[k - 1], pts[k], (0, int(200 * alpha), 255), thickness)

            # 3. Center tracking dot
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # 4. Label Badge
            badge_label = f"ID #{tid}  {det['conf']:.0%}" if (tracking_enabled and tid) else f"Milk Bottle {det['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, max(0, y1)), (0, 170, 85), -1)
            cv2.putText(frame, badge_label, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw virtual counting line (Glowing Gold)
        if counting_enabled:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 215, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (24, line_y), 4, (0, 215, 255), -1)
            cv2.circle(frame, (w - 24, line_y), 4, (0, 215, 255), -1)
            line_str = f"VIRTUAL COUNTING LINE (Y={line_y}) [{direction.upper()}]"
            cv2.putText(frame, line_str, (16, max(18, line_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        # Update telemetry state
        state.current_detections = active_in_frame
        state.active_tracks = len(active_track_ids_this_frame)
        if elapsed > 1.0 and state.total_counted > 0:
            state.throughput_bpm = (state.total_counted / (elapsed / 60.0))
        else:
            state.throughput_bpm = 0.0

        # Snapshot telemetry point
        state.telemetry_history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "detections": active_in_frame,
            "counted": state.total_counted,
            "throughput": round(state.throughput_bpm, 1),
            "fps": round(state.fps, 1),
            "active_tracks": state.active_tracks,
        })

    return frame


def generate_video_stream() -> Generator[bytes, None, None]:
    """Yield MJPEG stream for web display."""
    fps_deque = deque(maxlen=20)
    last_frame_t = time.time()
    camera_cap: Optional[cv2.VideoCapture] = None

    while True:
        loop_start = time.time()
        fps_deque.append(1.0 / max(0.001, loop_start - last_frame_t))
        last_frame_t = loop_start
        state.fps = sum(fps_deque) / len(fps_deque)
        state.frame_number += 1

        if state.feed_mode == "camera":
            if camera_cap is None or not camera_cap.isOpened():
                camera_cap = cv2.VideoCapture(state.camera_index)
            success, raw_frame = camera_cap.read()
            if not success:
                raw_frame = np.full((600, 800, 3), 24, dtype=np.uint8)
                cv2.putText(raw_frame, "LIVE CAMERA 0 NOT DETECTED", (150, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2)
                cv2.putText(raw_frame, "Switch to Conveyor Sim or connect camera", (180, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                detections = []
                state.inference_latency_ms = 0.0
            else:
                raw_frame = cv2.resize(raw_frame, (800, 600))
                if pipeline_manager.model_loaded:
                    detections, lat = pipeline_manager.infer_camera(
                        raw_frame,
                        state.conf_thresh,
                        tracking_enabled=state.tracking_enabled,
                    )
                    state.inference_latency_ms = lat
                else:
                    cv2.rectangle(raw_frame, (20, 20), (780, 90), (10, 15, 25), -1)
                    cv2.rectangle(raw_frame, (20, 20), (780, 90), (0, 140, 255), 2)
                    cv2.putText(raw_frame, "MODEL WEIGHTS NOT INCLUDED IN PUBLIC REPO", (36, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
                    cv2.putText(raw_frame, "Redistribution permission unconfirmed. Switch to Conveyor Sim or supply local weights.", (36, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                    detections = []
                    state.inference_latency_ms = 0.0
        else:
            if camera_cap is not None:
                camera_cap.release()
                camera_cap = None
            raw_frame, detections = simulator.update_and_render()
            state.inference_latency_ms = 11.2  # Nominal baseline YOLOv8n inference latency target (CPU)

        processed_frame = process_and_draw_frame(raw_frame, detections)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.025)  # Cap at ~40 FPS for efficient streaming


# ── Web Routes & REST Endpoints ───────────────────────────────────────────────
@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/telemetry")
def get_telemetry():
    """Return JSON telemetry for HUD and real-time graphs."""
    with state.lock:
        is_sim = (state.feed_mode == "simulation")
        if is_sim:
            status_text = "SIMULATOR ACTIVE"
            mode_type = "SIMULATED"
        else:
            if pipeline_manager.model_loaded:
                status_text = "CAMERA ACTIVE"
                mode_type = "MEASURED LOCALLY"
            else:
                status_text = "MODEL UNAVAILABLE"
                mode_type = "STANDBY"

        data = {
            "current_detections": state.current_detections,
            "total_counted": state.total_counted,
            "active_tracks": state.active_tracks,
            "throughput_bpm": round(state.throughput_bpm, 1),
            "fps": round(state.fps, 1),
            "inference_latency_ms": round(state.inference_latency_ms, 1),
            "conf_thresh": state.conf_thresh,
            "line_position": state.line_position,
            "direction": state.direction,
            "feed_mode": state.feed_mode,
            "tracking_enabled": state.tracking_enabled,
            "counting_enabled": state.counting_enabled,
            "elapsed_seconds": round(time.time() - state.start_time, 1),
            "events": list(state.events_log),
            "history": list(state.telemetry_history),
            "system_status": status_text,
            "mode_type": mode_type,
            "is_simulated": is_sim,
            "model_loaded": pipeline_manager.model_loaded,
        }
    return jsonify(data)


@app.route("/api/count_history")
def get_count_history():
    """
    Return persisted count events for the COUNT HISTORY dashboard section.

    Query params:
        from_time   HH:MM or HH:MM:SS  (default: session start)
        to_time     HH:MM or HH:MM:SS  (default: now)
        date        YYYY-MM-DD         (default: today)
        direction   DOWN|UP            (default: all)
        source      simulator|live_camera (default: all)
        limit       int                (default: 100, max 500)
    """
    from_time = request.args.get("from_time", "00:00:00")
    to_time = request.args.get("to_time", datetime.now().strftime("%H:%M:%S"))
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    direction_filter = request.args.get("direction", None)
    source_filter = request.args.get("source", None)
    limit = min(int(request.args.get("limit", 100)), 500)

    events = count_event_logger.query_window(
        from_time=from_time,
        to_time=to_time,
        date=date_str,
        direction_filter=direction_filter,
        source_filter=source_filter,
    )

    # Most-recent first for the UI table
    events_desc = list(reversed(events))

    return jsonify({
        "events": events_desc[:limit],
        "total_in_window": len(events),
        "from_time": from_time,
        "to_time": to_time,
        "date": date_str,
        "direction_filter": direction_filter,
        "source_filter": source_filter,
    })


@app.route("/api/count_history/all")
def get_all_count_events():
    """Return all stored count events (most recent first), up to limit."""
    limit = min(int(request.args.get("limit", 200)), 1000)
    events = count_event_logger.read_events()
    events_desc = list(reversed(events))
    return jsonify({
        "events": events_desc[:limit],
        "total_stored": len(events),
    })


@app.route("/api/production_history")
def get_production_history():
    """
    Production History analytics endpoint.
    Aggregates raw crossing events from count_events.csv into operational summaries.

    Query params:
        from        ISO datetime  YYYY-MM-DDTHH:MM or YYYY-MM-DD HH:MM (default: today 00:00)
        to          ISO datetime  YYYY-MM-DDTHH:MM or YYYY-MM-DD HH:MM (default: now)
        direction   DOWN|UP       (default: all)
        source      simulator|live_camera (default: all)
        include_events  0|1       include raw event rows in response (default: 0)
        event_limit int           max raw events to return (default: 200)

    Returns:
        {
          "total_objects": int,
          "down_count": int,
          "up_count": int,
          "average_rate": float,           # objects per minute over selected range
          "range_minutes": float,
          "bucket_size_minutes": int,
          "buckets": [{"start", "end", "label", "count"}, ...],
          "has_mixed_source": bool,
          "sources_present": [str],
          "from": str,
          "to": str,
          "direction_filter": str|null,
          "source_filter": str|null,
          "events": [...]                   # only if include_events=1
        }
    """
    preset = request.args.get("preset", "").strip().lower()
    now_dt = datetime.now()
    today_start_dt = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    direction_filter = request.args.get("direction", None) or None
    source_filter = request.args.get("source", None) or None
    include_events = request.args.get("include_events", "0") == "1"
    event_limit = min(int(request.args.get("event_limit", 200)), 1000)

    # Resolve from_dt and to_dt based on preset or explicit query params
    if preset == "today":
        from_dt = today_start_dt
        to_dt = now_dt
    elif preset in ("30m", "last_30_min", "last30m"):
        from_dt = now_dt - timedelta(minutes=30)
        to_dt = now_dt
    elif preset in ("1h", "last_1_hr", "last1h"):
        from_dt = now_dt - timedelta(hours=1)
        to_dt = now_dt
    else:
        from_str = request.args.get("from")
        to_str = request.args.get("to")
        from_dt = ProductionHistoryEngine._parse_datetime(from_str, is_end=False) if from_str else today_start_dt
        to_dt = ProductionHistoryEngine._parse_datetime(to_str, is_end=True) if to_str else now_dt

    if from_dt is None:
        return jsonify({"error": f"Cannot parse 'from' datetime: {request.args.get('from')!r}"}), 400
    if to_dt is None:
        return jsonify({"error": f"Cannot parse 'to' datetime: {request.args.get('to')!r}"}), 400

    if from_dt > to_dt:
        from_dt, to_dt = to_dt, from_dt

    # Load all events from CSV
    all_events = count_event_logger.read_events()

    # Run aggregation
    result = production_history_engine.aggregate(
        events=all_events,
        from_dt=from_dt,
        to_dt=to_dt,
        direction_filter=direction_filter,
        source_filter=source_filter,
    )

    result["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S")
    result["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%S")
    result["direction_filter"] = direction_filter
    result["source_filter"] = source_filter

    # Optionally include raw events (most-recent first)
    if include_events:
        filtered_events = [
            ev for ev in all_events
            if _event_in_range(ev, from_dt, to_dt, direction_filter, source_filter)
        ]
        result["events"] = list(reversed(filtered_events))[:event_limit]

    return jsonify(result)


def _event_in_range(
    ev: Dict[str, str],
    from_dt: datetime,
    to_dt: datetime,
    direction_filter: Optional[str],
    source_filter: Optional[str],
) -> bool:
    """Helper: return True if event falls within range and passes filters."""
    ev_dt = ProductionHistoryEngine._event_to_datetime(ev)
    if ev_dt is None:
        return False
    if not (from_dt <= ev_dt <= to_dt):
        return False
    if direction_filter and ev.get("direction", "").upper() != direction_filter.upper():
        return False
    if source_filter and ev.get("source_mode", "") != source_filter:
        return False
    return True



@app.route("/api/config", methods=["POST"])
def update_config():
    """Update runtime parameters."""
    req_data = request.get_json(force=True)
    with state.lock:
        if "conf_thresh" in req_data:
            state.conf_thresh = max(0.05, min(0.95, float(req_data["conf_thresh"])))
        if "line_position" in req_data:
            state.line_position = max(0.10, min(0.90, float(req_data["line_position"])))
        if "direction" in req_data:
            state.direction = str(req_data["direction"])
        if "feed_mode" in req_data:
            state.feed_mode = str(req_data["feed_mode"])
        if "tracking_enabled" in req_data:
            state.tracking_enabled = bool(req_data["tracking_enabled"])
        if "counting_enabled" in req_data:
            state.counting_enabled = bool(req_data["counting_enabled"])
    return jsonify({"status": "success", "message": "Configuration updated"})


@app.route("/api/reset", methods=["POST"])
def reset_counts():
    """
    Reset current SESSION counters only.
    Historical count_events.csv data is NOT deleted.
    """
    state.reset_counts()
    return jsonify({
        "status": "success",
        "message": "Session counters reset — historical CSV data preserved",
    })


@app.route("/api/metrics_image/<filename>")
def serve_metrics_image(filename: str):
    """Safely serve audited evaluation plots."""
    allowed_images = {
        "BoxPR_curve.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "BoxPR_curve.png",
        "BoxF1_curve.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "BoxF1_curve.png",
        "BoxP_curve.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "BoxP_curve.png",
        "BoxR_curve.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "BoxR_curve.png",
        "confusion_matrix.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "confusion_matrix.png",
        "confusion_matrix_normalized.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "confusion_matrix_normalized.png",
        "results.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "results.png",
        "training_graphs.png": BASE_DIR / "runs" / "detect" / "milk_bottle" / "training_graphs.png",
        "yolov8n_architecture.png": BASE_DIR / "yolov8n_architecture.png",
    }
    if filename in allowed_images and allowed_images[filename].exists():
        return send_file(str(allowed_images[filename]), mimetype="image/png")
    return jsonify({"error": "File not found or not permitted"}), 404


@app.route("/api/docs/<doc_name>")
def serve_doc(doc_name: str):
    """Return raw markdown documentation for browser rendering."""
    doc_map = {
        "readme": BASE_DIR / "README.md",
        "experiments": BASE_DIR / "EXPERIMENTS.md",
        "architecture": BASE_DIR / "docs" / "architecture.md",
        "evaluation": BASE_DIR / "docs" / "evaluation.md",
    }
    if doc_name in doc_map and doc_map[doc_name].exists():
        content = doc_map[doc_name].read_text(encoding="utf-8", errors="replace")
        return jsonify({"content": content})
    return jsonify({"error": "Documentation file not found"}), 404


if __name__ == "__main__":
    port = 8080
    print(f"\n=======================================================")
    print(f"🚀  Industrial Conveyor Vision Analytics Web Suite")
    print(f"📡  Running on: http://localhost:{port}")
    print(f"📋  Count events log: {COUNT_EVENTS_CSV}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
