# Results Directory

This directory holds runtime-generated outputs and training benchmark summaries for the
Industrial Conveyor Vision Analytics project.

---

## Runtime-Generated Files (Git-ignored)

All CSV files in this directory are excluded from version control via `.gitignore`.
They are generated locally during a running session and contain no proprietary imagery.

| File | Description |
|------|-------------|
| `count_events.csv` | Persistent event log — one row per valid line-crossing |
| `live_run.csv` | Per-frame telemetry from `detect_live.py` CLI sessions |

---

## `count_events.csv` — Count Event Log

**Purpose:** Record every valid virtual-line crossing event with operational metadata.
Enables time-window queries: *"how many objects crossed between 10:15 and 10:45?"*

**Lifecycle:**
- Created automatically on first server startup if missing.
- Appended to on every crossing event during simulation or live camera mode.
- **NOT deleted by session Reset** — `Reset Counts` in the dashboard resets only the
  in-session counter (OBJECTS COUNTED). CSV history accumulates across resets.
- Excluded from Git (`results/*.csv` in `.gitignore`).

**Schema:**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | HH:MM:SS | Time of the crossing event |
| `date` | YYYY-MM-DD | Date (for multi-day filtering) |
| `track_id` | integer | ByteTrack track ID (simulator or live) |
| `direction` | DOWN \| UP | Crossing direction |
| `confidence` | float (0–1) | Detection confidence at moment of crossing |
| `cumulative_count` | integer | Running count at time of this event |
| `source_mode` | simulator \| live_camera | Pipeline mode at crossing time |
| `line_position` | float (0–1) | Virtual counting line Y fraction |
| `frame_number` | integer | Frame index at crossing |

**Example row:**
```
timestamp,date,track_id,direction,confidence,cumulative_count,source_mode,line_position,frame_number
14:22:07,2026-10-02,103,DOWN,0.9231,5,simulator,0.500,412
```

**Privacy guarantee:** Only operational metadata is logged. No image frames, video,
faces, bounding box pixel coordinates, or personally identifiable information is stored.

---

## `count_events.csv` vs Frame Telemetry

These are **two distinct outputs** with different purposes:

| | `count_events.csv` | Frame telemetry (`live_run.csv`) |
|---|---|---|
| **Trigger** | One valid crossing event | Every frame |
| **Purpose** | Operational event history | Runtime diagnostics |
| **Query use** | "Objects in period A→B?" | "FPS at time X?" |
| **Rows/min** | ~4–8 (typical conveyor rate) | ~30–40 |
| **Source** | `web_server.py` CountEventLogger | `detect_live.py` CSVTelemetryLogger |

---

## Production History & Aggregation Layer

Raw computer-vision crossing events are stored locally in:
`results/count_events.csv`

The **Production History** analytics layer aggregates those raw events into selected time windows on-the-fly without requiring an external database.

This enables operational analysis answering:
- **Object count by period:** Total objects passed between any two timestamps (same-day, multi-hour, cross-midnight, multi-day)
- **Directional distribution:** UP vs. DOWN counts
- **Average production rate:** `selected_event_count / selected_window_duration_minutes` (e.g. 14.3 objects/min)
- **Time-bucket production trend:** Dynamic time-series bar chart showing production distribution over time

### Dynamic Time Bucketing Rules
The aggregation engine automatically selects optimal time buckets based on the requested window duration:
- ≤ 30 minutes → 1-minute buckets
- ≤ 2 hours → 5-minute buckets
- ≤ 6 hours → 15-minute buckets
- ≤ 24 hours → 30-minute buckets
- > 24 hours → 1-hour or daily buckets

### Backend Analytics API
```
GET /api/production_history?from=YYYY-MM-DDTHH:MM:SS&to=YYYY-MM-DDTHH:MM:SS&direction=both&source=all
```
Parameters:
- `from`: ISO datetime or HH:MM start timestamp
- `to`: ISO datetime or HH:MM end timestamp
- `direction`: `all`, `down`, or `up`
- `source`: `all`, `live_camera`, or `simulator`

Returns:
```json
{
  "total_objects": 357,
  "down_count": 349,
  "up_count": 8,
  "average_rate": 5.95,
  "bucket_size_minutes": 5,
  "range_minutes": 60.0,
  "sources_present": ["simulator"],
  "has_mixed_source": false,
  "buckets": [
    {
      "start": "2026-10-02T10:00:00",
      "end": "2026-10-02T10:05:00",
      "label": "10:00",
      "count": 27
    }
  ]
}
```

The raw event CSV remains the immutable source of truth, while detailed events (Track ID, Direction, Confidence, Timestamp) are available via an optional collapsed drill-down in the UI.

---

## Training Results Summary

All metrics below are derived from `runs/detect/milk_bottle/results.csv` logged over 100 training epochs.

| Metric | Final Epoch (100) | Peak Validation | Notes |
|:-------|:-----------------:|:---------------:|-------|
| **mAP @ 0.50** | **98.31%** | **98.38%** (Ep 94) | Near-perfect bottle detection under standard lighting |
| **mAP @ 0.50:0.95** | **63.89%** | **64.01%** (Ep 94) | Bounding box overlap across IoU thresholds |
| **Precision** | **92.07%** | **94.81%** (Ep 92) | High true-positive confidence |
| **Recall** | **96.56%** | **97.57%** (Ep 88) | Misses fewer than 3.5% of visible conveyor bottles |

> **Evaluation caveat:** Baseline validation metrics may be optimistic because temporally
> adjacent video frames were split across training and validation sets (frame-level, not
> sequence-level split). Calibrated results target background false-positive suppression only.

---

## Training Configuration

Logged in `runs/detect/milk_bottle/args.yaml`:
- **Optimizer:** SGD (`lr0=0.01`, `momentum=0.937`, `weight_decay=0.0005`)
- **Epochs:** 100 with Early Stopping patience of 20
- **Batch Size:** 16
- **Input Resolution:** 640×640
- **Augmentation:** Mosaic (1.0), HSV jitter, Rotation (5°), Translation (0.1), H-Flip (0.5)
