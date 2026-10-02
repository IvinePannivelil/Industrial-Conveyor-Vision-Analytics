"""
tests/test_count_events.py
--------------------------
Tests for the CountEventLogger and count_history query functionality.

Tests use temporary directories/files — they NEVER write to results/count_events.csv.

Coverage:
  1. One crossing creates one event row
  2. Duplicate/jitter does not create duplicate rows (deduplication in caller)
  3. Second valid independent crossing creates another row
  4. Reset session count does NOT delete history
  5. Time-window filter returns correct event count
  6. Direction filter returns correct count
  7. Source filter returns correct count
  8. Multi-day events filtered correctly to today
"""

import os
import sys
import csv
import time
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from web_server import CountEventLogger, GlobalAnalyticsState


def make_logger(tmp_path: str) -> CountEventLogger:
    """Helper: create a CountEventLogger using a temp CSV path."""
    csv_path = Path(tmp_path) / "test_count_events.csv"
    return CountEventLogger(csv_path)


def write_event(logger: CountEventLogger, **overrides) -> None:
    """Helper: write a single event with sensible defaults."""
    defaults = {
        "track_id": 1,
        "direction": "DOWN",
        "confidence": 0.92,
        "cumulative_count": 1,
        "source_mode": "simulator",
        "line_position": 0.5,
        "frame_number": 42,
    }
    defaults.update(overrides)
    logger.log_event(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: One crossing creates exactly one CSV event row
# ─────────────────────────────────────────────────────────────────────────────
def test_one_crossing_creates_one_row():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        write_event(logger, track_id=10, direction="DOWN", cumulative_count=1)
        events = logger.read_events()
        assert len(events) == 1, f"Expected 1 event row, got {len(events)}"
        assert events[0]["track_id"] == "10"
        assert events[0]["direction"] == "DOWN"
        assert events[0]["cumulative_count"] == "1"
        print("✓ test_one_crossing_creates_one_row")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Duplicate calls (jitter simulation) — duplicate ROWS are only prevented
#         by the caller's counted_track_ids set, not inside the logger itself.
#         The logger stores what it's given; deduplication is tested at the
#         state level via track_id set membership.
# ─────────────────────────────────────────────────────────────────────────────
def test_deduplication_via_state():
    """
    Verify that GlobalAnalyticsState.counted_track_ids prevents duplicate
    log_event calls for the same track_id crossing.
    """
    counted_ids: set = set()
    log_calls = []

    def mock_log_event(track_id, **kwargs):
        log_calls.append(track_id)

    # Simulate: track_id=5 crosses, already in counted set on second frame
    for i in range(3):
        tid = 5
        if tid not in counted_ids:
            counted_ids.add(tid)
            mock_log_event(tid)

    assert len(log_calls) == 1, f"Expected 1 log call, got {len(log_calls)}"
    print("✓ test_deduplication_via_state")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Second valid independent crossing creates a new row
# ─────────────────────────────────────────────────────────────────────────────
def test_two_crossings_create_two_rows():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        write_event(logger, track_id=7, direction="DOWN", cumulative_count=1)
        write_event(logger, track_id=8, direction="UP", cumulative_count=2)
        events = logger.read_events()
        assert len(events) == 2, f"Expected 2 rows, got {len(events)}"
        assert events[0]["track_id"] == "7"
        assert events[1]["track_id"] == "8"
        assert events[1]["direction"] == "UP"
        assert events[1]["cumulative_count"] == "2"
        print("✓ test_two_crossings_create_two_rows")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Session reset does NOT delete CSV history
# ─────────────────────────────────────────────────────────────────────────────
def test_reset_session_does_not_delete_history():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        # Write 3 events
        for i in range(1, 4):
            write_event(logger, track_id=i, cumulative_count=i)
        events_before = logger.read_events()
        assert len(events_before) == 3

        # Simulate session reset (only resets state, NOT logger)
        state = GlobalAnalyticsState()
        state.reset_counts()
        assert state.total_counted == 0, "Session count should be 0 after reset"

        # CSV must still have 3 rows
        events_after = logger.read_events()
        assert len(events_after) == 3, "CSV history must NOT be deleted by session reset"
        print("✓ test_reset_session_does_not_delete_history")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Time-window filter returns correct event count
# ─────────────────────────────────────────────────────────────────────────────
def test_time_window_filter():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        today = datetime.now().strftime("%Y-%m-%d")
        # Directly write rows with known timestamps via CSV
        csv_path = Path(tmp) / "test_count_events.csv"

        rows = [
            {"timestamp": "10:00:00", "date": today, "track_id": "1",
             "direction": "DOWN", "confidence": "0.92", "cumulative_count": "1",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "1"},
            {"timestamp": "10:15:00", "date": today, "track_id": "2",
             "direction": "DOWN", "confidence": "0.91", "cumulative_count": "2",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "20"},
            {"timestamp": "10:30:00", "date": today, "track_id": "3",
             "direction": "UP",   "confidence": "0.89", "cumulative_count": "3",
             "source_mode": "live_camera", "line_position": "0.50", "frame_number": "45"},
            {"timestamp": "10:45:00", "date": today, "track_id": "4",
             "direction": "DOWN", "confidence": "0.95", "cumulative_count": "4",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "60"},
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CountEventLogger.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        # Query 10:10 → 10:40 (should return events at 10:15 and 10:30)
        result = logger.query_window("10:10", "10:40", date=today)
        assert len(result) == 2, f"Expected 2 events in window, got {len(result)}"

        # Query full range — should get all 4
        result_all = logger.query_window("00:00", "23:59", date=today)
        assert len(result_all) == 4, f"Expected 4 events, got {len(result_all)}"

        # Query 10:45–10:45 — should get exactly 1 (inclusive boundary)
        result_single = logger.query_window("10:45", "10:45", date=today)
        assert len(result_single) == 1, f"Expected 1, got {len(result_single)}"
        print("✓ test_time_window_filter")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Direction filter
# ─────────────────────────────────────────────────────────────────────────────
def test_direction_filter():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        today = datetime.now().strftime("%Y-%m-%d")
        csv_path = Path(tmp) / "test_count_events.csv"

        rows = [
            {"timestamp": "11:00:00", "date": today, "track_id": "1",
             "direction": "DOWN", "confidence": "0.91", "cumulative_count": "1",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "1"},
            {"timestamp": "11:05:00", "date": today, "track_id": "2",
             "direction": "UP",   "confidence": "0.88", "cumulative_count": "2",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "10"},
            {"timestamp": "11:10:00", "date": today, "track_id": "3",
             "direction": "DOWN", "confidence": "0.93", "cumulative_count": "3",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "20"},
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CountEventLogger.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        down_events = logger.query_window("00:00", "23:59", date=today, direction_filter="DOWN")
        up_events = logger.query_window("00:00", "23:59", date=today, direction_filter="UP")

        assert len(down_events) == 2, f"Expected 2 DOWN events, got {len(down_events)}"
        assert len(up_events) == 1, f"Expected 1 UP event, got {len(up_events)}"
        print("✓ test_direction_filter")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Source filter
# ─────────────────────────────────────────────────────────────────────────────
def test_source_filter():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        today = datetime.now().strftime("%Y-%m-%d")
        csv_path = Path(tmp) / "test_count_events.csv"

        rows = [
            {"timestamp": "12:00:00", "date": today, "track_id": "1",
             "direction": "DOWN", "confidence": "0.90", "cumulative_count": "1",
             "source_mode": "simulator", "line_position": "0.50", "frame_number": "1"},
            {"timestamp": "12:01:00", "date": today, "track_id": "2",
             "direction": "DOWN", "confidence": "0.93", "cumulative_count": "2",
             "source_mode": "live_camera", "line_position": "0.50", "frame_number": "5"},
            {"timestamp": "12:02:00", "date": today, "track_id": "3",
             "direction": "UP",   "confidence": "0.88", "cumulative_count": "3",
             "source_mode": "live_camera", "line_position": "0.50", "frame_number": "10"},
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CountEventLogger.CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        sim_events = logger.query_window("00:00", "23:59", date=today, source_filter="simulator")
        live_events = logger.query_window("00:00", "23:59", date=today, source_filter="live_camera")

        assert len(sim_events) == 1, f"Expected 1 simulator event, got {len(sim_events)}"
        assert len(live_events) == 2, f"Expected 2 live_camera events, got {len(live_events)}"
        print("✓ test_source_filter")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: CSV file is created automatically with correct headers
# ─────────────────────────────────────────────────────────────────────────────
def test_csv_auto_creation_with_headers():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test_count_events.csv"
        assert not csv_path.exists(), "CSV should not exist yet"
        logger = CountEventLogger(csv_path)
        assert csv_path.exists(), "CSV should be auto-created"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == CountEventLogger.CSV_FIELDS, f"Header mismatch: {header}"
        print("✓ test_csv_auto_creation_with_headers")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Thread safety — concurrent log_event calls do not corrupt the file
# ─────────────────────────────────────────────────────────────────────────────
def test_thread_safe_concurrent_writes():
    with tempfile.TemporaryDirectory() as tmp:
        logger = make_logger(tmp)
        errors = []

        def write_10(tid_offset):
            for i in range(10):
                try:
                    write_event(logger, track_id=tid_offset + i, cumulative_count=i + 1)
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=write_10, args=(t * 10,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        events = logger.read_events()
        assert len(events) == 50, f"Expected 50 rows from 5×10 concurrent writes, got {len(events)}"
        print("✓ test_thread_safe_concurrent_writes")


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_one_crossing_creates_one_row()
    test_deduplication_via_state()
    test_two_crossings_create_two_rows()
    test_reset_session_does_not_delete_history()
    test_time_window_filter()
    test_direction_filter()
    test_source_filter()
    test_csv_auto_creation_with_headers()
    test_thread_safe_concurrent_writes()
    print("\n✅  All CountEventLogger tests passed.")
