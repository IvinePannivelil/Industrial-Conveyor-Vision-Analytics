"""
tests/test_production_history.py
---------------------------------
Tests for ProductionHistoryEngine aggregation logic.

All tests use temporary CSV files — never touch results/count_events.csv.

Coverage (all 10 required):
  1.  60-minute range returns correct total
  2.  UP/DOWN counts are correct
  3.  Average rate is correct
  4.  Source filter works
  5.  Direction filter works
  6.  5-minute bucket aggregation works
  7.  Cross-midnight query works
  8.  No-event range returns empty result
  9.  Reset does not affect historical data
  10. Simulator/live events remain distinguishable
"""

import sys
import csv
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from web_server import ProductionHistoryEngine, CountEventLogger, GlobalAnalyticsState

engine = ProductionHistoryEngine()


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_event(
    date: str,
    timestamp: str,
    track_id: str = "1",
    direction: str = "DOWN",
    confidence: str = "0.9200",
    cumulative_count: str = "1",
    source_mode: str = "simulator",
    line_position: str = "0.500",
    frame_number: str = "1",
) -> Dict[str, str]:
    return {
        "timestamp": timestamp,
        "date": date,
        "track_id": track_id,
        "direction": direction,
        "confidence": confidence,
        "cumulative_count": cumulative_count,
        "source_mode": source_mode,
        "line_position": line_position,
        "frame_number": frame_number,
    }


def write_events_csv(tmp_dir: str, events: List[Dict[str, str]]) -> CountEventLogger:
    """Write events to a temp CSV and return a logger bound to that file."""
    csv_path = Path(tmp_dir) / "test_prod_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CountEventLogger.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(events)
    return CountEventLogger(csv_path)


def dt(date: str, time_str: str) -> datetime:
    return datetime.strptime(f"{date} {time_str}", "%Y-%m-%d %H:%M:%S")


# ── Test 1: 60-minute range returns correct total ─────────────────────────────
def test_sixty_minute_range_correct_total():
    events = [
        make_event("2026-10-02", "10:00:00", track_id="1", direction="DOWN"),
        make_event("2026-10-02", "10:20:00", track_id="2", direction="DOWN"),
        make_event("2026-10-02", "10:45:00", track_id="3", direction="UP"),
        make_event("2026-10-02", "11:05:00", track_id="4", direction="DOWN"),  # outside
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "11:00:00")
    result = engine.aggregate(events, from_dt, to_dt)
    assert result["total_objects"] == 3, f"Expected 3, got {result['total_objects']}"
    print("✓ test_sixty_minute_range_correct_total")


# ── Test 2: UP/DOWN counts are correct ────────────────────────────────────────
def test_up_down_counts_correct():
    events = [
        make_event("2026-10-02", "10:00:00", direction="DOWN"),
        make_event("2026-10-02", "10:05:00", direction="DOWN"),
        make_event("2026-10-02", "10:10:00", direction="UP"),
        make_event("2026-10-02", "10:15:00", direction="DOWN"),
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "10:20:00")
    result = engine.aggregate(events, from_dt, to_dt)
    assert result["down_count"] == 3, f"Expected 3 DOWN, got {result['down_count']}"
    assert result["up_count"]   == 1, f"Expected 1 UP, got {result['up_count']}"
    assert result["total_objects"] == 4
    print("✓ test_up_down_counts_correct")


# ── Test 3: Average rate is correct ───────────────────────────────────────────
def test_average_rate_correct():
    # 30 events over 60 minutes → 0.5 objects/min
    events = [
        make_event("2026-10-02", f"{10 + i // 60:02d}:{i % 60:02d}:00", track_id=str(i))
        for i in range(30)
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "11:00:00")
    result = engine.aggregate(events, from_dt, to_dt)
    assert result["total_objects"] == 30
    expected_rate = round(30 / 60.0, 2)
    assert abs(result["average_rate"] - expected_rate) < 0.01, \
        f"Expected rate {expected_rate}, got {result['average_rate']}"
    print("✓ test_average_rate_correct")


# ── Test 4: Source filter works ────────────────────────────────────────────────
def test_source_filter_works():
    events = [
        make_event("2026-10-02", "10:00:00", source_mode="simulator", track_id="1"),
        make_event("2026-10-02", "10:05:00", source_mode="live_camera", track_id="2"),
        make_event("2026-10-02", "10:10:00", source_mode="live_camera", track_id="3"),
        make_event("2026-10-02", "10:15:00", source_mode="simulator", track_id="4"),
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "10:30:00")

    sim_result  = engine.aggregate(events, from_dt, to_dt, source_filter="simulator")
    live_result = engine.aggregate(events, from_dt, to_dt, source_filter="live_camera")
    all_result  = engine.aggregate(events, from_dt, to_dt)

    assert sim_result["total_objects"]  == 2, f"Sim: expected 2, got {sim_result['total_objects']}"
    assert live_result["total_objects"] == 2, f"Live: expected 2, got {live_result['total_objects']}"
    assert all_result["total_objects"]  == 4, f"All: expected 4, got {all_result['total_objects']}"
    assert all_result["has_mixed_source"] is True
    print("✓ test_source_filter_works")


# ── Test 5: Direction filter works ─────────────────────────────────────────────
def test_direction_filter_works():
    events = [
        make_event("2026-10-02", "10:00:00", direction="DOWN", track_id="1"),
        make_event("2026-10-02", "10:05:00", direction="UP",   track_id="2"),
        make_event("2026-10-02", "10:10:00", direction="DOWN", track_id="3"),
        make_event("2026-10-02", "10:15:00", direction="UP",   track_id="4"),
        make_event("2026-10-02", "10:20:00", direction="DOWN", track_id="5"),
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "10:30:00")

    down_result = engine.aggregate(events, from_dt, to_dt, direction_filter="DOWN")
    up_result   = engine.aggregate(events, from_dt, to_dt, direction_filter="UP")

    assert down_result["total_objects"] == 3, f"DOWN: expected 3, got {down_result['total_objects']}"
    assert up_result["total_objects"]   == 2, f"UP: expected 2, got {up_result['total_objects']}"
    assert down_result["up_count"]      == 0
    assert up_result["down_count"]      == 0
    print("✓ test_direction_filter_works")


# ── Test 6: 5-minute bucket aggregation works ─────────────────────────────────
def test_five_minute_bucket_aggregation():
    # 30-minute window → 5-minute buckets (BUCKET_RULES: ≤30 min → 1-min, ≤120 min → 5-min)
    # Use a 90-minute window to get 5-min buckets
    events = [
        make_event("2026-10-02", "10:00:00", track_id="1"),  # bucket 10:00
        make_event("2026-10-02", "10:01:00", track_id="2"),  # bucket 10:00
        make_event("2026-10-02", "10:05:00", track_id="3"),  # bucket 10:05
        make_event("2026-10-02", "10:06:00", track_id="4"),  # bucket 10:05
        make_event("2026-10-02", "10:07:00", track_id="5"),  # bucket 10:05
        make_event("2026-10-02", "10:15:00", track_id="6"),  # bucket 10:15
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "11:30:00")  # 90-minute window → 5-min buckets

    result = engine.aggregate(events, from_dt, to_dt)

    assert result["bucket_size_minutes"] == 5, \
        f"Expected 5-min buckets, got {result['bucket_size_minutes']}"

    # Check specific bucket counts
    bucket_map = {b["label"]: b["count"] for b in result["buckets"]}
    assert bucket_map.get("10:00", 0) == 2, f"Bucket 10:00: expected 2, got {bucket_map.get('10:00', 0)}"
    assert bucket_map.get("10:05", 0) == 3, f"Bucket 10:05: expected 3, got {bucket_map.get('10:05', 0)}"
    assert bucket_map.get("10:15", 0) == 1, f"Bucket 10:15: expected 1, got {bucket_map.get('10:15', 0)}"
    print("✓ test_five_minute_bucket_aggregation")


# ── Test 7: Cross-midnight query works ─────────────────────────────────────────
def test_cross_midnight_query():
    events = [
        make_event("2026-10-02", "23:50:00", track_id="1"),  # before midnight
        make_event("2026-10-02", "23:55:00", track_id="2"),  # before midnight
        make_event("2026-10-03", "00:05:00", track_id="3"),  # after midnight
        make_event("2026-10-03", "00:10:00", track_id="4"),  # after midnight
        make_event("2026-10-03", "00:20:00", track_id="5"),  # outside range
    ]
    from_dt = dt("2026-10-02", "23:45:00")
    to_dt   = dt("2026-10-03", "00:15:00")

    result = engine.aggregate(events, from_dt, to_dt)
    assert result["total_objects"] == 4, \
        f"Cross-midnight: expected 4 events, got {result['total_objects']}"
    print("✓ test_cross_midnight_query")


# ── Test 8: No-event range returns empty result ───────────────────────────────
def test_no_event_range_returns_empty():
    events = [
        make_event("2026-10-02", "09:00:00", track_id="1"),
        make_event("2026-10-02", "09:05:00", track_id="2"),
    ]
    # Query a completely different window
    from_dt = dt("2026-10-02", "14:00:00")
    to_dt   = dt("2026-10-02", "15:00:00")

    result = engine.aggregate(events, from_dt, to_dt)
    assert result["total_objects"] == 0, f"Expected 0, got {result['total_objects']}"
    assert result["down_count"]    == 0
    assert result["up_count"]      == 0
    assert result["average_rate"]  == 0.0
    assert all(b["count"] == 0 for b in result["buckets"]), "All buckets should be 0"
    print("✓ test_no_event_range_returns_empty")


# ── Test 9: Reset does not affect historical data ─────────────────────────────
def test_reset_does_not_affect_historical_data():
    with tempfile.TemporaryDirectory() as tmp:
        events = [
            make_event("2026-10-02", "10:00:00", track_id="1"),
            make_event("2026-10-02", "10:05:00", track_id="2"),
            make_event("2026-10-02", "10:10:00", track_id="3"),
        ]
        logger = write_events_csv(tmp, events)

        # Verify 3 events before reset
        all_before = logger.read_events()
        assert len(all_before) == 3

        # Simulate session reset
        state = GlobalAnalyticsState()
        state.reset_counts()
        assert state.total_counted == 0, "Session count should be 0"

        # CSV must still have 3 events
        all_after = logger.read_events()
        assert len(all_after) == 3, \
            f"CSV must have 3 events after reset, got {len(all_after)}"

        # Production history should still return 3
        from_dt = dt("2026-10-02", "10:00:00")
        to_dt   = dt("2026-10-02", "10:30:00")
        result = engine.aggregate(all_after, from_dt, to_dt)
        assert result["total_objects"] == 3
    print("✓ test_reset_does_not_affect_historical_data")


# ── Test 10: Simulator/live events remain distinguishable ─────────────────────
def test_simulator_and_live_events_distinguishable():
    events = [
        make_event("2026-10-02", "10:00:00", track_id="101", source_mode="simulator"),
        make_event("2026-10-02", "10:01:00", track_id="102", source_mode="simulator"),
        make_event("2026-10-02", "10:02:00", track_id="1",   source_mode="live_camera"),
        make_event("2026-10-02", "10:03:00", track_id="2",   source_mode="live_camera"),
        make_event("2026-10-02", "10:04:00", track_id="3",   source_mode="live_camera"),
    ]
    from_dt = dt("2026-10-02", "10:00:00")
    to_dt   = dt("2026-10-02", "10:10:00")

    # All combined
    combined = engine.aggregate(events, from_dt, to_dt)
    assert combined["total_objects"] == 5
    assert combined["has_mixed_source"] is True
    assert set(combined["sources_present"]) == {"simulator", "live_camera"}

    # Simulator only
    sim_only = engine.aggregate(events, from_dt, to_dt, source_filter="simulator")
    assert sim_only["total_objects"] == 2
    assert sim_only["has_mixed_source"] is False
    assert sim_only["sources_present"] == ["simulator"]

    # Live camera only
    live_only = engine.aggregate(events, from_dt, to_dt, source_filter="live_camera")
    assert live_only["total_objects"] == 3
    assert live_only["has_mixed_source"] is False
    assert live_only["sources_present"] == ["live_camera"]

    print("✓ test_simulator_and_live_events_distinguishable")


# ── Bonus: Bucket size selection rules ────────────────────────────────────────
def test_bucket_size_rules():
    assert engine.choose_bucket_size(15)   == 1,  "15-min → 1-min buckets"
    assert engine.choose_bucket_size(30)   == 1,  "30-min → 1-min buckets"
    assert engine.choose_bucket_size(60)   == 5,  "60-min → 5-min buckets"
    assert engine.choose_bucket_size(120)  == 5,  "120-min → 5-min buckets"
    assert engine.choose_bucket_size(200)  == 15, "200-min → 15-min buckets"
    assert engine.choose_bucket_size(360)  == 15, "360-min → 15-min buckets"
    assert engine.choose_bucket_size(720)  == 30, "720-min → 30-min buckets"
    assert engine.choose_bucket_size(1440) == 30, "1440-min → 30-min buckets"
    assert engine.choose_bucket_size(2000) == 60, ">24h → 60-min buckets"
    print("✓ test_bucket_size_rules")


# ── Bonus: Datetime parsing handles multiple formats ─────────────────────────
def test_datetime_parsing():
    assert ProductionHistoryEngine._parse_datetime("2026-10-02T10:30:00") == datetime(2026, 10, 2, 10, 30, 0)
    assert ProductionHistoryEngine._parse_datetime("2026-10-02T10:30")    == datetime(2026, 10, 2, 10, 30, 0)
    assert ProductionHistoryEngine._parse_datetime("2026-10-02 10:30:00") == datetime(2026, 10, 2, 10, 30, 0)
    assert ProductionHistoryEngine._parse_datetime("2026-10-02")          == datetime(2026, 10, 2, 0, 0, 0)
    assert ProductionHistoryEngine._parse_datetime("invalid")             is None
    print("✓ test_datetime_parsing")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sixty_minute_range_correct_total()
    test_up_down_counts_correct()
    test_average_rate_correct()
    test_source_filter_works()
    test_direction_filter_works()
    test_five_minute_bucket_aggregation()
    test_cross_midnight_query()
    test_no_event_range_returns_empty()
    test_reset_does_not_affect_historical_data()
    test_simulator_and_live_events_distinguishable()
    test_bucket_size_rules()
    test_datetime_parsing()
    print("\n✅  All ProductionHistoryEngine tests passed.")
