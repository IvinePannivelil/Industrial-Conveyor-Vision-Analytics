"""
tests/test_counter.py
---------------------
Unit tests for virtual counting line and tripwire logic in detect_live.py.

Covers:
  1. Single bottle crossing downwards increments count exactly once.
  2. Bottle jittering / oscillating back-and-forth across the tripwire line does NOT double-count.
  3. Bottle temporarily occluded (disappears and reappears) is counted correctly once.
  4. Centroid landing precisely on the boundary pixel (curr_y == line_y) is handled without ambiguity.
  5. Direction-aware crossing filters (down, up, both).
  6. Throughput calculation under edge conditions.
"""

import pytest
from collections import deque
from detect_live import (
    calculate_center,
    resolve_line_y,
    check_line_crossing,
    calculate_throughput,
    update_track_history_and_count,
)


def test_single_bottle_crossing_once_counts_once():
    """
    Real-World Scenario:
    A milk bottle moves steadily downstream along a conveyor belt.
    The bottle starts above the virtual counting line (y=500), advances
    frame-by-frame, crosses the line, and continues downstream.

    Expected Behavior:
    The counter transitions from 0 to 1 exactly at the crossing frame and
    does not increment again in subsequent downstream frames.
    """
    line_y = 500
    track_history = {}
    counted_track_ids = set()
    total_bottles = 0

    # Frame 1: Bottle approaching line from above
    det_f1 = [{"track_id": 1, "conf": 0.92, "bbox": (200, 420, 260, 480), "center": (230, 450)}]
    total_bottles = update_track_history_and_count(
        det_f1, track_history, counted_track_ids, line_y, total_bottles, direction="down"
    )
    assert total_bottles == 0
    assert 1 not in counted_track_ids

    # Frame 2: Bottle moves closer to line
    det_f2 = [{"track_id": 1, "conf": 0.94, "bbox": (200, 450, 260, 510), "center": (230, 480)}]
    total_bottles = update_track_history_and_count(
        det_f2, track_history, counted_track_ids, line_y, total_bottles, direction="down"
    )
    assert total_bottles == 0
    assert 1 not in counted_track_ids

    # Frame 3: Bottle crosses the line (480 -> 520, where line_y = 500)
    det_f3 = [{"track_id": 1, "conf": 0.95, "bbox": (200, 490, 260, 550), "center": (230, 520)}]
    total_bottles = update_track_history_and_count(
        det_f3, track_history, counted_track_ids, line_y, total_bottles, direction="down"
    )
    assert total_bottles == 1
    assert 1 in counted_track_ids

    # Frame 4: Bottle continues past line (520 -> 560)
    det_f4 = [{"track_id": 1, "conf": 0.93, "bbox": (200, 530, 260, 590), "center": (230, 560)}]
    total_bottles = update_track_history_and_count(
        det_f4, track_history, counted_track_ids, line_y, total_bottles, direction="down"
    )
    assert total_bottles == 1


def test_bottle_jittering_across_boundary_not_double_counted():
    """
    Real-World Scenario:
    Conveyor vibration, sensor noise, or an accumulation stop causes a bottle
    to oscillate back and forth directly across the counting line boundary
    (e.g., y = 490 -> 510 -> 492 -> 512 -> 488 -> 515).

    Expected Behavior:
    The persistent track ID registration ensures the bottle is counted exactly
    once on its initial crossing and never double-counted despite repeated
    oscillations across the tripwire.
    """
    line_y = 500
    track_history = {}
    counted_track_ids = set()
    total_bottles = 0

    trajectory_y = [470, 490, 510, 492, 512, 488, 515, 540]

    for frame_idx, cy in enumerate(trajectory_y):
        det = [{"track_id": 7, "conf": 0.91, "bbox": (100, cy - 30, 160, cy + 30), "center": (130, cy)}]
        total_bottles = update_track_history_and_count(
            det, track_history, counted_track_ids, line_y, total_bottles, direction="both"
        )
        if cy < line_y and frame_idx < 2:
            assert total_bottles == 0
        else:
            # Once crossed at frame_idx 2 (cy=510), count must remain strictly 1
            assert total_bottles == 1, f"Failed at frame {frame_idx} with cy={cy}: total_bottles={total_bottles}"

    assert total_bottles == 1
    assert counted_track_ids == {7}


def test_bottle_occlusion_dropout_and_reappearance():
    """
    Real-World Scenario:
    A milk bottle passes under an overhead structural bar or camera glare zone.
    The detector misses the bottle for 3 consecutive frames immediately around
    or after the crossing line, then the tracker re-acquires the bottle
    with the same persistent track ID downstream.

    Expected Behavior:
    Case A: The bottle crossed before disappearing. Upon reappearing downstream,
            it is recognized as already counted and not counted again.
    Case B: The bottle disappeared before crossing (y=480), and reappeared
            across the line (y=530). The delta between the last recorded position
            and the new detection triggers the crossing event, counting exactly once.
    """
    line_y = 500

    # Sub-case A: Crossed -> Occluded -> Reappears downstream
    track_history_a = {}
    counted_a = set()
    total_a = 0

    # Frame 1 & 2: Crosses line
    update_track_history_and_count(
        [{"track_id": 3, "conf": 0.9, "bbox": (100, 450, 150, 510), "center": (125, 480)}],
        track_history_a, counted_a, line_y, total_a
    )
    total_a = update_track_history_and_count(
        [{"track_id": 3, "conf": 0.9, "bbox": (100, 490, 150, 550), "center": (125, 520)}],
        track_history_a, counted_a, line_y, total_a
    )
    assert total_a == 1

    # Frames 3, 4, 5: Complete occlusion (empty detections list for this track)
    for _ in range(3):
        total_a = update_track_history_and_count([], track_history_a, counted_a, line_y, total_a)
    assert total_a == 1

    # Frame 6: Bottle re-emerges downstream at y=600
    total_a = update_track_history_and_count(
        [{"track_id": 3, "conf": 0.88, "bbox": (100, 570, 150, 630), "center": (125, 600)}],
        track_history_a, counted_a, line_y, total_a
    )
    assert total_a == 1, "Bottle double-counted after re-emerging from occlusion!"

    # Sub-case B: Occluded while straddling the line (y=475 -> Occluded -> y=525)
    track_history_b = {}
    counted_b = set()
    total_b = 0

    total_b = update_track_history_and_count(
        [{"track_id": 4, "conf": 0.9, "bbox": (100, 445, 150, 505), "center": (125, 475)}],
        track_history_b, counted_b, line_y, total_b
    )
    assert total_b == 0

    # Occluded for 2 frames
    for _ in range(2):
        total_b = update_track_history_and_count([], track_history_b, counted_b, line_y, total_b)
    assert total_b == 0

    # Reappears past line at y=525
    total_b = update_track_history_and_count(
        [{"track_id": 4, "conf": 0.85, "bbox": (100, 495, 150, 555), "center": (125, 525)}],
        track_history_b, counted_b, line_y, total_b
    )
    assert total_b == 1, "Bottle should be counted upon re-emerging across the tripwire line"


def test_boundary_pixel_crossing_handled_without_ambiguity():
    """
    Real-World Scenario:
    A bottle's centroid falls exactly on the integer boundary pixel of the
    counting line (i.e. curr_y == line_y).

    Expected Behavior:
    1. Downward motion landing exactly on line_y (e.g. 490 -> 500):
       check_line_crossing returns (True, 'DOWN').
    2. Remaining stationary exactly on the line (500 -> 500):
       check_line_crossing returns (False, None) to prevent duplicate triggering.
    3. Moving off the line downward (500 -> 510):
       does not double-count since track_id is already in counted_track_ids.
    4. Upward motion landing exactly on line_y (e.g. 510 -> 500):
       check_line_crossing returns (True, 'UP').
    """
    line_y = 300

    # Downward approach landing precisely on the boundary pixel
    crossed_down, direction = check_line_crossing(prev_y=295, curr_y=300, line_y=line_y, direction="down")
    assert crossed_down is True
    assert direction == "DOWN"

    # Stationary on the boundary pixel in consecutive frame
    crossed_stationary, _ = check_line_crossing(prev_y=300, curr_y=300, line_y=line_y, direction="down")
    assert crossed_stationary is False

    # Moving off the line downward
    crossed_off, direction_off = check_line_crossing(prev_y=300, curr_y=305, line_y=line_y, direction="down")
    # Note: prev_y < line_y is False when prev_y == line_y, so it doesn't re-trigger
    assert crossed_off is False

    # Upward approach landing precisely on the boundary pixel
    crossed_up, direction = check_line_crossing(prev_y=305, curr_y=300, line_y=line_y, direction="up")
    assert crossed_up is True
    assert direction == "UP"


def test_directional_crossing_filters():
    """
    Real-World Scenario:
    In industrial installations, conveyor lines often run in a dedicated
    forward direction (e.g., downstream packing). Occasionally operators
    manually pull bottles backwards or secondary return belts run in reverse.

    Expected Behavior:
    - direction='down' counts only downward crossings and ignores upward motion.
    - direction='up' counts only upward crossings and ignores downward motion.
    - direction='both' counts crossings in either direction.
    """
    line_y = 400

    # Downward movement
    assert check_line_crossing(380, 420, line_y, direction="down")[0] is True
    assert check_line_crossing(380, 420, line_y, direction="up")[0] is False
    assert check_line_crossing(380, 420, line_y, direction="both")[0] is True

    # Upward movement
    assert check_line_crossing(420, 380, line_y, direction="down")[0] is False
    assert check_line_crossing(420, 380, line_y, direction="up")[0] is True
    assert check_line_crossing(420, 380, line_y, direction="both")[0] is True


def test_throughput_calculation_edge_cases():
    """
    Real-World Scenario:
    Telemetry calculates bottles per minute throughout the run.
    At initialization, elapsed time is 0.0 or sub-second, and total bottles
    may be 0.

    Expected Behavior:
    - Returns 0.0 without DivisionByZeroError when elapsed time < 1.0s or count == 0.
    - Accurately computes throughput (e.g. 60 bottles in 60s -> 60.0 bottles/min).
    """
    # Sub-second initialization
    assert calculate_throughput(bottles_passed=0, elapsed_seconds=0.0) == 0.0
    assert calculate_throughput(bottles_passed=5, elapsed_seconds=0.5) == 0.0
    assert calculate_throughput(bottles_passed=0, elapsed_seconds=120.0) == 0.0

    # Standard production rates
    assert calculate_throughput(bottles_passed=30, elapsed_seconds=60.0) == 30.0
    assert calculate_throughput(bottles_passed=100, elapsed_seconds=120.0) == 50.0


def test_both_direction_crossing_regression():
    """
    Regression Test:
    Verify that in direction='both' mode, unique objects crossing downwards,
    upwards, jumping across the boundary, or crossing via boundary pixel
    increment count exactly once.
    """
    line_y = 300

    def run_trajectory(y_coords, direction="both"):
        th = {}
        counted = set()
        total = 0
        for cy in y_coords:
            det = [{"track_id": 1, "conf": 0.92, "bbox": (100, cy - 30, 160, cy + 30), "center": (130, cy)}]
            total = update_track_history_and_count(
                det, th, counted, line_y, total, direction=direction
            )
        return total

    # 1. Downward crossing: 260 -> 275 -> 290 -> 305 -> 320
    assert run_trajectory([260, 275, 290, 305, 320], direction="both") == 1

    # 2. Reverse (Upward) crossing: 320 -> 305 -> 290 -> 275 -> 260
    assert run_trajectory([320, 305, 290, 275, 260], direction="both") == 1

    # 3. Trajectory touching exact boundary pixel y == 300: 260 -> 280 -> 300 -> 320
    assert run_trajectory([260, 280, 300, 320], direction="both") == 1

    # 4. Trajectory with large frame jump across line: 285 -> 315
    assert run_trajectory([285, 315], direction="both") == 1

    # 5. Reverse jump across line: 315 -> 285
    assert run_trajectory([315, 285], direction="both") == 1


def test_track_loss_recovery_across_line():
    """
    Regression Test:
    Simulates a bottle approaching the line, dropping out for a frame
    during crossing (e.g. motion blur or transient confidence dip),
    and being re-acquired downstream with the same track ID.
    """
    line_y = 300
    th = {}
    counted = set()
    total = 0

    # Frame 1: y = 280
    total = update_track_history_and_count(
        [{"track_id": 9, "conf": 0.90, "bbox": (100, 250, 160, 310), "center": (130, 280)}],
        th, counted, line_y, total, direction="both"
    )
    assert total == 0

    # Frame 2: y = 290
    total = update_track_history_and_count(
        [{"track_id": 9, "conf": 0.91, "bbox": (100, 260, 160, 320), "center": (130, 290)}],
        th, counted, line_y, total, direction="both"
    )
    assert total == 0

    # Frame 3: Missing frame (occlusion / detection dropout)
    total = update_track_history_and_count(
        [], th, counted, line_y, total, direction="both"
    )
    assert total == 0

    # Frame 4: Re-acquired across line at y = 310 (same track ID #9)
    total = update_track_history_and_count(
        [{"track_id": 9, "conf": 0.88, "bbox": (100, 280, 160, 340), "center": (130, 310)}],
        th, counted, line_y, total, direction="both"
    )
    assert total == 1

    # Frame 5: y = 320
    total = update_track_history_and_count(
        [{"track_id": 9, "conf": 0.92, "bbox": (100, 290, 160, 350), "center": (130, 320)}],
        th, counted, line_y, total, direction="both"
    )
    assert total == 1

