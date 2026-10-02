"""
tests/test_tracker.py
---------------------
Unit tests for multi-object tracking (ByteTrack) logic in detect_live.py.

Covers:
  1. Persistent Track ID assignment across consecutive frames for moving bottles.
  2. Correct behavior when two bottles travel closely together (no ID swapping).
  3. Correct behavior when a tracked bottle temporarily drops below the primary confidence threshold (ByteTrack low-confidence matching).
  4. Correct re-identification of a bottle after a momentary 1-frame occlusion dropout.
"""

import pytest
import torch
from types import SimpleNamespace
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


@pytest.fixture
def tracker():
    """
    Instantiate a ByteTrack tracker matching the operational configuration
    used by YOLOv8 tracking in detect_live.py.
    """
    args = SimpleNamespace(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    return BYTETracker(args)


def create_boxes(box_list: list[list[float]], orig_shape=(640, 640)) -> Boxes:
    """Helper to convert list of [x1, y1, x2, y2, conf, cls] into Ultralytics Boxes."""
    if not box_list:
        return Boxes(torch.empty((0, 6)), orig_shape=orig_shape)
    tensor = torch.tensor(box_list, dtype=torch.float32)
    return Boxes(tensor, orig_shape=orig_shape)


def test_persistent_id_assignment_across_frames(tracker):
    """
    Real-World Scenario:
    A single milk bottle is transported downstream along a straight conveyor belt.
    As the camera captures consecutive frames at 30 FPS, the bottle's bounding box
    translates smoothly from top to bottom (y: 100 -> 125 -> 150 -> 175 -> 200).

    Expected Behavior:
    The tracker initializes a track ID on frame 1 (e.g., ID 1) and maintains
    the exact same track ID across every subsequent frame without resetting,
    flickering, or spawning redundant track IDs.
    """
    trajectory_y = [100, 125, 150, 175, 200]
    assigned_ids = []

    for frame_num, y1 in enumerate(trajectory_y, start=1):
        # Bottle dimensions: w=60, h=160
        y2 = y1 + 160
        boxes = create_boxes([[150.0, float(y1), 210.0, float(y2), 0.92, 0.0]])
        tracks = tracker.update(boxes)

        assert len(tracks) == 1, f"Expected 1 active track at frame {frame_num}, got {len(tracks)}"
        # tracks row: [x1, y1, x2, y2, track_id, conf, cls, det_idx]
        track_id = int(tracks[0][4])
        assigned_ids.append(track_id)

    # All frames should have assigned the exact same ID
    initial_id = assigned_ids[0]
    assert all(tid == initial_id for tid in assigned_ids), (
        f"Track ID fluctuated across frames: {assigned_ids}"
    )


def test_two_close_bottles_no_id_swapping(tracker):
    """
    Real-World Scenario:
    Two milk bottles travel side-by-side along adjacent conveyor lanes or in close
    formation approaching the packing station.
    Bottle A travels along x = [120, 180] while Bottle B travels along x = [195, 255].
    The lateral gap between the two bottles is only 15 pixels. Both bottles move
    downstream together across 5 frames.

    Expected Behavior:
    The tracker assigns two distinct IDs and tracks their independent spatial
    trajectories. At no point do the IDs swap between Bottle A and Bottle B,
    and neither track identity collapses into the other.
    """
    num_frames = 5
    assigned_track_history = {"bottle_a": [], "bottle_b": []}

    for frame_idx in range(num_frames):
        y_offset = frame_idx * 25.0
        # Bottle A (left) and Bottle B (right)
        box_a = [120.0, 100.0 + y_offset, 180.0, 260.0 + y_offset, 0.90, 0.0]
        box_b = [195.0, 100.0 + y_offset, 255.0, 260.0 + y_offset, 0.91, 0.0]

        boxes = create_boxes([box_a, box_b])
        tracks = tracker.update(boxes)

        assert len(tracks) == 2, f"Expected 2 tracks at frame {frame_idx}, got {len(tracks)}"

        # Sort tracks by x1 coordinate to separate Bottle A (left) and Bottle B (right)
        sorted_tracks = sorted(tracks, key=lambda t: t[0])
        track_id_left = int(sorted_tracks[0][4])
        track_id_right = int(sorted_tracks[1][4])

        assigned_track_history["bottle_a"].append(track_id_left)
        assigned_track_history["bottle_b"].append(track_id_right)

    # Verify ID consistency
    ids_a = assigned_track_history["bottle_a"]
    ids_b = assigned_track_history["bottle_b"]

    assert len(set(ids_a)) == 1, f"Bottle A switched ID during transit: {ids_a}"
    assert len(set(ids_b)) == 1, f"Bottle B switched ID during transit: {ids_b}"
    assert ids_a[0] != ids_b[0], f"Bottle A and B collapsed to the same ID: {ids_a[0]}"


def test_tracked_bottle_temporary_confidence_drop(tracker):
    """
    Real-World Scenario:
    A milk bottle on the conveyor moves under a harsh ceiling light causing specular
    glare across its plastic body. In frame 2, the detector confidence dips from 0.90
    down to 0.35 (below the high threshold of 0.50, but above the low threshold of 0.10).
    In frame 3, the bottle moves past the glare and confidence rebounds to 0.92.

    Expected Behavior:
    ByteTrack's two-stage association matches low-confidence detections against
    existing track priors, maintaining the active track through the glare frame
    without dropping the track or re-assigning a new ID in frame 3.
    """
    # Frame 1: High confidence detection
    b1 = create_boxes([[100.0, 100.0, 160.0, 260.0, 0.90, 0.0]])
    tracks_f1 = tracker.update(b1)
    assert len(tracks_f1) == 1
    orig_track_id = int(tracks_f1[0][4])

    # Frame 2: Glare dip (conf=0.35, below 0.50 high threshold)
    b2 = create_boxes([[100.0, 125.0, 160.0, 285.0, 0.35, 0.0]])
    tracks_f2 = tracker.update(b2)
    assert len(tracks_f2) == 1, "ByteTrack secondary association should retain low-conf track"
    assert int(tracks_f2[0][4]) == orig_track_id, "Track ID changed during low-confidence frame"

    # Frame 3: Confidence rebounds to 0.92
    b3 = create_boxes([[100.0, 150.0, 160.0, 310.0, 0.92, 0.0]])
    tracks_f3 = tracker.update(b3)
    assert len(tracks_f3) == 1
    assert int(tracks_f3[0][4]) == orig_track_id, "Track ID must remain identical after confidence recovery"


def test_tracked_bottle_momentary_occlusion_dropout(tracker):
    """
    Real-World Scenario:
    A bottle passes behind a thin metal cable or guide rail, causing the detector
    to miss the bottle completely for exactly 1 frame (0 detections). In the next frame,
    the bottle emerges on the other side of the obstruction at its expected downstream location.

    Expected Behavior:
    The tracker retains the lost track in its track buffer (track_buffer=30).
    When the detection reappears in frame 3, the Kalman filter motion state
    correctly links the new bounding box back to the original track ID.
    """
    # Frame 1: Bottle visible
    b1 = create_boxes([[200.0, 100.0, 260.0, 260.0, 0.88, 0.0]])
    tracks_f1 = tracker.update(b1)
    assert len(tracks_f1) == 1
    target_id = int(tracks_f1[0][4])

    # Frame 2: Complete dropout (camera blinded / occluded)
    b2 = create_boxes([])
    tracks_f2 = tracker.update(b2)
    assert len(tracks_f2) == 0, "No active detections on occlusion frame"

    # Frame 3: Bottle re-emerges downstream at predicted position
    b3 = create_boxes([[200.0, 150.0, 260.0, 310.0, 0.89, 0.0]])
    tracks_f3 = tracker.update(b3)
    assert len(tracks_f3) == 1
    assert int(tracks_f3[0][4]) == target_id, (
        f"Tracker failed to re-identify bottle after 1-frame occlusion. Expected ID {target_id}, got {int(tracks_f3[0][4])}"
    )
