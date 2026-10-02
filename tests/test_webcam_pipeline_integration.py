"""
Scratch script to verify webcam inference pipeline, ByteTrack tracking,
and line crossing counting in BOTH mode.
"""
import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from web_server import pipeline_manager, process_and_draw_frame, state

def test_webcam_counting_pipeline():
    print("Testing Webcam Vision Pipeline with ByteTrack & Line Crossing...")
    assert pipeline_manager.model_loaded, "Model should be loaded"
    
    # Reset global state to clean state
    state.reset_counts()
    state.direction = "both"
    state.tracking_enabled = True
    state.counting_enabled = True
    state.line_position = 0.50 # line_y = 300 for 600h frame
    
    # 1. Downward crossing simulation
    # Frame 1: bottle at y=260
    frame_h, frame_w = 600, 800
    line_y = 300
    
    # Feed frames directly into process_and_draw_frame with synthetic detection
    # Simulating tracker output for track_id=1 moving downwards
    # Frame 1: center y=270 (above line)
    dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    det1 = [{"track_id": 1, "conf": 0.92, "cls": 0, "bbox": (200, 240, 260, 300), "center": (230, 270)}]
    process_and_draw_frame(dummy_frame.copy(), det1)
    assert state.total_counted == 0, f"Expected 0, got {state.total_counted}"
    
    # Frame 2: center y=330 (crossed below line)
    det2 = [{"track_id": 1, "conf": 0.93, "cls": 0, "bbox": (200, 300, 260, 360), "center": (230, 330)}]
    process_and_draw_frame(dummy_frame.copy(), det2)
    assert state.total_counted == 1, f"Expected 1, got {state.total_counted}"
    print("✓ Downward crossing counted: total = 1")
    
    # Frame 3: continuing downward (should NOT double count)
    det3 = [{"track_id": 1, "conf": 0.91, "cls": 0, "bbox": (200, 340, 260, 400), "center": (230, 370)}]
    process_and_draw_frame(dummy_frame.copy(), det3)
    assert state.total_counted == 1, f"Expected 1, got {state.total_counted}"
    print("✓ Deduplication preserved: total remains 1")
    
    # 2. Upward crossing simulation with track_id=2
    # Frame 4: bottle 2 at center y=340 (below line)
    det4 = [{"track_id": 2, "conf": 0.89, "cls": 0, "bbox": (300, 310, 360, 370), "center": (330, 340)}]
    process_and_draw_frame(dummy_frame.copy(), det4)
    assert state.total_counted == 1, f"Expected 1, got {state.total_counted}"
    
    # Frame 5: bottle 2 moves upward across line to y=260 (above line)
    det5 = [{"track_id": 2, "conf": 0.90, "cls": 0, "bbox": (300, 230, 360, 290), "center": (330, 260)}]
    process_and_draw_frame(dummy_frame.copy(), det5)
    assert state.total_counted == 2, f"Expected 2, got {state.total_counted}"
    print("✓ Upward crossing counted in BOTH mode: total = 2")
    
    # 3. Test Reset
    state.reset_counts()
    assert state.total_counted == 0, f"Expected 0 after reset, got {state.total_counted}"
    assert len(state.counted_track_ids) == 0
    print("✓ Reset verified")
    
    # 4. Test infer_camera with tracking_enabled
    # Test on a blank frame or sample image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dets, lat = pipeline_manager.infer_camera(test_img, conf_thresh=0.40, tracking_enabled=True)
    print(f"✓ infer_camera executed successfully with tracking_enabled=True (latency: {lat:.1f}ms)")
    
    print("\nALL WEBCAM PIPELINE INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    test_webcam_counting_pipeline()
