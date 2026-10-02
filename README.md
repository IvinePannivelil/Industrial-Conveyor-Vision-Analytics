# Industrial Conveyor Vision Analytics

*Real-time object detection, tracking, counting, production-history analytics, and model reliability for industrial conveyor systems.*

---

## Project Overview

Industrial Conveyor Vision Analytics is a computer vision pipeline and observability system for industrial conveyor environments. Built around a YOLOv8n detector, the project extends baseline single-frame detection into an end-to-end production monitoring system featuring:

- **Multi-Object Tracking:** ByteTrack integration with persistent object IDs and occlusion handling
- **Unique Product Counting:** Virtual line crossing with trajectory raycasting and anti-jitter deduplication
- **Production History Analytics:** Dynamic time-window aggregation, directional counts (UP/DOWN), average production rate (objects/min), and time-bucketed distribution charts from local persistent event logs (`results/count_events.csv`) without requiring an external database
- **Operational Telemetry:** Rolling throughput calculation and structured per-frame CSV telemetry
- **Model Reliability Diagnostics:** Empirical failure mode investigation, hard-negative mining, false-positive suppression, and confidence calibration
- **Interactive Control Suite:** Dark-theme browser dashboard featuring a procedural conveyor simulator (runs with zero external model weights or private data) and live webcam support
- **Verification & Testing:** 34 / 34 automated tests passing across tracking, counting, event persistence, and production aggregation

The current industrial use case is milk bottle detection on a dairy conveyor line. The pipeline architecture is intentionally modular and applicable to any single-class conveyor monitoring task.

---

## System Architecture

### Runtime Pipeline

```
Camera / Video / Simulator Input
        ↓
YOLOv8n Detector  (confidence + IoU thresholds)
        ↓
ByteTrack Multi-Object Tracker  (Kalman filter + two-stage association)
        ↓
Persistent Track IDs  (occlusion-tolerant, ID-locked)
        ↓
Virtual Line Crossing  (configurable Y, direction: UP / DOWN / BOTH)
        ↓
Unique Object Counting  (each track ID counted once, jitter-resistant)
        ↓
Production History & Telemetry  (results/count_events.csv, obj/min, time buckets)
        ↓
Browser Dashboard / OpenCV HUD / Headless Output
```

### Model Reliability Workflow

```
Raw Dataset
        ↓
Dataset Quality Audit  (annotation consistency, bounding box distribution)
        ↓
Leakage Analysis  (sequence-aware vs. random frame split)
        ↓
Hard-Negative Mining  (factory floors, rails, machinery, 138 images)
        ↓
Calibration Fine-Tuning  (15-epoch transfer, false-positive suppression)
        ↓
Evaluation & Verification  (mAP, precision, recall, latency, unit tests)
```

See [`docs/architecture.md`](docs/architecture.md) for detailed diagrams.

---

## Core Capabilities

| Capability | Status | Notes |
| :--- | :---: | :--- |
| Procedural Conveyor Simulator | ✅ Working | Runs out of the box with zero model weights or proprietary assets |
| Live Camera Ingestion | ✅ Working | OpenCV VideoCapture with graceful fallback if weights absent |
| YOLOv8n inference (PyTorch) | ✅ Working | Bounding box prediction with configurable confidence threshold |
| ByteTrack multi-object tracking | ✅ Working | Persistent IDs with two-stage Kalman association |
| Virtual line crossing (UP / DOWN / BOTH) | ✅ Working | Bidirectional crossing state machine with boundary hysteresis |
| Unique object counting | ✅ Working | Single-count per track ID; oscillation jitter-resistant |
| Production History analytics | ✅ Working | Dynamic time bucketing (1m, 5m, 15m, 30m, 1h), rate math |
| CSV event persistence | ✅ Working | Append-safe logging to `results/count_events.csv` (gitignored) |
| Browser operations dashboard | ✅ Working | Real-time telemetry HUD, charts, controls, and markdown viewer |
| ONNX export & latency verification | ✅ Verified | Intel CPU ONNX Runtime (~17.8 ms) and PyTorch MPS (~35.9 ms) |
| Hard-negative mining | ✅ Completed | 138 factory-environment background images collected |
| Calibration fine-tuning | ✅ Completed | 15-epoch patch: whiteboard false positive 74.5% → 0.0% |
| Automated test suite | ✅ Passing | 34 / 34 automated unit and integration tests passing |
| Sequence-aware split (v2) | 🔄 Documented | Awaiting raw continuous video clips for retraining |

---

## Provenance & Engineering Contributions

### Baseline Detector
> The baseline YOLOv8n detector used in this project was originally trained separately by another member of the engineering team using authorized Muralya Dairy conveyor imagery.

### My Engineering Contributions
Building on that baseline detector, my engineering contributions encompass:

**Runtime System & Analytics**
- Integrated ByteTrack with persistent track lifecycle management and Kalman filtering
- Engineered the virtual line-crossing counter with bidirectional support (UP, DOWN, BOTH)
- Implemented state-machine hysteresis to eliminate double counting from boundary jitter
- Designed the Production History analytics layer: local time-window filtering, dynamic bucketing (1m, 5m, 15m, 30m, 1h), UP/DOWN separation, and average rate calculation
- Built thread-safe append-only event persistence to `results/count_events.csv` with session reset independence
- Developed the browser-based operations workbench with interactive parameter adjustments

**Model Reliability Engineering**
- Conducted dataset quality audit: identified annotation inconsistencies and bounding box distribution skew
- Identified temporal frame leakage in v1 random frame split; formulated sequence-aware grouped validation design (v2)
- Collected and augmented 138 factory hard-negative background images (conveyor rails, machine hoods, floor drains)
- Executed 15-epoch negative calibration fine-tuning, reducing whiteboard false-positive confidence from 74.5% to 0.0%
- Created comparative evaluation scripts (`scripts/evaluate_calibration.py`, `scripts/verify_v2_ready.py`)

**Testing & Verification**
- Authored 34 automated tests across tracker, counter, CSV lifecycle, and production history aggregation
- Built ONNX export pipeline with dynamic batching and latency benchmarking
- The procedural simulator is designed to run without proprietary weights or private assets, providing a reproducible public demonstration path

---

## Installation

**Prerequisites**: Python 3.9+, `pip`

```bash
# Clone the repository
git clone https://github.com/IvinePannivelil/Industrial-Conveyor-Vision-Analytics.git
cd Industrial-Conveyor-Vision-Analytics

# Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Model Weight Notice**: Model weights are not included in the public repository because redistribution permission has not been confirmed. The procedural conveyor simulator in the web dashboard runs out of the box with zero model weights or proprietary assets. For live camera tracking, place valid weights at `runs/detect/milk_bottle/weights/best.pt` or specify `--weights <path>`.

---

## Quick Start

```bash
# 1. Start the interactive web dashboard & procedural conveyor simulator
python web_server.py
# Open http://localhost:8080 in your browser

# 2. View all CLI options
python detect_live.py --help

# Live webcam — detection only
python detect_live.py --source 0

# Live webcam — tracking + counting
python detect_live.py --source 0 --track --count-line

# Video file — tracking + counting + telemetry
python detect_live.py \
  --source path/to/video.mp4 \
  --track --count-line \
  --csv-log results/session.csv

# Static image detection
python detect_live.py --source path/to/image.jpg

# Headless (no GUI window, e.g. server/CI)
python detect_live.py --source video.mp4 --no-window --track --count-line
```

---

## Configuration

All runtime parameters can be set via CLI arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source` | `0` | Camera index or path to image/video file |
| `--weights` | `runs/detect/milk_bottle/weights/best.pt` | Path to `.pt` weights |
| `--conf` | `0.40` | Detection confidence threshold |
| `--device` | auto | `cpu`, `cuda`, or `mps` |
| `--track` | off | Enable ByteTrack tracking |
| `--count-line` | off | Enable virtual line counting (enables `--track`) |
| `--line-position` | `0.5` | Line Y: relative (0.1–0.9) or absolute pixel |
| `--count-direction` | `both` | Crossing direction: `up`, `down`, or `both` |
| `--csv-log` | none | Path to CSV telemetry output |
| `--no-window` | off | Run headless without display |

**Runtime keyboard controls**:
- `q` — quit
- `s` — save screenshot
- `p` — pause / resume
- `+` / `-` — increase / decrease confidence threshold

---

## Telemetry

CSV telemetry output (`--csv-log`) logs one row per frame:

```
timestamp, elapsed_seconds, frame_number, current_detections,
bottles_passed, throughput_bottles_per_min, fps, confidence,
tracking_enabled, counting_enabled, count_direction
```

Example:
```bash
python detect_live.py --source conveyor.mp4 --track --count-line \
  --csv-log results/live_run.csv
```

---

## Evaluation

```bash
# Evaluate against validation split
python scripts/evaluate_model.py

# Evaluate specific model / split
python scripts/evaluate_model.py \
  --weights runs/detect/milk_bottle_calibrated/weights/best.pt \
  --split val

# Comparative baseline vs. calibrated
python scripts/evaluate_calibration.py
```

> **Note**: The proprietary dataset is withheld. Running evaluation without the
> dataset will display verified baseline metrics from the stored `results.csv`.

See [`docs/evaluation.md`](docs/evaluation.md) for full methodology and results.

---

## ONNX Export

```bash
# Export to ONNX
python export_model.py

# Export and verify with latency benchmark
python export_model.py --verify

# Verify an existing ONNX file
python export_model.py --verify-only exported_model/milk_bottle_detector.onnx
```

The exported model is saved to `exported_model/milk_bottle_detector.onnx`.

---

## Dataset & Model Reliability Findings

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full experiment log. Key findings:

### Temporal Frame Leakage (v1 Split)
Sequence-aware validation analysis identified that the v1 training split used
`random.shuffle()` across all 1,073 video frames. At 30 FPS, adjacent frames are
near-identical. The resulting validation metrics may reflect near-duplicate scene
memorisation rather than genuine generalisation. Experiment v2 addresses this with
sequence-level partitioning.

### False Positive Suppression (Calibration, Completed)
The v1 model produced a 74.5% confidence false positive on a blank whiteboard.
Root cause: 0% negative-image coverage in the training set. A 15-epoch
negative-calibration fine-tune (43 conveyor positives + 138 factory hard-negatives)
reduced whiteboard false-positive confidence to 0.0%.

### Annotation Inconsistency
Ground-truth bounding boxes used two incompatible annotation policies (cap-only vs.
full-body). This produced bimodal bounding box height distributions and conflicting
regression gradient signals. A canonical annotation policy and correction checklist
are documented in `scratch/annotation_fixes_needed.md` (local only, not tracked).

---

## Tests

```bash
# Run all unit tests
pytest

# Verbose output
pytest -v

# Specific module
pytest tests/test_counter.py -v
pytest tests/test_tracker.py -v
pytest tests/test_count_events.py -v
pytest tests/test_production_history.py -v
```

**Current status**: 34 / 34 automated tests passed in the final QA run.

Tests cover:
- Virtual line-crossing counter (single crossings, jitter resistance, occlusion recovery, directional filters)
- ByteTrack persistent track IDs and two-stage Kalman association
- Event CSV persistence, thread-safe writes, deduplication, and reset independence
- Production History time-range aggregation (60-minute window, UP/DOWN counts, average rate)
- Dynamic time-bucketing (1-min, 5-min, 15-min, 30-min, 1-hour rules)
- Cross-midnight queries and multi-day date/time support
- Source and direction filtering without data fabrication
- Simulator and live-camera event distinguishability

---

## Web Control & Reliability Suite

The repository includes a browser-based operations and model reliability workbench:

```bash
# Start the interactive web suite
python web_server.py
```
Open **`http://localhost:8080`** in your browser.

The web suite is organized into four distinct engineering views:

1. **Runtime Operations View (`Live Dashboard`)**:
   - **Procedural Conveyor Simulator**: Synthetic industrial conveyor generator with moving bottles, drop shadows, and procedural physics for complete public reproducibility without proprietary factory video or local weights.
   - **Live Camera Ingestion**: OpenCV camera stream with real-time YOLOv8 detection and ByteTrack tracking (when local weights are supplied).
   - **Interactive Parameter Deck**: Live adjustment of detection confidence threshold (10% to 95%), virtual trigger line position (15% to 85%), and directional crossing filters (`Both`, `Down`, `Up`).
   - **Production Telemetry**: Real-time KPI monitors (`SIMULATED FOV`, `SIMULATED`, `TARGET BUDGET`) and 60-second throughput wave chart (`objects/min`).
   - **Production History Analytics**: Local persisted event aggregation layer (`results/count_events.csv`) that answers operational questions without a database: objects passed in selected window, UP/DOWN counts, average production rate (objects/min), dynamic time bucketing (1m, 5m, 15m, 30m, 1h), and production distribution bar chart with optional collapsed raw event drill-down.

2. **Reliability Lab (`Reliability Lab`)**:
   - **Model Health Snapshot**: Compact technical status strip documenting model architecture, validation context, and dataset leakage status.
   - **Failure Mode Investigation Flow**: Vertical root-cause anatomy mapping Observed Failure → Root Cause Hypothesis → Empirical Evidence → Engineering Mitigation → Post-Mitigation Result across 5 verified failure cases.
   - **Experiment Comparison Matrix**: 3-column comparative view (Baseline v1 vs. Hard-Negative Calibration Patch vs. Sequence-Aware v2 Target) with empirical delta explanations.
   - **Evidence Ledger & Timeline**: 7-stage engineering investigation timeline, empirical scope audit table, and collapsible baseline training artifacts (PR curve, F1 curve, confusion matrix, loss curves).

3. **Deployment View (`Pipeline & Edge`)**:
   - **Architectural Flow**: 5-stage processing pipeline (Ingestion, YOLOv8n Inference, ByteTrack Association, Virtual Line Trigger, Telemetry).
   - **Measured Local Benchmarks**: Empirical ONNX CPU (~17.8 ms) and PyTorch MPS (~35.9 ms) timings.
   - **Target Production Budget**: 30 FPS edge latency budget (33.3 ms frame budget) and industrial edge hardware profiles.

4. **Technical Audit & Documentation (`Docs & Audit`)**:
   - In-browser markdown reader for `README.md`, `EXPERIMENTS.md`, `docs/architecture.md`, and `docs/evaluation.md`.

---

## Repository Structure

```
Industrial-Conveyor-Vision-Analytics/
├── detect_live.py              ← Core runtime: YOLOv8 + ByteTrack + virtual line counter
├── web_server.py               ← Web application server, conveyor simulator & telemetry API
├── train.py                    ← YOLOv8n baseline training pipeline
├── train_calibrated.py         ← 15-epoch negative calibration fine-tuning
├── export_model.py             ← ONNX export with dynamic batching & latency benchmark
├── pytest.ini                  ← Test configuration
│
├── web/
│   ├── templates/index.html    ← Control deck, telemetry HUD & diagnostics UI
│   └── static/
│       ├── style.css           ← Industrial dark mode & glassmorphism design system
│       └── app.js              ← Real-time telemetry polling & canvas charts
│
├── tests/
│   ├── test_counter.py         ← Deterministic line-crossing counter unit tests (8 tests)
│   ├── test_tracker.py         ← ByteTrack state & ID continuity tests (4 tests)
│   ├── test_count_events.py    ← Event persistence & CSV lifecycle tests (9 tests)
│   ├── test_production_history.py ← Production history aggregation & bucketing tests (12 tests)
│   └── test_webcam_pipeline_integration.py ← End-to-end webcam pipeline test (1 test)
│
├── configs/
│   ├── dataset.yaml            ← Baseline dataset schema definition
│   ├── dataset_calibrated.yaml ← Calibration dataset schema
│   └── dataset_v2.yaml         ← Sequence-aware v2 schema
│
├── scripts/
│   ├── build_hard_negatives.py ← Hard-negative mining pipeline
│   ├── evaluate_model.py       ← Standard YOLO validation script
│   ├── evaluate_calibration.py ← Comparative baseline vs. calibrated evaluation
│   ├── prepare_dataset_v2.py   ← Sequence-aware grouped dataset partitioner
│   └── verify_v2_ready.py      ← Pre-flight dataset readiness verification
│
├── docs/
│   ├── architecture.md         ← System design, pipeline topologies & state machines
│   └── evaluation.md           ← Complete evaluation methodology & metrics
│
├── data/
│   ├── README.md               ← Dataset structure & confidentiality notice
│   └── negatives/              ← 138 hard-negative images (local only, gitignored)
│
├── runs/                       ← Training run artifacts (partially gitignored)
├── exported_model/             ← ONNX export documentation & README
├── results/                    ← Stored evaluation results & telemetry README
├── samples/                    ← Local conveyor test samples (gitignored)
├── screenshots/                ← Captured frames (gitignored)
│
├── EXPERIMENTS.md              ← Full experiment log & engineering narrative
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Limitations

- **Counting accuracy is not ground-truth validated**: The virtual line-crossing counter and tracking system have been validated through 34 automated unit and integration tests under controlled conditions. Real-world factory counting accuracy has not been benchmarked against a manually labeled continuous video ground truth.
- **Baseline v1 metrics reflect temporal frame leakage**: Random uniform frame shuffling across adjacent 30 FPS video frames inflated baseline recall and precision. Sequence-aware grouped splitting (Experiment v2) is documented as the methodologically sound fix.
- **Negative calibration scope**: The 15-epoch calibration experiment utilized 181 controlled images (43 positive, 138 hard-negative) to eliminate specific false-positive failure modes (such as office whiteboards and reflective plates); generalization across diverse plant environments requires larger sequence data.
- **Hardware benchmarking boundaries**: Only local x86 CPU ONNX Runtime (~17.8 ms) and Apple Silicon PyTorch MPS (~35.9 ms) were measured directly. Embedded platforms (NVIDIA Jetson, Raspberry Pi) are target design profiles and have not been physically benchmarked.
- **Proprietary factory data withheld**: Raw conveyor video and derived model weights are not distributed publicly. Simulation and offline evaluation documentation allow full reproducibility of methodology without private assets.

---

## Future Improvements

### Must Fix
- Sequence-aware v2 retraining once raw annotated data is available
- Canonical annotation corrections (27 flagged images in `scratch/annotation_fixes_needed.md`)

### High Value
- Counting ground-truth validation with a manually annotated video sequence
- Multi-class support (e.g., different product types on shared conveyor)
- ROI zone support (polygon counting zones vs. single horizontal line)
- Output video saving with annotations (`--save-video` flag)

### Optional
- YAML-based configuration file (supplement to CLI args)
- RTSP stream support documentation
- Docker deployment guide

---

## Licensing

See [LICENSE](LICENSE). Licensing is pending formal review prior to public
release. The proprietary industrial dataset and factory imagery are **not
included** in this repository and are covered by a separate confidentiality notice
in `data/README.md`.

Model weights (`runs/detect/milk_bottle/weights/best.pt`,
`runs/detect/milk_bottle_calibrated/weights/best.pt`) are derived from a baseline
trained on proprietary data. Their redistribution status has not been formally
confirmed. See `models/README.md`.
