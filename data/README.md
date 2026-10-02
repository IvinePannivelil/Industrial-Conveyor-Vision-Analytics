# Dataset Documentation — Industrial Conveyor Vision Analytics

## Overview
This dataset contains high-resolution imagery of products passing on industrial conveyor belts at a proprietary dairy production facility. The images capture objects under variable industrial conditions, including motion blur, surface reflections, varying conveyor speeds, and clustered layouts. The current industrial use case is milk bottle detection.

## Confidentiality & Distribution Notice
> **Important**: The raw image files (`Milk_Bottles/`, `Annotated/`) and generated splits (`dataset/`) contain proprietary industrial footage from a dairy production facility and cannot be distributed publicly in this repository.

## Expected Directory Layout for Retraining
If you have authorized access to the raw data or are adapting this project to your own conveyor facility, place your Label Studio YOLO exports inside an `Annotated/` directory formatted as follows:

```
Annotated/
├── annotated1/
│   ├── images/       # Raw JPEG images
│   ├── labels/       # Normalized YOLO format txt annotations
│   └── classes.txt   # Class list (milk_bottle)
├── annotated2/
│   ├── images/
│   └── labels/
├── annotated3/
├── annotated4/
└── annotated5/
```

## Dataset Specifications
- **Classes**: `0: milk_bottle` (Single-class object detection)
- **Annotation Format**: YOLO normalized bounding box format:
  ```
  <class_id> <center_x> <center_y> <width> <height>
  ```
  *(All coordinates normalized to the range `[0.0, 1.0]` relative to image width and height).*
- **Total Annotated Images**: 1,073 image-label pairs across 5 sequential batches.
- **Dataset Split**: 80% Train (858 images), 10% Validation (107 images), 10% Test (108 images).
- **Random Seed**: Fixed random seed `42` used during splitting for reproducibility.

## Preparing the Dataset
To validate bounding boxes, remove empty/corrupt files, merge all batches, and generate the 80/10/10 split:

```bash
python scripts/prepare_dataset.py
```

This generates:
- `dataset/images/{train,val,test}`
- `dataset/labels/{train,val,test}`
- `configs/dataset.yaml` (canonical YOLO configuration) and `dataset/dataset.yaml` (legacy compatibility copy)
