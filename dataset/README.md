# Dataset Directory

This directory holds all artifacts related to the labeled training dataset
for the HAR (Human Activity Recognition) activity classifier.

---

## Directory Structure

```
dataset/
├── README.md            ← this file
├── raw/                 ← raw video files (NOT committed to git)
├── features/            ← extracted feature CSVs (one per collection session)
├── metadata/            ← per-session JSON metadata
└── models/              ← trained models (mirrors models/ in project root)

evaluation/              ← generated evaluation reports
├── metrics.json
├── classification_report.txt
├── confusion_matrix.csv
└── confusion_matrix.png
```

---

## Dataset Schema

Each row in a feature CSV represents one **temporal window** (sequence) for
one person in one video frame range.

### CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique sample UUID |
| `person_id` | int | Tracker-assigned person ID |
| `activity_label` | str | Ground-truth activity label |
| `source_video` | str | Source video filename or "webcam" |
| `frame_start` | int | First frame index of this window |
| `frame_end` | int | Last frame index of this window |
| `timestamp` | str | ISO-8601 collection timestamp |
| `f_00` … `f_33` | float32 | Flattened feature vector (see below) |

> **Feature ordering:** Each column `f_00` … `f_33` corresponds exactly to
> the 34-dimensional feature vector defined in `ai/features.py`.
> The vector is the **mean** of the temporal window (20 frames by default).
> See `ai/features.py` FEATURE_DIM and the docstring for the full index map.

---

## Activity Labels

| Label | Description |
|-------|-------------|
| `STANDING` | Person is upright and largely stationary |
| `SITTING` | Person is seated (low hip position, bent knees) |
| `WALKING` | Person is moving at walking pace |
| `RUNNING` | Person is moving at running pace |
| `FALLING` | Person is in or transitioning to a fallen posture |

---

## Collection Workflow

```bash
# 1. Record or obtain labeled video segments
#    Name files descriptively: walking_hallway_01.mp4

# 2. Extract features from each video
python collect_features.py \
    --input dataset/raw/walking_hallway_01.mp4 \
    --label WALKING \
    --output dataset/features

# 3. Validate the collected dataset
python dataset_validator.py --data dataset/features

# 4. Train the classifier (real data)
python train_classifier.py \
    --data dataset/features \
    --output models/activity_classifier_v1.joblib

# 5. Review evaluation reports
cat evaluation/classification_report.txt
```

---

## Data Leakage Prevention

**CRITICAL:** Adjacent frames in a video are nearly identical.
A random 80/20 split of individual frames creates data leakage.

The correct split is **by source video**:
- All frames from a given video go ENTIRELY to train, validation, OR test.
- Never mix frames from the same video across splits.

The split is performed in `train_classifier.py` using `GroupShuffleSplit`
keyed on `source_video`.

---

## Git Policy

- `dataset/raw/` — NOT committed (videos can be large)
- `dataset/features/*.csv` — CAN be committed (small)
- `dataset/metadata/*.json` — CAN be committed
- `models/*.joblib` — NOT committed by default (regenerate from data)

Add to `.gitignore`:
```
dataset/raw/
models/*.joblib
evaluation/*.png
```

---

## Limitations

- IoU-based tracker can swap person IDs if two people cross very closely.
  Feature samples near crossings may contain mixed-person data.
  Inspect videos manually if multiple people are present.
- Collection confidence: only frames where YOLO detects a valid person
  (mean keypoint confidence ≥ 0.30) are saved. Low-visibility frames
  are logged but discarded.
