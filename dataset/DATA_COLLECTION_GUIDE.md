# Data Collection Guide for On-Board BAS HAR

This guide specifies how to collect, organize, and label real video datasets for training the human activity recognition (HAR) classifier.

---

## 1. Supported Activity Classes

The classifier expects ground-truth labels from `ai.schemas.Activity.ALL`:

| Class Label | Description | Expected Posture / Motion Characteristics |
|-------------|-------------|-------------------------------------------|
| `STANDING`  | Person standing upright and stationary | High vertical torso, low velocity, feet aligned |
| `SITTING`   | Person seated on chair / bench / equipment | Torso upright, hips lowered, knees flexed |
| `WALKING`   | Person walking at normal pace | Upright torso, periodic leg movement, steady velocity |
| `RUNNING`   | Person running or jogging | Forward lean, high limb velocity, rapid foot transitions |
| `FALLING`   | Person falling, slipping, or lying fallen | Rapid downward velocity, large body lean (>60°), sudden impact |

> **Note:** Only the above 5 classes are valid training targets. Frames where activity is ambiguous should NOT be force-labeled.

---

## 2. Directory Layout for Raw Videos

Place recorded video files in `dataset/raw/<activity>/`:

```text
dataset/raw/
├── standing/
│   ├── standing_subject01_01.mp4
│   ├── standing_subject01_02.mp4
│   └── standing_subject02_01.mp4
├── sitting/
│   ├── sitting_subject01_01.mp4
│   ├── sitting_subject02_01.mp4
│   └── sitting_subject03_01.mp4
├── walking/
│   ├── walking_hallway_01.mp4
│   ├── walking_hallway_02.mp4
│   └── walking_lab_01.mp4
├── running/
│   ├── running_corridor_01.mp4
│   └── running_corridor_02.mp4
└── falling/
    ├── fall_slip_mat_01.mp4
    ├── fall_trip_mat_02.mp4
    └── fall_collapse_mat_03.mp4
```

### Video Recording Guidelines
1. **Multiple Short Clips:** Record 3–10 separate clips per class (10–30 seconds each) rather than one single continuous video. This ensures diverse source video IDs for leakage-safe splitting (`GroupShuffleSplit`).
2. **Subject Diversity:** Include at least 2–3 different participants across sessions.
3. **Camera Angle:** Fixed angle at 1.5–2.2m height, 3–6m from subject, representing on-board BAS monitoring perspective.
4. **Lighting & Environment:** Target environment background (lab / cabin / industrial setting), avoiding excessive motion blur or extreme backlighting.
5. **Safety First:** Falling actions should always be performed with crash mats / stunt padding in controlled conditions.

---

## 3. Feature Extraction

Run `collect_features.py` for each video file:

```bash
# Example extraction per video
python collect_features.py --input dataset/raw/standing/standing_subject01_01.mp4 --label STANDING
python collect_features.py --input dataset/raw/sitting/sitting_subject01_01.mp4   --label SITTING
python collect_features.py --input dataset/raw/walking/walking_hallway_01.mp4     --label WALKING
python collect_features.py --input dataset/raw/running/running_corridor_01.mp4   --label RUNNING
python collect_features.py --input dataset/raw/falling/fall_slip_mat_01.mp4       --label FALLING
```

The feature files will be saved in `dataset/features/<video_stem>_<label>_<timestamp>.csv`.

---

## 4. Dataset Validation

Before training, run the dataset validator to verify data integrity, sample counts, and feature dimensions:

```bash
python dataset_validator.py --data dataset/features
```

The validator checks:
- No NaN or infinite values
- All 34 features (`f_00` to `f_33`) are present
- Source video diversity (at least 3 unique source videos per class recommended for group splitting)
- Class balance and sample distribution

---

## 5. Model Training & Evaluation

Once validated, train the classifier using the leakage-safe training pipeline:

```bash
python train_classifier.py --data dataset/features --output models/activity_classifier_v1.joblib
```

This will automatically:
1. Split data by `source_video` using `GroupShuffleSplit` (no frame leakage).
2. Standardize features using training data only.
3. Train the model (Random Forest / Gradient Boost / Logistic Regression).
4. Evaluate on the held-out test videos and write reports to `evaluation/`:
   - `evaluation/metrics.json`
   - `evaluation/classification_report.txt`
   - `evaluation/confusion_matrix.csv`
   - `evaluation/confusion_matrix.png`
5. Save model, scaler, and metadata (`models/activity_classifier_v1_meta.json`).

---

## 6. Verification in Dashboard

Update `config.yaml` or pass the model path to point to the trained model:
```yaml
classifier:
  model_path: "models/activity_classifier_v1.joblib"
  scaler_path: "models/activity_classifier_v1_scaler.joblib"
```

Start the Streamlit dashboard:
```bash
streamlit run app.py
```
Open **AI DIAGNOSTICS** in the sidebar. It will show:
- `Mode: TRAINED ML`
- Model version and training timestamp
- Number of training samples and split strategy
