# Dataset Guidelines & Infrastructure

This document defines dataset recording protocols, structure, and validation workflows.

## Target Classes
The system targets 5 core human activity classes:
1. `STANDING` — Stationary upright posture.
2. `SITTING` — Stationary seated posture (chairs, stools, benches).
3. `WALKING` — Normal pacing across camera view.
4. `RUNNING` — Elevated speed gait.
5. `FALLING` — Controlled fall onto a soft surface or mat (safety protocol mandatory).

## Dataset Directory Structure
```text
dataset/
├── raw/
│   ├── STANDING/
│   ├── SITTING/
│   ├── WALKING/
│   ├── RUNNING/
│   └── FALLING/
├── features/
│   └── <source_video_stem>.csv
├── splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── manifest.csv
└── snapshot.json
```

## Collection Tools
- **Recorder:** `python tools/record_dataset.py --class <CLASS> --subject SUBJ_01 --duration 10`
- **Assistant:** `python tools/dataset_assistant.py`
- **Validator:** `python training_readiness.py`
- **Feature Extractor:** `python collect_features.py --data dataset/raw`
- **Leakage-Safe Splitter:** `python split_dataset.py`
