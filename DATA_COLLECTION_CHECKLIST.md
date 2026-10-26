# Real Data Collection Checklist

This checklist guides the collection of real-world video dataset clips for the **onboard-har** Human Activity Recognition and Fall Detection project.

> [!IMPORTANT]
> **Scientific Integrity Rules:**
> - **NO** synthetic videos or fake feature vectors.
> - **NO** fabricated metrics or hard-coded accuracy claims.
> - **NO** model training until real dataset requirements are met.

---

## 1. Required Activity Classes
Dataset recordings must cover all five target activities:
1. `STANDING` — Stationary upright posture.
2. `SITTING` — Stationary seated posture (chair, stool, bench).
3. `WALKING` — Pacing across or towards/away from camera.
4. `RUNNING` — Elevated speed gait.
5. `FALLING` — Staged controlled fall onto protective padding.

---

## 2. Recommended Collection Targets
- **Recordings:** Minimum 10 video clips per activity class.
- **Subjects:** At least 3 independent human participants.
- **Sessions:** Multiple distinct recording sessions.
- **Diversity:** Vary body orientation (frontal, $45^\circ$, side profile), movement direction, speed, and camera distance (2.0m – 5.0m).

---

## 3. Recording Command Format
Use the existing recording utility:

```bash
python tools/record_dataset.py --class <CLASS> --subject <SUBJECT_ID> --duration 10
```

### Examples:
```bash
python tools/record_dataset.py --class STANDING --subject SUBJ_01 --duration 10
python tools/record_dataset.py --class SITTING --subject SUBJ_01 --duration 10
python tools/record_dataset.py --class WALKING --subject SUBJ_01 --duration 10
python tools/record_dataset.py --class RUNNING --subject SUBJ_01 --duration 10
python tools/record_dataset.py --class FALLING --subject SUBJ_01 --duration 10
```

---

## 4. File Storage Location
Video clips are automatically routed to class subdirectories:
```text
dataset/raw/STANDING/
dataset/raw/SITTING/
dataset/raw/WALKING/
dataset/raw/RUNNING/
dataset/raw/FALLING/
```

---

## 5. Automatic Manifest Tracking
Every recording session automatically appends video metadata (ID, class, subject ID, session ID, resolution, FPS, duration, environment) to:
```text
dataset/manifest.csv
```

---

## 6. Safe FALLING Protocol
> [!WARNING]
> **Safety First for Controlled Falls:**
> - Perform staged falls **ONLY** onto heavy crash pads, gym mats, or soft mattresses.
> - Ensure surrounding area is free of sharp edges, furniture, or hard obstacles.
> - **NEVER** perform dangerous, unpadded, or uncontrolled falls.
> - Stop recording immediately if any participant feels unsafe.

---

## 7. Verification & Training Gate Execution
After collecting recordings, verify dataset readiness before initiating feature extraction or model training:

```bash
python training_readiness.py
```

If readiness requirements pass, execute feature extraction, dataset splitting, and model training:
```bash
python collect_features.py --data dataset/raw
python split_dataset.py
python train_classifier.py --compare
```
