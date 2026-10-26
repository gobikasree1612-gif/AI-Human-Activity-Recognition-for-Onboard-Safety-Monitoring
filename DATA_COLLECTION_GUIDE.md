# Practical Data Collection Guide for On-Board BAS HAR

This guide outlines practical instructions for recording real human activity video data for the `onboard-har` AI system.

---

## 1. Safety Guidelines & Rules

> [!CAUTION]
> **SAFETY FIRST FOR FALLING DATA:**
> - All `FALLING` recordings must be **staged safely in controlled environments**.
> - Use thick gymnastics mats, foam landing pads, or crash cushions.
> - Never instruct or allow participants to perform uncontrolled, painful, or dangerous falls onto hard surfaces.
> - If safe staging is not feasible, do not record real impact falls—record controlled descent or gentle floor-transition postures.

---

## 2. Supported Activity Classes & Recording Protocols

Record videos per class in separate directories under `dataset/raw/<CLASS>/`.

### 🧍 STANDING
- **Goal:** Capture natural upright postures across multiple individuals.
- **Variations to record:**
  - Stationary standing facing the camera, profile, and 45° angles.
  - Natural weight shifts, fidgeting, arms crossed, hands in pockets.
  - Holding equipment or a tablet/clipboard (typical onboard BAS inspection tasks).
- **Duration per clip:** 10–20 seconds. Record 5–10 distinct clips with at least 2–3 participants.

### 🪑 SITTING
- **Goal:** Distinguish sitting from standing and bending.
- **Variations to record:**
  - Static sitting on chairs of different heights, stools, or benches.
  - Transitions into sitting (walking up to a chair and sitting down).
  - Transitions out of sitting (rising to stand).
  - Leaning forward, leaning back, desk working postures.
- **Duration per clip:** 10–25 seconds.

### 🚶 WALKING
- **Goal:** Normal locomotive gait under varied speeds and paths.
- **Variations to record:**
  - Walking across the camera frame (left-to-right, right-to-left).
  - Walking towards and away from the camera.
  - Walking at slow, moderate, and brisk paces.
  - Turning around and changing direction mid-walk.
- **Duration per clip:** 10–20 seconds.

### 🏃 RUNNING
- **Goal:** High-velocity movements and forward lean.
- **Variations to record:**
  - Controlled jogging or rapid pace in a clear hallway or open area.
  - Ensure the recording area is completely clear of tripping hazards.
  - Running across the field of view and running towards the camera.
- **Duration per clip:** 8–15 seconds.

### ⚠️ FALLING (Controlled Staging Only)
- **Goal:** Detecting sudden collapse, slips, or trips followed by a horizontal posture.
- **Safe Staging Methods:**
  - **Slip on mat:** Controlled slip backwards onto a soft crash pad.
  - **Trip on mat:** Controlled forward stumble onto landing cushions.
  - **Faint / collapse:** Slump from standing/kneeling down onto mats.
  - **Post-fall stillness:** Remain prone or supine on the mat for 3–5 seconds after falling so the state machine can confirm the resting posture.
- **Duration per clip:** 8–15 seconds.

---

## 3. Recommended Variations & Environmental Conditions

To build an AI model that generalizes well and avoids overfitting to a single person or scene, introduce planned variations:

| Dimension | Recommended Variations |
|-----------|------------------------|
| **Participants** | At least 3 different subjects across age and build |
| **Clothing** | Loose vs. tight clothing, jackets, lab coats, different colors |
| **Distance** | 2 meters (near), 4 meters (mid), 6 meters (far) |
| **Camera Angle** | Eye-level (1.6m), elevated ceiling/wall mount (2.2m–2.5m) |
| **Lighting** | Standard fluorescent lab light, dimmed ambient light, window daylight |
| **Background** | Corridor, equipment rack area, open laboratory floor |
| **Orientation** | Frontal, lateral (profile), 45° oblique, facing away |
| **Partial Occlusion** | Subject walking behind a table, desk, or equipment cart |

---

## 4. File Organization & Naming Convention

Organize videos directly in `dataset/raw/`:

```text
dataset/raw/
├── STANDING/
│   ├── STANDING_S01_SESSION01_01.mp4
│   └── STANDING_S02_SESSION01_01.mp4
├── SITTING/
│   ├── SITTING_S01_SESSION01_01.mp4
│   └── SITTING_S03_SESSION02_01.mp4
├── WALKING/
│   ├── WALKING_S01_SESSION01_01.mp4
│   └── WALKING_S02_SESSION02_01.mp4
├── RUNNING/
│   ├── RUNNING_S02_SESSION01_01.mp4
│   └── RUNNING_S03_SESSION01_01.mp4
└── FALLING/
    ├── FALLING_S01_SESSION01_mat01.mp4
    └── FALLING_S02_SESSION02_mat02.mp4
```

### Naming Pattern:
`<CLASS>_<SUBJECT-ID>_<SESSION-ID>_<CLIP-INDEX>.mp4`
- Example: `WALKING_S01_SESSION01_01.mp4`
- This naming pattern allows `dataset_manifest.py` and `split_dataset.py` to automatically index subjects and sessions without data leakage.

---

## 5. Tooling Support

The repository includes utilities to record, validate, and extract features:

1. **Webcam Recorder Tool:**
   ```bash
   python tools/record_dataset.py --class STANDING --subject S01 --session SESSION01
   ```

2. **Generate / Update Dataset Manifest:**
   ```bash
   python dataset_manifest.py
   ```

3. **Check Quality & Inspect Videos:**
   ```bash
   python dataset_quality.py --data dataset/raw
   ```

4. **Verify Class Balance:**
   ```bash
   python dataset_balance.py
   ```

5. **Generate Leakage-Safe Splits:**
   ```bash
   python split_dataset.py
   ```

6. **Batch Extract Pose Features:**
   ```bash
   python collect_features.py --data dataset/raw
   ```
