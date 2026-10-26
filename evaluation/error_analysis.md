# Error Analysis & Failure Taxonomy

This document categorizes error failure modes and diagnostic procedures for the onboard-har AI engine.

> [!NOTE]
> Detailed per-class empirical error counts will be populated automatically following execution of `train_classifier.py` and `evaluate()` on a real held-out test split.

---

## 1. Failure Taxonomy

| Category | Primary Symptom | Root Cause / Contributing Factor | Mitigation / Remediation |
|---|---|---|---|
| **Pose Detection Failure** | Person detected but keypoints missing or displaced | Low light, heavy clothing, extreme occlusion | Omit low-confidence keypoints (<0.40); maintain temporal history |
| **Tracking ID Switch** | Person ID changes between consecutive frames | Bounding box overlap during multi-person crossing | Centroid distance tracking thresholding (`max_missing_frames=60`) |
| **Temporal Instability** | Rapid toggling between adjacent classes (e.g. STANDING $\leftrightarrow$ WALKING) | Single-frame feature noise near decision boundary | Exponential moving average & temporal buffer smoothing |
| **Class Confusion (Sitting vs Standing)** | Sitting classified as Standing | Torso verticality similar in high chair | Check knee angle ($\theta_{hip-knee-ankle}$) and hip height relative to bounding box |
| **Fall Detection Delay** | Delayed confirmation of fall event | Insufficient confirmation frames elapsed | Calibrate `confirmation_frames` (default: 15 frames) |

---

## 2. Real-World Fall Detection Diagnostics

The fall detector state machine evaluates three distinct physical triggers:
1. **Body Angle Deviation:** Torso lean angle $> 55^\circ$ from vertical.
2. **Downward Velocity:** Rapid downward displacement of hip center ($\Delta y_{hip} > 0.04$ normalized units/frame).
3. **Torso Height Compression:** Ratio of visible torso height to baseline $< 0.55$.

### State Machine Transitions
- **NORMAL State:** No fall indicators.
- **POSSIBLE_FALL State:** Triggered immediately when body angle $> 55^\circ$ OR downward velocity exceeds threshold.
- **FALL_CONFIRMED State:** Promoted from `POSSIBLE_FALL` when body remains in low vertical position for $\ge 15$ consecutive confirmation frames.
- **TIMEOUT / RECOVERY State:** If person returns to upright posture before 15 frames, system reverts to `NORMAL`.

---

## 3. Empirical Error Reporting Template

When real dataset evaluation is executed, `train_classifier.py` generates `evaluation/confusion_matrix_analysis.txt` detailing exact misclassifications:

```text
Observed confusion:
CLASS A → CLASS B: X samples
CLASS B → CLASS A: Y samples
```
