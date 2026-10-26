# Onboard-HAR Dataset Collection Checklist

Use this checklist during every video recording session to ensure data quality, reproducibility, and safety.

---

## 1. Pre-Recording Protocol

Before launching `python tools/record_dataset.py`, verify all setup parameters:

- [ ] **Camera Operational:** Camera index verified, live feed clear without severe lag or dropping frames.
- [ ] **Lighting Checked:** Scene is evenly lit; subject is not heavily silhouetted against bright windows or light sources.
- [ ] **Background Suitable:** Target environment floor and background free of clutter and irrelevant bystanders.
- [ ] **Camera Position Documented:** Mounting height (e.g., 1.8m eye-level or 2.4m elevated), tilt angle, and focal zone noted.
- [ ] **Subject ID Confirmed:** Distinct alphanumeric identifier assigned (e.g. `S01`, `S02`, `S03`) without duplicate confusion.
- [ ] **Session ID Confirmed:** Session identifier formatted consistently (e.g. `SESSION01`, `SESSION02`).
- [ ] **Activity Label Confirmed:** Verified that the action strictly represents one of `STANDING`, `SITTING`, `WALKING`, `RUNNING`, or `FALLING`.
- [ ] **Storage Available:** Sufficient disk space for uncompressed video clips.
- [ ] **Safety Review:** For `FALLING` sessions, thick impact-absorbing landing cushions/mats are verified in place.

---

## 2. Planned Environmental & Postural Variations

To ensure the trained model generalizes and does not overfit to specific individuals or settings, intentionally introduce variations across recordings:

- [ ] **Distance from Camera:**
  - Close range: 2 to 3 meters
  - Mid range: 3 to 5 meters
  - Far range: 5 to 7 meters
- [ ] **Movement Direction:**
  - Lateral motion: Left-to-Right and Right-to-Left
  - Longitudinal motion: Approaching camera and moving away from camera
  - Diagonal paths and turns
- [ ] **Body Orientation:**
  - Frontal (facing camera directly)
  - Profile (facing 90° lateral)
  - Oblique (facing 45°)
  - Posterior (back towards camera)
- [ ] **Clothing & Appearance:**
  - Different outerwear (jackets, lab coats, short-sleeve vs long-sleeve shirts)
  - Different footwear and pant styles
- [ ] **Lighting Variations:**
  - Standard overhead laboratory fluorescent/LED lighting
  - Dimmed or localized accent illumination
  - Natural ambient daylight variations
- [ ] **Backgrounds:**
  - Open laboratory floor
  - Equipment rack hallway
  - Desk / workstation perimeter
- [ ] **Movement Speeds:**
  - Deliberate / slow pace
  - Natural moderate tempo
  - Brisk / rapid pace (controlled and safe)
- [ ] **Camera Heights:**
  - Eye-level perspective (~1.6m)
  - Overhead / wall-mount perspective (~2.2m to 2.5m)
- [ ] **Partial Occlusion:**
  - Walking behind a lab cart, desk edge, or partition where legs/feet are momentarily obscured.

---

## 3. Post-Recording Verification

Immediately after stopping recording:

- [ ] Video playback verified: file opens cleanly, no corrupt frames or frozen streams.
- [ ] Video duration verified: at least 5–10 seconds of clear activity captured.
- [ ] Metadata logged: filename conforms to `<CLASS>_<SUBJECT>_<SESSION>_<TIMESTAMP>.mp4`.
- [ ] Manifest updated: entry exists in `dataset/manifest.csv`.
