# System Architecture Documentation

The **onboard-har** architecture is designed for real-time edge execution, privacy-first local processing, and high safety reliability.

## Core Modules

### 1. `ai/detector.py` — Human Pose Detector
- **Model:** YOLO11n-Pose (`yolo11n-pose.pt`)
- **Keypoints:** 17 COCO joints (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles).
- **Filtering:** Bounding box NMS (`iou_threshold=0.45`) and confidence thresholding (`detection_confidence=0.40`).

### 2. `ai/features.py` — 34D Feature Extractor
Computes a normalized 34-dimensional feature vector per frame:
- **Joint Angles (0–6):** Left/Right Elbow, Shoulder, Hip, Knee angles, Body Lean Angle.
- **Normalized Geometry (7–14):** Torso Height, Body Height, Shoulder Width, Hip Width, Knee-to-Hip ratios normalized by bounding box dimension.
- **Normalized Positions (15–21):** Bounding box center $(x, y)$, Hip center, Head position, Feet position normalized to $[0, 1]$.
- **Kinematic Velocities (22–33):** Frame-to-frame $\Delta x, \Delta y$ velocity of Keypoint Centers and Joints.

### 3. `ai/temporal.py` — Temporal Sequence Processor
- Maintains a 20-frame sliding window per tracked person.
- Applies Exponential Moving Average (EMA) smoothing ($\alpha = 0.3$) to minimize keypoint jitter.

### 4. `ai/classifier.py` — Activity Classifier Engine
- Dual-mode architecture:
  - **Trained ML:** Joblib-loaded Scikit-Learn RandomForest/XGBoost model.
  - **Rule-based Baseline:** Heuristic decision tree evaluated when no trained model artifact exists or when model integrity check fails.

### 5. `ai/fall_detector.py` — Fall Detector
- 3-stage state machine: `NORMAL` $\to$ `POSSIBLE_FALL` $\to$ `FALL_CONFIRMED`.
- Triggers `POSSIBLE_FALL` when body angle exceeds $55^\circ$ or downward velocity exceeds $0.04$ normalized units/frame.
- Confirms fall after 15 consecutive low-position frames.

### 6. `ai/safety.py` — Safety State & Explainability Engine
- Aggregates activity classification and fall state into 5 safety tiers (`SAFE`, `CAUTION`, `WARNING`, `DANGER`, `CRITICAL`).
- Generates human-readable natural language evidence for dashboard logging.
