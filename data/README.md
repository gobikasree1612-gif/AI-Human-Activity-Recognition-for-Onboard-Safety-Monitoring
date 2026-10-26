# Data Directory

This directory stores training data for the activity classifier.

## Expected Format

### Feature CSV (for training)

Each row = one temporal sequence sample:

```
label, f0, f1, f2, ..., f(seq_len * FEATURE_DIM - 1)
STANDING, 0.12, 0.03, ...
WALKING,  0.45, 0.22, ...
```

Where `seq_len = 20` and `FEATURE_DIM = 34` → 680 feature columns per row.

## How to Collect Training Data

Currently the system does not include an automated data collection script.

Suggested approach:

1. Run `demo_ai.py` and record video clips of each activity.
2. Label timestamps manually.
3. Use the `TemporalBuffer` + `extract_pose_features` to extract feature
   sequences from the labelled clips:

```python
import cv2
from ai.inference import HAREngine
from ai.features import extract_pose_features
from ai.temporal import TemporalBuffer

engine = HAREngine()
buffer = TemporalBuffer(sequence_length=20)
cap = cv2.VideoCapture("standing_clip.mp4")

X, y = [], []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    people = engine._detector.process(frame)
    for person in people:
        fv = extract_pose_features(person.keypoints, person.bbox)
        buffer.add(person.person_id, fv)
        if buffer.is_ready(person.person_id):
            seq = buffer.get_sequence(person.person_id)
            X.append(seq.flatten())
            y.append("STANDING")   # manual label

import numpy as np
X = np.array(X)
# Then train: clf.train(X, y)
```

## Activity Classes

- STANDING
- SITTING
- WALKING
- RUNNING
- FALLING

> [!IMPORTANT]
> Data balance matters. Try to collect roughly equal samples per class.
> FALLING data is rare — use careful simulation only.
