# Models Directory

This directory stores trained model artifacts:

| File | Description |
|------|-------------|
| `har_classifier.joblib` | Trained activity classifier (RandomForest/MLP/XGBoost) |
| `har_scaler.joblib` | Fitted `StandardScaler` for feature normalisation |

## Status

**No trained model is currently present.**

The system will automatically fall back to the **rule-based baseline** classifier.

All AI outputs in rule-based mode are labelled:
```json
"model_type": "rule_based_baseline"
```

## How to Train

1. Collect labelled feature vectors (see `data/` directory).
2. Run the training script:

```python
from ai.classifier import ActivityClassifier
import numpy as np

clf = ActivityClassifier(config={
    "classifier": {
        "model_path": "models/har_classifier.joblib",
        "scaler_path": "models/har_scaler.joblib",
    },
    "activity": {"confidence_threshold": 0.60},
})

# X: (N, sequence_length * FEATURE_DIM) – flattened sequences
# y: list of label strings e.g. ["STANDING", "WALKING", ...]
clf.train(X_train, y_train, model_type="random_forest")
clf.save()
```

3. Restart the engine. It will auto-detect and load the saved model.

## YOLO Pose Model

The YOLO Pose model (`yolo11n-pose.pt`) is **not stored here** – it is
automatically downloaded by Ultralytics on first run.

To pre-download:
```python
from ultralytics import YOLO
YOLO("yolo11n-pose.pt")   # downloads to ~/.ultralytics/
```
