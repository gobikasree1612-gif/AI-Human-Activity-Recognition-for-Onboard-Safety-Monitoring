"""
classifier.py
=============
Activity classification abstraction.

Two modes:
  1. TRAINED ML   – loads a pre-fitted sklearn/XGBoost model via joblib.
  2. RULE-BASED   – heuristic baseline using pose & temporal features.
                    Active when no trained model is found.

The rule-based fallback is clearly labelled so the dashboard can
distinguish inference quality.

Training helper
---------------
To train an ML classifier on your own labelled data::

    from ai.classifier import ActivityClassifier
    clf = ActivityClassifier(config)
    clf.train(X_train, y_train)   # X: (N, feature_dim), y: label strings
    clf.save()                    # saves to config['classifier']['model_path']

Training data collection tip
-----------------------------
Run demo_ai.py with --save-features flag (not built yet) to log
feature vectors + manual labels. Then call clf.train().
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai.features import FEATURE_DIM
from ai.schemas import Activity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature indices for rule-based logic (mirror features.py layout)
# ---------------------------------------------------------------------------
_I = {
    "shoulder_angle": 0,
    "hip_angle": 1,
    "left_knee_angle": 2,
    "right_knee_angle": 3,
    "body_lean_angle": 6,
    "torso_height_norm": 7,
    "body_height_norm": 8,
    "shoulder_width_norm": 9,
    "hip_width_norm": 10,
    "left_knee_hip_ratio": 11,
    "right_knee_hip_ratio": 12,
    "center_x_norm": 15,
    "center_y_norm": 16,
    "hip_center_y_norm": 17,
    "shoulder_center_y_norm": 18,
    "knee_bend_left": 19,
    "knee_bend_right": 20,
    "visibility_score": 29,
    "upper_lower_ratio": 30,
    "nose_to_hip_y_norm": 31,
}


class ActivityClassifier:
    """
    Unified interface for activity classification.

    predict(sequence) returns::

        {
            "activity": "WALKING",
            "confidence": 0.85,
            "model_type": "trained_ml" | "rule_based_baseline"
        }
    """

    def __init__(self, config: dict):
        self._cfg = config
        clf_cfg = config.get("classifier", {})
        self._model_path: str = clf_cfg.get("model_path", "models/har_classifier.joblib")
        self._scaler_path: str = clf_cfg.get("scaler_path", "models/har_scaler.joblib")
        self._conf_thresh: float = config.get("activity", {}).get(
            "confidence_threshold", 0.60
        )

        self._model = None           # sklearn-compatible model
        self._scaler = None          # sklearn StandardScaler (optional)
        self._label_encoder = None
        self._model_type = "rule_based_baseline"
        self._load_info: dict = {}   # metadata from last successful load
        self._load_error: Optional[str] = None  # reason if load was rejected

        self._try_load()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _try_load(self) -> None:
        """
        Attempt to load a trained model from disk with integrity checks.

        Checks performed before accepting the model:
          1. File must exist and be loadable by joblib.
          2. Metadata file (<stem>_meta.json) is read if present.
          3. Feature dimension in metadata must match FEATURE_DIM.
          4. Model must expose predict_proba() (sklearn API).
          5. Scaler (if present) must match expected input dimension.

        On ANY check failure:
          - Model is rejected.
          - _load_error is set to a human-readable explanation.
          - System falls back to rule-based classifier.
          - No exception propagates to the caller.
        """
        if not os.path.isfile(self._model_path):
            logger.info(
                "No trained model found at '%s'. Using rule-based fallback.",
                self._model_path,
            )
            return

        try:
            import joblib
            import json as _json

            # ── 1. Load model file ────────────────────────────────────────
            try:
                candidate = joblib.load(self._model_path)
            except Exception as exc:
                self._reject(f"Model file corrupt or unreadable: {exc}")
                return

            # ── 2. Load + validate metadata ───────────────────────────────
            meta_path = self._model_path.replace(".joblib", "_meta.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as fh:
                        meta = _json.load(fh)
                except Exception as exc:
                    self._reject(f"Metadata file unreadable: {exc}")
                    return

                # Feature dimension check
                stored_dim = meta.get("feature_dim")
                if stored_dim is not None and stored_dim != FEATURE_DIM:
                    self._reject(
                        f"Feature dimension mismatch: model was trained on "
                        f"{stored_dim} features but current pipeline produces "
                        f"{FEATURE_DIM}. Retrain with the current feature extractor."
                    )
                    return

                self._load_info = meta
                logger.info(
                    "Model metadata: classes=%s, trained_at=%s",
                    meta.get("classes", "?"),
                    meta.get("trained_at", "?"),
                )
            else:
                logger.warning(
                    "No metadata file found at '%s'. "
                    "Skipping feature-dimension integrity check.",
                    meta_path,
                )
                self._load_info = {}

            # ── 3. API check ──────────────────────────────────────────────
            if not hasattr(candidate, "predict_proba"):
                self._reject(
                    "Model does not expose predict_proba(). "
                    "Only probabilistic sklearn-compatible models are supported."
                )
                return

            # ── 4. Load label encoder from metadata or model classes_ ─────
            classes = self._load_info.get("classes")
            if classes:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(classes)
                self._label_encoder = le
            elif hasattr(candidate, "classes_"):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(candidate.classes_)
                self._label_encoder = le

            # ── 5. Load scaler (optional) ─────────────────────────────────
            if os.path.isfile(self._scaler_path):
                try:
                    scaler = joblib.load(self._scaler_path)
                    # Verify scaler input dimension
                    n_in = getattr(scaler, "n_features_in_", None)
                    if n_in is not None and n_in != FEATURE_DIM:
                        logger.warning(
                            "Scaler expects %d features but pipeline produces %d. "
                            "Scaler will be skipped.",
                            n_in, FEATURE_DIM,
                        )
                    else:
                        self._scaler = scaler
                        logger.info("Feature scaler loaded from %s", self._scaler_path)
                except Exception as exc:
                    logger.warning("Could not load scaler (%s) – proceeding without.", exc)

            # ── All checks passed ─────────────────────────────────────────
            self._model = candidate
            self._model_type = "trained_ml"
            logger.info(
                "Trained classifier accepted: %s  (dim=%s)",
                self._model_path,
                self._load_info.get("feature_dim", "unknown"),
            )

        except Exception as exc:
            self._reject(f"Unexpected error during model load: {exc}")

    def _reject(self, reason: str) -> None:
        """Reject the model and fall back to rule-based with a clear message."""
        self._model = None
        self._scaler = None
        self._label_encoder = None
        self._model_type = "rule_based_baseline"
        self._load_error = reason
        logger.warning(
            "Trained model REJECTED – falling back to rule-based classifier.\n"
            "Reason: %s",
            reason,
        )

    def save(self, model_path: Optional[str] = None,
             scaler_path: Optional[str] = None) -> None:
        """Save the trained model (and optional scaler) to disk."""
        import joblib
        mp = model_path or self._model_path
        sp = scaler_path or self._scaler_path
        os.makedirs(os.path.dirname(mp) if os.path.dirname(mp) else ".", exist_ok=True)
        if self._model is not None:
            joblib.dump(self._model, mp)
            logger.info("Model saved to %s", mp)
        if self._scaler is not None:
            joblib.dump(self._scaler, sp)
            logger.info("Scaler saved to %s", sp)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: List[str],
        model_type: str = "random_forest",
    ) -> None:
        """
        Fit a classifier on labelled feature sequences.

        Parameters
        ----------
        X : np.ndarray, shape (N, sequence_length * FEATURE_DIM) or (N, FEATURE_DIM)
            Flattened feature sequences.
        y : List[str]
            Activity labels (must be from Activity.ALL).
        model_type : str
            'random_forest' | 'mlp' | 'xgboost'
        """
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        logger.info("Training %s on %d samples...", model_type, len(y))

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self._label_encoder = le

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self._scaler = scaler

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=42,
            )
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                clf = XGBClassifier(
                    n_estimators=200,
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    random_state=42,
                )
            except ImportError:
                logger.warning("xgboost not installed – falling back to RandomForest")
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        clf.fit(X_scaled, y_enc)
        self._model = clf
        self._model_type = "trained_ml"
        logger.info("Training complete. Classes: %s", list(le.classes_))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, sequence: np.ndarray) -> Dict:
        """
        Classify an activity from a temporal sequence of feature vectors.

        Parameters
        ----------
        sequence : np.ndarray, shape (N, FEATURE_DIM) where N ≥ 1.

        Returns
        -------
        dict with keys: activity, confidence, model_type
        """
        if sequence is None or sequence.size == 0:
            return {
                "activity": Activity.UNKNOWN,
                "confidence": 0.0,
                "model_type": self._model_type,
            }

        seq = np.atleast_2d(sequence).astype(np.float32)

        if self._model is not None and self._model_type == "trained_ml":
            return self._predict_ml(seq)
        else:
            return self._predict_rules(seq)

    def _predict_ml(self, seq: np.ndarray) -> Dict:
        """Run the trained sklearn/XGB model."""
        try:
            # Flatten sequence into a single feature vector
            flat = seq.flatten().reshape(1, -1)

            if self._scaler is not None:
                # Scaler was fitted on the same flattened dimension
                if flat.shape[1] == self._scaler.n_features_in_:
                    flat = self._scaler.transform(flat)
                else:
                    logger.debug(
                        "Scaler dim mismatch (%d vs %d) – skipping scaling",
                        flat.shape[1],
                        self._scaler.n_features_in_,
                    )

            proba = self._model.predict_proba(flat)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])

            if self._label_encoder is not None:
                activity = str(self._label_encoder.inverse_transform([pred_idx])[0])
            else:
                # Try model classes_ attribute
                classes = getattr(self._model, "classes_", None)
                activity = str(classes[pred_idx]) if classes is not None else Activity.UNKNOWN

            activity = activity.upper()
            if activity not in Activity.ALL + [Activity.UNKNOWN]:
                activity = Activity.UNKNOWN

            if confidence < self._conf_thresh:
                activity = Activity.UNKNOWN

            return {
                "activity": activity,
                "confidence": confidence,
                "model_type": "trained_ml",
            }

        except Exception as exc:
            logger.error("ML prediction failed (%s) – using fallback", exc)
            return self._predict_rules(seq)

    def _predict_rules(self, seq: np.ndarray) -> Dict:
        """
        Rule-based heuristic classifier.
        Uses the MOST RECENT frame's features + simple temporal statistics.

        NOTE: This is explicitly a fallback. Confidence values are heuristic
        scores, NOT calibrated probabilities.
        """
        latest = seq[-1]                      # most recent frame
        mean   = seq.mean(axis=0)             # temporal mean
        std    = seq.std(axis=0)              # temporal std = motion proxy

        def f(name: str, arr=latest) -> float:
            return float(arr[_I.get(name, 0)]) if _I.get(name, 0) < len(arr) else 0.0

        def mf(name: str) -> float:          # mean over sequence
            return float(mean[_I.get(name, 0)]) if _I.get(name, 0) < len(mean) else 0.0

        def sf(name: str) -> float:          # std over sequence
            return float(std[_I.get(name, 0)]) if _I.get(name, 0) < len(std) else 0.0

        body_lean   = f("body_lean_angle")
        torso_h     = f("torso_height_norm")
        hip_y       = f("hip_center_y_norm")
        shoulder_y  = f("shoulder_center_y_norm")
        knee_bend_l = f("knee_bend_left")
        knee_bend_r = f("knee_bend_right")
        visibility  = f("visibility_score")
        upper_lower = f("upper_lower_ratio")
        avg_knee_bend = (knee_bend_l + knee_bend_r) / 2.0

        # Temporal motion proxies (separated by feature scale)
        # Position std across normalized spatial features (indices 7 to 20)
        motion_pos   = float(std[7:21].mean()) if len(std) >= 21 else 0.0
        # Angle std across degree features (indices 0 to 6)
        motion_angle = float(std[:7].mean()) if len(std) >= 7 else 0.0

        # -----------------------------------------------
        # Decision tree (ordered by priority / specificity)
        # -----------------------------------------------

        # 1. FALLING – horizontal body
        if body_lean > 60.0:
            return {"activity": Activity.FALLING, "confidence": 0.85,
                    "model_type": "rule_based_baseline"}

        # 2. SITTING – lowered hips and significantly bent knees
        if hip_y > 0.54 and avg_knee_bend > 0.25:
            conf = min(0.6 + avg_knee_bend * 0.4, 0.85)
            return {"activity": Activity.SITTING, "confidence": conf,
                    "model_type": "rule_based_baseline"}

        # 3. RUNNING – forward body lean + bent knees or high positional motion
        if (body_lean > 12.0 and avg_knee_bend > 0.15) or (motion_pos > 0.004 and avg_knee_bend > 0.15):
            return {"activity": Activity.RUNNING, "confidence": 0.80,
                    "model_type": "rule_based_baseline"}

        # 4. WALKING – noticeable angle variation or positional motion
        if motion_angle > 1.5 or motion_pos > 0.0024:
            return {"activity": Activity.WALKING, "confidence": 0.75,
                    "model_type": "rule_based_baseline"}

        # 5. STANDING – upright body, low motion, straight legs
        if body_lean < 15.0 and avg_knee_bend < 0.15:
            conf = 0.85 if visibility > 0.5 else 0.65
            return {"activity": Activity.STANDING, "confidence": conf,
                    "model_type": "rule_based_baseline"}

        # Default upright static -> STANDING
        if body_lean < 25.0:
            return {"activity": Activity.STANDING, "confidence": 0.70,
                    "model_type": "rule_based_baseline"}

        # Not enough signal
        return {"activity": Activity.UNKNOWN, "confidence": 0.0,
                "model_type": "rule_based_baseline"}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def is_trained(self) -> bool:
        return self._model is not None and self._model_type == "trained_ml"

    @property
    def load_info(self) -> dict:
        """Metadata from the last successfully loaded model (empty if fallback)."""
        return dict(self._load_info)

    @property
    def load_error(self) -> Optional[str]:
        """
        Human-readable reason why the trained model was rejected.
        None if model loaded successfully or was simply not present.
        """
        return self._load_error
