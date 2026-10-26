"""
test_classifier_integrity.py
=============================
Step 11 & Step 10 verification:
  - Model present: loads, sets trained_ml, generates valid predictions.
  - Model absent: graceful fallback to rule_based_baseline, no crash.
  - Corrupt model: invalid binary rejected gracefully, fallback active.
  - Feature mismatch: wrong feature dimension rejected with clear error.
  - Non-probabilistic model: rejected if predict_proba is missing.
  - Scaler mismatch: handled safely without crash.
  - Label mapping: predicted class labels match training classes.
  - Diagnostics exposure: load_info and load_error reflected accurately.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from ai.classifier import ActivityClassifier, FEATURE_DIM
from ai.schemas import Activity


class NonProbabilisticModel:
    """Model lacking predict_proba for testing rejection."""
    def predict(self, X):
        return [0] * len(X)


class TestClassifierIntegrity(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_clf.joblib")
        self.scaler_path = os.path.join(self.test_dir, "test_scaler.joblib")
        self.meta_path = os.path.join(self.test_dir, "test_clf_meta.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_model(self, feature_dim=FEATURE_DIM, classes=None):
        if classes is None:
            classes = [Activity.STANDING, Activity.SITTING, Activity.WALKING]
        from sklearn.tree import DecisionTreeClassifier
        X = []
        y = []
        for i, c in enumerate(classes):
            for _ in range(5):
                row = np.zeros(feature_dim, dtype=np.float32)
                row[0] = float(i)
                X.append(row)
                y.append(c)
        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, self.model_path)

        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, self.scaler_path)

        meta = {
            "model_type": "decision_tree",
            "feature_dim": feature_dim,
            "classes": classes,
            "train_samples": len(X),
            "test_accuracy": 1.0,
            "split_strategy": "by_source_video (GroupShuffleSplit)",
        }
        with open(self.meta_path, "w") as fh:
            json.dump(meta, fh)
        return clf, scaler, meta

    # ── 1. Model absent ───────────────────────────────────────────────────────
    def test_model_absent_fallback(self):
        cfg = {"classifier": {"model_path": os.path.join(self.test_dir, "nonexistent.joblib")}}
        clf = ActivityClassifier(config=cfg)
        self.assertEqual(clf.model_type, "rule_based_baseline")
        self.assertFalse(clf.is_trained)
        self.assertIsNone(clf.load_error)
        self.assertEqual(clf.load_info, {})

        # Predict should run without error
        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        res = clf.predict(fv)
        self.assertEqual(res["model_type"], "rule_based_baseline")
        self.assertIn("activity", res)
        self.assertIn("confidence", res)

    # ── 2. Model present ──────────────────────────────────────────────────────
    def test_model_present_loaded_and_predicts(self):
        self._create_dummy_model()
        cfg = {
            "classifier": {
                "model_path": self.model_path,
                "scaler_path": self.scaler_path,
            }
        }
        clf = ActivityClassifier(config=cfg)
        self.assertEqual(clf.model_type, "trained_ml")
        self.assertTrue(clf.is_trained)
        self.assertIsNone(clf.load_error)
        self.assertEqual(clf.load_info.get("feature_dim"), FEATURE_DIM)

        # Predict should return trained_ml
        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        res = clf.predict(fv)
        self.assertEqual(res["model_type"], "trained_ml")
        self.assertIn(res["activity"], [Activity.STANDING, Activity.SITTING, Activity.WALKING])
        self.assertGreater(res["confidence"], 0.6)

    # ── 3. Corrupt model ──────────────────────────────────────────────────────
    def test_corrupt_model_rejected(self):
        with open(self.model_path, "wb") as fh:
            fh.write(b"NOT_A_VALID_JOBLIB_FILE_GARBAGE_BYTES_1234567890")
        cfg = {"classifier": {"model_path": self.model_path}}
        clf = ActivityClassifier(config=cfg)

        self.assertEqual(clf.model_type, "rule_based_baseline")
        self.assertFalse(clf.is_trained)
        self.assertIsNotNone(clf.load_error)
        self.assertIn("corrupt", clf.load_error.lower())

        # Prediction still works via fallback
        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        res = clf.predict(fv)
        self.assertEqual(res["model_type"], "rule_based_baseline")

    # ── 4. Feature mismatch in metadata ───────────────────────────────────────
    def test_feature_mismatch_rejected(self):
        self._create_dummy_model(feature_dim=99)
        cfg = {"classifier": {"model_path": self.model_path}}
        clf = ActivityClassifier(config=cfg)

        self.assertEqual(clf.model_type, "rule_based_baseline")
        self.assertFalse(clf.is_trained)
        self.assertIsNotNone(clf.load_error)
        self.assertIn("mismatch", clf.load_error.lower())

    # ── 5. Missing predict_proba ──────────────────────────────────────────────
    def test_model_missing_predict_proba_rejected(self):
        joblib.dump(NonProbabilisticModel(), self.model_path)
        with open(self.meta_path, "w") as fh:
            json.dump({"feature_dim": FEATURE_DIM, "classes": ["STANDING"]}, fh)

        cfg = {"classifier": {"model_path": self.model_path}}
        clf = ActivityClassifier(config=cfg)

        self.assertEqual(clf.model_type, "rule_based_baseline")
        self.assertFalse(clf.is_trained)
        self.assertIsNotNone(clf.load_error)
        self.assertIn("predict_proba", clf.load_error)

    # ── 6. Scaler dimension mismatch handled gracefully ───────────────────────
    def test_scaler_dimension_mismatch_skipped(self):
        self._create_dummy_model(feature_dim=FEATURE_DIM)
        # Overwrite scaler with incompatible dimension
        bad_scaler = StandardScaler()
        bad_scaler.fit(np.ones((10, 50)))
        joblib.dump(bad_scaler, self.scaler_path)

        cfg = {
            "classifier": {
                "model_path": self.model_path,
                "scaler_path": self.scaler_path,
            }
        }
        # Model should still load, scaler skipped safely
        clf = ActivityClassifier(config=cfg)
        self.assertEqual(clf.model_type, "trained_ml")
        self.assertTrue(clf.is_trained)

    # ── 7. Label mapping matches training classes ─────────────────────────────
    def test_label_mapping_matches(self):
        expected_classes = [Activity.STANDING, Activity.WALKING, Activity.FALLING]
        self._create_dummy_model(classes=expected_classes)
        cfg = {"classifier": {"model_path": self.model_path}}
        clf = ActivityClassifier(config=cfg)

        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        res = clf.predict(fv)
        self.assertIn(res["activity"], expected_classes)

    # ── 8. Diagnostics integration ────────────────────────────────────────────
    def test_diagnostics_integration_with_integrity_status(self):
        from ai.inference import HAREngine

        # Test with absent model
        cfg_absent = {"classifier": {"model_path": os.path.join(self.test_dir, "none.joblib")}}
        engine_absent = HAREngine(config=cfg_absent)
        diag = engine_absent.get_diagnostics()
        clf_diag = diag["classifier"]

        self.assertEqual(clf_diag["mode"], "rule_based_baseline")
        self.assertFalse(clf_diag["is_trained"])
        self.assertFalse(clf_diag["integrity_check_passed"])
        self.assertIsNone(clf_diag["load_error"])

        # Test with corrupt model
        with open(self.model_path, "wb") as fh:
            fh.write(b"GARBAGE")
        cfg_corrupt = {"classifier": {"model_path": self.model_path}}
        engine_corrupt = HAREngine(config=cfg_corrupt)
        diag_corrupt = engine_corrupt.get_diagnostics()
        clf_corrupt = diag_corrupt["classifier"]

        self.assertEqual(clf_corrupt["mode"], "rule_based_baseline")
        self.assertFalse(clf_corrupt["is_trained"])
        self.assertFalse(clf_corrupt["integrity_check_passed"])
        self.assertIsNotNone(clf_corrupt["load_error"])
        self.assertIn("REJECTED", clf_corrupt["note"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
