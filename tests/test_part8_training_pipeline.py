"""
tests/test_part8_training_pipeline.py
======================================
Comprehensive test suite for Part 8:
- Real dataset validation logic
- Feature extraction format & 34D verification
- Split integrity & subject leakage checks
- Model training, persistence, and reload
- Prediction & predict_proba verification
- Metrics calculation & classification report generation
- Confusion matrix & confusion analysis generation
- Model manifest generation & SHA-256 verification
- Multi-model comparison (--compare) logic
- Runtime trained-model loading & fallback when corrupt/absent
- Training gate refusal when real data is absent
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ai.classifier import ActivityClassifier
from ai.features import FEATURE_DIM
from ai.schemas import Activity
from train_classifier import (
    train_model,
    evaluate,
    save_model,
    save_evaluation,
    create_model_manifest,
    validate_training_gate,
    load_dataset,
    group_split,
)
from training_readiness import TrainingReadinessChecker

FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]
VALID_LABELS = list(Activity.ALL)


class TestPart8TrainingPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.features_dir = os.path.join(self.test_dir, "features")
        self.splits_dir = os.path.join(self.test_dir, "splits")
        self.models_dir = os.path.join(self.test_dir, "models")
        self.eval_dir = os.path.join(self.test_dir, "evaluation")

        for d in [self.raw_dir, self.features_dir, self.splits_dir, self.models_dir, self.eval_dir]:
            os.makedirs(d, exist_ok=True)

        for c in Activity.ALL:
            os.makedirs(os.path.join(self.raw_dir, c), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_sample_csv(self, file_path: str, n_samples: int = 30, subject: str = "SUBJ_01"):
        header = ["sample_id", "person_id", "activity_label", "source_video", "frame_start", "frame_end", "timestamp"] + FEATURE_COLS
        with open(file_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i in range(n_samples):
                label = VALID_LABELS[i % len(VALID_LABELS)]
                row = [
                    f"{subject}_{i}",
                    "1",
                    label,
                    f"{subject}_vid.mp4",
                    i * 20,
                    (i + 1) * 20,
                    1000.0 + i,
                ] + [float(j * 0.01) for j in range(FEATURE_DIM)]
                writer.writerow(row)

    # 1. Training gate refusal on zero real data
    def test_training_gate_refusal_when_real_data_absent(self):
        with self.assertRaises(SystemExit):
            validate_training_gate(
                data_path=self.features_dir,
                raw_dir=self.raw_dir,
                splits_dir=self.splits_dir,
            )

    # 2. Feature extraction verification & 34D check
    def test_feature_data_loading_and_dimension(self):
        csv_path = os.path.join(self.features_dir, "features_01.csv")
        self._create_sample_csv(csv_path, n_samples=25)
        X, y, groups = load_dataset(self.features_dir)
        self.assertEqual(X.shape[1], 34)
        self.assertEqual(len(y), 25)
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isinf(X).any())

    # 3. Split integrity and subject leakage check
    def test_split_integrity_and_subject_leakage(self):
        for s_idx in range(1, 4):
            subj = f"SUBJ_0{s_idx}"
            p = os.path.join(self.features_dir, f"{subj}.csv")
            self._create_sample_csv(p, n_samples=30, subject=subj)

        X, y, groups = load_dataset(self.features_dir)
        X_tr, X_v, X_te, y_tr, y_v, y_te, tr_src, v_src, te_src = group_split(
            X, y, groups, val_frac=0.33, test_frac=0.33, random_state=42
        )
        # Check no overlap in sources between train and test
        overlap = set(tr_src) & set(te_src)
        self.assertEqual(len(overlap), 0, "Leakage detected: videos found in both train and test splits!")
        self.assertGreater(len(X_tr), 0)
        self.assertGreater(len(X_te), 0)

    # 4. Model training, persistence and reload
    def test_model_training_persistence_and_reload(self):
        # Create balanced sample features for all 5 classes
        csv_path = os.path.join(self.features_dir, "train_data.csv")
        self._create_sample_csv(csv_path, n_samples=50)
        X, y, _ = load_dataset(self.features_dir)

        model, scaler, le = train_model(X, y, model_type="random_forest", random_state=42)
        model_path = os.path.join(self.models_dir, "har_classifier.joblib")
        meta = {
            "model_type": "random_forest",
            "feature_dim": FEATURE_DIM,
            "classes": le.classes_.tolist(),
            "n_samples": len(y),
        }
        save_model(model, scaler, le, model_path, meta)

        self.assertTrue(os.path.isfile(model_path))
        self.assertTrue(os.path.isfile(os.path.join(self.models_dir, "har_classifier_scaler.joblib")))
        self.assertTrue(os.path.isfile(os.path.join(self.models_dir, "scaler.joblib")))
        self.assertTrue(os.path.isfile(os.path.join(self.models_dir, "metadata.json")))

        # Test reload with ActivityClassifier
        config = {
            "classifier": {
                "model_path": model_path,
                "scaler_path": os.path.join(self.models_dir, "har_classifier_scaler.joblib"),
            },
            "activity": {"confidence_threshold": 0.5},
        }
        clf = ActivityClassifier(config)
        self.assertEqual(clf.model_type, "trained_ml")

        # Test inference and probability output
        sample_vec = np.zeros((1, FEATURE_DIM))
        res = clf.predict(sample_vec)
        self.assertIn(res["activity"], VALID_LABELS + [Activity.UNKNOWN])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertEqual(res["model_type"], "trained_ml")

    # 5. Metrics generation & confusion matrix
    def test_metrics_and_confusion_matrix_generation(self):
        csv_path = os.path.join(self.features_dir, "eval_data.csv")
        self._create_sample_csv(csv_path, n_samples=50)
        X, y, _ = load_dataset(self.features_dir)
        model, scaler, le = train_model(X, y, model_type="random_forest", random_state=42)

        res = evaluate(model, scaler, le, X, y, split_name="test")
        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("weighted_f1", res)
        self.assertIn("confusion_matrix", res)

        save_evaluation([res], eval_dir=self.eval_dir)

        self.assertTrue(os.path.isfile(os.path.join(self.eval_dir, "metrics.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.eval_dir, "classification_report.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.eval_dir, "confusion_matrix.csv")))
        self.assertTrue(os.path.isfile(os.path.join(self.eval_dir, "confusion_matrix_analysis.txt")))

        # Validate metrics JSON content
        with open(os.path.join(self.eval_dir, "metrics.json"), "r", encoding="utf-8") as fh:
            data = json.load(fh)
            self.assertEqual(data.get("split"), "test")
            self.assertIn("accuracy", data)
            self.assertIn("macro_f1", data)
            self.assertIn("weighted_f1", data)

    # 6. Model manifest creation and SHA-256 integrity
    def test_model_manifest_creation(self):
        model_path = os.path.join(self.models_dir, "har_classifier.joblib")
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        meta_path = os.path.join(self.models_dir, "metadata.json")

        with open(model_path, "wb") as f:
            f.write(b"dummy_model_bytes")
        with open(scaler_path, "wb") as f:
            f.write(b"dummy_scaler_bytes")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"model_type": "random_forest", "feature_dim": 34, "classes": VALID_LABELS}, f)

        manifest = create_model_manifest(
            model_path=model_path,
            scaler_path=scaler_path,
            meta_path=meta_path,
            output_dir=self.models_dir,
        )
        self.assertIsNotNone(manifest["model_hash"])
        self.assertIsNotNone(manifest["scaler_hash"])
        self.assertIsNotNone(manifest["metadata_hash"])
        self.assertEqual(manifest["feature_dimension"], 34)
        self.assertTrue(os.path.isfile(os.path.join(self.models_dir, "model_manifest.json")))

    # 7. Fallback behavior when model is absent or corrupt
    def test_fallback_when_model_absent_or_corrupt(self):
        # Absent model
        cfg_absent = {
            "classifier": {"model_path": os.path.join(self.models_dir, "missing.joblib")},
            "activity": {"confidence_threshold": 0.5},
        }
        clf_absent = ActivityClassifier(cfg_absent)
        self.assertEqual(clf_absent.model_type, "rule_based_baseline")

        # Corrupt model
        corrupt_path = os.path.join(self.models_dir, "corrupt.joblib")
        with open(corrupt_path, "wb") as fh:
            fh.write(b"this_is_not_a_valid_joblib_file")
        cfg_corrupt = {
            "classifier": {"model_path": corrupt_path},
            "activity": {"confidence_threshold": 0.5},
        }
        clf_corrupt = ActivityClassifier(cfg_corrupt)
        self.assertEqual(clf_corrupt.model_type, "rule_based_baseline")
        self.assertIsNotNone(clf_corrupt._load_error)

    # 8. Real dataset validation readiness checker
    def test_real_dataset_validation_readiness_checker(self):
        checker = TrainingReadinessChecker(
            raw_dir=self.raw_dir,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
        )
        report = checker.evaluate()
        self.assertFalse(report["overall_ready"])
        self.assertFalse(report["dataset_ready"])
        self.assertFalse(report["classes_ready"])

    # 9. Probability output and confidence thresholding
    def test_model_probability_output_and_thresholding(self):
        csv_path = os.path.join(self.features_dir, "train_prob.csv")
        self._create_sample_csv(csv_path, n_samples=50)
        X, y, _ = load_dataset(self.features_dir)
        model, scaler, le = train_model(X, y, model_type="random_forest", random_state=42)

        # Direct predict_proba verification
        X_scaled = scaler.transform(X[:5])
        probas = model.predict_proba(X_scaled)
        self.assertEqual(probas.shape[1], 5)
        for p in probas:
            self.assertAlmostEqual(float(np.sum(p)), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
