"""
test_training_readiness_suite.py
================================
Unit tests for Part 7:
- Empty dataset assistant behavior
- Collection progress reporting vs configurable targets
- Class coverage and missing class detection
- Subject diversity evaluation
- Session diversity evaluation
- Training readiness verification (all phases)
- Missing features handling
- Missing splits handling
- Leakage detection in readiness evaluator
- Snapshot generation and schema validation
- Streamlit dataset status formatting
- Zero-data behavior
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from ai.schemas import Activity
from tools.dataset_assistant import inspect_readiness, format_assistant_report, load_dataset_config
from training_readiness import TrainingReadinessChecker, format_readiness_report
from dataset_snapshot import generate_snapshot
from dataset_balance import get_dataset_status_summary
from ai.features import FEATURE_DIM

FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]


class TestTrainingReadinessSuite(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")
        self.features_dir = os.path.join(self.test_dir, "features")
        self.splits_dir = os.path.join(self.test_dir, "splits")
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        self.snapshot_path = os.path.join(self.test_dir, "snapshot.json")

        for d in [self.raw_dir, self.features_dir, self.splits_dir]:
            os.makedirs(d, exist_ok=True)

        for c in Activity.ALL:
            os.makedirs(os.path.join(self.raw_dir, c), exist_ok=True)

        # Create test config
        with open(self.config_path, "w", encoding="utf-8") as fh:
            fh.write("dataset:\n  target_videos_per_class: 5\n  target_subjects_per_class: 2\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_video(self, class_name: str, filename: str):
        p = os.path.join(self.raw_dir, class_name, filename)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(p, fourcc, 30.0, (160, 120))
        for i in range(30):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()
        return p

    # ── 1. Empty Dataset Assistant ────────────────────────────────────────────
    def test_empty_dataset_assistant(self):
        res = inspect_readiness(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        self.assertEqual(res["total_videos"], 0)
        self.assertFalse(res["training_ready"])
        self.assertEqual(len(res["missing_classes"]), len(Activity.ALL))

        report = format_assistant_report(res)
        self.assertIn("COLLECTION REQUIRED", report)
        self.assertIn("Training:            BLOCKED", report)

    # ── 2. Collection Progress Reporting ──────────────────────────────────────
    def test_collection_progress_reporting(self):
        # Add 3 videos for STANDING
        self._create_mock_video("STANDING", "STANDING_S01_SESSION01_01.mp4")
        self._create_mock_video("STANDING", "STANDING_S01_SESSION01_02.mp4")
        self._create_mock_video("STANDING", "STANDING_S02_SESSION01_01.mp4")

        res = inspect_readiness(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        self.assertEqual(res["total_videos"], 3)
        self.assertEqual(res["total_subjects"], 2)
        st_standing = res["class_status"]["STANDING"]
        self.assertEqual(st_standing["videos"], 3)
        self.assertEqual(st_standing["subjects"], 2)
        self.assertEqual(st_standing["status"], "NEEDS MORE DATA")

    # ── 3. Class Coverage & Missing Class Detection ───────────────────────────
    def test_class_coverage_and_missing_class(self):
        self._create_mock_video("STANDING", "STANDING_S01_SESSION01_01.mp4")
        self._create_mock_video("WALKING", "WALKING_S01_SESSION01_01.mp4")

        res = inspect_readiness(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        self.assertIn("SITTING", res["missing_classes"])
        self.assertIn("RUNNING", res["missing_classes"])
        self.assertIn("FALLING", res["missing_classes"])
        self.assertFalse(res["all_classes_ready"])

    # ── 4. Subject Diversity Evaluation ───────────────────────────────────────
    def test_subject_diversity_evaluation(self):
        checker = TrainingReadinessChecker(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        eval_res = checker.evaluate()
        self.assertFalse(eval_res["diversity_ready"])
        self.assertTrue(any("diversity" in issue.lower() for issue in eval_res["issues"]))

    # ── 5. Missing Features Blocks Readiness ──────────────────────────────────
    def test_missing_features_blocks_readiness(self):
        # Create videos for all 5 classes
        for i, c in enumerate(Activity.ALL):
            self._create_mock_video(c, f"{c}_S01_SESSION01_{i}.mp4")

        checker = TrainingReadinessChecker(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        res = checker.evaluate()
        self.assertFalse(res["features_ready"])
        self.assertFalse(res["overall_ready"])
        self.assertTrue(any("feature" in issue.lower() for issue in res["issues"]))

    # ── 6. Missing Splits Blocks Readiness ────────────────────────────────────
    def test_missing_splits_blocks_readiness(self):
        checker = TrainingReadinessChecker(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        res = checker.evaluate()
        self.assertFalse(res["splits_ready"])
        self.assertFalse(res["training_gate_open"])

    # ── 7. Leakage Detection in Evaluator ─────────────────────────────────────
    def test_leakage_detection_in_evaluator(self):
        # Create splits with intentional video leakage
        leak_vid = "leak_video_01.mp4"
        tr_p = os.path.join(self.splits_dir, "train.csv")
        va_p = os.path.join(self.splits_dir, "validation.csv")
        te_p = os.path.join(self.splits_dir, "test.csv")

        for p in [tr_p, va_p]:
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["video_id", "filename", "subject_id"])
                w.writeheader()
                w.writerow({"video_id": leak_vid, "filename": leak_vid, "subject_id": "S01"})

        with open(te_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["video_id", "filename", "subject_id"])
            w.writeheader()
            w.writerow({"video_id": "other.mp4", "filename": "other.mp4", "subject_id": "S02"})

        checker = TrainingReadinessChecker(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            config_path=self.config_path,
        )
        res = checker.evaluate()
        self.assertFalse(res["leakage_passed"])
        self.assertTrue(any("leakage" in issue.lower() for issue in res["issues"]))

    # ── 8. Snapshot Generation & Schema Validation ────────────────────────────
    def test_snapshot_generation(self):
        snap = generate_snapshot(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
            splits_dir=self.splits_dir,
            output_path=self.snapshot_path,
        )
        self.assertTrue(os.path.isfile(self.snapshot_path))
        self.assertIn("timestamp", snap)
        self.assertIn("video_count", snap)
        self.assertIn("class_counts", snap)
        self.assertIn("split_counts", snap)
        self.assertEqual(snap["video_count"], 0)
        self.assertTrue(snap["is_dataset_empty"])

    # ── 9. Streamlit Dataset Status Integration ───────────────────────────────
    def test_streamlit_status_summary(self):
        summary = get_dataset_status_summary(
            raw_dir=self.raw_dir,
            manifest_path=self.manifest_path,
            features_dir=self.features_dir,
        )
        self.assertFalse(summary["has_real_data"])
        self.assertEqual(summary["dataset_label"], "NO DATA")
        self.assertEqual(summary["training_status"], "BLOCKED")
        self.assertIn("Training blocked", summary["blocked_reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
