"""
test_dataset_quality_pipeline.py
================================
Unit tests for Part 6:
- Valid / invalid class detection
- Empty dataset handling
- Manifest generation
- Duplicate detection (filename & content hash)
- Dataset balance reporting
- Missing metadata reporting
- Corrupted video handling
- Split leakage prevention (no video appears in >1 split)
- Subject / session separation
- Feature dimension validation
- Training gate enforcement
- No-data behavior & fallback
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
from dataset_manifest import (
    VALID_CLASSES,
    compute_file_hash,
    check_duplicates,
    generate_manifest,
    scan_raw_directory,
    parse_filename_metadata,
)
from dataset_quality import DatasetQualityChecker
from dataset_balance import analyze_balance, format_balance_report, get_dataset_status_summary
from dataset.split_dataset import perform_leakage_safe_split
from train_classifier import validate_training_gate, FEATURE_DIM, FEATURE_COLS


class TestDatasetQualityPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.splits_dir = os.path.join(self.test_dir, "splits")
        self.features_dir = os.path.join(self.test_dir, "features")
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")

        for d in [self.raw_dir, self.splits_dir, self.features_dir]:
            os.makedirs(d, exist_ok=True)

        for c in Activity.ALL:
            os.makedirs(os.path.join(self.raw_dir, c), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dummy_video(self, filepath: str, num_frames: int = 30, fps: float = 30.0, w: int = 160, h: int = 120):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        for i in range(num_frames):
            frame = np.full((h, w, 3), (i * 5) % 255, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    # ── 1. Valid and Invalid Class Detection ──────────────────────────────────
    def test_valid_classes_match_activity_all(self):
        self.assertEqual(VALID_CLASSES, set(Activity.ALL))
        for act in ["STANDING", "SITTING", "WALKING", "RUNNING", "FALLING"]:
            self.assertIn(act, VALID_CLASSES)

    def test_invalid_class_rejected_by_scanner(self):
        invalid_folder = os.path.join(self.raw_dir, "JUMPING")
        os.makedirs(invalid_folder, exist_ok=True)
        vid_path = os.path.join(invalid_folder, "test.mp4")
        self._create_dummy_video(vid_path)

        records, warnings = scan_raw_directory(self.raw_dir)
        # Should not be in records
        self.assertNotIn("JUMPING", [r["class"] for r in records])
        # Warning should mention invalid class
        self.assertTrue(any("JUMPING" in w for w in warnings))

    # ── 2. Empty Dataset Handling ─────────────────────────────────────────────
    def test_empty_dataset_handling(self):
        # Empty manifest generation
        count, warnings = generate_manifest(raw_dir=self.raw_dir, manifest_path=self.manifest_path)
        self.assertEqual(count, 0)
        self.assertTrue(os.path.isfile(self.manifest_path))

        # Quality check on empty
        checker = DatasetQualityChecker(raw_dir=self.raw_dir)
        summary = checker.run_check()
        self.assertEqual(summary["total_videos"], 0)
        self.assertEqual(summary["critical_errors"], 0)

        # Balance on empty
        balance = analyze_balance(raw_dir=self.raw_dir, manifest_path=self.manifest_path)
        rep, warn = format_balance_report(balance)
        self.assertIn("No real recordings found", rep)

        # Status summary on empty
        status = get_dataset_status_summary(raw_dir=self.raw_dir, manifest_path=self.manifest_path, features_dir=self.features_dir)
        self.assertFalse(status["has_real_data"])
        self.assertEqual(status["dataset_label"], "NO DATA")
        self.assertEqual(status["training_status"], "BLOCKED")

    # ── 3. Manifest Generation & Metadata Parsing ─────────────────────────────
    def test_manifest_generation_with_metadata_parsing(self):
        vid_file = os.path.join(self.raw_dir, "WALKING", "WALKING_S01_SESSION02_clip1.mp4")
        self._create_dummy_video(vid_file, num_frames=30, fps=30.0, w=320, h=240)

        count, warnings = generate_manifest(raw_dir=self.raw_dir, manifest_path=self.manifest_path)
        self.assertEqual(count, 1)

        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["class"], "WALKING")
        self.assertEqual(r["subject_id"], "S01")
        self.assertEqual(r["session_id"], "SESSION02")
        self.assertEqual(r["resolution"], "320x240")
        self.assertEqual(float(r["fps"]), 30.0)

    # ── 4. Duplicate Detection (Filename & Content) ───────────────────────────
    def test_duplicate_detection(self):
        vid1 = os.path.join(self.raw_dir, "STANDING", "stand_01.mp4")
        self._create_dummy_video(vid1, num_frames=20)

        # Exact copy with another name
        vid2 = os.path.join(self.raw_dir, "STANDING", "stand_02_copy.mp4")
        shutil.copyfile(vid1, vid2)

        records, warnings = scan_raw_directory(self.raw_dir)
        dup_warn = check_duplicates(records)
        self.assertTrue(any("Duplicate" in w or "duplicate" in w for w in dup_warn))

        # Check with DatasetQualityChecker
        checker = DatasetQualityChecker(raw_dir=self.raw_dir)
        summary = checker.run_check()
        self.assertGreater(summary["duplicate_content_count"], 0)

    # ── 5. Corrupted Video Handling ───────────────────────────────────────────
    def test_corrupted_video_handling(self):
        corrupt_path = os.path.join(self.raw_dir, "FALLING", "corrupt_fall.mp4")
        with open(corrupt_path, "wb") as fh:
            fh.write(b"CORRUPT_NOT_A_VIDEO_FILE_DATA_987654321")

        checker = DatasetQualityChecker(raw_dir=self.raw_dir)
        summary = checker.run_check()
        self.assertEqual(summary["corrupted_count"], 1)
        self.assertGreater(summary["critical_errors"], 0)

    # ── 6. Missing Metadata Reporting ─────────────────────────────────────────
    def test_missing_metadata_reporting(self):
        # Video without subject/session in name
        vid = os.path.join(self.raw_dir, "SITTING", "sitting_unlabeled.mp4")
        self._create_dummy_video(vid, num_frames=30)

        generate_manifest(raw_dir=self.raw_dir, manifest_path=self.manifest_path)
        checker = DatasetQualityChecker(manifest_path=self.manifest_path, raw_dir=self.raw_dir)
        summary = checker.run_check()
        self.assertGreaterEqual(summary["missing_metadata_count"], 1)

    # ── 7. Leakage-Safe Splitting (Group by Video / Subject) ───────────────────
    def test_split_leakage_prevention(self):
        # Create 6 distinct videos across classes
        test_rows = []
        for i in range(6):
            cls = Activity.ALL[i % len(Activity.ALL)]
            vid_id = f"video_{i:02d}"
            fn = f"{cls}/{vid_id}.mp4"
            subj = f"S0{i+1}"
            sess = f"SESSION{i+1}"
            test_rows.append({
                "video_id": vid_id,
                "filename": fn,
                "class": cls,
                "subject_id": subj,
                "session_id": sess,
            })

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["video_id", "filename", "class", "subject_id", "session_id"])
            writer.writeheader()
            for r in test_rows:
                writer.writerow(r)

        success, msg, splits = perform_leakage_safe_split(
            manifest_path=self.manifest_path,
            splits_dir=self.splits_dir,
            val_ratio=0.2,
            test_ratio=0.2,
            random_seed=42,
        )
        self.assertTrue(success)

        # Verify NO overlap between splits
        tr_vids = {r["video_id"] for r in splits["train"]}
        va_vids = {r["video_id"] for r in splits["validation"]}
        te_vids = {r["video_id"] for r in splits["test"]}

        self.assertEqual(len(tr_vids & va_vids), 0, "Train and Val share video IDs (LEAKAGE)")
        self.assertEqual(len(tr_vids & te_vids), 0, "Train and Test share video IDs (LEAKAGE)")
        self.assertEqual(len(va_vids & te_vids), 0, "Val and Test share video IDs (LEAKAGE)")

    def test_split_subject_separation(self):
        # Create videos with 3 distinct subjects
        test_rows = [
            {"video_id": "v1", "filename": "c/v1.mp4", "class": "STANDING", "subject_id": "S01", "session_id": "SES1"},
            {"video_id": "v2", "filename": "c/v2.mp4", "class": "STANDING", "subject_id": "S01", "session_id": "SES2"},
            {"video_id": "v3", "filename": "c/v3.mp4", "class": "WALKING", "subject_id": "S02", "session_id": "SES1"},
            {"video_id": "v4", "filename": "c/v4.mp4", "class": "WALKING", "subject_id": "S02", "session_id": "SES2"},
            {"video_id": "v5", "filename": "c/v5.mp4", "class": "FALLING", "subject_id": "S03", "session_id": "SES1"},
            {"video_id": "v6", "filename": "c/v6.mp4", "class": "FALLING", "subject_id": "S03", "session_id": "SES2"},
        ]
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["video_id", "filename", "class", "subject_id", "session_id"])
            writer.writeheader()
            for r in test_rows:
                writer.writerow(r)

        success, msg, splits = perform_leakage_safe_split(
            manifest_path=self.manifest_path,
            splits_dir=self.splits_dir,
            val_ratio=0.33,
            test_ratio=0.33,
            random_seed=42,
        )
        self.assertTrue(success)

        tr_subjs = {r["subject_id"] for r in splits["train"]}
        va_subjs = {r["subject_id"] for r in splits["validation"]}
        te_subjs = {r["subject_id"] for r in splits["test"]}

        self.assertEqual(len(tr_subjs & va_subjs), 0, "Train and Val share subjects")
        self.assertEqual(len(tr_subjs & te_subjs), 0, "Train and Test share subjects")
        self.assertEqual(len(va_subjs & te_subjs), 0, "Val and Test share subjects")

    # ── 8. Training Gate Enforcement ──────────────────────────────────────────
    def test_training_gate_blocks_when_empty(self):
        with self.assertRaises(SystemExit) as cm:
            validate_training_gate(
                data_path=self.features_dir,
                raw_dir=self.raw_dir,
                splits_dir=self.splits_dir,
            )
        self.assertEqual(cm.exception.code, 1)

    def test_training_gate_passes_when_prerequisites_satisfied(self):
        # 1. Add mock video to raw
        vid_p = os.path.join(self.raw_dir, "STANDING", "dummy.mp4")
        self._create_dummy_video(vid_p)

        # 2. Add mock feature CSV with all 5 classes and 34 features
        import pandas as pd
        rows = []
        for cls in Activity.ALL:
            for i in range(10):
                row = {
                    "sample_id": f"{cls}_{i}",
                    "person_id": 1,
                    "activity_label": cls,
                    "source_video": f"vid_{cls}_{i}.mp4",
                }
                for fcol in FEATURE_COLS:
                    row[fcol] = 0.5
                rows.append(row)
        df = pd.DataFrame(rows)
        feat_csv = os.path.join(self.features_dir, "features_all.csv")
        df.to_csv(feat_csv, index=False)

        # 3. Create non-leaked splits
        for split_name in ["train", "validation", "test"]:
            split_p = os.path.join(self.splits_dir, f"{split_name}.csv")
            with open(split_p, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["video_id", "filename", "class"])
                writer.writeheader()
                writer.writerow({"video_id": f"vid_{split_name}", "filename": f"{split_name}.mp4", "class": "STANDING"})

        # Should validate without raising SystemExit
        validate_training_gate(
            data_path=self.features_dir,
            raw_dir=self.raw_dir,
            splits_dir=self.splits_dir,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
