"""
tests/test_dataset.py
======================
Tests for the Part 4 dataset pipeline:

- Dataset CSV schema validation
- Feature collector (FeatureCollector) — no webcam / video needed
- Dataset validator
- Missing/invalid feature values
- Leakage-safe group split
- Model save / load round-trip
- Trained classifier inference
- Fallback classifier still works
- Diagnostics classifier mode reporting

All tests run WITHOUT a webcam, real videos, or GPU.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from ai.features import FEATURE_DIM
from ai.schemas import Activity

FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]

VALID_LABELS = list(Activity.ALL)

CSV_HEADER = [
    "sample_id", "person_id", "activity_label",
    "source_video", "frame_start", "frame_end", "timestamp",
] + FEATURE_COLS


# ── CSV helpers ────────────────────────────────────────────────────────────

def _make_csv(path: str, rows: list, header=None) -> str:
    h = header or CSV_HEADER
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(h)
        for row in rows:
            w.writerow(row)
    return path


def _make_row(
    label="WALKING",
    source="video_a.mp4",
    person_id=1,
    fv=None,
):
    fv = fv if fv is not None else np.zeros(FEATURE_DIM, dtype=np.float32).tolist()
    return [
        str(uuid.uuid4()), person_id, label, source,
        0, 20, "2026-01-01T00:00:00",
    ] + list(fv)


def _make_dataset_dir(
    tmpdir: str,
    n_per_class: int = 10,
    n_sources: int = 3,
) -> str:
    """Write a minimal valid dataset CSV into tmpdir."""
    rows = []
    sources = [f"video_{i}.mp4" for i in range(n_sources)]
    for act in VALID_LABELS:
        for i in range(n_per_class):
            src = sources[i % n_sources]
            fv = np.abs(np.random.randn(FEATURE_DIM).astype(np.float32)) * 0.3
            rows.append(_make_row(label=act, source=src, fv=fv.tolist()))
    csv_path = os.path.join(tmpdir, "WALKING_20260929.csv")
    _make_csv(csv_path, rows)
    return tmpdir


# ── 1. Dataset schema ──────────────────────────────────────────────────────

class TestDatasetSchema(unittest.TestCase):
    """Test that the CSV schema matches the documented contract."""

    def test_header_has_correct_columns(self):
        required = {
            "sample_id", "person_id", "activity_label",
            "source_video", "frame_start", "frame_end", "timestamp",
        } | set(FEATURE_COLS)
        self.assertEqual(required, set(CSV_HEADER))

    def test_feature_cols_count(self):
        self.assertEqual(len(FEATURE_COLS), FEATURE_DIM)

    def test_feature_cols_ordered(self):
        for i, col in enumerate(FEATURE_COLS):
            self.assertEqual(col, f"f_{i:02d}")

    def test_valid_labels_from_activity(self):
        self.assertIn("STANDING", VALID_LABELS)
        self.assertIn("SITTING",  VALID_LABELS)
        self.assertIn("WALKING",  VALID_LABELS)
        self.assertIn("RUNNING",  VALID_LABELS)
        self.assertIn("FALLING",  VALID_LABELS)

    def test_row_length_matches_header(self):
        row = _make_row()
        self.assertEqual(len(row), len(CSV_HEADER))


# ── 2. _PersonBuffer ──────────────────────────────────────────────────────

class TestPersonBuffer(unittest.TestCase):
    """Test the temporal sliding window buffer in collect_features.py."""

    def _make_buf(self, window=5):
        from collect_features import _PersonBuffer
        return _PersonBuffer(person_id=1, window_size=window)

    def test_buffer_empty_returns_none(self):
        buf = self._make_buf(5)
        result = buf.push(np.zeros(FEATURE_DIM), 0)
        self.assertIsNone(result)

    def test_buffer_full_returns_array(self):
        buf = self._make_buf(5)
        result = None
        for i in range(5):
            result = buf.push(np.ones(FEATURE_DIM) * i, i)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (FEATURE_DIM,))

    def test_buffer_mean_is_correct(self):
        buf = self._make_buf(4)
        values = [0.0, 1.0, 2.0, 3.0]
        result = None
        for v in values:
            result = buf.push(np.full(FEATURE_DIM, v), 0)
        expected_mean = np.mean(values)
        self.assertAlmostEqual(float(result[0]), expected_mean, places=5)

    def test_buffer_result_is_finite(self):
        buf = self._make_buf(5)
        result = None
        for i in range(5):
            result = buf.push(np.random.randn(FEATURE_DIM).astype(np.float32), i)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_frame_start_end_tracked(self):
        buf = self._make_buf(3)
        for i in range(3):
            buf.push(np.zeros(FEATURE_DIM), i * 10)
        self.assertEqual(buf.frame_start, 0)
        self.assertEqual(buf.frame_end, 20)

    def test_sliding_window_produces_result_on_every_push(self):
        """After the buffer is full, every new push should also return a result."""
        buf = self._make_buf(3)
        results = []
        for i in range(6):
            r = buf.push(np.ones(FEATURE_DIM) * i, i)
            results.append(r)
        # First 2 pushes: None; pushes 3-6: result
        self.assertIsNone(results[0])
        self.assertIsNone(results[1])
        for r in results[2:]:
            self.assertIsNotNone(r)


# ── 3. FeatureCollector build_row ─────────────────────────────────────────

class TestFeatureCollectorBuildRow(unittest.TestCase):
    """Test the row builder without needing a video source."""

    def _make_collector(self, tmpdir):
        from collect_features import FeatureCollector
        cfg = {
            "model": {"pose_model": "yolo11n-pose.pt",
                      "detection_confidence": 0.40,
                      "iou_threshold": 0.45},
            "tracking": {"max_missing_frames": 60},
        }
        return FeatureCollector(
            label="WALKING",
            output_dir=tmpdir,
            config=cfg,
            source_name="test_video.mp4",
            window_size=5,
        )

    def test_row_has_correct_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._make_collector(tmp)
            fv = np.zeros(FEATURE_DIM, dtype=np.float32)
            row = c._build_row(1, fv, 0, 4)
            self.assertEqual(len(row), len(CSV_HEADER))

    def test_row_label_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._make_collector(tmp)
            fv = np.zeros(FEATURE_DIM, dtype=np.float32)
            row = c._build_row(1, fv, 0, 4)
            label_idx = CSV_HEADER.index("activity_label")
            self.assertEqual(row[label_idx], "WALKING")

    def test_row_source_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._make_collector(tmp)
            fv = np.zeros(FEATURE_DIM, dtype=np.float32)
            row = c._build_row(1, fv, 0, 4)
            src_idx = CSV_HEADER.index("source_video")
            self.assertEqual(row[src_idx], "test_video.mp4")

    def test_feature_values_in_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._make_collector(tmp)
            fv = np.arange(FEATURE_DIM, dtype=np.float32)
            row = c._build_row(1, fv, 0, 4)
            feat_start = len(CSV_HEADER) - FEATURE_DIM
            row_feats = np.array(row[feat_start:], dtype=np.float32)
            np.testing.assert_array_almost_equal(row_feats, fv)

    def test_sample_id_is_uuid(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self._make_collector(tmp)
            fv = np.zeros(FEATURE_DIM, dtype=np.float32)
            row = c._build_row(1, fv, 0, 4)
            sample_id = row[0]
            self.assertEqual(str(uuid.UUID(sample_id)), sample_id)


# ── 4. Dataset validator ──────────────────────────────────────────────────

class TestDatasetValidator(unittest.TestCase):

    def _validate(self, tmpdir):
        """Helper: load CSVs from tmpdir and run validator."""
        import pandas as pd
        from dataset_validator import validate, _load_csvs
        df, errors = _load_csvs(tmpdir)
        if df is None:
            return None, errors
        return validate(df), errors

    def test_valid_dataset_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dataset_dir(tmp, n_per_class=5, n_sources=3)
            report, _ = self._validate(tmp)
            self.assertTrue(report["valid"], msg=str(report["errors"]))

    def test_missing_column_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Write CSV without feature columns
            bad_header = ["sample_id", "person_id", "activity_label",
                          "source_video", "frame_start", "frame_end", "timestamp"]
            rows = [_make_row()[:7]]  # only metadata columns
            _make_csv(os.path.join(tmp, "bad.csv"), rows, header=bad_header)
            report, _ = self._validate(tmp)
            self.assertFalse(report["valid"])
            self.assertTrue(any("Missing columns" in e or "mismatch" in e
                                for e in report["errors"]))

    def test_invalid_label_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_make_row(label="FLYING")]
            _make_csv(os.path.join(tmp, "bad.csv"), rows)
            report, _ = self._validate(tmp)
            self.assertFalse(report["valid"])
            self.assertTrue(any("label" in e.lower() for e in report["errors"]))

    def test_nan_values_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fv = [float("nan")] * FEATURE_DIM
            rows = [_make_row(fv=fv)]
            _make_csv(os.path.join(tmp, "nan.csv"), rows)
            report, _ = self._validate(tmp)
            self.assertFalse(report["valid"])

    def test_inf_values_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            fv = [float("inf")] * FEATURE_DIM
            rows = [_make_row(fv=fv)]
            _make_csv(os.path.join(tmp, "inf.csv"), rows)
            report, _ = self._validate(tmp)
            self.assertFalse(report["valid"])

    def test_per_activity_counts_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 7
            _make_dataset_dir(tmp, n_per_class=n, n_sources=3)
            report, _ = self._validate(tmp)
            for act in VALID_LABELS:
                self.assertIn(act, report["stats"]["per_activity"])
                self.assertEqual(report["stats"]["per_activity"][act], n)

    def test_feature_dim_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dataset_dir(tmp, n_per_class=3, n_sources=3)
            report, _ = self._validate(tmp)
            self.assertEqual(report["stats"]["feature_dim_found"], FEATURE_DIM)

    def test_source_video_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_dataset_dir(tmp, n_per_class=5, n_sources=4)
            report, _ = self._validate(tmp)
            self.assertEqual(report["stats"]["n_source_videos"], 4)

    def test_empty_directory_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            df, errors = __import__("dataset_validator")._load_csvs(tmp)
            self.assertIsNone(df)
            self.assertTrue(len(errors) > 0)


# ── 5. Leakage-safe split ─────────────────────────────────────────────────

class TestGroupSplit(unittest.TestCase):

    def _make_data(self, n_per_class=30, n_sources=5):
        rng = np.random.default_rng(0)
        X, y, groups = [], [], []
        sources = [f"vid_{i}.mp4" for i in range(n_sources)]
        for act in VALID_LABELS:
            for i, src in enumerate(sources):
                n = n_per_class // n_sources
                X.append(rng.random((n, FEATURE_DIM)).astype(np.float32))
                y.extend([act] * n)
                groups.extend([src] * n)
        return np.vstack(X), np.array(y), np.array(groups)

    def test_no_overlap_between_splits(self):
        from train_classifier import group_split
        X, y, groups = self._make_data(n_sources=6)
        (_, _, _,
         _, _, _,
         tr_src, v_src, te_src) = group_split(X, y, groups)
        tr = set(tr_src)
        v  = set(v_src)
        te = set(te_src)
        self.assertTrue(tr.isdisjoint(te),  "Train/test source overlap!")
        self.assertTrue(tr.isdisjoint(v),   "Train/val source overlap!")
        self.assertTrue(v.isdisjoint(te),   "Val/test source overlap!")

    def test_all_samples_accounted_for(self):
        from train_classifier import group_split
        X, y, groups = self._make_data(n_sources=6)
        (X_tr, X_v, X_te,
         y_tr, y_v, y_te,
         *_) = group_split(X, y, groups)
        self.assertEqual(len(X_tr) + len(X_v) + len(X_te), len(X))

    def test_train_is_largest_split(self):
        from train_classifier import group_split
        X, y, groups = self._make_data(n_sources=6)
        (X_tr, X_v, X_te, *_) = group_split(X, y, groups)
        self.assertGreater(len(X_tr), len(X_te))

    def test_fallback_with_few_sources(self):
        """When < 3 source videos, fallback split runs without crashing."""
        from train_classifier import group_split
        X, y, groups = self._make_data(n_sources=2)
        result = group_split(X, y, groups)
        X_tr, X_v, X_te = result[0], result[1], result[2]
        self.assertGreater(len(X_tr), 0)


# ── 6. Model save / load ──────────────────────────────────────────────────

class TestModelPersistence(unittest.TestCase):

    def _train_minimal(self, tmpdir):
        """Train a tiny RF on synthetic data and save it."""
        from train_classifier import train_model, save_model
        rng = np.random.default_rng(1)
        X = rng.random((50, FEATURE_DIM)).astype(np.float32)
        y = np.array(["STANDING"] * 20 + ["WALKING"] * 20 + ["FALLING"] * 10)
        model, scaler, le = train_model(X, y, "random_forest", random_state=1)
        out = os.path.join(tmpdir, "test_model.joblib")
        meta = {"feature_dim": FEATURE_DIM, "classes": list(le.classes_),
                "trained_at": "test", "performance_note": "test"}
        save_model(model, scaler, le, out, meta)
        return out, model, scaler, le

    def test_model_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, *_ = self._train_minimal(tmp)
            self.assertTrue(os.path.isfile(out))

    def test_scaler_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, *_ = self._train_minimal(tmp)
            scaler_path = out.replace(".joblib", "_scaler.joblib")
            self.assertTrue(os.path.isfile(scaler_path))

    def test_meta_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, *_ = self._train_minimal(tmp)
            meta_path = out.replace(".joblib", "_meta.json")
            self.assertTrue(os.path.isfile(meta_path))
            with open(meta_path) as fh:
                meta = json.load(fh)
            self.assertIn("feature_dim", meta)
            self.assertIn("classes", meta)

    def test_model_can_be_reloaded(self):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            out, orig_model, *_ = self._train_minimal(tmp)
            loaded = joblib.load(out)
            self.assertIsNotNone(loaded)

    def test_loaded_model_predicts_same(self):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            out, model, scaler, le = self._train_minimal(tmp)
            loaded_model  = joblib.load(out)
            loaded_scaler = joblib.load(out.replace(".joblib", "_scaler.joblib"))
            X_test = np.random.randn(5, FEATURE_DIM).astype(np.float32)
            pred_orig   = model.predict(scaler.transform(X_test))
            pred_loaded = loaded_model.predict(loaded_scaler.transform(X_test))
            np.testing.assert_array_equal(pred_orig, pred_loaded)

    def test_overwrite_creates_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, *_ = self._train_minimal(tmp)
            # Write again — should create a backup
            self._train_minimal(tmp)
            backups = [f for f in os.listdir(tmp) if "backup" in f]
            self.assertGreater(len(backups), 0, "No backup created on overwrite")


# ── 7. Trained classifier integration ─────────────────────────────────────

class TestTrainedClassifierIntegration(unittest.TestCase):
    """
    Train a minimal RF, save it, then load it via ActivityClassifier
    and verify inference works end-to-end.
    """

    @classmethod
    def setUpClass(cls):
        import joblib
        from ai.classifier import ActivityClassifier
        from train_classifier import train_model, save_model

        cls._tmpdir = tempfile.mkdtemp()
        rng = np.random.default_rng(7)
        X = rng.random((60, FEATURE_DIM)).astype(np.float32)
        y = np.array(
            ["STANDING"] * 15 + ["WALKING"] * 15 +
            ["SITTING"] * 15 + ["RUNNING"] * 10 + ["FALLING"] * 5
        )
        model, scaler, le = train_model(X, y, "random_forest", random_state=7)

        model_path  = os.path.join(cls._tmpdir, "test_clf.joblib")
        scaler_path = os.path.join(cls._tmpdir, "test_scaler.joblib")
        meta = {"feature_dim": FEATURE_DIM, "classes": list(le.classes_),
                "trained_at": "test", "performance_note": "test"}
        save_model(model, scaler, le, model_path, meta)
        # save_model saves scaler as stem + _scaler.joblib
        scaler_saved = model_path.replace(".joblib", "_scaler.joblib")

        cfg = {
            "classifier": {
                "model_path":  model_path,
                "scaler_path": scaler_saved,
            },
            "activity": {"confidence_threshold": 0.0},  # accept any conf
        }
        cls._clf = ActivityClassifier(cfg)

    def test_classifier_is_trained(self):
        self.assertTrue(self._clf.is_trained)

    def test_model_type_is_trained_ml(self):
        self.assertEqual(self._clf.model_type, "trained_ml")

    def test_predict_returns_valid_activity(self):
        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        result = self._clf.predict(fv)
        self.assertIn(result["activity"], Activity.ALL + [Activity.UNKNOWN])

    def test_predict_confidence_in_range(self):
        fv = np.random.randn(1, FEATURE_DIM).astype(np.float32)
        result = self._clf.predict(fv)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0 + 1e-6)

    def test_predict_model_type_field(self):
        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        result = self._clf.predict(fv)
        self.assertEqual(result["model_type"], "trained_ml")

    def test_predict_sequence_shape(self):
        """Classifier accepts (N, FEATURE_DIM) sequence input."""
        seq = np.random.randn(20, FEATURE_DIM).astype(np.float32)
        result = self._clf.predict(seq)
        self.assertIn(result["activity"], Activity.ALL + [Activity.UNKNOWN])

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)


# ── 8. Fallback classifier still works ────────────────────────────────────

class TestFallbackClassifier(unittest.TestCase):

    def _make_fallback_clf(self):
        from ai.classifier import ActivityClassifier
        cfg = {
            "classifier": {
                "model_path":  "/nonexistent/path/model.joblib",
                "scaler_path": "/nonexistent/path/scaler.joblib",
            },
            "activity": {"confidence_threshold": 0.60},
        }
        return ActivityClassifier(cfg)

    def test_fallback_mode_when_no_model(self):
        clf = self._make_fallback_clf()
        self.assertEqual(clf.model_type, "rule_based_baseline")
        self.assertFalse(clf.is_trained)

    def test_fallback_predict_does_not_crash(self):
        clf = self._make_fallback_clf()
        fv = np.zeros((5, FEATURE_DIM), dtype=np.float32)
        result = clf.predict(fv)
        self.assertIn("activity", result)
        self.assertIn("confidence", result)
        self.assertIn("model_type", result)

    def test_fallback_activity_is_valid(self):
        clf = self._make_fallback_clf()
        fv = np.random.randn(10, FEATURE_DIM).astype(np.float32)
        result = clf.predict(fv)
        self.assertIn(result["activity"], Activity.ALL + [Activity.UNKNOWN])

    def test_fallback_model_type_label(self):
        clf = self._make_fallback_clf()
        fv = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        result = clf.predict(fv)
        self.assertEqual(result["model_type"], "rule_based_baseline")


# ── 9. Diagnostics classifier mode reporting ──────────────────────────────

class TestDiagnosticsClassifierMode(unittest.TestCase):

    def test_diagnostics_reports_rule_based(self):
        from ai.inference import HAREngine
        import yaml, os
        cfg_path = os.path.join(_HERE, "config.yaml")
        # Override to point to a nonexistent model
        cfg = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh) or {}
        cfg.setdefault("classifier", {})["model_path"] = "/nonexistent/model.joblib"
        engine = HAREngine(config=cfg)
        diag = engine.get_diagnostics()
        clf_section = diag["classifier"]
        self.assertIn("mode", clf_section)
        self.assertEqual(clf_section["mode"], "rule_based_baseline")
        self.assertFalse(clf_section["is_trained"])

    def test_diagnostics_note_mentions_heuristic_for_rule_based(self):
        from ai.inference import HAREngine
        cfg = {"classifier": {"model_path": "/nonexistent/model.joblib"},
               "activity": {"confidence_threshold": 0.6}}
        engine = HAREngine(config=cfg)
        diag = engine.get_diagnostics()
        note = diag["classifier"]["note"].lower()
        self.assertIn("heuristic", note)

    def test_diagnostics_activities_list(self):
        from ai.inference import HAREngine
        cfg = {"classifier": {"model_path": "/nonexistent/model.joblib"}}
        engine = HAREngine(config=cfg)
        diag = engine.get_diagnostics()
        acts = diag["classifier"]["activities_supported"]
        self.assertIsInstance(acts, list)
        for a in Activity.ALL:
            self.assertIn(a, acts)

    def test_diagnostics_feature_dim_correct(self):
        from ai.inference import HAREngine
        cfg = {"classifier": {"model_path": "/nonexistent/model.joblib"}}
        engine = HAREngine(config=cfg)
        diag = engine.get_diagnostics()
        self.assertEqual(diag["classifier"]["feature_dim"], FEATURE_DIM)


# ── 10. Guided Recording Workflow Tests ────────────────────────────────────

class TestGuidedRecordingWorkflow(unittest.TestCase):

    @patch("tools.record_dataset.record_session")
    def test_run_guided_session_sequence(self, mock_record):
        from tools.record_dataset import run_guided_session
        tmp_dir = tempfile.mkdtemp()
        try:
            mock_record.side_effect = lambda activity_class, **kwargs: os.path.join(tmp_dir, f"{activity_class}.mp4")
            # Create dummy paths to satisfy isfile check in test mock
            for c in VALID_LABELS:
                open(os.path.join(tmp_dir, f"{c}.mp4"), "w").close()

            res = run_guided_session(
                subject_id="TEST_SUBJ",
                session_id="TEST_SESS",
                output_root=tmp_dir,
                manifest_path=os.path.join(tmp_dir, "manifest.csv"),
                interactive_prompt=False,
            )
            self.assertEqual(len(res), len(VALID_LABELS))
            self.assertEqual(mock_record.call_count, len(VALID_LABELS))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── 11. Record Dataset Loop & Stop Condition Tests ───────────────────────

class TestRecordDatasetLoop(unittest.TestCase):
    """Test recording loop stop conditions, duration handling, and camera FPS fallback."""

    @patch("tools.record_dataset.open_camera_capture")
    @patch("cv2.VideoWriter")
    def test_actual_fps_fallback_when_cap_returns_negative(self, mock_writer_cls, mock_open_cam):
        from tools.record_dataset import record_session

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        # DirectShow on Windows returns -1.0 for CAP_PROP_FPS
        def mock_get(prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640.0
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480.0
            elif prop == cv2.CAP_PROP_FPS:
                return -1.0
            return 0.0

        mock_cap.get.side_effect = mock_get
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_open_cam.return_value = (mock_cap, None)

        mock_writer = unittest.mock.MagicMock()
        mock_writer.isOpened.return_value = True
        def make_writer(path, *args, **kwargs):
            open(path, "w").close()
            return mock_writer
        mock_writer_cls.side_effect = make_writer

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_p = os.path.join(tmp_dir, "manifest.csv")
            saved = record_session(
                activity_class="STANDING",
                subject_id="SUBJ_TEST",
                session_id="SESS_TEST",
                fps=30.0,
                output_root=tmp_dir,
                manifest_path=manifest_p,
                headless_frames=5,
            )
            self.assertIsNotNone(saved)
            # Verify VideoWriter was constructed with 30.0 FPS, NOT -1.0 FPS
            mock_writer_cls.assert_called_once()
            args, _ = mock_writer_cls.call_args
            # args: (out_path, fourcc, actual_fps, (w, h))
            used_fps = args[2]
            self.assertEqual(used_fps, 30.0)

    @patch("tools.record_dataset.open_camera_capture")
    @patch("cv2.VideoWriter")
    def test_target_duration_auto_stop(self, mock_writer_cls, mock_open_cam):
        from tools.record_dataset import record_session

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_open_cam.return_value = (mock_cap, None)

        mock_writer = unittest.mock.MagicMock()
        mock_writer.isOpened.return_value = True
        def make_writer(path, *args, **kwargs):
            open(path, "w").close()
            return mock_writer
        mock_writer_cls.side_effect = make_writer

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_p = os.path.join(tmp_dir, "manifest.csv")

            t = [100.0]
            def mock_time():
                t[0] += 0.5
                return t[0]

            with patch("time.monotonic", side_effect=mock_time), patch("cv2.imshow"):
                saved = record_session(
                    activity_class="STANDING",
                    subject_id="SUBJ_TEST",
                    session_id="SESS_TEST",
                    output_root=tmp_dir,
                    manifest_path=manifest_p,
                    target_duration=2.0,
                )
                self.assertIsNotNone(saved)
                self.assertTrue(os.path.isfile(manifest_p))
                with open(manifest_p) as fh:
                    content = fh.read()
                    self.assertIn("STANDING", content)
                    self.assertIn("SUBJ_TEST", content)

    @patch("tools.record_dataset.open_camera_capture")
    @patch("cv2.VideoWriter")
    def test_manual_space_start_and_stop(self, mock_writer_cls, mock_open_cam):
        from tools.record_dataset import record_session

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_open_cam.return_value = (mock_cap, None)

        mock_writer = unittest.mock.MagicMock()
        mock_writer.isOpened.return_value = True
        def make_writer(path, *args, **kwargs):
            open(path, "w").close()
            return mock_writer
        mock_writer_cls.side_effect = make_writer

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_p = os.path.join(tmp_dir, "manifest.csv")

            # Frame 1: -1 (preview), Frame 2: 32 (SPACE start), Frame 3: -1 (recording), Frame 4: 32 (SPACE stop after >0.5s)
            sim_time = [100.0]
            def mock_time():
                sim_time[0] += 0.6  # >0.5s between key checks
                return sim_time[0]

            keys = iter([-1, 32, -1, 32])
            def mock_key(delay):
                try:
                    return next(keys)
                except StopIteration:
                    return 27

            with patch("time.monotonic", side_effect=mock_time), patch("cv2.waitKey", side_effect=mock_key), patch("cv2.imshow"):
                saved = record_session(
                    activity_class="WALKING",
                    subject_id="SUBJ_TEST",
                    session_id="SESS_TEST",
                    output_root=tmp_dir,
                    manifest_path=manifest_p,
                    target_duration=0.0,
                )
                self.assertIsNotNone(saved)
                self.assertTrue(os.path.isfile(manifest_p))

    @patch("tools.record_dataset.open_camera_capture")
    @patch("cv2.VideoWriter")
    def test_ten_second_target_does_not_stop_at_one_second(self, mock_writer_cls, mock_open_cam):
        from tools.record_dataset import record_session

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_open_cam.return_value = (mock_cap, None)

        mock_writer = unittest.mock.MagicMock()
        mock_writer.isOpened.return_value = True
        def make_writer(path, *args, **kwargs):
            open(path, "w").close()
            return mock_writer
        mock_writer_cls.side_effect = make_writer

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_p = os.path.join(tmp_dir, "manifest.csv")

            # Simulate 1/60th second per time.monotonic() call so each 30 FPS frame advances time
            sim_clock = [100.0]
            def mock_time():
                sim_clock[0] += (1.0 / 60.0)
                return sim_clock[0]

            # Simulate SPACE key (32) returned by cv2.waitKey (which previously triggered early stop at ~1s)
            def mock_key(delay):
                return 32

            with patch("time.monotonic", side_effect=mock_time), patch("cv2.waitKey", side_effect=mock_key), patch("cv2.imshow"):
                saved = record_session(
                    activity_class="STANDING",
                    subject_id="SUBJ_TEST",
                    session_id="SESS_TEST",
                    output_root=tmp_dir,
                    manifest_path=manifest_p,
                    target_duration=10.0,
                )
                self.assertIsNotNone(saved)
                # Over 10 seconds at 30 FPS, write is called ~300 times despite key 32
                self.assertGreater(mock_writer.write.call_count, 250)

    @patch("tools.record_dataset.open_camera_capture")
    @patch("cv2.VideoWriter")
    def test_five_second_target_recording(self, mock_writer_cls, mock_open_cam):
        from tools.record_dataset import record_session

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_open_cam.return_value = (mock_cap, None)

        mock_writer = unittest.mock.MagicMock()
        mock_writer.isOpened.return_value = True
        def make_writer(path, *args, **kwargs):
            open(path, "w").close()
            return mock_writer
        mock_writer_cls.side_effect = make_writer

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_p = os.path.join(tmp_dir, "manifest.csv")

            sim_clock = [100.0]
            def mock_time():
                sim_clock[0] += (1.0 / 60.0)
                return sim_clock[0]

            with patch("time.monotonic", side_effect=mock_time), patch("cv2.imshow"):
                saved = record_session(
                    activity_class="STANDING",
                    subject_id="SUBJ_TEST",
                    session_id="SESS_TEST",
                    output_root=tmp_dir,
                    manifest_path=manifest_p,
                    target_duration=5.0,
                )
                self.assertIsNotNone(saved)
                # Over 5 seconds at 30 FPS, write is called ~150 times
                self.assertGreater(mock_writer.write.call_count, 120)

    @patch("tools.record_dataset.open_camera_capture")
    @patch("cv2.VideoWriter")
    def test_videowriter_failure_returns_none(self, mock_writer_cls, mock_open_cam):
        from tools.record_dataset import record_session

        mock_cap = unittest.mock.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: 640.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (480.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_open_cam.return_value = (mock_cap, None)

        mock_writer = unittest.mock.MagicMock()
        mock_writer.isOpened.return_value = False  # VideoWriter failed to open
        mock_writer_cls.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_p = os.path.join(tmp_dir, "manifest.csv")
            saved = record_session(
                activity_class="STANDING",
                subject_id="SUBJ_TEST",
                session_id="SESS_TEST",
                output_root=tmp_dir,
                manifest_path=manifest_p,
                target_duration=10.0,
            )
            self.assertIsNone(saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)



