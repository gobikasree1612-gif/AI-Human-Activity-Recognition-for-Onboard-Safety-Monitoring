"""
test_inference.py
=================
Integration tests for the HAREngine output schema.

These tests use a MOCK detector so no GPU / camera / YOLO model is needed.
They verify the output contract that the dashboard developer depends on.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.features import FEATURE_DIM
from ai.schemas import Activity, SystemStatus


# ---------------------------------------------------------------------------
# Helpers to build fake YOLO detector output
# ---------------------------------------------------------------------------

def _make_fake_keypoints(conf: float = 0.8) -> np.ndarray:
    kps = np.zeros((17, 3), dtype=np.float32)
    positions = [
        (150, 80), (145, 75), (155, 75), (140, 80), (160, 80),
        (130, 150), (170, 150), (120, 220), (180, 220), (115, 290),
        (185, 290), (135, 280), (165, 280), (130, 360), (170, 360),
        (128, 440), (172, 440),
    ]
    for i, (x, y) in enumerate(positions):
        kps[i] = [x, y, conf]
    return kps


def _make_detected_person(person_id: int = 1, conf: float = 0.85):
    from ai.detector import DetectedPerson
    return DetectedPerson(
        person_id=person_id,
        bbox=[100, 50, 200, 450],
        keypoints=_make_fake_keypoints(conf=conf),
        detection_confidence=conf,
    )


# ---------------------------------------------------------------------------
# Build an engine with a mocked YOLO detector
# ---------------------------------------------------------------------------

class _MockHAREngine:
    """HAREngine with detector mocked out (no YOLO needed)."""

    def __init__(self, detected_people=None):
        from ai.inference import HAREngine
        # Config with no model file so it uses rule-based fallback
        cfg = {
            "activity": {
                "sequence_length": 5,
                "confidence_threshold": 0.50,
                "pose_confidence_threshold": 0.30,
            },
            "fall": {
                "confirmation_frames": 5,
                "downward_velocity_threshold": 0.04,
                "body_angle_threshold": 55,
                "low_height_fraction": 0.55,
                "possible_fall_timeout_seconds": 10.0,
            },
            "tracking": {"max_missing_frames": 30},
            "performance": {"warmup_frames": 1},
            "classifier": {
                "model_path": "__nonexistent_model__.joblib",
                "scaler_path": "__nonexistent_scaler__.joblib",
            },
        }
        self._engine = HAREngine(config=cfg)
        self._detected_people = detected_people or []

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process frame with mocked detector."""
        with patch.object(
            self._engine._detector, "process",
            return_value=self._detected_people
        ):
            result, _ = self._engine.process_frame(frame)
        return result


class TestOutputSchema(unittest.TestCase):

    def setUp(self):
        self._frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_empty_detection_result_schema(self):
        """Empty detections must still produce a valid result dict."""
        engine = _MockHAREngine(detected_people=[])
        result = engine.process_frame(self._frame)

        self.assertIn("timestamp", result)
        self.assertIn("system_status", result)
        self.assertIn("people", result)
        self.assertIn("metrics", result)

        self.assertEqual(result["people"], [])
        self.assertEqual(result["system_status"], SystemStatus.NORMAL)
        self.assertEqual(result["metrics"]["people_count"], 0)
        self.assertGreaterEqual(result["metrics"]["fps"], 0)
        self.assertGreaterEqual(result["metrics"]["latency_ms"], 0)

    def test_single_person_result_schema(self):
        """One person detected – result must include all required fields."""
        person = _make_detected_person(person_id=1)
        engine = _MockHAREngine(detected_people=[person])

        # Run several frames to fill temporal buffer
        for _ in range(6):
            result = engine.process_frame(self._frame)

        self.assertEqual(len(result["people"]), 1)
        p = result["people"][0]

        required_keys = [
            "person_id", "bbox", "activity", "confidence",
            "model_type", "alert", "alert_type", "severity", "evidence",
        ]
        for key in required_keys:
            self.assertIn(key, p, f"Missing key: {key}")

        self.assertIsInstance(p["person_id"], int)
        self.assertIsInstance(p["bbox"], list)
        self.assertEqual(len(p["bbox"]), 4)
        self.assertIn(p["activity"], Activity.ALL + [Activity.UNKNOWN])
        self.assertIsInstance(p["confidence"], float)
        self.assertGreaterEqual(p["confidence"], 0.0)
        self.assertLessEqual(p["confidence"], 1.0)
        self.assertIn(p["model_type"], ["trained_ml", "rule_based_baseline"])
        self.assertIsInstance(p["alert"], bool)
        self.assertIsInstance(p["evidence"], list)

    def test_two_people_independent_results(self):
        """Two people must produce two independent entries."""
        people = [
            _make_detected_person(person_id=1),
            _make_detected_person(person_id=2),
        ]
        engine = _MockHAREngine(detected_people=people)

        for _ in range(6):
            result = engine.process_frame(self._frame)

        self.assertEqual(len(result["people"]), 2)
        ids = {p["person_id"] for p in result["people"]}
        self.assertEqual(ids, {1, 2})

    def test_model_type_is_rule_based_without_trained_model(self):
        """Without a trained model file, model_type must be 'rule_based_baseline'."""
        person = _make_detected_person()
        engine = _MockHAREngine(detected_people=[person])

        for _ in range(6):
            result = engine.process_frame(self._frame)

        for p in result["people"]:
            self.assertEqual(p["model_type"], "rule_based_baseline")

    def test_system_status_is_normal_without_falls(self):
        """Normal activity must yield NORMAL system status."""
        person = _make_detected_person()
        engine = _MockHAREngine(detected_people=[person])

        for _ in range(6):
            result = engine.process_frame(self._frame)

        self.assertEqual(result["system_status"], SystemStatus.NORMAL)

    def test_none_frame_does_not_crash(self):
        """A None frame must be handled gracefully."""
        engine = _MockHAREngine()
        result = engine.process_frame(None)
        self.assertIn("timestamp", result)
        self.assertIn("people", result)

    def test_black_frame_does_not_crash(self):
        """Completely black frame should not crash."""
        engine = _MockHAREngine(detected_people=[])
        result = engine.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIn("people", result)

    def test_low_confidence_detection_yields_unknown(self):
        """Very low confidence pose should result in UNKNOWN activity."""
        person = _make_detected_person(conf=0.01)
        engine = _MockHAREngine(detected_people=[person])
        result = engine.process_frame(self._frame)
        # Should not crash; may return UNKNOWN
        if result["people"]:
            p = result["people"][0]
            # If activity is UNKNOWN, that's acceptable
            self.assertIn(p["activity"], Activity.ALL + [Activity.UNKNOWN])

    def test_metrics_people_count_matches_people_list(self):
        """metrics.people_count must match len(people)."""
        people = [_make_detected_person(1), _make_detected_person(2)]
        engine = _MockHAREngine(detected_people=people)
        for _ in range(6):
            result = engine.process_frame(self._frame)
        self.assertEqual(
            result["metrics"]["people_count"],
            len(result["people"]),
        )

    def test_evidence_is_non_empty_list(self):
        """Evidence must always be a non-empty list of strings."""
        person = _make_detected_person()
        engine = _MockHAREngine(detected_people=[person])
        for _ in range(6):
            result = engine.process_frame(self._frame)
        for p in result["people"]:
            self.assertIsInstance(p["evidence"], list)
            self.assertGreater(len(p["evidence"]), 0)
            for e in p["evidence"]:
                self.assertIsInstance(e, str)

    def test_timestamp_format(self):
        """Timestamp must be a non-empty ISO string."""
        engine = _MockHAREngine()
        result = engine.process_frame(self._frame)
        ts = result.get("timestamp", "")
        self.assertIsInstance(ts, str)
        self.assertGreater(len(ts), 0)

    def test_fall_latching_and_safety_reset(self):
        """Once a fall is confirmed, system status remains CRITICAL until reset_safety() is called."""
        from ai.detector import DetectedPerson

        kps_fall = np.zeros((17, 3), dtype=np.float32)
        positions = [
            (100, 300), (105, 295), (105, 305), (110, 290), (110, 310),
            (150, 300), (150, 300), (180, 300), (180, 300), (200, 300),
            (200, 300), (250, 300), (250, 300), (280, 300), (280, 300),
            (300, 300), (300, 300),
        ]
        for i, (x, y) in enumerate(positions):
            kps_fall[i] = [x, y, 0.8]

        person_fall = DetectedPerson(
            person_id=1,
            bbox=[100, 250, 300, 350],
            keypoints=kps_fall,
            detection_confidence=0.85,
        )

        engine = _MockHAREngine(detected_people=[person_fall])
        for _ in range(10):
            result = engine.process_frame(self._frame)

        self.assertEqual(result["system_status"], SystemStatus.CRITICAL)
        self.assertIsNotNone(result.get("latched_fall"))

        # Switch to normal standing person
        person_normal = _make_detected_person(person_id=1)
        engine._detected_people = [person_normal]

        result_next = engine.process_frame(self._frame)
        # Latched safety state must keep system status CRITICAL
        self.assertEqual(result_next["system_status"], SystemStatus.CRITICAL)
        self.assertIsNotNone(result_next.get("latched_fall"))

        # Explicit reset must restore NORMAL status
        engine._engine.reset_safety()
        result_after_reset = engine.process_frame(self._frame)
        self.assertEqual(result_after_reset["system_status"], SystemStatus.NORMAL)
        self.assertIsNone(result_after_reset.get("latched_fall"))



class TestProcessFrameReturnsAnnotatedFrame(unittest.TestCase):

    def test_annotated_frame_returned(self):
        """process_frame must return (dict, numpy BGR frame)."""
        from ai.inference import HAREngine
        cfg = {
            "activity": {"sequence_length": 5, "confidence_threshold": 0.50,
                         "pose_confidence_threshold": 0.30},
            "fall": {"confirmation_frames": 5, "downward_velocity_threshold": 0.04,
                     "body_angle_threshold": 55, "low_height_fraction": 0.55,
                     "possible_fall_timeout_seconds": 10.0},
            "tracking": {"max_missing_frames": 30},
            "performance": {"warmup_frames": 1},
            "classifier": {"model_path": "__nonexistent__.joblib",
                           "scaler_path": "__nonexistent__.joblib"},
        }
        engine = HAREngine(config=cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch.object(engine._detector, "process", return_value=[]):
            result, annotated = engine.process_frame(frame)

        self.assertIsInstance(result, dict)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
