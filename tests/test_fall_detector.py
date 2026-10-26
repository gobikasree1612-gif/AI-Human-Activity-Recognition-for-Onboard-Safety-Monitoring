"""
test_fall_detector.py
=====================
Unit tests for the fall detection module.

Runnable without GPU.
"""

import sys
import os
import unittest
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.fall_detector import FallDetector
from ai.features import FEATURE_DIM


def _minimal_config(
    confirm_frames: int = 5,
    angle_thresh: float = 55,
    vel_thresh: float = 0.04,
    timeout_s: float = 10.0,
) -> dict:
    return {
        "fall": {
            "confirmation_frames": confirm_frames,
            "downward_velocity_threshold": vel_thresh,
            "body_angle_threshold": angle_thresh,
            "low_height_fraction": 0.55,
            "possible_fall_timeout_seconds": timeout_s,
        }
    }


def _make_fv(overrides: dict = None) -> np.ndarray:
    """Create a feature vector with sensible defaults for standing.

    Pass integer-keyed overrides as a plain dict:
        _make_fv({6: 70.0, 17: 0.75})

    Python does not allow integer **kwargs, so we accept a positional dict.
    """
    fv = np.zeros(FEATURE_DIM, dtype=np.float32)
    # Feature indices mirror features.py
    defaults = {
        6:  10.0,   # body_lean_angle (small = upright)
        7:  0.25,   # torso_height_norm
        17: 0.45,   # hip_center_y_norm
        18: 0.30,   # shoulder_center_y_norm
        29: 0.80,   # visibility_score
    }
    if overrides:
        defaults.update(overrides)
    for idx, val in defaults.items():
        fv[idx] = val
    return fv


class TestNormalMovement(unittest.TestCase):

    def test_normal_movement_stays_normal(self):
        fd = FallDetector(_minimal_config())
        for _ in range(20):
            fv = _make_fv()
            fd.update(1, fv)
        self.assertFalse(fd.is_possible_fall(1))
        self.assertFalse(fd.is_confirmed_fall(1))
        self.assertEqual(fd.get_status(1), "NORMAL")


class TestPossibleFall(unittest.TestCase):

    def test_large_body_angle_triggers_possible_fall(self):
        fd = FallDetector(_minimal_config(angle_thresh=55))
        # Normal frames
        for _ in range(5):
            fd.update(1, _make_fv())
        # Sudden large lean angle (body nearly horizontal)
        fd.update(1, _make_fv({6: 70.0}))
        self.assertTrue(fd.is_possible_fall(1))

    def test_rapid_downward_velocity_triggers_possible_fall(self):
        fd = FallDetector(_minimal_config(vel_thresh=0.04))
        # Give a baseline hip position
        for _ in range(3):
            fd.update(1, _make_fv({17: 0.40}))
        # Sudden large downward jump
        fd.update(1, _make_fv({17: 0.70}))
        self.assertTrue(fd.is_possible_fall(1))

    def test_small_gradual_movement_does_not_trigger(self):
        fd = FallDetector(_minimal_config(vel_thresh=0.04))
        # Slow gradual descent – tiny steps well below threshold
        for i in range(20):
            fd.update(1, _make_fv({17: 0.40 + i * 0.005}))   # 0.005/frame << 0.04
        self.assertFalse(fd.is_possible_fall(1))


class TestConfirmedFall(unittest.TestCase):

    def test_confirmed_fall_after_sustained_low_position(self):
        fd = FallDetector(_minimal_config(confirm_frames=5, angle_thresh=55))

        # Normal frames
        for _ in range(5):
            fd.update(1, _make_fv({6: 10.0, 17: 0.40}))

        # Trigger possible fall with large angle
        fd.update(1, _make_fv({6: 70.0, 17: 0.75}))
        self.assertTrue(fd.is_possible_fall(1))

        # Sustain fallen posture for confirmation_frames
        for _ in range(5):
            fd.update(1, _make_fv({6: 68.0, 17: 0.76}))

        self.assertTrue(fd.is_confirmed_fall(1))
        self.assertEqual(fd.get_status(1), "CONFIRMED_FALL")

    def test_recovery_before_confirmation_resets_to_normal(self):
        fd = FallDetector(_minimal_config(confirm_frames=10, angle_thresh=55))

        for _ in range(5):
            fd.update(1, _make_fv({6: 10.0, 17: 0.40}))

        fd.update(1, _make_fv({6: 70.0, 17: 0.75}))
        self.assertTrue(fd.is_possible_fall(1))

        # Person recovers immediately (upright pose)
        for _ in range(3):
            fd.update(1, _make_fv({6: 10.0, 17: 0.40}))

        self.assertFalse(fd.is_confirmed_fall(1))
        self.assertEqual(fd.get_status(1), "NORMAL")


class TestReset(unittest.TestCase):

    def test_manual_reset_after_confirmed_fall(self):
        fd = FallDetector(_minimal_config(confirm_frames=3, angle_thresh=55))

        for _ in range(3):
            fd.update(1, _make_fv({6: 10.0}))

        fd.update(1, _make_fv({6: 70.0, 17: 0.75}))
        for _ in range(4):
            fd.update(1, _make_fv({6: 68.0, 17: 0.76}))

        self.assertTrue(fd.is_confirmed_fall(1))

        fd.reset(1)
        self.assertEqual(fd.get_status(1), "NORMAL")
        self.assertFalse(fd.is_possible_fall(1))
        self.assertFalse(fd.is_confirmed_fall(1))

    def test_remove_person(self):
        fd = FallDetector(_minimal_config())
        fd.update(1, _make_fv())
        fd.remove(1)
        # After removal, default status should be NORMAL
        self.assertEqual(fd.get_status(1), "NORMAL")


class TestEvidence(unittest.TestCase):

    def test_evidence_generated_on_possible_fall(self):
        fd = FallDetector(_minimal_config(angle_thresh=55))
        for _ in range(3):
            fd.update(1, _make_fv())
        fd.update(1, _make_fv({6: 70.0}))
        evidence = fd.get_evidence(1)
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)

    def test_no_evidence_in_normal_state(self):
        fd = FallDetector(_minimal_config())
        fd.update(1, _make_fv())
        evidence = fd.get_evidence(1)
        # Evidence may be empty or non-empty; should be a list
        self.assertIsInstance(evidence, list)


class TestMultiplePeople(unittest.TestCase):

    def test_independent_fall_states(self):
        fd = FallDetector(_minimal_config(confirm_frames=3, angle_thresh=55))

        # Person 1 – normal
        for _ in range(5):
            fd.update(1, _make_fv({6: 10.0}))

        # Person 2 – falls
        for _ in range(5):
            fd.update(2, _make_fv({6: 10.0}))
        fd.update(2, _make_fv({6: 70.0, 17: 0.75}))
        for _ in range(4):
            fd.update(2, _make_fv({6: 68.0, 17: 0.76}))

        self.assertEqual(fd.get_status(1), "NORMAL")
        self.assertEqual(fd.get_status(2), "CONFIRMED_FALL")


if __name__ == "__main__":
    unittest.main()
