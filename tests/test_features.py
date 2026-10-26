"""
test_features.py
================
Unit tests for the feature extraction module.

Designed to run WITHOUT a GPU (no YOLO model required).
"""

import sys
import os
import math
import unittest

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.features import extract_pose_features, FEATURE_DIM


def _make_keypoints(visible: bool = True, conf: float = 0.9) -> np.ndarray:
    """Create a realistic standing-pose keypoints array (17, 3)."""
    kps = np.zeros((17, 3), dtype=np.float32)
    # Approximate COCO keypoints for a standing person at (100,50)-(200,450)
    positions = [
        (150, 80),    # 0  nose
        (145, 75),    # 1  left_eye
        (155, 75),    # 2  right_eye
        (140, 80),    # 3  left_ear
        (160, 80),    # 4  right_ear
        (130, 150),   # 5  left_shoulder
        (170, 150),   # 6  right_shoulder
        (120, 220),   # 7  left_elbow
        (180, 220),   # 8  right_elbow
        (115, 290),   # 9  left_wrist
        (185, 290),   # 10 right_wrist
        (135, 280),   # 11 left_hip
        (165, 280),   # 12 right_hip
        (130, 360),   # 13 left_knee
        (170, 360),   # 14 right_knee
        (128, 440),   # 15 left_ankle
        (172, 440),   # 16 right_ankle
    ]
    for i, (x, y) in enumerate(positions):
        kps[i, 0] = x
        kps[i, 1] = y
        kps[i, 2] = conf if visible else 0.0
    return kps


BBOX = [100, 50, 200, 450]   # [x1, y1, x2, y2]


class TestExtractPoseFeatures(unittest.TestCase):

    def test_output_shape(self):
        """Feature vector must have exactly FEATURE_DIM elements."""
        kps = _make_keypoints()
        fv = extract_pose_features(kps, BBOX)
        self.assertEqual(fv.shape, (FEATURE_DIM,))
        self.assertEqual(fv.dtype, np.float32)

    def test_no_nan_inf(self):
        """Feature vector must never contain NaN or Inf."""
        kps = _make_keypoints()
        fv = extract_pose_features(kps, BBOX)
        self.assertTrue(np.all(np.isfinite(fv)), "NaN or Inf found in features")

    def test_all_missing_keypoints(self):
        """Zero-confidence keypoints should return a safe zero vector."""
        kps = np.zeros((17, 3), dtype=np.float32)
        fv = extract_pose_features(kps, BBOX)
        self.assertEqual(fv.shape, (FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(fv)))

    def test_partial_missing_keypoints(self):
        """Partial visibility must not crash and must return finite values."""
        kps = _make_keypoints(conf=0.9)
        # Zero out legs
        kps[13:17, 2] = 0.0
        fv = extract_pose_features(kps, BBOX)
        self.assertEqual(fv.shape, (FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(fv)))

    def test_visibility_score(self):
        """Visibility score (index 29) should reflect keypoint confidence."""
        kps_good = _make_keypoints(conf=0.9)
        kps_bad  = _make_keypoints(conf=0.1)
        fv_good  = extract_pose_features(kps_good, BBOX)
        fv_bad   = extract_pose_features(kps_bad, BBOX)
        # Good keypoints should yield higher visibility score
        self.assertGreater(fv_good[29], fv_bad[29])

    def test_body_lean_small_for_standing(self):
        """Standing pose should have a small body lean angle (index 6)."""
        kps = _make_keypoints()
        fv = extract_pose_features(kps, BBOX)
        # Body lean angle < 30 for upright standing
        self.assertLess(fv[6], 30.0)

    def test_normalisation_is_resolution_independent(self):
        """
        Features normalised by bbox should be identical for the same
        relative pose at different absolute resolutions.
        """
        kps1 = _make_keypoints()
        bbox1 = [100, 50, 200, 450]

        # Scale everything by 2×
        kps2 = kps1.copy()
        kps2[:, :2] *= 2
        bbox2 = [b * 2 for b in bbox1]

        fv1 = extract_pose_features(kps1, bbox1)
        fv2 = extract_pose_features(kps2, bbox2)

        # Normalised features should be (approximately) the same
        np.testing.assert_allclose(fv1, fv2, atol=1e-4)

    def test_empty_bbox_does_not_crash(self):
        """A degenerate bbox must not raise an exception."""
        kps = _make_keypoints()
        fv = extract_pose_features(kps, [0, 0, 0, 0])
        self.assertEqual(fv.shape, (FEATURE_DIM,))

    def test_wrong_keypoint_shape_does_not_crash(self):
        """A truncated keypoints array must not crash."""
        kps = np.zeros((5, 3), dtype=np.float32)   # Only 5 keypoints
        fv = extract_pose_features(kps, BBOX)
        self.assertEqual(fv.shape, (FEATURE_DIM,))


class TestKneeAngles(unittest.TestCase):

    def test_straight_legs_give_large_knee_angle(self):
        """Straight legs → large knee angle (close to 180°)."""
        kps = _make_keypoints()
        fv = extract_pose_features(kps, BBOX)
        # Knee bend (features[19,20]) = 1 - angle/180; small for straight legs
        self.assertLess(fv[19], 0.5)
        self.assertLess(fv[20], 0.5)

    def test_sitting_gives_larger_knee_bend(self):
        """Bent knees (sitting-like) → higher knee_bend values."""
        kps = _make_keypoints()
        # Move knees forward (simulate sitting)
        kps[13, 0] += 40   # left knee forward
        kps[14, 0] += 40   # right knee forward
        kps[13, 1] = kps[11, 1]   # left knee at hip height
        kps[14, 1] = kps[12, 1]   # right knee at hip height
        fv = extract_pose_features(kps, BBOX)
        self.assertGreater(fv[19], 0.3)


if __name__ == "__main__":
    unittest.main()
