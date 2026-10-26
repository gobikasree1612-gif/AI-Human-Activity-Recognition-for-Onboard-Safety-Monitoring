"""
features.py
===========
Keypoint → Feature-vector engineering.

Takes COCO-17 keypoints from YOLO Pose and extracts meaningful,
normalised, body-size-invariant features suitable for activity
classification.

IMPORTANT:
- All coordinates are normalised relative to the person's bounding box
  so the feature vector is resolution-independent.
- Missing / low-confidence keypoints are handled gracefully; the
  feature value is set to a safe default (0.0) rather than crashing.
- Every public function is pure (no side-effects) for easy testing.

Feature vector layout (34 values):
  [0]   shoulder_angle
  [1]   hip_angle
  [2]   left_knee_angle
  [3]   right_knee_angle
  [4]   left_elbow_angle
  [5]   right_elbow_angle
  [6]   body_lean_angle      (torso tilt from vertical)
  [7]   torso_height_norm    (normalised torso height)
  [8]   body_height_norm     (head-to-ankle height / bbox height)
  [9]   shoulder_width_norm
  [10]  hip_width_norm
  [11]  left_knee_hip_ratio  (knee y / hip y, normalised)
  [12]  right_knee_hip_ratio
  [13]  left_ankle_hip_ratio
  [14]  right_ankle_hip_ratio
  [15]  center_x_norm        (body centre x, normalised)
  [16]  center_y_norm        (body centre y, normalised)
  [17]  hip_center_y_norm
  [18]  shoulder_center_y_norm
  [19]  knee_bend_left       (0→straight, 1→fully bent)
  [20]  knee_bend_right
  [21]  left_hip_conf        (keypoint confidence passthrough)
  [22]  right_hip_conf
  [23]  left_knee_conf
  [24]  right_knee_conf
  [25]  left_ankle_conf
  [26]  right_ankle_conf
  [27]  left_shoulder_conf
  [28]  right_shoulder_conf
  [29]  visibility_score     (mean confidence of core keypoints)
  [30]  upper_lower_ratio    (torso height / leg height)
  [31]  nose_to_hip_y_norm
  [32]  left_wrist_y_norm
  [33]  right_wrist_y_norm
"""

from __future__ import annotations

import math
import logging
from typing import List, Optional

import numpy as np

from ai.detector import COCO_KEYPOINTS

logger = logging.getLogger(__name__)

# Total number of features produced by extract_pose_features()
FEATURE_DIM = 34

# Core keypoints used for visibility / confidence scoring
_CORE_KP = [
    COCO_KEYPOINTS["left_shoulder"],
    COCO_KEYPOINTS["right_shoulder"],
    COCO_KEYPOINTS["left_hip"],
    COCO_KEYPOINTS["right_hip"],
    COCO_KEYPOINTS["left_knee"],
    COCO_KEYPOINTS["right_knee"],
    COCO_KEYPOINTS["left_ankle"],
    COCO_KEYPOINTS["right_ankle"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kp(keypoints: np.ndarray, name: str, min_conf: float = 0.0
        ) -> Optional[np.ndarray]:
    """
    Return (x, y) for a named keypoint if confidence ≥ min_conf,
    else return None.
    """
    idx = COCO_KEYPOINTS[name]
    if idx >= len(keypoints):
        return None
    x, y, c = keypoints[idx]
    if c < min_conf:
        return None
    return np.array([x, y], dtype=np.float32)


def _kp_conf(keypoints: np.ndarray, name: str) -> float:
    idx = COCO_KEYPOINTS[name]
    if idx >= len(keypoints):
        return 0.0
    return float(keypoints[idx, 2])


def _angle_3pts(a: Optional[np.ndarray],
                b: Optional[np.ndarray],
                c: Optional[np.ndarray]) -> float:
    """
    Angle at vertex B formed by A-B-C in degrees [0, 180].
    Returns 0.0 if any point is None.
    """
    if a is None or b is None or c is None:
        return 0.0
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_val)))


def _norm_y(y: float, bbox: List[int]) -> float:
    """Normalise a pixel y-coordinate into [0, 1] using bbox height."""
    _, y1, _, y2 = bbox
    h = max(y2 - y1, 1)
    return (y - y1) / h


def _norm_x(x: float, bbox: List[int]) -> float:
    x1, _, x2, _ = bbox
    w = max(x2 - x1, 1)
    return (x - x1) / w


def _norm_length(length: float, bbox: List[int]) -> float:
    """Normalise a pixel length using bbox height."""
    _, y1, _, y2 = bbox
    h = max(y2 - y1, 1)
    return length / h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pose_features(
    keypoints: np.ndarray,
    bbox: List[int],
    min_conf: float = 0.30,
) -> np.ndarray:
    """
    Extract a (FEATURE_DIM,) float32 feature vector from COCO-17 keypoints.

    Parameters
    ----------
    keypoints : np.ndarray, shape (17, 3)
        YOLO Pose output: (x, y, confidence) per keypoint.
    bbox : List[int]
        Bounding box [x1, y1, x2, y2] in pixel coordinates.
    min_conf : float
        Minimum keypoint confidence to use a keypoint.

    Returns
    -------
    np.ndarray, shape (FEATURE_DIM,)
        Normalised, resolution-independent feature vector.
        All values are finite floats (no NaN / Inf).
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    try:
        # ------------------------------------------------------------------
        # Fetch keypoints (returns None if confidence too low)
        # ------------------------------------------------------------------
        c = min_conf  # shorthand

        nose       = _kp(keypoints, "nose",           c)
        l_shoulder = _kp(keypoints, "left_shoulder",  c)
        r_shoulder = _kp(keypoints, "right_shoulder", c)
        l_elbow    = _kp(keypoints, "left_elbow",     c)
        r_elbow    = _kp(keypoints, "right_elbow",    c)
        l_wrist    = _kp(keypoints, "left_wrist",     c)
        r_wrist    = _kp(keypoints, "right_wrist",    c)
        l_hip      = _kp(keypoints, "left_hip",       c)
        r_hip      = _kp(keypoints, "right_hip",      c)
        l_knee     = _kp(keypoints, "left_knee",      c)
        r_knee     = _kp(keypoints, "right_knee",     c)
        l_ankle    = _kp(keypoints, "left_ankle",     c)
        r_ankle    = _kp(keypoints, "right_ankle",    c)

        # ------------------------------------------------------------------
        # [0] Shoulder angle (torso width angle)
        # ------------------------------------------------------------------
        if l_shoulder is not None and r_shoulder is not None:
            dx = r_shoulder[0] - l_shoulder[0]
            dy = r_shoulder[1] - l_shoulder[1]
            features[0] = float(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-8)))

        # ------------------------------------------------------------------
        # [1] Hip angle
        # ------------------------------------------------------------------
        if l_hip is not None and r_hip is not None:
            dx = r_hip[0] - l_hip[0]
            dy = r_hip[1] - l_hip[1]
            features[1] = float(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-8)))

        # ------------------------------------------------------------------
        # [2] Left knee angle, [3] Right knee angle
        # ------------------------------------------------------------------
        features[2] = _angle_3pts(l_hip, l_knee, l_ankle)
        features[3] = _angle_3pts(r_hip, r_knee, r_ankle)

        # ------------------------------------------------------------------
        # [4] Left elbow angle, [5] Right elbow angle
        # ------------------------------------------------------------------
        features[4] = _angle_3pts(l_shoulder, l_elbow, l_wrist)
        features[5] = _angle_3pts(r_shoulder, r_elbow, r_wrist)

        # ------------------------------------------------------------------
        # [6] Body lean angle (torso tilt from vertical)
        # ------------------------------------------------------------------
        if l_shoulder is not None and r_shoulder is not None and \
                l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2.0
            hip_mid = (l_hip + r_hip) / 2.0
            dx = hip_mid[0] - shoulder_mid[0]
            dy = hip_mid[1] - shoulder_mid[1]
            # 0° = upright, 90° = horizontal
            features[6] = float(
                math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8))
            )

        # ------------------------------------------------------------------
        # [7] Torso height (normalised by bbox height)
        # ------------------------------------------------------------------
        if l_shoulder is not None and r_shoulder is not None and \
                l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2.0
            hip_mid = (l_hip + r_hip) / 2.0
            torso_h = abs(hip_mid[1] - shoulder_mid[1])
            features[7] = _norm_length(torso_h, bbox)

        # ------------------------------------------------------------------
        # [8] Body height (nose or shoulder → ankle, normalised)
        # ------------------------------------------------------------------
        top_pt = nose if nose is not None else (
            l_shoulder if l_shoulder is not None else r_shoulder
        )
        bot_pt = None
        if l_ankle is not None and r_ankle is not None:
            bot_pt = (l_ankle + r_ankle) / 2.0
        elif l_ankle is not None:
            bot_pt = l_ankle
        elif r_ankle is not None:
            bot_pt = r_ankle

        if top_pt is not None and bot_pt is not None:
            features[8] = _norm_length(abs(bot_pt[1] - top_pt[1]), bbox)

        # ------------------------------------------------------------------
        # [9] Shoulder width, [10] Hip width (normalised)
        # ------------------------------------------------------------------
        if l_shoulder is not None and r_shoulder is not None:
            sw = abs(r_shoulder[0] - l_shoulder[0])
            features[9] = _norm_length(sw, bbox)

        if l_hip is not None and r_hip is not None:
            hw = abs(r_hip[0] - l_hip[0])
            features[10] = _norm_length(hw, bbox)

        # ------------------------------------------------------------------
        # [11-14] Relative joint positions (knee / ankle y relative to hip y)
        # ------------------------------------------------------------------
        if l_hip is not None:
            if l_knee is not None:
                features[11] = _norm_y(l_knee[1], bbox) - _norm_y(l_hip[1], bbox)
            if l_ankle is not None:
                features[13] = _norm_y(l_ankle[1], bbox) - _norm_y(l_hip[1], bbox)

        if r_hip is not None:
            if r_knee is not None:
                features[12] = _norm_y(r_knee[1], bbox) - _norm_y(r_hip[1], bbox)
            if r_ankle is not None:
                features[14] = _norm_y(r_ankle[1], bbox) - _norm_y(r_hip[1], bbox)

        # ------------------------------------------------------------------
        # [15-18] Body centre positions (normalised)
        # ------------------------------------------------------------------
        if l_shoulder is not None and r_shoulder is not None and \
                l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2.0
            hip_mid = (l_hip + r_hip) / 2.0
            center = (shoulder_mid + hip_mid) / 2.0
            features[15] = _norm_x(center[0], bbox)
            features[16] = _norm_y(center[1], bbox)
            features[17] = _norm_y(hip_mid[1], bbox)
            features[18] = _norm_y(shoulder_mid[1], bbox)

        # ------------------------------------------------------------------
        # [19-20] Knee bend (0=straight, 1=fully bent ~90°)
        # ------------------------------------------------------------------
        if features[2] > 0:
            features[19] = 1.0 - min(features[2] / 180.0, 1.0)
        if features[3] > 0:
            features[20] = 1.0 - min(features[3] / 180.0, 1.0)

        # ------------------------------------------------------------------
        # [21-28] Keypoint confidence pass-through (core joints)
        # ------------------------------------------------------------------
        features[21] = _kp_conf(keypoints, "left_hip")
        features[22] = _kp_conf(keypoints, "right_hip")
        features[23] = _kp_conf(keypoints, "left_knee")
        features[24] = _kp_conf(keypoints, "right_knee")
        features[25] = _kp_conf(keypoints, "left_ankle")
        features[26] = _kp_conf(keypoints, "right_ankle")
        features[27] = _kp_conf(keypoints, "left_shoulder")
        features[28] = _kp_conf(keypoints, "right_shoulder")

        # ------------------------------------------------------------------
        # [29] Overall visibility score (mean of core keypoints)
        # ------------------------------------------------------------------
        core_confs = [float(keypoints[i, 2]) for i in _CORE_KP
                      if i < len(keypoints)]
        features[29] = float(np.mean(core_confs)) if core_confs else 0.0

        # ------------------------------------------------------------------
        # [30] Upper / lower body ratio
        # ------------------------------------------------------------------
        if l_shoulder is not None and r_shoulder is not None and \
                l_hip is not None and r_hip is not None:
            shoulder_mid = (l_shoulder + r_shoulder) / 2.0
            hip_mid = (l_hip + r_hip) / 2.0
            torso_h = abs(hip_mid[1] - shoulder_mid[1]) + 1e-6
            if bot_pt is not None:
                leg_h = abs(bot_pt[1] - hip_mid[1]) + 1e-6
                features[30] = torso_h / leg_h

        # ------------------------------------------------------------------
        # [31] Nose to hip y (normalised) – body compactness
        # ------------------------------------------------------------------
        if nose is not None and l_hip is not None and r_hip is not None:
            hip_mid = (l_hip + r_hip) / 2.0
            features[31] = _norm_length(abs(hip_mid[1] - nose[1]), bbox)

        # ------------------------------------------------------------------
        # [32-33] Wrist y positions (normalised)
        # ------------------------------------------------------------------
        if l_wrist is not None:
            features[32] = _norm_y(l_wrist[1], bbox)
        if r_wrist is not None:
            features[33] = _norm_y(r_wrist[1], bbox)

    except Exception as exc:
        # Never crash – return a zero vector and log the issue
        logger.warning("Feature extraction error (returning zeros): %s", exc)
        features = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Sanitise: replace NaN / Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features
