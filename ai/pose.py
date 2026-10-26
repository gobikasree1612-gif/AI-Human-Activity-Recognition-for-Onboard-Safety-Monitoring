"""
pose.py
=======
Thin wrapper kept for future pose-model abstraction.

Currently HumanPoseDetector (in detector.py) already handles pose
estimation internally via YOLO Pose. This module re-exports it and
provides COCO keypoint index constants for convenient use across the
rest of the AI package.
"""

from ai.detector import COCO_KEYPOINTS, DetectedPerson, HumanPoseDetector

__all__ = ["HumanPoseDetector", "DetectedPerson", "COCO_KEYPOINTS"]
