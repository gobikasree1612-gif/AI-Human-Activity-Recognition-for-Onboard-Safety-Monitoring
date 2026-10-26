"""
detector.py
===========
Human detection and pose estimation using YOLO Pose (Ultralytics).

Supports:
- Webcam frames
- MP4 / video frames
- Single still images

Assigns stable person IDs using simple IoU-based tracking when
YOLO's built-in tracker is not available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-keypoint indices (for reference by other modules)
COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass
class DetectedPerson:
    """Pose detection result for a single person."""

    person_id: int
    bbox: List[int]               # [x1, y1, x2, y2]
    keypoints: np.ndarray         # shape (17, 3) – (x, y, confidence)
    detection_confidence: float


class _SimpleTracker:
    """
    Lightweight IoU-based tracker to maintain stable person IDs across frames
    without requiring a GPU tracker.

    Strategy:
      - Each known person is represented by their last bounding box.
      - New detections are matched to existing persons greedily by max IoU.
      - Unmatched detections get a new ID.
      - Persons missing for `max_missing_frames` are evicted.
    """

    def __init__(self, max_missing_frames: int = 60, iou_threshold: float = 0.30):
        self._next_id = 1
        self._tracks: Dict[int, dict] = {}   # id → {bbox, missing}
        self._max_missing = max_missing_frames
        self._iou_thresh = iou_threshold

    # ------------------------------------------------------------------
    def update(self, bboxes: List[List[int]]) -> List[int]:
        """
        Given a list of bboxes for the current frame, return a list of
        person IDs in the same order.
        """
        # Increment missing counter for all existing tracks
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["missing"] += 1

        assigned_ids: List[int] = []
        used_track_ids: set = set()

        for bbox in bboxes:
            best_id = None
            best_iou = self._iou_thresh
            for tid, track in self._tracks.items():
                if tid in used_track_ids:
                    continue
                iou = self._iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                self._tracks[best_id]["bbox"] = bbox
                self._tracks[best_id]["missing"] = 0
                used_track_ids.add(best_id)
                assigned_ids.append(best_id)
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = {"bbox": bbox, "missing": 0}
                used_track_ids.add(new_id)
                assigned_ids.append(new_id)

        # Remove stale tracks
        for tid in list(self._tracks.keys()):
            if self._tracks[tid]["missing"] > self._max_missing:
                del self._tracks[tid]

        return assigned_ids

    @staticmethod
    def _iou(a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)


class HumanPoseDetector:
    """
    Detects humans and estimates pose keypoints using YOLO Pose.

    Usage::

        detector = HumanPoseDetector(config)
        people: List[DetectedPerson] = detector.process(frame)
    """

    def __init__(self, config: dict):
        self._cfg = config
        model_path: str = config.get("model", {}).get("pose_model", "yolo11n-pose.pt")
        device: str = config.get("model", {}).get("device", "")
        det_conf: float = config.get("model", {}).get("detection_confidence", 0.40)
        iou_thresh: float = config.get("model", {}).get("iou_threshold", 0.45)
        max_missing: int = config.get("tracking", {}).get("max_missing_frames", 60)

        self._det_conf = det_conf
        self._iou_thresh = iou_thresh
        self._device = device if device else None
        self._model = self._load_model(model_path)
        self._tracker = _SimpleTracker(
            max_missing_frames=max_missing,
            iou_threshold=0.30,
        )
        logger.info("HumanPoseDetector initialised with model: %s", model_path)

    # ------------------------------------------------------------------
    def _load_model(self, model_path: str):
        """Load YOLO Pose model; auto-download if not present."""
        try:
            from ultralytics import YOLO
            kwargs = {}
            if self._device:
                kwargs["device"] = self._device
            model = YOLO(model_path)
            logger.info("YOLO Pose model loaded: %s", model_path)
            return model
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO Pose model '{model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> List[DetectedPerson]:
        """
        Run detection + pose estimation on a single BGR frame.

        Returns a list of DetectedPerson objects (may be empty).
        Never raises – errors are logged and an empty list is returned.
        """
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame – skipping detection.")
            return []

        try:
            device_arg = self._device if self._device else None
            results = self._model.predict(
                source=frame,
                conf=self._det_conf,
                iou=self._iou_thresh,
                device=device_arg,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference failed: %s", exc)
            return []

        if not results or len(results) == 0:
            return []

        result = results[0]
        people = self._parse_result(result)
        return people

    # ------------------------------------------------------------------
    def _parse_result(self, result) -> List[DetectedPerson]:
        """Parse a single YOLO result into DetectedPerson objects."""
        people: List[DetectedPerson] = []

        boxes = result.boxes
        keypoints_data = result.keypoints

        if boxes is None or len(boxes) == 0:
            return []

        # Collect raw bboxes and confs
        raw_bboxes: List[List[int]] = []
        raw_confs: List[float] = []

        for box in boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                raw_bboxes.append(xyxy)
                raw_confs.append(conf)
            except Exception as exc:
                logger.debug("Failed to parse box: %s", exc)
                continue

        if not raw_bboxes:
            return []

        # Assign / update track IDs
        person_ids = self._tracker.update(raw_bboxes)

        for idx, (bbox, conf, pid) in enumerate(
            zip(raw_bboxes, raw_confs, person_ids)
        ):
            kps = self._extract_keypoints(keypoints_data, idx)
            people.append(
                DetectedPerson(
                    person_id=pid,
                    bbox=bbox,
                    keypoints=kps,
                    detection_confidence=conf,
                )
            )

        return people

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_keypoints(keypoints_data, idx: int) -> np.ndarray:
        """
        Extract (17, 3) keypoints array for person at index `idx`.
        Returns zeros array on failure.
        """
        empty = np.zeros((17, 3), dtype=np.float32)
        try:
            if keypoints_data is None:
                return empty
            kp = keypoints_data[idx]
            xy = kp.xy[0].cpu().numpy()    # (17, 2)
            conf = kp.conf[0].cpu().numpy()  # (17,)
            out = np.zeros((17, 3), dtype=np.float32)
            out[:, :2] = xy
            out[:, 2] = conf
            return out
        except Exception as exc:
            logger.debug("Keypoint extraction failed for person %d: %s", idx, exc)
            return empty
