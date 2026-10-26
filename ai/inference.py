"""
inference.py
============
HAREngine – the single entry point for the dashboard developer.

Dashboard integration (ONLY interface the dashboard team needs)::

    from ai.inference import HAREngine

    engine = HAREngine()                    # loads config automatically
    # OR
    engine = HAREngine(config_path="config.yaml")

    # For each camera frame (OpenCV BGR numpy array):
    result, annotated_frame = engine.process_frame(frame)

    # result is a plain dict following the standard schema (see schemas.py)
    # annotated_frame is the BGR frame with overlays drawn

The dashboard team does NOT need to know anything about YOLO, keypoints,
temporal buffers, or fall algorithms.

Performance metrics are measured from REAL timing – nothing is fabricated.
"""

from __future__ import annotations

import datetime
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from ai.classifier import ActivityClassifier
from ai.detector import HumanPoseDetector
from ai.features import extract_pose_features
from ai.safety import SafetyEngine
from ai.schemas import (
    Activity,
    FrameResult,
    PerformanceMetrics,
    PersonResult,
    SystemStatus,
)
from ai.temporal import TemporalBuffer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config.yaml",
)


def _load_config(path: str) -> dict:
    """Load YAML config. Returns empty dict on failure (safe defaults used)."""
    try:
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh)
            logger.info("Config loaded from %s", path)
            return cfg or {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s – using defaults.", path)
        return {}
    except Exception as exc:
        logger.error("Failed to load config (%s) – using defaults.", exc)
        return {}


# ---------------------------------------------------------------------------
# Performance monitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """Measures real frame processing time, FPS, and latency."""

    _ALPHA = 0.1   # EMA smoothing factor

    def __init__(self, warmup_frames: int = 5):
        self._warmup = warmup_frames
        self._frame_count = 0
        self._last_time = time.perf_counter()
        self._fps_ema: float = 0.0
        self._latency_ema: float = 0.0

    def start_frame(self) -> float:
        return time.perf_counter()

    def end_frame(self, t_start: float) -> Tuple[float, float]:
        """Call after frame processing. Returns (fps, latency_ms)."""
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        if self._frame_count == 0:
            elapsed = t_end - t_start
        else:
            elapsed = t_end - self._last_time
        self._last_time = t_end
        fps = 1.0 / max(elapsed, 1e-6)

        self._frame_count += 1
        if self._frame_count <= self._warmup:
            self._fps_ema = fps
            self._latency_ema = latency_ms
        else:
            self._fps_ema = (
                self._ALPHA * fps + (1 - self._ALPHA) * self._fps_ema
            )
            self._latency_ema = (
                self._ALPHA * latency_ms + (1 - self._ALPHA) * self._latency_ema
            )

        return round(self._fps_ema, 2), round(self._latency_ema, 2)


# ---------------------------------------------------------------------------
# HAREngine – main engine
# ---------------------------------------------------------------------------

class HAREngine:
    """
    End-to-end Human Activity Recognition engine.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. Defaults to the config.yaml in the project root.
    config : dict, optional
        Pre-loaded config dict. Overrides config_path if supplied.
    """

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG,
        config: Optional[dict] = None,
    ):
        self._cfg = config if config is not None else _load_config(config_path)

        act_cfg = self._cfg.get("activity", {})
        seq_len: int = act_cfg.get("sequence_length", 20)
        self._pose_conf_thresh: float = act_cfg.get("pose_confidence_threshold", 0.40)
        self._act_conf_thresh: float  = act_cfg.get("confidence_threshold", 0.60)
        self._min_feat_conf: float    = act_cfg.get("pose_confidence_threshold", 0.30)

        self._detector   = HumanPoseDetector(self._cfg)
        self._buffer     = TemporalBuffer(sequence_length=seq_len)
        self._classifier = ActivityClassifier(self._cfg)
        self._safety     = SafetyEngine(self._cfg)
        self._perf       = PerformanceMonitor(
            warmup_frames=self._cfg.get("performance", {}).get("warmup_frames", 5)
        )

        # Track which person IDs were seen in last frame for buffer housekeeping
        self._prev_person_ids: set = set()

        logger.info(
            "HAREngine ready. Classifier mode: %s", self._classifier.model_type
        )

    # ------------------------------------------------------------------
    # PRIMARY INTERFACE (dashboard calls this)
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[dict, np.ndarray]:
        """
        Process a single BGR frame through the full HAR pipeline.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV BGR frame (from cv2.VideoCapture or a still image).

        Returns
        -------
        result : dict
            Standardised output conforming to the documented schema.
        annotated_frame : np.ndarray
            Copy of the frame with bounding boxes, activity labels,
            keypoints, and FPS overlay drawn.
        """
        t_start = self._perf.start_frame()

        annotated = frame.copy() if frame is not None else np.zeros((480, 640, 3), np.uint8)

        if frame is None or frame.size == 0:
            logger.warning("process_frame received an empty frame.")
            fps, lat = self._perf.end_frame(t_start)
            return self._empty_result(fps, lat), annotated

        # 1. Detect people + pose
        try:
            detected_people = self._detector.process(frame)
        except Exception as exc:
            logger.error("Detection failed: %s", exc)
            detected_people = []

        current_ids = {p.person_id for p in detected_people}

        # 2. Remove stale buffers for people who left the scene
        self._buffer.remove_stale(current_ids)
        stale_ids = self._prev_person_ids - current_ids
        for pid in stale_ids:
            self._safety.remove_person(pid)
        self._prev_person_ids = current_ids

        # 3. Process each person
        person_results: List[PersonResult] = []
        for person in detected_people:
            pr = self._process_person(person)
            if pr is not None:
                person_results.append(pr)

        # 4. System status
        sys_status = SafetyEngine.system_status(
            person_results, latched_fall=self._safety._latched_fall
        )

        # 5. Performance metrics
        fps, lat = self._perf.end_frame(t_start)

        # 6. Annotate frame
        self._draw_annotations(annotated, person_results, fps, sys_status)

        # 7. Build result dict
        metrics = PerformanceMetrics(
            people_count=len(person_results),
            fps=fps,
            latency_ms=lat,
        )
        frame_result = FrameResult(
            timestamp=datetime.datetime.now().isoformat(),
            system_status=sys_status,
            people=person_results,
            metrics=metrics,
        )
        res_dict = frame_result.to_dict()
        res_dict["latched_fall"] = self._safety._latched_fall
        return res_dict, annotated

    def reset_safety(self) -> None:
        """Reset latched fall alerts and internal safety state."""
        if hasattr(self, "_safety") and self._safety is not None:
            self._safety.reset_all()


    # ------------------------------------------------------------------
    # Per-person processing
    # ------------------------------------------------------------------

    def _process_person(self, person) -> Optional[PersonResult]:
        """Run feature extraction, temporal buffering, classification
        and safety evaluation for one detected person."""

        pid    = person.person_id
        bbox   = person.bbox
        kps    = person.keypoints    # (17, 3)
        det_cf = person.detection_confidence

        # Visibility gate
        mean_kp_conf = float(kps[:, 2].mean()) if kps is not None else 0.0
        if mean_kp_conf < self._pose_conf_thresh and det_cf < 0.5:
            # Very low quality detection – emit UNKNOWN without buffering
            return PersonResult(
                person_id=pid,
                bbox=bbox,
                activity=Activity.UNKNOWN,
                confidence=0.0,
                model_type=self._classifier.model_type,
                alert=False,
                alert_type=None,
                severity=None,
                evidence=["Low pose visibility – detection below threshold"],
                keypoints=kps.tolist() if kps is not None else None,
            )

        # Feature extraction
        try:
            fv = extract_pose_features(kps, bbox, min_conf=self._min_feat_conf)
        except Exception as exc:
            logger.error("Feature extraction failed for person %d: %s", pid, exc)
            fv = np.zeros(34, dtype=np.float32)

        # Temporal buffer update
        self._buffer.add(pid, fv)

        # Get sequence (full window if ready, otherwise partial)
        sequence = self._buffer.get_sequence(pid)
        if sequence is None or sequence.shape[0] == 0:
            sequence = fv.reshape(1, -1)

        # Activity classification
        try:
            act_result = self._classifier.predict(sequence)
        except Exception as exc:
            logger.error("Classifier failed for person %d: %s", pid, exc)
            act_result = {
                "activity": Activity.UNKNOWN,
                "confidence": 0.0,
                "model_type": self._classifier.model_type,
            }

        # Safety evaluation (includes fall detection)
        try:
            person_result = self._safety.evaluate(
                person_id=pid,
                bbox=bbox,
                activity_result=act_result,
                feature_vector=fv,
                keypoints=kps,
                sequence=sequence,
            )
        except Exception as exc:
            logger.error("Safety evaluation failed for person %d: %s", pid, exc)
            person_result = PersonResult(
                person_id=pid,
                bbox=bbox,
                activity=act_result.get("activity", Activity.UNKNOWN),
                confidence=float(act_result.get("confidence", 0.0)),
                model_type=act_result.get("model_type", "rule_based_baseline"),
                alert=False,
                alert_type=None,
                severity=None,
                evidence=["Safety evaluation error"],
                keypoints=kps.tolist() if kps is not None else None,
            )

        return person_result

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def _draw_annotations(
        self,
        frame: np.ndarray,
        person_results: List[PersonResult],
        fps: float,
        system_status: str,
    ) -> None:
        """Draw bounding boxes, activity labels, and overlays on the frame."""
        h, w = frame.shape[:2]

        # Status colours
        STATUS_COLORS = {
            SystemStatus.NORMAL:   (0, 200, 80),
            SystemStatus.WARNING:  (0, 165, 255),
            SystemStatus.CRITICAL: (0, 0, 220),
        }
        status_color = STATUS_COLORS.get(system_status, (200, 200, 200))

        # --- HUD: top-left ---
        cv2.rectangle(frame, (0, 0), (280, 55), (20, 20, 20), -1)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 180), 2,
        )
        cv2.putText(
            frame,
            f"People: {len(person_results)}",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 180), 2,
        )

        # --- System status banner ---
        cv2.rectangle(frame, (w - 220, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(
            frame,
            system_status,
            (w - 210, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75, status_color, 2,
        )

        for pr in person_results:
            x1, y1, x2, y2 = pr.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # BBox colour based on alert
            if pr.severity == "CRITICAL":
                box_color = (0, 0, 220)
            elif pr.severity == "WARNING":
                box_color = (0, 140, 255)
            else:
                box_color = (0, 200, 80)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # --- Label background ---
            label = f"#{pr.person_id} {pr.activity} {pr.confidence:.0%}"
            if pr.alert_type:
                label += f" [{pr.alert_type}]"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(y1 - 4, lh + 4)
            cv2.rectangle(
                frame,
                (x1, ly - lh - 4),
                (x1 + lw + 4, ly + 2),
                (20, 20, 20), -1,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 2, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, box_color, 1,
            )

            # --- Model type tag ---
            mt_tag = "ML" if pr.model_type == "trained_ml" else "RB"
            cv2.putText(
                frame,
                mt_tag,
                (x1 + 2, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 50), 1,
            )

            # --- Draw keypoints ---
            if pr.keypoints is not None:
                self._draw_keypoints(frame, pr.keypoints)

    def _draw_keypoints(self, frame: np.ndarray, keypoints) -> None:
        """Draw COCO-17 keypoints and basic skeleton on the frame."""
        KP_PAIRS = [
            (5, 6),    # shoulders
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (11, 12),  # hips
            (5, 11), (6, 12), # torso
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
        ]

        kps = np.array(keypoints)
        for a, b in KP_PAIRS:
            if a < len(kps) and b < len(kps):
                xa, ya, ca = kps[a]
                xb, yb, cb = kps[b]
                if ca > 0.30 and cb > 0.30:
                    cv2.line(
                        frame,
                        (int(xa), int(ya)),
                        (int(xb), int(yb)),
                        (100, 200, 255), 1,
                    )

        for kp in kps:
            x, y, c = kp
            if c > 0.30:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 200, 0), -1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(self, fps: float, lat: float) -> dict:
        return FrameResult(
            timestamp=datetime.datetime.now().isoformat(),
            system_status=SystemStatus.NORMAL,
            people=[],
            metrics=PerformanceMetrics(people_count=0, fps=fps, latency_ms=lat),
        ).to_dict()

    @property
    def classifier_mode(self) -> str:
        """Return 'trained_ml' or 'rule_based_baseline'."""
        return self._classifier.model_type

    # ------------------------------------------------------------------
    # Diagnostics (developer / debug mode)
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        """
        Return a structured diagnostics snapshot describing every pipeline
        component: what model is loaded, thresholds, feature dimensions,
        and which components are real vs. placeholder.

        Intended for developer/debug mode in the dashboard.
        NOT called during normal frame processing – zero overhead in production.

        Returns
        -------
        dict with sections:
            detector, classifier, fall_detector, pipeline
        """
        from ai.features import FEATURE_DIM
        from ai.schemas import Activity

        # Detector info
        model_obj = self._detector._model
        try:
            model_name = str(getattr(model_obj, "model_name", None)
                             or getattr(model_obj, "pt_path", None)
                             or self._cfg.get("model", {}).get("pose_model", "yolo11n-pose.pt"))
        except Exception:
            model_name = self._cfg.get("model", {}).get("pose_model", "yolo11n-pose.pt")

        detector_info = {
            "model_name":                    model_name,
            "model_type":                    "YOLO Pose (Ultralytics)",
            "detection_confidence_threshold": self._detector._det_conf,
            "iou_threshold":                 self._detector._iou_thresh,
            "tracker_type":                  "IoU-based (SimpleTracker)",
            "max_missing_frames":            self._detector._tracker._max_missing,
            "multi_person_supported":        True,
            "pose_keypoints":                17,
            "keypoint_format":               "COCO-17 (x, y, confidence)",
            "inference_device":              self._detector._device or "auto (CPU/GPU)",
        }

        # Classifier info
        clf = self._classifier
        load_info = clf.load_info  # non-empty only if trained model loaded
        classifier_info = {
            "mode":                 clf.model_type,
            "is_trained":           clf.is_trained,
            "model_path":           clf._model_path,
            "scaler_path":          clf._scaler_path,
            "activities_supported": Activity.ALL,
            "feature_dim":          FEATURE_DIM,
            "sequence_length":      self._cfg.get("activity", {}).get("sequence_length", 20),
            "confidence_threshold": self._act_conf_thresh,
            # Load-integrity fields
            "integrity_check_passed": clf.is_trained,
            "load_error":             clf.load_error,
            # From saved metadata (if trained model loaded)
            "model_version":          load_info.get("model_type", "N/A"),
            "trained_at":             load_info.get("trained_at", "N/A"),
            "training_samples":       load_info.get("train_samples", "N/A"),
            "test_accuracy":          load_info.get("test_accuracy", "N/A"),
            "split_strategy":         load_info.get("split_strategy", "N/A"),
            "synthetic_data":         load_info.get("synthetic_data", "N/A"),
            "note": (
                "Confidence values are calibrated probabilities from sklearn model."
                if clf.is_trained
                else (
                    f"Model REJECTED: {clf.load_error}"
                    if clf.load_error else
                    "No trained model found. Using heuristic rule-based classifier."
                )
            ),
        }


        # Fall detector info
        fd = self._safety._fall_detector
        fall_detector_info = {
            "body_angle_threshold":        fd._angle_thresh,
            "downward_velocity_threshold": fd._vel_thresh,
            "confirmation_frames":         fd._confirm_frames,
            "possible_fall_timeout_s":     fd._timeout_s,
            "low_height_fraction":         fd._low_h_frac,
            "states":                      "NORMAL → POSSIBLE_FALL → CONFIRMED_FALL",
            "recovery_supported":          True,
            "multi_person_supported":      True,
        }

        # Pipeline status – what is real vs. placeholder
        pipeline_status = {
            "detector_real":       True,   # Real YOLO Pose via Ultralytics
            "pose_real":           True,   # Real 17-keypoint COCO pose from YOLO
            "features_real":       True,   # Computed from real keypoints
            "temporal_real":       True,   # Sliding ring buffer of real features
            "classifier_real":     True,   # ML (if trained) or rule-based (if not)
            "fall_detection_real": True,   # Rule-based physics thresholds
            "safety_engine_real":  True,   # Aggregates real fall detector output
        }

        # Performance metrics (from PerformanceMonitor)
        perf = self._perf
        performance_info = {
            "current_fps_ema":     round(perf._fps_ema, 2),
            "current_latency_ema": round(perf._latency_ema, 2),
            "frames_processed":    perf._frame_count,
            "warmup_frames":       perf._warmup,
            "note": "All metrics are from real timing (time.perf_counter). Nothing is fabricated.",
        }

        return {
            "detector":      detector_info,
            "classifier":    classifier_info,
            "fall_detector": fall_detector_info,
            "pipeline":      pipeline_status,
            "performance":   performance_info,
        }

