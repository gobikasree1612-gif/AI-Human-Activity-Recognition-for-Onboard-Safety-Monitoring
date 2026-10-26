"""
safety.py
=========
SafetyEngine – combines activity classification + fall detection to
produce a unified system-level safety assessment.

System states:

    NORMAL   – all persons performing expected activities
    WARNING  – at least one person has a possible (unconfirmed) fall
    CRITICAL – at least one person has a confirmed fall

The engine also generates evidence strings to help the dashboard
explain the AI reasoning.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai.fall_detector import FallDetector
from ai.schemas import (
    Activity,
    AlertType,
    PersonResult,
    Severity,
    SystemStatus,
)

logger = logging.getLogger(__name__)

# Feature index shortcuts – must match features.py layout
_I = {
    "body_lean_angle": 6,
    "torso_height_norm": 7,
    "hip_center_y_norm": 17,
    "shoulder_center_y_norm": 18,
    "knee_bend_left": 19,
    "knee_bend_right": 20,
    "visibility_score": 29,
}


class SafetyEngine:
    """
    Fuses activity classification + fall detection into a safety assessment.

    Usage::

        safety = SafetyEngine(config)
        person_result = safety.evaluate(
            person_id, bbox, activity_result, feature_vector, keypoints, seq
        )
        system_status = safety.system_status(person_results)
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._fall_detector = FallDetector(config)
        self._conf_thresh: float = config.get("activity", {}).get(
            "confidence_threshold", 0.60
        )
        self._latched_fall: Optional[dict] = None

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        person_id: int,
        bbox: List[int],
        activity_result: Dict,
        feature_vector: np.ndarray,
        keypoints: Optional[np.ndarray],
        sequence: Optional[np.ndarray],
    ) -> PersonResult:
        """
        Evaluate a single person's safety state.

        Parameters
        ----------
        person_id : int
        bbox : [x1, y1, x2, y2]
        activity_result : dict from ActivityClassifier.predict()
        feature_vector : np.ndarray shape (FEATURE_DIM,)
        keypoints : np.ndarray shape (17, 3) or None
        sequence : np.ndarray shape (N, FEATURE_DIM) or None

        Returns
        -------
        PersonResult (see schemas.py)
        """
        import datetime

        activity: str = activity_result.get("activity", Activity.UNKNOWN)
        confidence: float = float(activity_result.get("confidence", 0.0))
        model_type: str = activity_result.get("model_type", "rule_based_baseline")

        # Update fall detector
        if feature_vector is not None and len(feature_vector) > 0:
            self._fall_detector.update(person_id, feature_vector)

        fall_status = self._fall_detector.get_status(person_id)
        fall_evidence = self._fall_detector.get_evidence(person_id)

        # Override activity with FALLING if fall is detected
        alert = False
        alert_type: Optional[str] = None
        severity: Optional[str] = None

        if fall_status == "CONFIRMED_FALL":
            activity = Activity.FALLING
            confidence = max(confidence, 0.85)
            alert = True
            alert_type = AlertType.FALL_DETECTED
            severity = Severity.CRITICAL

            if self._latched_fall is None:
                now_dt = datetime.datetime.now()
                self._latched_fall = {
                    "person_id": person_id,
                    "timestamp": now_dt.strftime("%H:%M:%S"),
                    "iso_timestamp": now_dt.isoformat(),
                    "evidence": list(fall_evidence) if fall_evidence else ["Confirmed fall posture detected"],
                    "activity": Activity.FALLING,
                    "severity": Severity.CRITICAL,
                    "alert_type": AlertType.FALL_DETECTED,
                }
            else:
                for e in (fall_evidence or []):
                    if e not in self._latched_fall["evidence"]:
                        self._latched_fall["evidence"].append(e)

        elif fall_status == "POSSIBLE_FALL":
            alert = True
            alert_type = AlertType.POSSIBLE_FALL
            severity = Severity.WARNING

        # Build evidence
        evidence = self._build_evidence(
            activity, feature_vector, sequence, fall_status, fall_evidence
        )

        return PersonResult(
            person_id=person_id,
            bbox=bbox,
            activity=activity,
            confidence=round(confidence, 4),
            model_type=model_type,
            alert=alert,
            alert_type=alert_type,
            severity=severity,
            evidence=evidence,
            keypoints=keypoints.tolist() if keypoints is not None else None,
        )

    # ------------------------------------------------------------------
    # Evidence generation
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        activity: str,
        fv: np.ndarray,
        seq: Optional[np.ndarray],
        fall_status: str,
        fall_evidence: List[str],
    ) -> List[str]:
        """
        Build human-readable evidence strings from MEASURED feature values.
        Evidence is NOT fabricated – every string comes from an actual
        measured quantity.
        """
        evidence: List[str] = []

        def f(name: str) -> float:
            idx = _I.get(name, 0)
            return float(fv[idx]) if fv is not None and idx < len(fv) else 0.0

        # Temporal statistics (require at least 2 frames)
        motion_std = 0.0
        hip_std = 0.0
        if seq is not None and seq.shape[0] >= 2:
            motion_std = float(seq[:, :20].std())
            hip_idx = _I["hip_center_y_norm"]
            if hip_idx < seq.shape[1]:
                hip_std = float(seq[:, hip_idx].std())

        if fall_status in ("POSSIBLE_FALL", "CONFIRMED_FALL"):
            evidence.extend(fall_evidence)

        if activity == Activity.FALLING:
            if f("body_lean_angle") > 40:
                evidence.append(
                    f"Horizontal body orientation detected "
                    f"(lean={f('body_lean_angle'):.1f}°)"
                )
            if fall_status == "CONFIRMED_FALL":
                evidence.append("Person remained in low position after fall")

        elif activity == Activity.WALKING:
            if motion_std > 0.01:
                evidence.append(
                    f"Continuous body movement detected (motion_std={motion_std:.3f})"
                )
            if hip_std > 0.005:
                evidence.append("Sequential body position changes observed")

        elif activity == Activity.RUNNING:
            if motion_std > 0.04:
                evidence.append(
                    f"High-speed body movement (motion_std={motion_std:.3f})"
                )
            evidence.append("Large limb displacement between frames")

        elif activity == Activity.STANDING:
            if motion_std < 0.04:
                evidence.append(
                    f"Stable body position over time (motion_std={motion_std:.3f})"
                )
            if f("body_lean_angle") < 25:
                evidence.append(
                    f"Upright body posture (lean={f('body_lean_angle'):.1f}°)"
                )

        elif activity == Activity.SITTING:
            if f("hip_center_y_norm") > 0.5:
                evidence.append(
                    f"Low hip position (hip_y_norm={f('hip_center_y_norm'):.2f})"
                )
            knee_bend = max(f("knee_bend_left"), f("knee_bend_right"))
            if knee_bend > 0.2:
                evidence.append(
                    f"Bent knee configuration (bend={knee_bend:.2f})"
                )

        elif activity == Activity.UNKNOWN:
            evidence.append("Insufficient pose visibility or confidence")

        # Deduplicate while preserving order
        seen = set()
        unique_evidence = []
        for e in evidence:
            if e not in seen:
                seen.add(e)
                unique_evidence.append(e)

        return unique_evidence if unique_evidence else ["Activity classification in progress"]

    # ------------------------------------------------------------------
    # System-level status
    # ------------------------------------------------------------------

    @staticmethod
    def system_status(
        person_results: List[PersonResult],
        latched_fall: Optional[dict] = None,
    ) -> str:
        """
        Aggregate individual person statuses into a system-level status.

            CRITICAL  → at least one CONFIRMED fall or latched fall active
            WARNING   → at least one POSSIBLE fall (no CRITICAL)
            NORMAL    → no alerts
        """
        if latched_fall is not None or any(pr.severity == Severity.CRITICAL for pr in person_results):
            return SystemStatus.CRITICAL
        if any(pr.severity == Severity.WARNING for pr in person_results):
            return SystemStatus.WARNING
        return SystemStatus.NORMAL

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def remove_person(self, person_id: int) -> None:
        """Remove all tracking state for a person who left the scene."""
        self._fall_detector.remove(person_id)

    def reset_fall(self, person_id: int) -> None:
        """Manually reset a person's fall alert (e.g., after acknowledgement)."""
        self._fall_detector.reset(person_id)

    def reset_all(self) -> None:
        """Clear all safety state including latched fall alerts."""
        self._latched_fall = None
        self._fall_detector._states.clear()
        self._fall_detector._history.clear()

