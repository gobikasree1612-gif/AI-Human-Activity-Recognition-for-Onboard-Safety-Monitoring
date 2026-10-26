"""
schemas.py
==========
Dataclass definitions for the standardised output contract.

The dashboard developer only needs to import these to understand
the data structures – they do NOT need to touch any other AI module.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional


# ---------------------------------------------------------------------------
# Per-person result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PersonResult:
    """All information about a single tracked person in one frame."""

    person_id: int
    """Stable tracking ID assigned by the detection/tracking pipeline."""

    bbox: List[int]
    """Bounding box as [x1, y1, x2, y2] in pixel coordinates."""

    activity: str
    """Recognised activity label.
    One of: STANDING | SITTING | WALKING | RUNNING | FALLING | UNKNOWN
    """

    confidence: float
    """Classifier confidence in [0, 1].
    For rule-based fallback this is a heuristic score, not a probability.
    """

    model_type: str
    """'trained_ml'  – result comes from a trained sklearn/XGBoost model.
       'rule_based_baseline' – fallback heuristic rules (no trained model).
    """

    alert: bool
    """True when a safety alert is active for this person."""

    alert_type: Optional[str]
    """'FALL_DETECTED' | 'POSSIBLE_FALL' | None"""

    severity: Optional[str]
    """'CRITICAL' | 'WARNING' | None"""

    evidence: List[str]
    """Human-readable feature observations that explain the prediction.
    Generated from measured signal values, NOT fabricated text.
    """

    keypoints: Optional[List[List[float]]] = None
    """Raw YOLO pose keypoints [[x, y, conf], ...] – 17 COCO keypoints.
    Included for downstream visualisation; may be None if pose failed.
    """

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PerformanceMetrics:
    """Real-time performance measurements for the current frame."""

    people_count: int
    fps: float
    latency_ms: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Top-level frame result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameResult:
    """
    Standardised result for a single processed frame.

    Dashboard integration example::

        result, annotated_frame = engine.process_frame(frame)
        # result is a dict (from FrameResult.to_dict())
        print(result["system_status"])
        for person in result["people"]:
            print(person["activity"], person["confidence"])
    """

    timestamp: str
    """ISO-8601 timestamp of this frame."""

    system_status: str
    """Overall system safety status.
    'NORMAL' | 'WARNING' | 'CRITICAL'
    """

    people: List[PersonResult]
    """One entry per detected person."""

    metrics: PerformanceMetrics
    """FPS, latency, people count."""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Ensure nested lists are plain Python (not numpy arrays)
        return d


# ---------------------------------------------------------------------------
# Safety state enum (used internally)
# ---------------------------------------------------------------------------

class SystemStatus:
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType:
    FALL_DETECTED = "FALL_DETECTED"
    POSSIBLE_FALL = "POSSIBLE_FALL"


class Severity:
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class Activity:
    STANDING = "STANDING"
    SITTING = "SITTING"
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    FALLING = "FALLING"
    UNKNOWN = "UNKNOWN"

    ALL = [STANDING, SITTING, WALKING, RUNNING, FALLING]
