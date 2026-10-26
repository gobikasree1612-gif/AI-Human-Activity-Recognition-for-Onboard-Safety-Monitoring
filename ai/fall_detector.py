"""
fall_detector.py
================
Multi-stage fall detection with temporal confirmation.

States per person:

    NORMAL → POSSIBLE_FALL → CONFIRMED_FALL

A single unusual frame NEVER triggers an alert.
Confirmation requires the person to remain in a fallen posture for
`confirmation_frames` consecutive frames after the initial trigger.

Algorithm:

  Possible fall indicators (ANY threshold breach):
    - Body lean angle > body_angle_threshold (nearly horizontal)
    - Rapid downward centre-of-body displacement
    - Sudden reduction in torso height (collapse)

  Confirmed fall:
    - After a possible-fall event, the person remains in a low/down
      position for `confirmation_frames` frames without standing up.

  Reset:
    - If the person stands up (low body lean + increasing torso height)
      for several frames, the state is reset to NORMAL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _FallState:
    """Per-person fall detection state."""
    status: str = "NORMAL"          # NORMAL | POSSIBLE_FALL | CONFIRMED_FALL
    possible_fall_frame: int = 0    # frame counter when possible fall was flagged
    confirmation_count: int = 0     # frames spent in low position after flag
    pre_fall_hip_y: Optional[float] = None   # hip y just before possible fall
    possible_fall_time: float = field(default_factory=time.time)
    evidence: List[str] = field(default_factory=list)


# Feature vector index map (must match features.py layout)
_I = {
    "body_lean_angle": 6,
    "torso_height_norm": 7,
    "hip_center_y_norm": 17,
    "shoulder_center_y_norm": 18,
    "visibility_score": 29,
}


class FallDetector:
    """
    Per-person, multi-stage fall detection with temporal confirmation.

    Usage::

        fd = FallDetector(config)

        # Each frame, after feature extraction:
        fd.update(person_id, feature_vector)

        if fd.is_confirmed_fall(person_id):
            # raise alert
        elif fd.is_possible_fall(person_id):
            # show warning
    """

    def __init__(self, config: dict):
        fall_cfg = config.get("fall", {})
        self._confirm_frames: int  = fall_cfg.get("confirmation_frames", 15)
        self._vel_thresh: float    = fall_cfg.get("downward_velocity_threshold", 0.04)
        self._angle_thresh: float  = fall_cfg.get("body_angle_threshold", 55)
        self._low_h_frac: float    = fall_cfg.get("low_height_fraction", 0.55)
        self._timeout_s: float     = fall_cfg.get("possible_fall_timeout_seconds", 5.0)

        self._states: Dict[int, _FallState] = {}
        # Store the most recent 3 feature vectors per person for velocity calc
        self._history: Dict[int, list] = {}

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, person_id: int, feature_vector: np.ndarray) -> None:
        """
        Process a new feature vector for a person.
        Updates the internal fall state machine.
        """
        fv = np.asarray(feature_vector, dtype=np.float32)

        if person_id not in self._states:
            self._states[person_id] = _FallState()

        if person_id not in self._history:
            self._history[person_id] = []

        hist = self._history[person_id]
        hist.append(fv)
        if len(hist) > 5:
            hist.pop(0)

        state = self._states[person_id]

        # Compute indicators
        indicators = self._compute_indicators(fv, hist)

        if state.status == "NORMAL":
            self._check_possible_fall(person_id, fv, indicators)

        elif state.status == "POSSIBLE_FALL":
            # Timeout check – if too much time has passed, reset
            if time.time() - state.possible_fall_time > self._timeout_s:
                logger.debug(
                    "Person %d: possible-fall timed out → resetting", person_id
                )
                self._reset_state(person_id)
                return

            # Check if person is still in a fallen posture
            if self._is_fallen_posture(fv, indicators, state):
                state.confirmation_count += 1
                if state.confirmation_count >= self._confirm_frames:
                    state.status = "CONFIRMED_FALL"
                    state.evidence.append(
                        f"Person remained down for {state.confirmation_count} frames"
                    )
                    logger.warning(
                        "Person %d: FALL CONFIRMED after %d confirmation frames",
                        person_id,
                        state.confirmation_count,
                    )
            else:
                # Person stood up – reset
                logger.debug(
                    "Person %d: recovered from possible fall → resetting", person_id
                )
                self._reset_state(person_id)

        # CONFIRMED_FALL stays until explicit reset() call or recovery

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def _compute_indicators(
        self, fv: np.ndarray, hist: list
    ) -> dict:
        """Compute fall-relevant indicators from current + recent frames."""
        def f(name: str) -> float:
            idx = _I.get(name, 0)
            return float(fv[idx]) if idx < len(fv) else 0.0

        # Downward velocity of hip centre (positive = moving down in image)
        downward_vel = 0.0
        if len(hist) >= 2:
            prev = hist[-2]
            curr_hip_y = f("hip_center_y_norm")
            prev_hip_y = float(prev[_I["hip_center_y_norm"]]) if _I["hip_center_y_norm"] < len(prev) else curr_hip_y
            downward_vel = curr_hip_y - prev_hip_y   # positive = downward

        return {
            "body_lean_angle": f("body_lean_angle"),
            "torso_height_norm": f("torso_height_norm"),
            "hip_center_y_norm": f("hip_center_y_norm"),
            "shoulder_center_y_norm": f("shoulder_center_y_norm"),
            "visibility_score": f("visibility_score"),
            "downward_vel": downward_vel,
        }

    def _check_possible_fall(
        self,
        person_id: int,
        fv: np.ndarray,
        indicators: dict,
    ) -> None:
        """Evaluate whether this frame is a possible-fall trigger."""
        state = self._states[person_id]
        evidence: List[str] = []

        triggered = False

        # Indicator 1: body nearly horizontal
        if indicators["body_lean_angle"] > self._angle_thresh:
            triggered = True
            evidence.append(
                f"Significant body angle change "
                f"({indicators['body_lean_angle']:.1f}° > {self._angle_thresh}°)"
            )

        # Indicator 2: rapid downward movement
        if indicators["downward_vel"] > self._vel_thresh:
            triggered = True
            evidence.append(
                f"Rapid downward movement "
                f"(Δhip_y={indicators['downward_vel']:.3f})"
            )

        # Indicator 3: sudden torso height collapse (< low_h_frac of baseline)
        # We use a generous threshold because the baseline changes
        if indicators["torso_height_norm"] < 0.10 and indicators["visibility_score"] > 0.3:
            triggered = True
            evidence.append(
                f"Sudden reduction in torso height "
                f"({indicators['torso_height_norm']:.3f})"
            )

        if triggered:
            state.status = "POSSIBLE_FALL"
            state.possible_fall_time = time.time()
            state.confirmation_count = 0
            state.pre_fall_hip_y = indicators["hip_center_y_norm"]
            state.evidence = evidence
            logger.info(
                "Person %d: POSSIBLE FALL detected. Evidence: %s",
                person_id,
                evidence,
            )

    def _is_fallen_posture(
        self, fv: np.ndarray, indicators: dict, state: _FallState
    ) -> bool:
        """
        Returns True if the current pose is consistent with a fallen person:
          - body still lean angle is large, OR
          - hip position is significantly lower than pre-fall baseline
        """
        if indicators["body_lean_angle"] > self._angle_thresh * 0.8:
            return True

        if state.pre_fall_hip_y is not None:
            hip_diff = indicators["hip_center_y_norm"] - state.pre_fall_hip_y
            # Hip moved down significantly and stayed there
            if hip_diff > 0.10:
                return True

        return False

    def _reset_state(self, person_id: int) -> None:
        """Reset a person's fall state to NORMAL."""
        self._states[person_id] = _FallState()

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def is_possible_fall(self, person_id: int) -> bool:
        state = self._states.get(person_id)
        return state is not None and state.status == "POSSIBLE_FALL"

    def is_confirmed_fall(self, person_id: int) -> bool:
        state = self._states.get(person_id)
        return state is not None and state.status == "CONFIRMED_FALL"

    def get_status(self, person_id: int) -> str:
        """Return 'NORMAL' | 'POSSIBLE_FALL' | 'CONFIRMED_FALL'."""
        state = self._states.get(person_id)
        return state.status if state is not None else "NORMAL"

    def get_evidence(self, person_id: int) -> List[str]:
        state = self._states.get(person_id)
        return list(state.evidence) if state is not None else []

    def reset(self, person_id: int) -> None:
        """Manually reset fall state (e.g., after an alert is acknowledged)."""
        self._reset_state(person_id)
        if person_id in self._history:
            del self._history[person_id]

    def remove(self, person_id: int) -> None:
        """Remove all state for a person who is no longer tracked."""
        self._states.pop(person_id, None)
        self._history.pop(person_id, None)
