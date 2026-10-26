"""
dashboard/history.py
====================
In-memory session history and statistics tracker.

Stores per-frame results for the current session only (no database).
Provides aggregated statistics for the analytics panel.
"""

from __future__ import annotations

import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class HistoryRecord:
    """One record per person per frame."""
    timestamp: str
    person_id: int
    activity: str
    confidence: float
    alert: bool
    alert_type: Optional[str]
    severity: Optional[str]
    system_status: str
    fps: float
    latency_ms: float


class SessionHistory:
    """
    Records and aggregates HAR results for the active monitoring session.

    All data is in-memory. Reset by calling reset().
    """

    MAX_RECORDS = 5000   # cap to avoid memory bloat in long sessions

    def __init__(self):
        self._records: List[HistoryRecord] = []
        self._fps_samples: List[float] = []
        self._latency_samples: List[float] = []
        self._alert_count: int = 0
        self._total_frames: int = 0
        self._people_seen: set = set()
        self._activity_counts: Dict[str, int] = defaultdict(int)
        self.start_time: datetime.datetime = datetime.datetime.now()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(self, result: dict) -> None:
        """
        Record a single frame result from engine.process_frame().

        Parameters
        ----------
        result : dict
            The plain-dict output of FrameResult.to_dict().
        """
        self._total_frames += 1

        metrics = result.get("metrics", {})
        fps = float(metrics.get("fps", 0.0))
        latency = float(metrics.get("latency_ms", 0.0))
        system_status = result.get("system_status", "NORMAL")
        timestamp = result.get("timestamp", datetime.datetime.now().isoformat())

        if fps > 0:
            self._fps_samples.append(fps)
        if latency > 0:
            self._latency_samples.append(latency)

        people = result.get("people", [])
        for person in people:
            pid = person.get("person_id", 0)
            activity = person.get("activity", "UNKNOWN")
            confidence = float(person.get("confidence", 0.0))
            alert = bool(person.get("alert", False))
            alert_type = person.get("alert_type")
            severity = person.get("severity")

            self._people_seen.add(pid)
            self._activity_counts[activity] += 1

            if alert:
                self._alert_count += 1

            if len(self._records) < self.MAX_RECORDS:
                self._records.append(
                    HistoryRecord(
                        timestamp=timestamp,
                        person_id=pid,
                        activity=activity,
                        confidence=confidence,
                        alert=alert,
                        alert_type=alert_type,
                        severity=severity,
                        system_status=system_status,
                        fps=fps,
                        latency_ms=latency,
                    )
                )

    # ------------------------------------------------------------------
    # Read – statistics
    # ------------------------------------------------------------------

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def alert_count(self) -> int:
        return self._alert_count

    @property
    def people_observed(self) -> int:
        return len(self._people_seen)

    @property
    def avg_fps(self) -> float:
        if not self._fps_samples:
            return 0.0
        return round(sum(self._fps_samples) / len(self._fps_samples), 2)

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_samples:
            return 0.0
        return round(sum(self._latency_samples) / len(self._latency_samples), 2)

    @property
    def activity_distribution(self) -> Dict[str, int]:
        return dict(self._activity_counts)

    @property
    def session_duration_s(self) -> float:
        return (datetime.datetime.now() - self.start_time).total_seconds()

    def get_records(self) -> List[HistoryRecord]:
        """Return all records (most-recent first)."""
        return list(reversed(self._records))

    def get_recent_records(self, n: int = 50) -> List[HistoryRecord]:
        """Return the most recent n records."""
        return list(reversed(self._records[-n:]))

    def to_dataframe(self, n: int = 200) -> pd.DataFrame:
        """
        Return the last n records as a Pandas DataFrame for display.
        Returns an empty DataFrame if there is no data.
        """
        records = self._records[-n:]
        if not records:
            return pd.DataFrame(
                columns=[
                    "Time", "Person", "Activity", "Confidence",
                    "Alert", "Status", "FPS", "Latency (ms)",
                ]
            )
        rows = []
        for r in records:
            try:
                ts = datetime.datetime.fromisoformat(r.timestamp)
                time_str = ts.strftime("%H:%M:%S")
            except Exception:
                time_str = r.timestamp[:8]

            rows.append(
                {
                    "Time": time_str,
                    "Person": f"#{r.person_id}",
                    "Activity": r.activity,
                    "Confidence": f"{r.confidence:.0%}",
                    "Alert": "⚠" if r.alert else "",
                    "Status": r.system_status,
                    "FPS": f"{r.fps:.1f}",
                    "Latency (ms)": f"{r.latency_ms:.1f}",
                }
            )
        df = pd.DataFrame(rows)
        return df.iloc[::-1].reset_index(drop=True)   # most-recent first

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all session data."""
        self._records.clear()
        self._fps_samples.clear()
        self._latency_samples.clear()
        self._alert_count = 0
        self._total_frames = 0
        self._people_seen.clear()
        self._activity_counts.clear()
        self.start_time = datetime.datetime.now()
