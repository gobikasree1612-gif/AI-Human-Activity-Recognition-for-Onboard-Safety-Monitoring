"""
tests/test_dashboard.py
=======================
Unit tests for dashboard/history.py and dashboard/analytics.py.

No webcam, Streamlit, or GPU required.
"""

import datetime
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.history import SessionHistory
from dashboard.analytics import (
    activity_distribution_chart,
    activity_timeline_chart,
    performance_chart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    activity: str = "WALKING",
    person_id: int = 1,
    confidence: float = 0.85,
    alert: bool = False,
    alert_type: str = None,
    severity: str = None,
    system_status: str = "NORMAL",
    fps: float = 24.5,
    latency_ms: float = 41.2,
) -> dict:
    """Build a minimal FrameResult-like dict."""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_status": system_status,
        "people": [
            {
                "person_id": person_id,
                "bbox": [100, 50, 200, 450],
                "activity": activity,
                "confidence": confidence,
                "model_type": "rule_based_baseline",
                "alert": alert,
                "alert_type": alert_type,
                "severity": severity,
                "evidence": ["Test evidence item"],
            }
        ],
        "metrics": {
            "people_count": 1,
            "fps": fps,
            "latency_ms": latency_ms,
        },
    }


def _make_empty_result(system_status: str = "NORMAL") -> dict:
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "system_status": system_status,
        "people": [],
        "metrics": {"people_count": 0, "fps": 22.0, "latency_ms": 45.0},
    }


# ---------------------------------------------------------------------------
# SessionHistory tests
# ---------------------------------------------------------------------------

class TestSessionHistoryInit(unittest.TestCase):

    def test_initial_state(self):
        h = SessionHistory()
        self.assertEqual(h.total_frames, 0)
        self.assertEqual(h.alert_count, 0)
        self.assertEqual(h.people_observed, 0)
        self.assertEqual(h.avg_fps, 0.0)
        self.assertEqual(h.avg_latency_ms, 0.0)
        self.assertEqual(h.activity_distribution, {})

    def test_initial_records_empty(self):
        h = SessionHistory()
        self.assertEqual(len(h.get_records()), 0)


class TestSessionHistoryRecord(unittest.TestCase):

    def test_records_frame_count(self):
        h = SessionHistory()
        for _ in range(5):
            h.record(_make_result())
        self.assertEqual(h.total_frames, 5)

    def test_records_fps_samples(self):
        h = SessionHistory()
        h.record(_make_result(fps=30.0))
        h.record(_make_result(fps=20.0))
        self.assertAlmostEqual(h.avg_fps, 25.0, delta=0.1)

    def test_records_latency_samples(self):
        h = SessionHistory()
        h.record(_make_result(latency_ms=40.0))
        h.record(_make_result(latency_ms=60.0))
        self.assertAlmostEqual(h.avg_latency_ms, 50.0, delta=0.1)

    def test_activity_count(self):
        h = SessionHistory()
        h.record(_make_result(activity="WALKING"))
        h.record(_make_result(activity="STANDING"))
        h.record(_make_result(activity="WALKING"))
        dist = h.activity_distribution
        self.assertEqual(dist.get("WALKING"), 2)
        self.assertEqual(dist.get("STANDING"), 1)

    def test_people_observed(self):
        h = SessionHistory()
        h.record(_make_result(person_id=1))
        h.record(_make_result(person_id=2))
        h.record(_make_result(person_id=1))
        self.assertEqual(h.people_observed, 2)

    def test_alert_counting_no_alert(self):
        h = SessionHistory()
        h.record(_make_result(alert=False))
        self.assertEqual(h.alert_count, 0)

    def test_alert_counting_with_alert(self):
        h = SessionHistory()
        h.record(_make_result(alert=True, alert_type="FALL_DETECTED", severity="CRITICAL"))
        self.assertEqual(h.alert_count, 1)

    def test_multiple_alerts_accumulate(self):
        h = SessionHistory()
        for _ in range(5):
            h.record(_make_result(alert=True, alert_type="POSSIBLE_FALL", severity="WARNING"))
        self.assertEqual(h.alert_count, 5)

    def test_empty_people_does_not_crash(self):
        h = SessionHistory()
        h.record(_make_empty_result())
        self.assertEqual(h.total_frames, 1)
        self.assertEqual(h.people_observed, 0)

    def test_records_list_length(self):
        h = SessionHistory()
        for _ in range(10):
            h.record(_make_result())
        self.assertEqual(len(h.get_records()), 10)

    def test_most_recent_first(self):
        """get_records() should return newest first."""
        h = SessionHistory()
        h.record(_make_result(activity="WALKING"))
        h.record(_make_result(activity="STANDING"))
        records = h.get_records()
        self.assertEqual(records[0].activity, "STANDING")

    def test_max_records_cap(self):
        h = SessionHistory()
        h.MAX_RECORDS = 10   # override for test
        for _ in range(20):
            h.record(_make_result())
        # _records list is capped
        self.assertLessEqual(len(h._records), 10)


class TestSessionHistoryDataFrame(unittest.TestCase):

    def test_empty_returns_dataframe(self):
        import pandas as pd
        h = SessionHistory()
        df = h.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)

    def test_populated_dataframe(self):
        h = SessionHistory()
        h.record(_make_result(activity="WALKING"))
        df = h.to_dataframe()
        self.assertGreater(len(df), 0)
        self.assertIn("Activity", df.columns)
        self.assertIn("Person", df.columns)

    def test_alert_shown_in_dataframe(self):
        h = SessionHistory()
        h.record(_make_result(alert=True, alert_type="FALL_DETECTED", severity="CRITICAL"))
        df = h.to_dataframe()
        self.assertEqual(df.iloc[0]["Alert"], "⚠")


class TestSessionHistoryReset(unittest.TestCase):

    def test_reset_clears_all(self):
        h = SessionHistory()
        for _ in range(10):
            h.record(_make_result(alert=True))
        h.reset()
        self.assertEqual(h.total_frames, 0)
        self.assertEqual(h.alert_count, 0)
        self.assertEqual(h.people_observed, 0)
        self.assertEqual(len(h.get_records()), 0)
        self.assertEqual(h.activity_distribution, {})

    def test_reset_restarts_timer(self):
        import time
        h = SessionHistory()
        time.sleep(0.05)
        h.reset()
        self.assertLess(h.session_duration_s, 0.1)


# ---------------------------------------------------------------------------
# Analytics chart tests (smoke tests – verify no crash)
# ---------------------------------------------------------------------------

class TestAnalyticsCharts(unittest.TestCase):

    def test_empty_activity_distribution(self):
        import plotly.graph_objects as go
        fig = activity_distribution_chart({})
        self.assertIsInstance(fig, go.Figure)

    def test_activity_distribution_with_data(self):
        import plotly.graph_objects as go
        data = {"STANDING": 50, "WALKING": 80, "FALLING": 3}
        fig = activity_distribution_chart(data)
        self.assertIsInstance(fig, go.Figure)

    def test_empty_timeline(self):
        import plotly.graph_objects as go
        fig = activity_timeline_chart([])
        self.assertIsInstance(fig, go.Figure)

    def test_timeline_with_records(self):
        import plotly.graph_objects as go
        h = SessionHistory()
        for act in ["WALKING", "STANDING", "WALKING", "RUNNING"]:
            h.record(_make_result(activity=act))
        fig = activity_timeline_chart(h.get_records())
        self.assertIsInstance(fig, go.Figure)

    def test_empty_performance_chart(self):
        import plotly.graph_objects as go
        fig = performance_chart([])
        self.assertIsInstance(fig, go.Figure)

    def test_performance_chart_with_records(self):
        import plotly.graph_objects as go
        h = SessionHistory()
        for _ in range(10):
            h.record(_make_result(fps=25.0, latency_ms=40.0))
        fig = performance_chart(h.get_records())
        self.assertIsInstance(fig, go.Figure)


class TestCameraSourceBackendFallback(unittest.TestCase):

    def test_open_camera_capture_file_not_found(self):
        from dashboard.camera import open_camera_capture
        cap, err = open_camera_capture("nonexistent_video_file.mp4")
        self.assertIsNone(cap)
        self.assertIsNotNone(err)
        self.assertIn("Cannot open video file", err)

    def test_open_camera_capture_mock_fallback(self):
        from unittest.mock import patch, MagicMock
        from dashboard.camera import open_camera_capture

        # Mock cv2.VideoCapture so the first attempt (e.g. DirectShow) fails frame grab
        # and the second attempt succeeds
        mock_cap_fail = MagicMock()
        mock_cap_fail.isOpened.return_value = True
        mock_cap_fail.read.return_value = (False, None)

        mock_cap_success = MagicMock()
        mock_cap_success.isOpened.return_value = True
        mock_cap_success.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        with patch("cv2.VideoCapture", side_effect=[mock_cap_fail, mock_cap_success]):
            cap, err = open_camera_capture(0)
            self.assertIsNotNone(cap)
            self.assertIsNone(err)

    def test_camera_source_open_failure(self):
        from unittest.mock import patch
        from dashboard.camera import CameraSource

        with patch("dashboard.camera.open_camera_capture", return_value=(None, "Mock error")):
            cam = CameraSource(source=99)
            ok = cam.open()
            self.assertFalse(ok)
            self.assertEqual(cam.error, "Mock error")


if __name__ == "__main__":
    unittest.main()
