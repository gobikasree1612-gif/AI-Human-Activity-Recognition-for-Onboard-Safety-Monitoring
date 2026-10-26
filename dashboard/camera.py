"""
dashboard/camera.py
===================
Camera and video source abstraction.

Wraps OpenCV VideoCapture with clean error handling.
Supports webcam index and uploaded video file.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def open_camera_capture(
    source,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
) -> Tuple[Optional[cv2.VideoCapture], Optional[str]]:
    """
    Open a VideoCapture device with robust backend selection.

    On Windows (os.name == 'nt'):
      1. Tries DirectShow backend (cv2.CAP_DSHOW) first to prevent MSMF frame grab failures (-1072875772).
      2. If DirectShow fails to open or grab a frame, falls back to MSMF / default (cv2.CAP_ANY).
    On other OS platforms:
      1. Tries default backend (cv2.CAP_ANY).

    Returns
    -------
    (cap, error_message)
    """
    if isinstance(source, str) and not source.isdigit():
        norm_source = os.path.normpath(source)
        cap = cv2.VideoCapture(norm_source)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap, None
            cap.release()
            return None, f"Video file opened but failed to read frames: {source}"
        return None, f"Cannot open video file: {source}"

    try:
        camera_index = int(source)
    except (ValueError, TypeError):
        return None, f"Invalid camera source: {source}"

    backends = []
    if os.name == "nt":
        cap_dshow = getattr(cv2, "CAP_DSHOW", 700)
        cap_msmf = getattr(cv2, "CAP_MSMF", 1400)
        cap_any = getattr(cv2, "CAP_ANY", 0)
        backends = [
            (cap_dshow, "DirectShow"),
            (cap_msmf, "Media Foundation (MSMF)"),
            (cap_any, "Default"),
        ]
    else:
        cap_any = getattr(cv2, "CAP_ANY", 0)
        backends = [(cap_any, "Default")]

    last_error = None
    for api_pref, name in backends:
        try:
            logger.info("Attempting to open camera index %d via %s backend...", camera_index, name)
            if api_pref == 0:
                cap = cv2.VideoCapture(camera_index)
            else:
                cap = cv2.VideoCapture(camera_index, api_pref)

            if not cap.isOpened():
                last_error = f"Camera {camera_index} could not be opened via {name}."
                continue

            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)

            # Test frame grab to verify backend functionality
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(
                    "Camera %d opened via %s but failed to grab frame. Trying next backend...",
                    camera_index, name,
                )
                cap.release()
                last_error = f"Camera {camera_index} opened via {name} but failed to grab frame."
                continue

            logger.info("Successfully initialized camera %d via %s backend.", camera_index, name)
            return cap, None

        except Exception as exc:
            logger.warning("Exception opening camera %d via %s: %s", camera_index, name, exc)
            last_error = str(exc)

    return None, last_error or f"Failed to open camera index {camera_index} across all available backends."


class CameraSource:
    """
    Unified frame source: webcam index or video file path.

    Usage::

        cam = CameraSource(source=0)
        cam.open()
        ok, frame = cam.read()
        cam.release()

    Or as a context manager::

        with CameraSource(source="video.mp4") as cam:
            ok, frame = cam.read()
    """

    def __init__(self, source=0, loop_video: bool = True):
        """
        Parameters
        ----------
        source : int | str
            Webcam index (int) or path to video file (str).
        loop_video : bool
            If True, video files restart from the beginning when they end.
        """
        self._source = source
        self._loop = loop_video
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = isinstance(source, str) and not str(source).isdigit() and os.path.isfile(str(source))
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        try:
            cap, err = open_camera_capture(self._source)
            if cap is None or not cap.isOpened():
                self._error = err or (
                    f"Cannot open source: {self._source}. "
                    "Check that the webcam is connected or the file path is valid."
                )
                logger.error(self._error)
                return False
            self._cap = cap
            logger.info("Camera/video source opened: %s", self._source)
            self._error = None
            return True
        except Exception as exc:
            self._error = f"Failed to open source '{self._source}': {exc}"
            logger.error(self._error)
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns (True, frame) on success, (False, None) on failure.
        For video files, loops automatically if loop_video=True.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()

        if not ret:
            if self._is_file and self._loop:
                # Loop video
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    self._error = "Video file cannot be read."
                    return False, None
            else:
                return False, None

        return True, frame

    def release(self) -> None:
        """Release the video capture resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera/video source released.")

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def fps(self) -> float:
        """Native FPS of the source (0 for webcam live stream)."""
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        """Total frame count for video files (-1 for webcams)."""
        if self._cap is None:
            return -1
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ------------------------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# Helper: save an uploaded Streamlit file to a temp path
# ---------------------------------------------------------------------------

def save_uploaded_video(uploaded_file) -> Optional[str]:
    """
    Write a Streamlit UploadedFile to a named temporary file.

    Returns the file path, or None on error.
    """
    try:
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            logger.warning("Unsupported video format: %s", suffix)
            return None

        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="har_upload_"
        )
        tmp.write(uploaded_file.read())
        tmp.close()
        logger.info("Uploaded video saved to: %s", tmp.name)
        return tmp.name
    except Exception as exc:
        logger.error("Failed to save uploaded video: %s", exc)
        return None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Test camera or video source.")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video file path")
    parser.add_argument("--preview", action="store_true", help="Display live OpenCV window until 'q' is pressed")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    print(f"\n========================================")
    print(f" Testing CameraSource: {src}")
    print(f"========================================")

    cam = CameraSource(source=src)
    if not cam.open():
        print(f"[FAIL] Could not open camera/source: {cam.error}")
        sys.exit(1)

    ok, frame = cam.read()
    if not ok or frame is None:
        print("[FAIL] Camera opened but failed to read a frame.")
        cam.release()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[PASS] Camera successfully opened!")
    print(f"       Resolution : {w}x{h}")
    print(f"       Native FPS : {cam.fps}")
    print(f"       Total Frames: {cam.frame_count}")

    if args.preview:
        print("\nOpening live preview window. Press 'q' to quit...")
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                break
            cv2.putText(
                frame,
                f"Camera {src} - Press Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Source Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    cam.release()
    print("[PASS] Camera released cleanly.\n")

