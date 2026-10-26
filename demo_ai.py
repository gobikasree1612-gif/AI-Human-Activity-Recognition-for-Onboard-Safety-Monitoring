"""
demo_ai.py
==========
AI Engine Demo – runs the full HAR pipeline on webcam or video file
with OpenCV overlay for debugging.

Usage:
    python demo_ai.py --source 0               # webcam
    python demo_ai.py --source video.mp4       # video file
    python demo_ai.py --source image.jpg       # single image
    python demo_ai.py --source 0 --config path/to/config.yaml

Controls (when running on webcam/video):
    Q  – quit
    R  – reset all fall states
    S  – save current frame as PNG
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np

# Ensure the project root is on the path when running from any directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ai.inference import HAREngine


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------

def _run_image(engine: HAREngine, path: str) -> None:
    """Process a single image file and display the result."""
    frame = cv2.imread(path)
    if frame is None:
        logging.error("Cannot read image: %s", path)
        return

    result, annotated = engine.process_frame(frame)

    # Print result JSON to stdout
    import json
    print(json.dumps(result, indent=2))

    cv2.imshow("HAR Engine – Image Demo", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Video / webcam mode
# ---------------------------------------------------------------------------

def _run_video(engine: HAREngine, source) -> None:
    """Stream frames from webcam or video file."""
    import json

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error(
            "Cannot open video source: %s\n"
            "  For webcam use: --source 0\n"
            "  For video file: --source path/to/file.mp4",
            source,
        )
        return

    logging.info("Video source opened: %s", source)
    logging.info("Press Q to quit, R to reset fall states, S to save frame.")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("End of video stream.")
            break

        frame_idx += 1

        # Run AI pipeline
        result, annotated = engine.process_frame(frame)

        # Log significant events (not every frame)
        status = result.get("system_status", "NORMAL")
        if status != "NORMAL":
            logging.warning(
                "Frame %d | System: %s | %d person(s)",
                frame_idx,
                status,
                result.get("metrics", {}).get("people_count", 0),
            )
        elif frame_idx % 60 == 0:
            logging.info(
                "Frame %d | FPS: %.1f | People: %d",
                frame_idx,
                result.get("metrics", {}).get("fps", 0),
                result.get("metrics", {}).get("people_count", 0),
            )

        cv2.imshow("HAR Engine – AI Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            logging.info("User quit.")
            break
        elif key == ord("r"):
            logging.info("Resetting fall states for all persons.")
            for person in result.get("people", []):
                engine._safety.reset_fall(person["person_id"])
        elif key == ord("s"):
            fname = f"har_frame_{frame_idx:06d}.png"
            cv2.imwrite(fname, annotated)
            logging.info("Saved frame: %s", fname)

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HAR AI Engine Demo – webcam / video / image"
    )
    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Video source: webcam index (e.g. 0) | video path | image path. "
            "Default: 0 (first webcam)"
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    logging.info("=" * 60)
    logging.info("  Onboard HAR – AI Engine Demo")
    logging.info("=" * 60)

    # Build engine
    config_path = args.config
    if not os.path.isfile(config_path):
        logging.warning("Config not found at '%s', using defaults.", config_path)

    try:
        engine = HAREngine(config_path=config_path)
    except Exception as exc:
        logging.critical("Failed to initialise HAREngine: %s", exc)
        sys.exit(1)

    logging.info("Classifier mode: %s", engine.classifier_mode)

    # Determine source type
    source_str = args.source.strip()

    # Check if it's an image file
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(source_str)[-1].lower()
    if ext in img_exts:
        _run_image(engine, source_str)
        return

    # Webcam index or video file
    if source_str.isdigit():
        source = int(source_str)
    else:
        source = source_str
        if not os.path.isfile(source_str):
            logging.error("Video file not found: %s", source_str)
            sys.exit(1)

    _run_video(engine, source)


if __name__ == "__main__":
    main()
