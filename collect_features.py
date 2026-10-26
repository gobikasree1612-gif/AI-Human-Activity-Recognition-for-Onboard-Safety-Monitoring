"""
collect_features.py
===================
Real feature collection tool for the HAR activity classifier.

Reads a video file (or webcam), runs YOLO11n-Pose on every frame,
extracts 34-dimensional pose features, accumulates temporal windows,
and saves labeled samples to a CSV for classifier training.

IMPORTANT
---------
- Every sample originates from real YOLO keypoints on real video frames.
- No synthetic or fabricated data is ever created.
- Frames with no valid person or insufficient keypoint confidence are
  SKIPPED and logged — never filled with invented values.
- Each person is tracked independently (IoU tracker).

Usage
-----
# From a video file
python collect_features.py \\
    --input dataset/raw/walking_hallway_01.mp4 \\
    --label WALKING \\
    --output dataset/features

# From webcam (press Q to stop)
python collect_features.py \\
    --input 0 \\
    --label STANDING \\
    --output dataset/features

# Optional flags
python collect_features.py \\
    --input video.mp4 \\
    --label WALKING \\
    --output dataset/features \\
    --config config.yaml \\
    --min-kp-conf 0.30 \\
    --max-persons 3 \\
    --show            # display live preview window

Output
------
dataset/features/<label>_<timestamp>.csv
dataset/metadata/<label>_<timestamp>.json

CSV columns
-----------
sample_id, person_id, activity_label, source_video,
frame_start, frame_end, timestamp, f_00 … f_33
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import uuid
import datetime
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

# Ensure project root is on the path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ai.detector import HumanPoseDetector
from ai.features import extract_pose_features, FEATURE_DIM
from ai.schemas import Activity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

VALID_LABELS = set(Activity.ALL)   # STANDING, SITTING, WALKING, RUNNING, FALLING

FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]

CSV_HEADER = [
    "sample_id",
    "person_id",
    "activity_label",
    "source_video",
    "frame_start",
    "frame_end",
    "timestamp",
] + FEATURE_COLS


# ── Per-person temporal buffer ─────────────────────────────────────────────

class _PersonBuffer:
    """
    Accumulates feature vectors for a single tracked person.
    Emits a sample when the buffer reaches `window_size` frames.
    Uses a sliding window with stride 1 (dense sampling).
    """

    def __init__(self, person_id: int, window_size: int = 20):
        self.person_id = person_id
        self.window_size = window_size
        self._buf: deque = deque(maxlen=window_size)
        self.frame_indices: deque = deque(maxlen=window_size)

    def push(self, fv: np.ndarray, frame_idx: int) -> Optional[np.ndarray]:
        """
        Add one feature vector. Returns the mean feature vector of the
        full window if the buffer is full, else None.
        """
        self._buf.append(fv.copy())
        self.frame_indices.append(frame_idx)
        if len(self._buf) == self.window_size:
            window = np.stack(list(self._buf), axis=0)   # (window_size, 34)
            # Use per-feature MEAN across the temporal window as the sample.
            # This gives a single (34,) vector per sample – same input shape
            # as what ActivityClassifier._predict_rules / _predict_ml expects
            # when given a single-frame sequence.
            return window.mean(axis=0).astype(np.float32)
        return None

    @property
    def frame_start(self) -> int:
        return int(self.frame_indices[0]) if self.frame_indices else 0

    @property
    def frame_end(self) -> int:
        return int(self.frame_indices[-1]) if self.frame_indices else 0


# ── Main collector ─────────────────────────────────────────────────────────

class FeatureCollector:
    """
    Processes a video source and writes labeled feature samples to CSV.
    """

    def __init__(
        self,
        label: str,
        output_dir: str,
        config: dict,
        source_name: str,
        min_kp_conf: float = 0.30,
        max_persons: int = 5,
        window_size: int = 20,
        show_preview: bool = False,
    ):
        self.label       = label.upper()
        self.output_dir  = output_dir
        self.source_name = source_name
        self.min_kp_conf = min_kp_conf
        self.max_persons = max_persons
        self.window_size = window_size
        self.show        = show_preview

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir.replace("features", "metadata"), exist_ok=True)

        self._detector = HumanPoseDetector(config)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path  = os.path.join(output_dir, f"{self.label}_{ts}.csv")
        self._meta_path = os.path.join(
            output_dir.replace("features", "metadata"),
            f"{self.label}_{ts}.json",
        )

        self._buffers: Dict[int, _PersonBuffer] = {}
        self._samples_written = 0
        self._frames_processed = 0
        self._frames_skipped   = 0
        self._start_time       = time.time()

    # ------------------------------------------------------------------

    def collect(self, source) -> None:
        """Open video source and collect features until end / 'Q' pressed."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {source!r}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        logger.info(
            "Collecting '%s' from '%s' (%d frames)",
            self.label, self.source_name, total_frames,
        )

        with open(self._csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                samples = self._process_frame(frame, frame_idx)
                for row in samples:
                    writer.writerow(row)
                    self._samples_written += 1

                self._frames_processed += 1
                frame_idx += 1

                if self.show:
                    cv2.imshow(f"Collecting [{self.label}]", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User pressed Q – stopping collection.")
                        break

                if frame_idx % 100 == 0:
                    logger.info(
                        "Frame %d / %s | samples: %d | skipped: %d",
                        frame_idx,
                        str(total_frames) if total_frames else "?",
                        self._samples_written,
                        self._frames_skipped,
                    )

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

        self._write_metadata()
        self._print_summary()

    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> List[list]:
        """Run detector + features, update buffers. Return any complete samples."""
        rows = []
        try:
            people = self._detector.process(frame)
        except Exception as exc:
            logger.debug("Detector error on frame %d: %s", frame_idx, exc)
            self._frames_skipped += 1
            return rows

        if not people:
            self._frames_skipped += 1
            return rows

        # Honour max_persons limit
        people = people[: self.max_persons]
        seen_ids = set()

        for person in people:
            pid = person.person_id
            seen_ids.add(pid)

            # Quality gate – skip if keypoints are too low confidence
            mean_conf = float(person.keypoints[:, 2].mean())
            if mean_conf < self.min_kp_conf:
                logger.debug(
                    "Frame %d person %d: low KP conf %.2f – skipping",
                    frame_idx, pid, mean_conf,
                )
                continue

            # Extract features
            try:
                fv = extract_pose_features(
                    person.keypoints, person.bbox, min_conf=self.min_kp_conf
                )
            except Exception as exc:
                logger.debug("Feature error person %d: %s", pid, exc)
                continue

            # Sanity check – no NaN/Inf
            if not np.all(np.isfinite(fv)):
                logger.debug("Frame %d person %d: non-finite features", frame_idx, pid)
                continue

            # Temporal buffer
            if pid not in self._buffers:
                self._buffers[pid] = _PersonBuffer(pid, self.window_size)

            buf = self._buffers[pid]
            mean_fv = buf.push(fv, frame_idx)

            if mean_fv is not None:
                row = self._build_row(pid, mean_fv, buf.frame_start, buf.frame_end)
                rows.append(row)

        # Clean up buffers for persons who left the scene
        stale = set(self._buffers.keys()) - seen_ids
        for pid in stale:
            del self._buffers[pid]

        return rows

    # ------------------------------------------------------------------

    def _build_row(
        self,
        person_id: int,
        mean_fv: np.ndarray,
        frame_start: int,
        frame_end: int,
    ) -> list:
        return [
            str(uuid.uuid4()),           # sample_id
            person_id,                   # person_id
            self.label,                  # activity_label
            self.source_name,            # source_video
            frame_start,                 # frame_start
            frame_end,                   # frame_end
            datetime.datetime.now().isoformat(),  # timestamp
        ] + mean_fv.tolist()             # f_00 … f_33

    # ------------------------------------------------------------------

    def _write_metadata(self) -> None:
        elapsed = time.time() - self._start_time
        meta = {
            "label":            self.label,
            "source_video":     self.source_name,
            "csv_path":         self._csv_path,
            "samples_written":  self._samples_written,
            "frames_processed": self._frames_processed,
            "frames_skipped":   self._frames_skipped,
            "window_size":      self.window_size,
            "feature_dim":      FEATURE_DIM,
            "min_kp_conf":      self.min_kp_conf,
            "elapsed_s":        round(elapsed, 2),
            "collection_time":  datetime.datetime.now().isoformat(),
        }
        with open(self._meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Metadata written to %s", self._meta_path)

    def _print_summary(self) -> None:
        elapsed = time.time() - self._start_time
        logger.info(
            "\n"
            "─" * 50 + "\n"
            "COLLECTION COMPLETE\n"
            "  Label:            %s\n"
            "  Source:           %s\n"
            "  Frames processed: %d\n"
            "  Frames skipped:   %d\n"
            "  Samples saved:    %d\n"
            "  CSV output:       %s\n"
            "  Elapsed:          %.1fs\n"
            "─" * 50,
            self.label,
            self.source_name,
            self._frames_processed,
            self._frames_skipped,
            self._samples_written,
            self._csv_path,
            elapsed,
        )


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect labeled HAR feature samples from video/webcam or batch directory."
    )
    p.add_argument(
        "--data", "-d", default=None,
        help="Path to raw dataset directory (e.g. dataset/raw) for batch processing.",
    )
    p.add_argument(
        "--input", "-i", default=None,
        help="Single video file path or webcam index (e.g. 0).",
    )
    p.add_argument(
        "--label", "-l", default=None,
        choices=list(VALID_LABELS),
        type=str.upper,
        help="Activity label for all samples in this session (required if using --input).",
    )
    p.add_argument(
        "--output", "-o", default="dataset/features",
        help="Directory to write feature CSV (default: dataset/features).",
    )
    p.add_argument(
        "--config", "-c", default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    p.add_argument(
        "--min-kp-conf", type=float, default=0.30,
        help="Minimum mean keypoint confidence to accept a detection (default: 0.30).",
    )
    p.add_argument(
        "--max-persons", type=int, default=5,
        help="Maximum number of persons to track per frame (default: 5).",
    )
    p.add_argument(
        "--window-size", type=int, default=20,
        help="Temporal window size in frames (default: 20, matches config.yaml).",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Display a live preview window during collection.",
    )
    return p.parse_args()


def _load_config(path: str) -> dict:
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config not found at %s – using defaults.", path)
        return {}


def run_batch_collection(
    data_dir: str,
    output_dir: str,
    cfg: dict,
    min_kp_conf: float = 0.30,
    max_persons: int = 5,
    window_size: int = 20,
    show_preview: bool = False,
) -> Dict[str, any]:
    """Process all valid video recordings under data_dir/<CLASS>/*."""
    supported_exts = {".mp4", ".avi", ".mov", ".mkv"}
    discovered = []

    if not os.path.isdir(data_dir):
        logger.error("Data directory '%s' does not exist.", data_dir)
        return {"successful": 0, "failed": 0, "skipped": 0, "samples": 0}

    for entry in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(class_dir):
            continue
        norm_class = entry.upper()
        if norm_class not in VALID_LABELS:
            logger.warning("Skipping unknown class folder: '%s'", entry)
            continue

        for root, _, files in os.walk(class_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_exts:
                    full_p = os.path.join(root, f)
                    discovered.append((full_p, norm_class))

    total = len(discovered)
    if total == 0:
        print("\n" + "=" * 60)
        print("BATCH FEATURE COLLECTION REPORT")
        print("=" * 60)
        print(f"No real recordings found in '{data_dir}'.")
        print("=" * 60)
        return {"successful": 0, "failed": 0, "skipped": 0, "samples": 0, "class_distribution": {}}

    print(f"\nStarting batch feature extraction on {total} video(s)...")

    successful_videos = []
    failed_videos = []
    skipped_videos = []
    total_samples = 0
    class_distribution = {c: 0 for c in VALID_LABELS}

    for idx, (vid_path, act_label) in enumerate(discovered, start=1):
        vid_name = os.path.basename(vid_path)
        print(f"[{idx}/{total}] processing {vid_name} ({act_label})...")
        try:
            collector = FeatureCollector(
                label=act_label,
                output_dir=output_dir,
                config=cfg,
                source_name=vid_name,
                min_kp_conf=min_kp_conf,
                max_persons=max_persons,
                window_size=window_size,
                show_preview=show_preview,
            )
            collector.collect(vid_path)
            samples = collector._samples_written
            if samples > 0:
                successful_videos.append(vid_name)
                total_samples += samples
                class_distribution[act_label] += samples
            else:
                skipped_videos.append((vid_name, "0 samples extracted (low confidence or short video)"))
        except Exception as exc:
            logger.error("Failed processing %s: %s", vid_name, exc)
            failed_videos.append((vid_name, str(exc)))

    print("\n" + "=" * 60)
    print("BATCH FEATURE EXTRACTION REPORT")
    print("=" * 60)
    print(f"Total videos processed:   {total}")
    print(f"Successful videos:        {len(successful_videos)}")
    print(f"Failed videos:            {len(failed_videos)}")
    print(f"Skipped videos:           {len(skipped_videos)}")
    print(f"Total samples extracted:  {total_samples}")
    print("-" * 60)
    print("CLASS DISTRIBUTION (SAMPLES):")
    for act in sorted(VALID_LABELS):
        print(f"  - {act:<12}: {class_distribution.get(act, 0):>6} sample(s)")

    if failed_videos:
        print("\nFailed Video Details:")
        for fn, err in failed_videos:
            print(f"  [!] {fn}: {err}")

    print("=" * 60)
    return {
        "successful": len(successful_videos),
        "failed": len(failed_videos),
        "skipped": len(skipped_videos),
        "samples": total_samples,
        "class_distribution": class_distribution,
    }


def main() -> None:
    args = _parse_args()
    cfg  = _load_config(args.config)

    # Resolve window size from config if not overridden
    window_size = args.window_size
    cfg_seq_len = cfg.get("activity", {}).get("sequence_length", 20)
    if window_size == 20 and cfg_seq_len != 20:
        window_size = cfg_seq_len
        logger.info("Using sequence_length=%d from config.yaml", window_size)

    if args.data is not None:
        # Batch mode
        run_batch_collection(
            data_dir=args.data,
            output_dir=args.output,
            cfg=cfg,
            min_kp_conf=args.min_kp_conf,
            max_persons=args.max_persons,
            window_size=window_size,
            show_preview=args.show,
        )
        return

    # Single-video or webcam mode
    if not args.input or not args.label:
        logger.error("Either --data <dir> OR (--input <source> and --label <LABEL>) is required.")
        sys.exit(1)

    source_raw = args.input
    try:
        source = int(source_raw)
        source_name = f"webcam_{source}"
    except ValueError:
        source = source_raw
        source_name = os.path.basename(source_raw)

    collector = FeatureCollector(
        label       = args.label,
        output_dir  = args.output,
        config      = cfg,
        source_name = source_name,
        min_kp_conf = args.min_kp_conf,
        max_persons = args.max_persons,
        window_size = window_size,
        show_preview= args.show,
    )

    try:
        collector.collect(source)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except IOError as exc:
        logger.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
