"""
dataset_quality.py
==================
Real Video Data Quality Checker for onboard-har.

Validates the integrity, format, durations, resolutions, and metadata
of real video recordings across the dataset.

Features:
- File readability & corruption detection (tests actual frame decoding)
- Detection of extremely short videos (< 1.0s or < 15 frames)
- Exact duplicate file detection (SHA-256 hash)
- Filename collisions
- Class distribution & validation against Activity.ALL
- Subject and session coverage & missing metadata detection
- Supports direct raw directory scanning or manifest-based inspection
- Returns non-zero exit code on critical errors (corrupted files, unknown classes, duplicates)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2

# Project root in path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ai.schemas import Activity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_quality")

VALID_CLASSES = set(Activity.ALL)
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_DURATION_SECONDS = 1.0
MIN_FRAMES = 15


class DatasetQualityChecker:
    def __init__(
        self,
        raw_dir: Optional[str] = None,
        manifest_path: Optional[str] = None,
        min_duration: float = MIN_DURATION_SECONDS,
        min_frames: int = MIN_FRAMES,
    ):
        self.raw_dir = raw_dir
        self.manifest_path = manifest_path
        self.min_duration = min_duration
        self.min_frames = min_frames

        # Findings
        self.total_videos = 0
        self.corrupted_videos: List[Tuple[str, str]] = []
        self.short_videos: List[Tuple[str, float, int]] = []
        self.duplicate_hashes: Dict[str, List[str]] = defaultdict(list)
        self.duplicate_filenames: Dict[str, List[str]] = defaultdict(list)
        self.invalid_classes: List[Tuple[str, str]] = []
        self.unsupported_formats: List[str] = []
        self.missing_metadata: List[Tuple[str, List[str]]] = []

        self.class_counts: Counter = Counter()
        self.subject_counts: Counter = Counter()
        self.session_counts: Counter = Counter()
        self.total_duration_by_class: Dict[str, float] = defaultdict(float)

    def check_file(self, full_path: str, class_label: str, metadata: Optional[Dict[str, str]] = None) -> bool:
        """Inspect a single video file. Returns True if healthy, False if critical problem."""
        self.total_videos += 1
        fn = os.path.basename(full_path)
        self.duplicate_filenames[fn].append(full_path)

        # Check format
        ext = os.path.splitext(full_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            self.unsupported_formats.append(full_path)
            return False

        # Check class
        norm_class = class_label.upper()
        if norm_class not in VALID_CLASSES:
            self.invalid_classes.append((full_path, class_label))
            return False

        # Check file hash for duplicates
        try:
            sha = hashlib.sha256()
            with open(full_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha.update(chunk)
            fhash = sha.hexdigest()
            self.duplicate_hashes[fhash].append(full_path)
        except Exception as e:
            self.corrupted_videos.append((full_path, f"Failed to read file: {e}"))
            return False

        # Open and decode frames with OpenCV
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            self.corrupted_videos.append((full_path, "cv2.VideoCapture failed to open"))
            return False

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try reading at least 1 frame to verify decode
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None or width <= 0 or height <= 0:
            self.corrupted_videos.append((full_path, "Could not decode first video frame"))
            return False

        fps_val = fps if fps > 0 else 0.0
        duration = (frame_count / fps_val) if (fps_val > 0 and frame_count > 0) else 0.0

        if duration < self.min_duration or frame_count < self.min_frames:
            self.short_videos.append((full_path, duration, frame_count))

        self.class_counts[norm_class] += 1
        self.total_duration_by_class[norm_class] += duration

        # Metadata validation
        if metadata:
            missing = []
            subj = metadata.get("subject_id", "").strip()
            sess = metadata.get("session_id", "").strip()
            if not subj or subj.upper() == "UNKNOWN":
                missing.append("subject_id")
            else:
                self.subject_counts[subj] += 1

            if not sess or sess.upper() == "UNKNOWN":
                missing.append("session_id")
            else:
                self.session_counts[sess] += 1

            if missing:
                self.missing_metadata.append((full_path, missing))

        return True

    def run_check(self) -> Dict[str, any]:
        """Execute quality checks using raw_dir or manifest_path."""
        if self.manifest_path and os.path.isfile(self.manifest_path):
            self._check_from_manifest()
        elif self.raw_dir and os.path.isdir(self.raw_dir):
            self._check_from_raw_dir()
        else:
            logger.warning("No valid raw_dir or manifest_path provided for quality check.")

        critical_errors = len(self.corrupted_videos) + len(self.invalid_classes)
        # Duplicate hashes having more than 1 entry
        dup_count = sum(1 for v in self.duplicate_hashes.values() if len(v) > 1)
        critical_errors += dup_count

        return {
            "total_videos": self.total_videos,
            "corrupted_count": len(self.corrupted_videos),
            "short_count": len(self.short_videos),
            "duplicate_content_count": dup_count,
            "invalid_class_count": len(self.invalid_classes),
            "missing_metadata_count": len(self.missing_metadata),
            "critical_errors": critical_errors,
            "class_counts": dict(self.class_counts),
            "subject_counts": dict(self.subject_counts),
            "session_counts": dict(self.session_counts),
            "total_duration_by_class": dict(self.total_duration_by_class),
        }

    def _check_from_manifest(self) -> None:
        raw_root = self.raw_dir or "dataset/raw"
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rel = row.get("filename", "")
                cls = row.get("class", "")
                if not rel:
                    continue
                full_path = rel if os.path.isabs(rel) else os.path.join(raw_root, rel)
                self.check_file(full_path, cls, metadata=row)

    def _check_from_raw_dir(self) -> None:
        for entry in os.listdir(self.raw_dir):
            class_path = os.path.join(self.raw_dir, entry)
            if not os.path.isdir(class_path):
                continue
            for root, _, files in os.walk(class_path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        full_path = os.path.join(root, f)
                        self.check_file(full_path, entry)


def format_report(checker: DatasetQualityChecker, summary: Dict[str, any]) -> str:
    lines = [
        "=" * 65,
        "REAL VIDEO DATASET QUALITY REPORT",
        "=" * 65,
        f"Total videos inspected:       {summary['total_videos']}",
        f"Corrupted / unreadable:       {summary['corrupted_count']}",
        f"Extremely short (<{checker.min_duration}s):       {summary['short_count']}",
        f"Duplicate content matches:    {summary['duplicate_content_count']}",
        f"Invalid activity classes:     {summary['invalid_class_count']}",
        f"Videos with missing metadata: {summary['missing_metadata_count']}",
        "-" * 65,
        "CLASS DISTRIBUTION & DURATIONS:",
    ]
    if summary["total_videos"] == 0:
        lines.append("  (No real recordings found in target path)")
    else:
        for act in sorted(VALID_CLASSES):
            count = summary["class_counts"].get(act, 0)
            duration = summary["total_duration_by_class"].get(act, 0.0)
            lines.append(f"  - {act:<12}: {count:>4} video(s) | {duration:>7.1f}s total duration")

    if summary["subject_counts"]:
        lines.append("-" * 65)
        lines.append("SUBJECT DISTRIBUTION:")
        for subj, count in sorted(summary["subject_counts"].items()):
            lines.append(f"  - {subj:<12}: {count:>4} video(s)")

    if summary["corrupted_count"] > 0:
        lines.append("-" * 65)
        lines.append("CRITICAL: Corrupted Videos:")
        for path, err in checker.corrupted_videos:
            lines.append(f"  [!] {os.path.basename(path)}: {err}")

    if summary["duplicate_content_count"] > 0:
        lines.append("-" * 65)
        lines.append("CRITICAL: Duplicate Content Detected:")
        for fhash, paths in checker.duplicate_hashes.items():
            if len(paths) > 1:
                lines.append(f"  [!] Duplicate pair: {', '.join(os.path.basename(p) for p in paths)}")

    lines.append("=" * 65)
    status_str = "PASSED" if summary["critical_errors"] == 0 else f"FAILED ({summary['critical_errors']} critical issues)"
    lines.append(f"OVERALL QUALITY STATUS: {status_str}")
    lines.append("=" * 65)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check video dataset quality and integrity.")
    parser.add_argument("--data", default="dataset/raw", help="Path to raw video directory")
    parser.add_argument("--manifest", default=None, help="Path to manifest.csv (optional)")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_SECONDS)
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = parser.parse_args()

    checker = DatasetQualityChecker(
        raw_dir=args.data if not args.manifest else None,
        manifest_path=args.manifest,
        min_duration=args.min_duration,
        min_frames=args.min_frames,
    )

    summary = checker.run_check()
    report = format_report(checker, summary)
    print(report)

    if summary["critical_errors"] > 0:
        logger.error("Dataset quality check failed with %d critical errors.", summary["critical_errors"])
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
