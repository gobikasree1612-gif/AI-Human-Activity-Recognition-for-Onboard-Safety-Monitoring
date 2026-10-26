"""
dataset_snapshot.py
===================
Dataset State Snapshot Generator for onboard-har.

Produces a machine-readable JSON snapshot (dataset/snapshot.json) summarizing
the exact current repository dataset state:
- Timestamp
- Real video counts and class distributions
- Subject and session coverage
- Total duration
- Feature sample counts
- Split sizes
- Manifest cryptographic SHA-256 hash

RULE:
- Reflects only actual files on disk without synthetic/fabricated placeholders.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
import sys
from typing import Dict, List, Optional

# Ensure project root in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ai.schemas import Activity
from dataset_balance import analyze_balance, VALID_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_snapshot")


def compute_file_sha256(filepath: str) -> Optional[str]:
    """Return SHA-256 hash of a file if it exists, else None."""
    if not os.path.isfile(filepath):
        return None
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def generate_snapshot(
    raw_dir: str = "dataset/raw",
    manifest_path: str = "dataset/manifest.csv",
    features_dir: str = "dataset/features",
    splits_dir: str = "dataset/splits",
    output_path: str = "dataset/snapshot.json",
) -> Dict[str, any]:
    """Compute and serialize the dataset snapshot."""
    balance = analyze_balance(raw_dir=raw_dir, manifest_path=manifest_path)

    total_videos = balance["total_videos"]
    class_counts = {c: len(balance["class_videos"][c]) for c in VALID_CLASSES}
    total_dur = round(sum(balance["class_durations"].values()), 2)

    all_subjects = set()
    for s in balance["class_subjects"].values():
        all_subjects.update(s)

    all_sessions = set()
    for s in balance["class_sessions"].values():
        all_sessions.update(s)

    # Feature sample counts
    sample_count = 0
    if os.path.isdir(features_dir):
        for f in os.listdir(features_dir):
            if f.endswith(".csv"):
                try:
                    with open(os.path.join(features_dir, f), "r", encoding="utf-8") as fh:
                        sample_count += max(0, sum(1 for _ in fh) - 1)
                except Exception:
                    pass

    # Split counts
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for s_name in ["train", "validation", "test"]:
        s_file = os.path.join(splits_dir, f"{s_name}.csv")
        if os.path.isfile(s_file):
            try:
                with open(s_file, "r", encoding="utf-8") as fh:
                    split_counts[s_name] = max(0, sum(1 for _ in fh) - 1)
            except Exception:
                split_counts[s_name] = 0

    manifest_hash = compute_file_sha256(manifest_path)

    snapshot_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "video_count": total_videos,
        "class_counts": class_counts,
        "subject_count": len(all_subjects),
        "session_count": len(all_sessions),
        "total_duration_seconds": total_dur,
        "feature_sample_count": sample_count,
        "split_counts": split_counts,
        "manifest_hash": manifest_hash,
        "is_dataset_empty": (total_videos == 0 and sample_count == 0),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(snapshot_data, fh, indent=2)

    logger.info("Dataset snapshot saved: %s", output_path)
    return snapshot_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Create machine-readable dataset snapshot.")
    parser.add_argument("--data", default="dataset/raw", help="Path to raw dataset directory")
    parser.add_argument("--manifest", default="dataset/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--features", default="dataset/features", help="Path to features directory")
    parser.add_argument("--splits", default="dataset/splits", help="Path to splits directory")
    parser.add_argument("--output", default="dataset/snapshot.json", help="Output path for snapshot.json")
    args = parser.parse_args()

    data = generate_snapshot(
        raw_dir=args.data,
        manifest_path=args.manifest,
        features_dir=args.features,
        splits_dir=args.splits,
        output_path=args.output,
    )

    print("\n" + "=" * 50)
    print("DATASET SNAPSHOT SUMMARY")
    print("=" * 50)
    print(f"Timestamp:       {data['timestamp']}")
    print(f"Total Videos:    {data['video_count']}")
    print(f"Total Duration:  {data['total_duration_seconds']}s")
    print(f"Subjects:        {data['subject_count']}")
    print(f"Sessions:        {data['session_count']}")
    print(f"Feature Samples: {data['feature_sample_count']}")
    print(f"Splits (T/V/Te): {data['split_counts']['train']}/{data['split_counts']['validation']}/{data['split_counts']['test']}")
    print(f"Manifest Hash:   {data['manifest_hash'][:16] if data['manifest_hash'] else 'None'}")
    print(f"Snapshot File:   {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
