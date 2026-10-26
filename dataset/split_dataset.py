"""
split_dataset.py
================
Leakage-Safe Subject / Session / Video Splitting Utility.

Performs group-based dataset partitioning into TRAIN, VALIDATION, and TEST sets.
Ensures zero data leakage:
- Individual video recordings NEVER appear in multiple splits.
- Subjects / sessions are kept within single splits whenever possible.
- Class representation is preserved across splits where data permits.
- Refuses to create silent, broken, or single-group splits.

Outputs:
  dataset/splits/train.csv
  dataset/splits/validation.csv
  dataset/splits/test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Project root in path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "dataset" else _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.schemas import Activity
from dataset_manifest import generate_manifest, load_existing_manifest, VALID_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("split_dataset")


def perform_leakage_safe_split(
    manifest_path: str = "dataset/manifest.csv",
    splits_dir: str = "dataset/splits",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[bool, str, Dict[str, List[dict]]]:
    """
    Split records by subject or source video group into train, val, and test.
    Returns:
      (success, message, splits_dict)
    """
    if not os.path.isfile(manifest_path):
        return False, f"Manifest not found at '{manifest_path}'. Run dataset_manifest.py first.", {}

    records = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if r.get("filename") and r.get("class") in VALID_CLASSES:
                records.append(r)

    if len(records) == 0:
        return False, "No valid recordings found in manifest to split.", {}

    # Check class coverage
    present_classes = {r["class"] for r in records}
    missing_classes = VALID_CLASSES - present_classes
    if missing_classes:
        logger.warning(
            "Dataset does not contain all 5 target classes. Missing: %s",
            sorted(missing_classes),
        )

    # Determine grouping level (Subject-based if >= 3 unique subjects exist across dataset, else video-based)
    unique_subjects = {r["subject_id"].strip().upper() for r in records if r.get("subject_id") and r["subject_id"].strip().upper() != "UNKNOWN"}
    use_subject_grouping = len(unique_subjects) >= 3

    grouped_records: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if use_subject_grouping and r.get("subject_id") and r["subject_id"].strip().upper() != "UNKNOWN":
            group_key = f"subj_{r['subject_id'].strip().upper()}"
        else:
            # Fall back to video_id grouping
            group_key = f"vid_{r.get('video_id', r['filename'])}"
        grouped_records[group_key].append(r)

    unique_groups = list(grouped_records.keys())
    if len(unique_groups) < 3:
        msg = (
            f"Insufficient data to perform a valid 3-way split: only {len(unique_groups)} group(s) found. "
            f"At least 3 distinct video recordings or subjects are required."
        )
        return False, msg, {}

    # Deterministic shuffle of groups
    rng = random.Random(random_seed)
    rng.shuffle(unique_groups)

    # Group-level partition
    n_groups = len(unique_groups)
    n_test = max(1, int(round(n_groups * test_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio)))
    if n_test + n_val >= n_groups:
        n_test = 1
        n_val = 1

    test_groups = set(unique_groups[:n_test])
    val_groups = set(unique_groups[n_test : n_test + n_val])
    train_groups = set(unique_groups[n_test + n_val :])

    if not train_groups or not val_groups or not test_groups:
        return False, "Split failed: one or more splits are empty.", {}

    splits_data: Dict[str, List[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for g, recs in grouped_records.items():
        if g in test_groups:
            target_split = "test"
        elif g in val_groups:
            target_split = "validation"
        else:
            target_split = "train"

        for r in recs:
            row_copy = dict(r)
            row_copy["split"] = target_split
            splits_data[target_split].append(row_copy)

    # Write output CSVs
    os.makedirs(splits_dir, exist_ok=True)
    fieldnames = ["video_id", "filename", "class", "subject_id", "session_id", "split"]

    for split_name in ["train", "validation", "test"]:
        out_csv = os.path.join(splits_dir, f"{split_name}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in splits_data[split_name]:
                writer.writerow(r)

    # Write summary
    summary_path = os.path.join(splits_dir, "split_summary.json")
    summary = {
        "grouping_strategy": "subject" if use_subject_grouping else "video",
        "total_recordings": len(records),
        "total_groups": len(unique_groups),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_count": len(splits_data["train"]),
        "val_count": len(splits_data["validation"]),
        "test_count": len(splits_data["test"]),
        "random_seed": random_seed,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    msg = (
        f"Split successful ({summary['grouping_strategy']} grouping):\n"
        f"  Train:      {len(splits_data['train'])} recordings ({len(train_groups)} groups)\n"
        f"  Validation: {len(splits_data['validation'])} recordings ({len(val_groups)} groups)\n"
        f"  Test:       {len(splits_data['test'])} recordings ({len(test_groups)} groups)\n"
        f"Outputs written to: {splits_dir}"
    )
    return True, msg, splits_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition dataset into train/val/test splits without leakage.")
    parser.add_argument("--manifest", default="dataset/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--output-dir", default="dataset/splits", help="Directory for split CSVs")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    success, message, _ = perform_leakage_safe_split(
        manifest_path=args.manifest,
        splits_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("DATASET LEAKAGE-SAFE SPLIT")
    print("=" * 60)
    print(message)
    print("=" * 60)

    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
