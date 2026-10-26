"""
dataset_manifest.py
===================
Dataset Manifest Generator and Validator for onboard-har.

Scans `dataset/raw/`, extracts actual video properties (resolution, FPS,
duration), validates class labels, tracks subject/session metadata,
detects duplicate recordings, and generates or updates `dataset/manifest.csv`.

CRITICAL RULES:
- Never invent fake records, fake subjects, or fake sessions.
- Preserve existing manual metadata when updating an existing manifest.
- Strictly validate activity classes against Activity.ALL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import cv2

# Ensure project root is in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ai.schemas import Activity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_manifest")

MANIFEST_HEADERS = [
    "video_id",
    "filename",
    "class",
    "subject_id",
    "session_id",
    "camera_id",
    "environment",
    "resolution",
    "fps",
    "duration_seconds",
    "lighting",
    "camera_angle",
    "distance",
    "notes",
]

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
VALID_CLASSES = set(Activity.ALL)


def compute_file_hash(filepath: str, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for exact duplicate detection."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()


def inspect_video_file(filepath: str) -> Optional[Dict[str, any]]:
    """Open video file with cv2.VideoCapture and retrieve actual properties."""
    if not os.path.isfile(filepath):
        return None

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    fps_val = fps if fps > 0 else 0.0
    duration = (frame_count / fps_val) if (fps_val > 0 and frame_count > 0) else 0.0

    return {
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": round(fps_val, 2),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
    }


def parse_filename_metadata(filename: str) -> Dict[str, str]:
    """
    Attempt to extract subject or session tokens from standard filename conventions.
    E.g., STANDING_S01_SESSION01_20260929.mp4 -> subject: S01, session: SESSION01.
    If not explicitly formatted, leaves them empty (does NOT invent data).
    """
    stem = os.path.splitext(filename)[0]
    tokens = re.split(r"[_\-\s]+", stem)

    subject_id = ""
    session_id = ""

    for tok in tokens:
        # Matches patterns like S01, SUBJ02, Subject1
        if re.match(r"^(S|SUBJ|SUBJECT)\d+$", tok, re.IGNORECASE):
            subject_id = tok.upper()
        # Matches patterns like SESSION01, SESS1, SES02
        elif re.match(r"^(SESSION|SESS|SES)\d+$", tok, re.IGNORECASE):
            session_id = tok.upper()

    return {
        "subject_id": subject_id,
        "session_id": session_id,
    }


def scan_raw_directory(raw_dir: str) -> Tuple[List[Dict[str, any]], List[str]]:
    """
    Scan raw directory for class folders and video recordings.
    Returns:
      (discovered_records, warnings_list)
    """
    records: List[Dict[str, any]] = []
    warnings: List[str] = []

    if not os.path.isdir(raw_dir):
        warnings.append(f"Raw directory '{raw_dir}' does not exist.")
        return records, warnings

    for entry in os.listdir(raw_dir):
        class_dir = os.path.join(raw_dir, entry)
        if not os.path.isdir(class_dir):
            continue

        normalized_class = entry.upper()
        if normalized_class not in VALID_CLASSES:
            warnings.append(
                f"Invalid class folder found: '{entry}'. Expected one of {sorted(VALID_CLASSES)}."
            )
            continue

        for root, _, files in os.walk(class_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    warnings.append(f"Ignored unsupported format: '{file}' in '{entry}'.")
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, raw_dir).replace("\\", "/")

                vinfo = inspect_video_file(full_path)
                if not vinfo or vinfo["frame_count"] == 0:
                    warnings.append(f"Corrupted or unreadable video: '{rel_path}'.")
                    continue

                parsed = parse_filename_metadata(file)

                record = {
                    "video_id": os.path.splitext(file)[0],
                    "filename": rel_path,
                    "class": normalized_class,
                    "subject_id": parsed["subject_id"],
                    "session_id": parsed["session_id"],
                    "camera_id": "",
                    "environment": "",
                    "resolution": vinfo["resolution"],
                    "fps": str(vinfo["fps"]),
                    "duration_seconds": str(vinfo["duration_seconds"]),
                    "lighting": "",
                    "camera_angle": "",
                    "distance": "",
                    "notes": "",
                    "full_path": full_path,
                }
                records.append(record)

    return records, warnings


def load_existing_manifest(manifest_path: str) -> Dict[str, Dict[str, str]]:
    """Load existing manifest indexed by filename and video_id."""
    existing = {}
    if not os.path.isfile(manifest_path):
        return existing

    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fn = row.get("filename", "").strip()
                vid = row.get("video_id", "").strip()
                if fn:
                    existing[fn] = row
                elif vid:
                    existing[vid] = row
    except Exception as exc:
        logger.warning("Could not read existing manifest '%s': %s", manifest_path, exc)

    return existing


def check_duplicates(records: List[Dict[str, any]]) -> List[str]:
    """Detect duplicate filenames and identical file contents."""
    warnings = []
    seen_filenames: Dict[str, str] = {}
    seen_hashes: Dict[str, str] = {}

    for r in records:
        fn = os.path.basename(r["filename"])
        if fn in seen_filenames:
            warnings.append(
                f"Duplicate filename: '{fn}' at '{r['filename']}' and '{seen_filenames[fn]}'."
            )
        else:
            seen_filenames[fn] = r["filename"]

        full_path = r.get("full_path")
        if full_path and os.path.isfile(full_path):
            try:
                fhash = compute_file_hash(full_path)
                if fhash in seen_hashes:
                    warnings.append(
                        f"Suspicious duplicate content: '{r['filename']}' matches '{seen_hashes[fhash]}'."
                    )
                else:
                    seen_hashes[fhash] = r["filename"]
            except Exception as e:
                warnings.append(f"Could not hash '{r['filename']}': {e}")

    return warnings


def generate_manifest(
    raw_dir: str = "dataset/raw",
    manifest_path: str = "dataset/manifest.csv",
    preserve_existing: bool = True,
) -> Tuple[int, List[str]]:
    """
    Generate or update the manifest file.
    Returns:
      (count_of_valid_records, list_of_warnings)
    """
    records, warnings = scan_raw_directory(raw_dir)
    dup_warnings = check_duplicates(records)
    warnings.extend(dup_warnings)

    existing_manifest = load_existing_manifest(manifest_path) if preserve_existing else {}

    merged_records = []
    missing_metadata_count = 0

    for r in records:
        fn = r["filename"]
        vid = r["video_id"]
        # Merge manual annotations if already present in manifest
        existing_row = existing_manifest.get(fn) or existing_manifest.get(vid)
        if existing_row:
            for field in [
                "subject_id",
                "session_id",
                "camera_id",
                "environment",
                "lighting",
                "camera_angle",
                "distance",
                "notes",
            ]:
                if existing_row.get(field):
                    r[field] = existing_row[field]

        if not r["subject_id"] or not r["session_id"]:
            missing_metadata_count += 1

        clean_row = {h: r.get(h, "") for h in MANIFEST_HEADERS}
        merged_records.append(clean_row)

    if missing_metadata_count > 0:
        warnings.append(
            f"{missing_metadata_count} recording(s) missing subject_id and/or session_id metadata."
        )

    # Write output manifest
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_HEADERS)
        writer.writeheader()
        for row in merged_records:
            writer.writerow(row)

    logger.info(
        "Manifest written: %s (%d records, %d warnings)",
        manifest_path,
        len(merged_records),
        len(warnings),
    )
    return len(merged_records), warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate dataset manifest.")
    parser.add_argument("--data", default="dataset/raw", help="Path to raw dataset directory")
    parser.add_argument("--output", default="dataset/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--no-preserve", action="store_true", help="Overwrite existing manifest metadata")
    args = parser.parse_args()

    count, warnings = generate_manifest(
        raw_dir=args.data,
        manifest_path=args.output,
        preserve_existing=not args.no_preserve,
    )

    print("\n" + "=" * 60)
    print(f"DATASET MANIFEST SUMMARY")
    print("=" * 60)
    print(f"Recordings detected: {count}")
    if count == 0:
        print("Status: No real video recordings found in target raw directory.")
    else:
        print(f"Manifest written to: {args.output}")

    if warnings:
        print("\nWarnings / Quality Notices:")
        for w in warnings:
            print(f"  - {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
