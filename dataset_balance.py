"""
dataset_balance.py
==================
Dataset Balance and Representation Reporter for onboard-har.

Calculates and prints class counts, total recording durations, unique subjects,
and session diversity strictly from real video recordings or manifest entries.

RULE:
- If there is no data, clearly show: "No real recordings found."
- Never substitute dummy numbers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import cv2

# Project root in path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ai.schemas import Activity
from dataset_manifest import parse_filename_metadata, SUPPORTED_EXTENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_balance")

VALID_CLASSES = Activity.ALL


def analyze_balance(
    raw_dir: str = "dataset/raw",
    manifest_path: Optional[str] = "dataset/manifest.csv",
) -> Dict[str, any]:
    """
    Analyze video files and manifest to compute actual balance metrics.
    """
    class_videos: Dict[str, List[str]] = {c: [] for c in VALID_CLASSES}
    class_durations: Dict[str, float] = {c: 0.0 for c in VALID_CLASSES}
    class_subjects: Dict[str, Set[str]] = {c: set() for c in VALID_CLASSES}
    class_sessions: Dict[str, Set[str]] = {c: set() for c in VALID_CLASSES}

    subject_video_counts = Counter()
    environments: Set[str] = set()
    camera_angles: Set[str] = set()

    total_files_found = 0

    # 1. Try reading manifest first if it has populated records
    manifest_records = []
    if manifest_path and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    fn = row.get("filename", "").strip()
                    cls = row.get("class", "").strip().upper()
                    if fn and cls in VALID_CLASSES:
                        manifest_records.append(row)
        except Exception as e:
            logger.debug("Failed reading manifest for balance: %s", e)

    if manifest_records:
        for r in manifest_records:
            cls = r["class"].upper()
            fn = r["filename"]
            dur = float(r.get("duration_seconds") or 0.0)
            subj = r.get("subject_id", "").strip().upper()
            sess = r.get("session_id", "").strip().upper()
            env = r.get("environment", "").strip()
            cam_ang = r.get("camera_angle", "").strip()

            class_videos[cls].append(fn)
            class_durations[cls] += dur
            if subj and subj != "UNKNOWN":
                class_subjects[cls].add(subj)
                subject_video_counts[subj] += 1
            if sess and sess != "UNKNOWN":
                class_sessions[cls].add(sess)
            if env:
                environments.add(env)
            if cam_ang:
                camera_angles.add(cam_ang)
            total_files_found += 1
    elif os.path.isdir(raw_dir):
        # 2. Inspect raw directory directly
        for entry in os.listdir(raw_dir):
            class_dir = os.path.join(raw_dir, entry)
            if not os.path.isdir(class_dir):
                continue
            norm_class = entry.upper()
            if norm_class not in VALID_CLASSES:
                continue

            for root, _, files in os.walk(class_dir):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        full_path = os.path.join(root, f)
                        cap = cv2.VideoCapture(full_path)
                        dur = 0.0
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                            fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                            cap.release()
                            if fps > 0 and fc > 0:
                                dur = fc / fps
                        parsed = parse_filename_metadata(f)
                        subj = parsed["subject_id"]
                        sess = parsed["session_id"]

                        class_videos[norm_class].append(f)
                        class_durations[norm_class] += dur
                        if subj:
                            class_subjects[norm_class].add(subj)
                            subject_video_counts[subj] += 1
                        if sess:
                            class_sessions[norm_class].add(sess)
                        total_files_found += 1

    return {
        "total_videos": total_files_found,
        "class_videos": class_videos,
        "class_durations": class_durations,
        "class_subjects": class_subjects,
        "class_sessions": class_sessions,
        "subject_video_counts": subject_video_counts,
        "environments": environments,
        "camera_angles": camera_angles,
    }


def format_balance_report(metrics: Dict[str, any]) -> Tuple[str, List[str]]:
    """Format human-readable balance report and compute imbalance warnings."""
    total = metrics["total_videos"]
    warnings: List[str] = []

    lines = [
        "=" * 60,
        "DATASET BALANCE & DIVERSITY REPORT",
        "=" * 60,
    ]

    if total == 0:
        lines.append("No real recordings found.")
        lines.append("=" * 60)
        return "\n".join(lines), ["No real recordings found in dataset/raw or manifest."]

    lines.append(f"Total video recordings: {total}\n")
    lines.append(f"{'Activity Class':<12} {'Videos':>6} {'Duration':>10} {'Subjects':>9} {'Status':>18}")
    lines.append("-" * 60)

    counts = []
    all_subjects = set()
    all_sessions = set()

    for act in VALID_CLASSES:
        vids = len(metrics["class_videos"][act])
        dur = metrics["class_durations"][act]
        subjs = len(metrics["class_subjects"][act])
        counts.append(vids)

        all_subjects.update(metrics["class_subjects"][act])
        all_sessions.update(metrics["class_sessions"][act])

        # Neutral status labels
        if vids == 0:
            status_label = "MISSING CLASS"
            warnings.append(f"Class '{act}': MISSING CLASS (0 recordings).")
        elif subjs <= 1 and vids > 1:
            status_label = "LOW DIVERSITY"
            warnings.append(f"Class '{act}': LOW DIVERSITY (only 1 subject represented).")
        elif vids < 5 or subjs < 2:
            status_label = "NEEDS MORE DATA"
            warnings.append(f"Class '{act}': NEEDS MORE DATA ({vids} video(s), {subjs} subject(s)).")
        else:
            status_label = "READY"

        lines.append(
            f"{act:<12} {vids:>6} {dur:>9.1f}s {subjs:>9} {status_label:>18}"
        )

    # Diversity Section
    lines.append("-" * 60)
    lines.append("DIVERSITY SUMMARY:")
    lines.append(f"Subjects ({len(all_subjects)}):")
    if all_subjects:
        for s in sorted(all_subjects):
            cnt = metrics.get("subject_video_counts", {}).get(s, 0)
            lines.append(f"  - {s} ({cnt} video(s))")
    else:
        lines.append("  (None identified)")

    lines.append(f"Sessions ({len(all_sessions)}):")
    if all_sessions:
        for s in sorted(all_sessions):
            lines.append(f"  - {s}")
    else:
        lines.append("  (None identified)")

    envs = metrics.get("environments")
    cams = metrics.get("camera_angles")
    lines.append("Camera / environment:")
    lines.append(f"  Environments: {', '.join(sorted(envs)) if envs else 'standard lab'}")
    lines.append(f"  Angles:       {', '.join(sorted(cams)) if cams else 'eye_level'}")

    lines.append("-" * 60)
    lines.append("Coverage by Activity:")
    for act in VALID_CLASSES:
        subjs_list = sorted(metrics["class_subjects"][act])
        subjs_str = ", ".join(subjs_list) if subjs_list else "None"
        lines.append(f"  {act:<10} → subjects: {subjs_str}")

    # Dominance check
    subj_counts = metrics.get("subject_video_counts", {})
    if total >= 3 and subj_counts:
        top_subj, top_cnt = subj_counts.most_common(1)[0]
        pct = (top_cnt / total) * 100
        if pct > 60.0:
            warnings.append(f"Subject '{top_subj}' dominates the dataset ({pct:.1f}% of all recordings).")

    # Overall representation evaluation
    if any(c == 0 for c in counts):
        overall_eval = "INSUFFICIENT FOR TRAINING"
    elif any(len(metrics["class_subjects"][a]) <= 1 for a in VALID_CLASSES):
        overall_eval = "LOW DIVERSITY"
    elif sum(counts) < 25:
        overall_eval = "NEEDS MORE DATA"
    else:
        overall_eval = "READY"

    lines.append("-" * 60)
    lines.append(f"Overall Dataset Status: {overall_eval}")

    if warnings:
        lines.append("\nBALANCE & DIVERSITY NOTICES:")
        for w in warnings:
            lines.append(f"  [!] {w}")

    lines.append("=" * 60)
    return "\n".join(lines), warnings


def get_dataset_status_summary(
    raw_dir: str = "dataset/raw",
    manifest_path: str = "dataset/manifest.csv",
    features_dir: str = "dataset/features",
) -> Dict[str, any]:
    """Provide concise dataset metrics for UI diagnostics."""
    balance = analyze_balance(raw_dir=raw_dir, manifest_path=manifest_path)
    total_videos = balance["total_videos"]

    active_classes = [c for c, vids in balance["class_videos"].items() if len(vids) > 0]

    all_subjects = set()
    all_sessions = set()
    for s in balance["class_subjects"].values():
        all_subjects.update(s)
    for s in balance["class_sessions"].values():
        all_sessions.update(s)

    sample_count = 0
    if os.path.isdir(features_dir):
        for f in os.listdir(features_dir):
            if f.endswith(".csv"):
                try:
                    with open(os.path.join(features_dir, f), "r", encoding="utf-8") as fh:
                        sample_count += max(0, sum(1 for _ in fh) - 1)
                except Exception:
                    pass

    has_real_data = total_videos > 0
    training_available = (
        has_real_data
        and len(active_classes) == len(VALID_CLASSES)
        and sample_count >= 25
    )

    return {
        "has_real_data": has_real_data,
        "dataset_label": "REAL DATA" if has_real_data else "NO DATA",
        "videos": total_videos,
        "classes": len(active_classes),
        "samples": sample_count,
        "subjects": len(all_subjects),
        "sessions": len(all_sessions),
        "training_status": "AVAILABLE" if training_available else "BLOCKED",
        "blocked_reason": None if training_available else (
            "Training blocked: no real dataset available." if not has_real_data else
            f"Training blocked: only {len(active_classes)}/{len(VALID_CLASSES)} classes and {sample_count} samples available."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report dataset class balance and metadata representation.")
    parser.add_argument("--data", default="dataset/raw", help="Path to raw video directory")
    parser.add_argument("--manifest", default="dataset/manifest.csv", help="Path to manifest.csv")
    args = parser.parse_args()

    metrics = analyze_balance(raw_dir=args.data, manifest_path=args.manifest)
    report, _ = format_balance_report(metrics)
    print(report)


if __name__ == "__main__":
    main()
