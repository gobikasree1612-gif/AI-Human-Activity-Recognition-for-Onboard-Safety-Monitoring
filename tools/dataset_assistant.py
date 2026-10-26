"""
dataset_assistant.py
====================
Dataset Collection Assistant & Training Readiness Guide for onboard-har.

Inspects the current state of real recordings, extracted features, and
splits in the repository. Compares against configurable collection targets
in config.yaml and provides actionable next steps.

RULE:
- All values must come from the actual repository.
- Never invent or assume synthetic/fake data.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Set, Tuple

import yaml

# Ensure project root in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.schemas import Activity
from dataset_balance import analyze_balance, VALID_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dataset_assistant")


def load_dataset_config(config_path: str = "config.yaml") -> dict:
    """Load dataset target configuration with sensible defaults."""
    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception as e:
            logger.warning("Could not read config.yaml: %s", e)

    ds_cfg = cfg.get("dataset", {})
    return {
        "target_videos_per_class": ds_cfg.get("target_videos_per_class", 10),
        "target_subjects_per_class": ds_cfg.get("target_subjects_per_class", 3),
        "minimum_duration_seconds": ds_cfg.get("minimum_duration_seconds", 5.0),
    }


def inspect_readiness(
    raw_dir: str = "dataset/raw",
    manifest_path: str = "dataset/manifest.csv",
    features_dir: str = "dataset/features",
    splits_dir: str = "dataset/splits",
    config_path: str = "config.yaml",
) -> Dict[str, any]:
    """Inspect the repository and compute readiness metrics."""
    targets = load_dataset_config(config_path)
    balance = analyze_balance(raw_dir=raw_dir, manifest_path=manifest_path)

    total_videos = balance["total_videos"]
    class_videos = balance["class_videos"]
    class_durations = balance["class_durations"]
    class_subjects = balance["class_subjects"]
    class_sessions = balance["class_sessions"]

    # Overall unique subjects and sessions
    all_subjects: Set[str] = set()
    all_sessions: Set[str] = set()
    for s in class_subjects.values():
        all_subjects.update(s)
    for s in class_sessions.values():
        all_sessions.update(s)

    # Class-level status
    class_status: Dict[str, Dict[str, any]] = {}
    missing_classes: List[str] = []
    insufficient_classes: List[str] = []

    for act in sorted(VALID_CLASSES):
        v_count = len(class_videos[act])
        s_count = len(class_subjects[act])
        dur = class_durations[act]

        if v_count == 0:
            st = "NEED RECORDINGS"
            missing_classes.append(act)
        elif v_count < targets["target_videos_per_class"] or s_count < targets["target_subjects_per_class"]:
            st = "NEEDS MORE DATA"
            insufficient_classes.append(act)
        else:
            st = "READY"

        class_status[act] = {
            "videos": v_count,
            "subjects": s_count,
            "duration": dur,
            "status": st,
        }

    # Features check
    feature_samples = 0
    feature_classes: Set[str] = set()
    if os.path.isdir(features_dir):
        for f in os.listdir(features_dir):
            if f.endswith(".csv"):
                try:
                    with open(os.path.join(features_dir, f), "r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        for r in reader:
                            feature_samples += 1
                            if r.get("activity_label"):
                                feature_classes.add(r["activity_label"].upper())
                except Exception:
                    pass

    # Splits check
    splits_present = False
    train_count = 0
    val_count = 0
    test_count = 0
    tr_path = os.path.join(splits_dir, "train.csv")
    va_path = os.path.join(splits_dir, "validation.csv")
    te_path = os.path.join(splits_dir, "test.csv")

    if os.path.isfile(tr_path) and os.path.isfile(va_path) and os.path.isfile(te_path):
        try:
            with open(tr_path, encoding="utf-8") as fh:
                train_count = sum(1 for _ in csv.DictReader(fh))
            with open(va_path, encoding="utf-8") as fh:
                val_count = sum(1 for _ in csv.DictReader(fh))
            with open(te_path, encoding="utf-8") as fh:
                test_count = sum(1 for _ in csv.DictReader(fh))
            if train_count > 0 and val_count > 0 and test_count > 0:
                splits_present = True
        except Exception:
            splits_present = False

    # Check for leakage in splits
    has_leakage = False
    if splits_present:
        try:
            def _get_ids(p):
                with open(p, encoding="utf-8") as fh:
                    return {r.get("video_id") or r.get("filename") for r in csv.DictReader(fh)}
            tr_v = _get_ids(tr_path)
            va_v = _get_ids(va_path)
            te_v = _get_ids(te_path)
            if (tr_v & va_v) or (tr_v & te_v) or (va_v & te_v):
                has_leakage = True
        except Exception:
            pass

    # Readiness flags
    real_dataset_ready = total_videos > 0
    all_classes_ready = len(missing_classes) == 0 and len(insufficient_classes) == 0
    subject_diversity_ready = len(all_subjects) >= targets["target_subjects_per_class"]
    features_ready = (
        feature_samples > 0
        and len(feature_classes.intersection(VALID_CLASSES)) == len(VALID_CLASSES)
    )
    splits_ready = splits_present and not has_leakage

    training_ready = (
        real_dataset_ready
        and all_classes_ready
        and features_ready
        and splits_ready
        and not has_leakage
    )

    # Next action recommendation
    next_actions = []
    if total_videos == 0:
        next_actions.append("Record real videos using tools/record_dataset.py")
        next_actions.append("Review instructions in DATA_COLLECTION_GUIDE.md")
    elif missing_classes:
        next_actions.append(f"Record missing classes: {', '.join(missing_classes)}")
    elif insufficient_classes:
        next_actions.append(f"Record more sessions for classes with low coverage: {', '.join(insufficient_classes)}")
    elif not features_ready:
        next_actions.append("Extract features from raw videos: python collect_features.py --data dataset/raw")
    elif not splits_ready:
        next_actions.append("Generate leakage-safe splits: python split_dataset.py")
    else:
        next_actions.append("Prerequisites satisfied! Ready to train: python train_classifier.py --data dataset/features")

    return {
        "targets": targets,
        "total_videos": total_videos,
        "total_duration": sum(class_durations.values()),
        "total_subjects": len(all_subjects),
        "total_sessions": len(all_sessions),
        "class_status": class_status,
        "missing_classes": missing_classes,
        "insufficient_classes": insufficient_classes,
        "feature_samples": feature_samples,
        "splits_present": splits_present,
        "has_leakage": has_leakage,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "real_dataset_ready": real_dataset_ready,
        "all_classes_ready": all_classes_ready,
        "subject_diversity_ready": subject_diversity_ready,
        "features_ready": features_ready,
        "splits_ready": splits_ready,
        "training_ready": training_ready,
        "next_actions": next_actions,
    }


def format_assistant_report(res: Dict[str, any]) -> str:
    """Format the user-facing report."""
    targets = res["targets"]
    lines = [
        "=" * 44,
        "       HAR DATASET ASSISTANT",
        "=" * 44,
        "",
        f"Dataset status: {'READY FOR TRAINING' if res['training_ready'] else 'COLLECTION REQUIRED'}",
        f"Collection targets: {targets['target_videos_per_class']} vids/class | {targets['target_subjects_per_class']} subjects/class",
        "",
    ]

    for act in sorted(VALID_CLASSES):
        st = res["class_status"][act]
        lines.append(f"{act}")
        lines.append(f"  Videos:   {st['videos']}/{targets['target_videos_per_class']}")
        lines.append(f"  Subjects: {st['subjects']}/{targets['target_subjects_per_class']}")
        lines.append(f"  Duration: {st['duration']:.1f}s")
        lines.append(f"  Status:   {st['status']}")
        lines.append("")

    lines.append("-" * 44)
    lines.append("COLLECTION PROGRESS")
    lines.append("-" * 44)
    lines.append(f"Total videos:    {res['total_videos']}")
    lines.append(f"Total duration:  {res['total_duration']:.1f}s")
    lines.append(f"Total subjects:  {res['total_subjects']}")
    lines.append(f"Total sessions:  {res['total_sessions']}")
    lines.append(f"Feature samples: {res['feature_samples']}")
    lines.append("")

    lines.append("-" * 44)
    lines.append("TRAINING READINESS")
    lines.append("-" * 44)
    lines.append(f"Real dataset:        {'READY' if res['real_dataset_ready'] else 'NOT READY'}")
    lines.append(f"All classes:         {'READY' if res['all_classes_ready'] else 'NOT READY'}")
    lines.append(f"Subject diversity:   {'READY' if res['subject_diversity_ready'] else 'NOT READY'}")
    lines.append(f"Feature extraction:  {'READY' if res['features_ready'] else 'NOT READY'}")
    lines.append(f"Leakage-safe split:  {'READY' if res['splits_ready'] else 'NOT READY'}")
    lines.append(f"Training:            {'OPEN' if res['training_ready'] else 'BLOCKED'}")
    lines.append("")

    lines.append("Next action:")
    for a in res["next_actions"]:
        lines.append(f"  - {a}")
    lines.append("=" * 44)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HAR dataset status and training readiness.")
    parser.add_argument("--data", default="dataset/raw", help="Path to raw video directory")
    parser.add_argument("--manifest", default="dataset/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--features", default="dataset/features", help="Path to features directory")
    parser.add_argument("--splits", default="dataset/splits", help="Path to splits directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    res = inspect_readiness(
        raw_dir=args.data,
        manifest_path=args.manifest,
        features_dir=args.features,
        splits_dir=args.splits,
        config_path=args.config,
    )
    print(format_assistant_report(res))


if __name__ == "__main__":
    main()
