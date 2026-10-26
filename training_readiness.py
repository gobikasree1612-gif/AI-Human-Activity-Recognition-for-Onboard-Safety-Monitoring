"""
training_readiness.py
=====================
Training Readiness Verification Tool for onboard-har.

Validates all prerequisites across real video recordings, class balance,
subject/session diversity, feature extraction integrity, and leakage-safe
train/validation/test splits before allowing training to proceed.

RULE:
- Never fabricate data or create dummy artifacts to force a passing status.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Project root in path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ai.schemas import Activity
from ai.features import FEATURE_DIM
from dataset_balance import analyze_balance
from dataset_manifest import check_duplicates, scan_raw_directory
from tools.dataset_assistant import load_dataset_config

VALID_CLASSES = set(Activity.ALL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training_readiness")

FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]


class TrainingReadinessChecker:
    def __init__(
        self,
        raw_dir: str = "dataset/raw",
        manifest_path: str = "dataset/manifest.csv",
        features_dir: str = "dataset/features",
        splits_dir: str = "dataset/splits",
        config_path: str = "config.yaml",
    ):
        self.raw_dir = raw_dir
        self.manifest_path = manifest_path
        self.features_dir = features_dir
        self.splits_dir = splits_dir
        self.config_path = config_path
        self.targets = load_dataset_config(config_path)

        self.issues: List[str] = []
        self.next_actions: List[str] = []

    def evaluate(self) -> Dict[str, any]:
        """Run all readiness checks and return structured status dict."""
        self.issues.clear()
        self.next_actions.clear()

        # 1. Dataset Check
        balance = analyze_balance(raw_dir=self.raw_dir, manifest_path=self.manifest_path)
        total_videos = balance["total_videos"]
        records, scan_warns = scan_raw_directory(self.raw_dir)
        dup_warns = check_duplicates(records)

        has_raw_videos = total_videos > 0 and len(records) > 0
        no_duplicates = len(dup_warns) == 0
        dataset_ready = has_raw_videos and no_duplicates

        if not has_raw_videos:
            self.issues.append("No real video recordings found in raw dataset directory.")
            self.next_actions.append("Record real video clips: python tools/record_dataset.py --class <CLASS>")
        if not no_duplicates:
            self.issues.append(f"Duplicate recordings detected ({len(dup_warns)} duplicate notices).")
            self.next_actions.append("Resolve duplicate videos in dataset/raw")

        # 2. Classes Check
        present_raw_classes = {c for c, vids in balance["class_videos"].items() if len(vids) > 0}
        missing_classes = VALID_CLASSES - present_raw_classes
        classes_ready = len(missing_classes) == 0 and has_raw_videos

        if missing_classes:
            self.issues.append(f"Missing target activity classes: {sorted(missing_classes)}")
            self.next_actions.append(f"Record missing classes: {', '.join(sorted(missing_classes))}")

        # 3. Diversity Check
        all_subjects = set()
        for s in balance["class_subjects"].values():
            all_subjects.update(s)
        all_sessions = set()
        for s in balance["class_sessions"].values():
            all_sessions.update(s)

        min_subjs = self.targets["target_subjects_per_class"]
        diversity_ready = (
            len(all_subjects) >= min_subjs
            and len(all_sessions) >= 2
            and not any(len(balance["class_subjects"][c]) < 2 for c in VALID_CLASSES)
        )
        if not diversity_ready:
            if len(all_subjects) < min_subjs:
                self.issues.append(f"Subject diversity low: {len(all_subjects)} subject(s) found (target: {min_subjs}+).")
            self.next_actions.append("Involve multiple independent subjects and recording sessions.")

        # 4. Features Check
        features_ready = False
        csv_files = []
        if os.path.isdir(self.features_dir):
            csv_files = [
                os.path.join(self.features_dir, f)
                for f in os.listdir(self.features_dir)
                if f.endswith(".csv")
            ]

        if not csv_files:
            self.issues.append("No extracted feature CSV files found in dataset/features.")
            self.next_actions.append("Extract features: python collect_features.py --data dataset/raw")
        else:
            try:
                frames = [pd.read_csv(f) for f in csv_files]
                if frames and any(len(df) > 0 for df in frames):
                    df_all = pd.concat(frames, ignore_index=True)
                    missing_feat_cols = set(FEATURE_COLS) - set(df_all.columns)
                    if missing_feat_cols:
                        self.issues.append(f"Feature dimension mismatch: missing {len(missing_feat_cols)} columns.")
                    elif not np.all(np.isfinite(df_all[FEATURE_COLS].to_numpy())):
                        self.issues.append("Features contain NaN or Infinite values.")
                    else:
                        feat_classes = set(df_all.get("activity_label", []).dropna().unique())
                        if VALID_CLASSES - feat_classes:
                            self.issues.append(f"Features missing classes: {sorted(VALID_CLASSES - feat_classes)}")
                        else:
                            features_ready = True
            except Exception as e:
                self.issues.append(f"Failed parsing feature files: {e}")

        # 5. Splits & Leakage Check
        splits_ready = False
        leakage_passed = True
        tr_path = os.path.join(self.splits_dir, "train.csv")
        va_path = os.path.join(self.splits_dir, "validation.csv")
        te_path = os.path.join(self.splits_dir, "test.csv")

        if not (os.path.isfile(tr_path) and os.path.isfile(va_path) and os.path.isfile(te_path)):
            self.issues.append("Train/Validation/Test split files missing in dataset/splits.")
            self.next_actions.append("Generate splits: python split_dataset.py")
            splits_ready = False
            leakage_passed = False
        else:
            try:
                def _get_vids_subjs(path):
                    with open(path, encoding="utf-8") as fh:
                        r = list(csv.DictReader(fh))
                        vids = {x.get("video_id") or x.get("filename") for x in r if x.get("video_id") or x.get("filename")}
                        subjs = {x.get("subject_id") for x in r if x.get("subject_id") and x.get("subject_id") != "UNKNOWN"}
                        return vids, subjs, len(r)

                tr_v, tr_s, tr_len = _get_vids_subjs(tr_path)
                va_v, va_s, va_len = _get_vids_subjs(va_path)
                te_v, te_s, te_len = _get_vids_subjs(te_path)

                if tr_len == 0 or va_len == 0 or te_len == 0:
                    self.issues.append("One or more split CSVs are empty.")
                    splits_ready = False
                else:
                    splits_ready = True

                # Check video leakage
                if (tr_v & va_v) or (tr_v & te_v) or (va_v & te_v):
                    self.issues.append("CRITICAL: Video leakage detected across train/val/test splits.")
                    leakage_passed = False

                # Check subject leakage
                if tr_s and te_s and (tr_s & te_s):
                    self.issues.append("Subject leakage detected: same subject appears in both train and test splits.")
                    leakage_passed = False

            except Exception as e:
                self.issues.append(f"Error checking split files: {e}")
                splits_ready = False
                leakage_passed = False

        # 6. Overall Training Gate Status
        training_gate_open = (
            dataset_ready
            and classes_ready
            and features_ready
            and splits_ready
            and leakage_passed
        )

        overall_ready = training_gate_open and diversity_ready

        # Deduplicate next actions
        dedup_actions = []
        for a in self.next_actions:
            if a not in dedup_actions:
                dedup_actions.append(a)

        return {
            "dataset_ready": dataset_ready,
            "classes_ready": classes_ready,
            "diversity_ready": diversity_ready,
            "features_ready": features_ready,
            "splits_ready": splits_ready,
            "leakage_passed": leakage_passed,
            "training_gate_open": training_gate_open,
            "overall_ready": overall_ready,
            "issues": self.issues,
            "next_actions": dedup_actions,
        }


def format_readiness_report(res: Dict[str, any]) -> str:
    lines = [
        "=" * 40,
        "TRAINING READINESS REPORT",
        "=" * 40,
        "",
        f"Dataset:              {'READY' if res['dataset_ready'] else 'NOT READY'}",
        f"Classes:              {'READY' if res['classes_ready'] else 'NOT READY'}",
        f"Diversity:            {'READY' if res['diversity_ready'] else 'NOT READY'}",
        f"Features:             {'READY' if res['features_ready'] else 'NOT READY'}",
        f"Splits:               {'READY' if res['splits_ready'] else 'NOT READY'}",
        f"Leakage check:        {'PASS' if res['leakage_passed'] else 'FAIL'}",
        f"Training gate:        {'OPEN' if res['training_gate_open'] else 'BLOCKED'}",
        "",
        "Overall:",
        f"{'TRAINING READY' if res['overall_ready'] else 'TRAINING NOT READY'}",
    ]

    if not res["overall_ready"]:
        lines.append("")
        lines.append("Required next actions:")
        if res["next_actions"]:
            for i, act in enumerate(res["next_actions"], start=1):
                lines.append(f"{i}. {act}")
        else:
            lines.append("1. Collect real recordings and extract features.")

    lines.append("=" * 40)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate training readiness of the HAR dataset.")
    parser.add_argument("--data", default="dataset/raw", help="Path to raw dataset directory")
    parser.add_argument("--manifest", default="dataset/manifest.csv", help="Path to manifest.csv")
    parser.add_argument("--features", default="dataset/features", help="Path to features directory")
    parser.add_argument("--splits", default="dataset/splits", help="Path to splits directory")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    checker = TrainingReadinessChecker(
        raw_dir=args.data,
        manifest_path=args.manifest,
        features_dir=args.features,
        splits_dir=args.splits,
        config_path=args.config,
    )
    res = checker.evaluate()
    report = format_readiness_report(res)
    print(report)

    if not res["overall_ready"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
