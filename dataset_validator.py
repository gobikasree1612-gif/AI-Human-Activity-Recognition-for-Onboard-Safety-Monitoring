"""
dataset_validator.py
====================
Validates a collected HAR feature dataset.

Reads all CSV files from the dataset/features directory and reports:

  - Total samples
  - Samples per activity
  - Missing values (NaN)
  - Invalid feature values (Inf, out-of-range)
  - Feature dimension consistency
  - Number of source videos
  - Approximate number of unique persons

All statistics are computed from the ACTUAL data.
Nothing is hard-coded.

Usage
-----
python dataset_validator.py --data dataset/features

# Optional: validate a single CSV
python dataset_validator.py --data dataset/features/WALKING_20260929.csv

# JSON output for programmatic use
python dataset_validator.py --data dataset/features --json

Exit codes
----------
0 = dataset is valid
1 = validation errors found
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Ensure project root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ai.features import FEATURE_DIM
from ai.schemas import Activity

VALID_LABELS = set(Activity.ALL)
FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]

REQUIRED_COLS = {
    "sample_id", "person_id", "activity_label",
    "source_video", "frame_start", "frame_end", "timestamp",
} | set(FEATURE_COLS)


# ── Loader ─────────────────────────────────────────────────────────────────

def _load_csvs(path: str):
    """Load one CSV file or all CSVs in a directory. Returns (DataFrame, errors)."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas", file=sys.stderr)
        sys.exit(1)

    p = Path(path)
    if p.is_file():
        csv_files = [p]
    elif p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
    else:
        return None, [f"Path does not exist: {path}"]

    if not csv_files:
        return None, [f"No CSV files found in: {path}"]

    frames = []
    errors = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, dtype={"sample_id": str, "source_video": str})
            df["_source_file"] = f.name
            frames.append(df)
        except Exception as exc:
            errors.append(f"Cannot read {f.name}: {exc}")

    if not frames:
        return None, errors

    combined = pd.concat(frames, ignore_index=True)
    return combined, errors


# ── Validators ─────────────────────────────────────────────────────────────

def validate(df, verbose: bool = True) -> dict:
    """
    Run all validation checks on the combined DataFrame.
    Returns a structured report dict.
    """
    import pandas as pd

    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    def _err(msg: str):
        report["errors"].append(msg)
        report["valid"] = False

    def _warn(msg: str):
        report["warnings"].append(msg)

    # ── 1. Required columns ───────────────────────────────────────────────
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        _err(f"Missing columns: {sorted(missing_cols)}")
        return report  # cannot continue without schema

    # ── 2. Basic stats ────────────────────────────────────────────────────
    total = len(df)
    report["stats"]["total_samples"] = total

    if total == 0:
        _err("Dataset is empty.")
        return report

    # ── 3. Activity label distribution ────────────────────────────────────
    per_activity = df["activity_label"].value_counts().to_dict()
    report["stats"]["per_activity"] = per_activity

    invalid_labels = set(df["activity_label"].unique()) - VALID_LABELS
    if invalid_labels:
        _err(f"Unknown activity labels: {sorted(invalid_labels)}")

    # ── 4. Feature dimension check ────────────────────────────────────────
    feat_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    report["stats"]["feature_dim_expected"] = FEATURE_DIM
    report["stats"]["feature_dim_found"] = len(feat_cols_present)

    if len(feat_cols_present) != FEATURE_DIM:
        _err(
            f"Feature dimension mismatch: expected {FEATURE_DIM}, "
            f"found {len(feat_cols_present)}"
        )

    # ── 5. Missing values ─────────────────────────────────────────────────
    feat_df = df[feat_cols_present]
    missing_count = int(feat_df.isnull().sum().sum())
    report["stats"]["missing_values"] = missing_count
    if missing_count > 0:
        _err(f"Missing feature values: {missing_count}")

    # ── 6. Non-finite values (Inf / NaN after fill) ───────────────────────
    feat_arr = feat_df.to_numpy(dtype=np.float32, na_value=np.nan)
    inf_count = int(np.sum(~np.isfinite(feat_arr)))
    report["stats"]["invalid_values"] = inf_count
    if inf_count > 0:
        _err(f"Non-finite feature values (NaN/Inf): {inf_count}")

    # ── 7. Feature value range check ──────────────────────────────────────
    # Most normalised features are in [0, 1]; angles (indices 0-6) can be [0, 90]
    # We flag values > 1000 as likely errors
    large_vals = int(np.sum(np.abs(feat_arr) > 1000))
    if large_vals > 0:
        _warn(f"Feature values > 1000 detected ({large_vals} cells). Check angles.")

    # ── 8. Source video diversity ─────────────────────────────────────────
    n_sources = df["source_video"].nunique()
    report["stats"]["n_source_videos"] = n_sources
    if n_sources < 2:
        _warn(
            f"Only {n_sources} source video(s). Data leakage risk: "
            "ensure train/val/test split by source video, not by row."
        )

    # ── 9. Unique persons ─────────────────────────────────────────────────
    n_persons = df["person_id"].nunique()
    report["stats"]["n_unique_person_ids"] = n_persons

    # ── 10. Sample-ID uniqueness ──────────────────────────────────────────
    dup_ids = total - df["sample_id"].nunique()
    if dup_ids > 0:
        _warn(f"Duplicate sample_id values: {dup_ids}")

    # ── 11. Class balance ─────────────────────────────────────────────────
    counts = list(per_activity.values())
    if counts:
        max_c, min_c = max(counts), min(counts)
        ratio = max_c / max(min_c, 1)
        report["stats"]["class_imbalance_ratio"] = round(ratio, 2)
        if ratio > 5:
            _warn(
                f"Severe class imbalance (ratio {ratio:.1f}). "
                "Consider collecting more data for minority classes."
            )

    return report


# ── Printer ────────────────────────────────────────────────────────────────

def _print_report(report: dict, df) -> None:
    valid_str = "✅ VALID" if report["valid"] else "❌ INVALID"
    s = report["stats"]

    print("\n" + "─" * 55)
    print(f"  DATASET VALIDATION  {valid_str}")
    print("─" * 55)
    print(f"  Total samples:         {s.get('total_samples', 0):,}")
    print()
    print("  Per activity:")
    for act, cnt in sorted(s.get("per_activity", {}).items()):
        bar = "█" * (cnt * 20 // max(s["per_activity"].values(), default=1))
        print(f"    {act:<12} {cnt:>6,}  {bar}")
    print()
    print(f"  Feature dim expected:  {s.get('feature_dim_expected', '?')}")
    print(f"  Feature dim found:     {s.get('feature_dim_found', '?')}")
    print(f"  Missing values:        {s.get('missing_values', 0)}")
    print(f"  Invalid values:        {s.get('invalid_values', 0)}")
    print(f"  Source videos:         {s.get('n_source_videos', '?')}")
    print(f"  Unique person IDs:     {s.get('n_unique_person_ids', '?')}")
    if "class_imbalance_ratio" in s:
        print(f"  Imbalance ratio:       {s['class_imbalance_ratio']:.1f}x")
    print()

    if report["errors"]:
        print("  ERRORS:")
        for e in report["errors"]:
            print(f"    ✖ {e}")

    if report["warnings"]:
        print("  WARNINGS:")
        for w in report["warnings"]:
            print(f"    ⚠ {w}")

    print("─" * 55 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a collected HAR feature dataset."
    )
    p.add_argument(
        "--data", "-d", required=True,
        help="Path to a feature CSV or directory containing feature CSVs.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Output the report as JSON to stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df, load_errors = _load_csvs(args.data)

    if load_errors:
        for e in load_errors:
            print(f"ERROR: {e}", file=sys.stderr)

    if df is None:
        sys.exit(1)

    report = validate(df)
    report["load_errors"] = load_errors

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report, df)

    sys.exit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
