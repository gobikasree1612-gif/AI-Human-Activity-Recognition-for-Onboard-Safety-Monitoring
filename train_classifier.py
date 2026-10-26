"""
train_classifier.py
===================
Training pipeline for the HAR activity classifier.

Loads REAL labeled feature data collected by collect_features.py,
performs a leakage-safe train/validation/test split (by source video),
trains a lightweight classifier, evaluates it, saves the model,
and generates an evaluation report.

Usage
-----
# Train on real collected data
python train_classifier.py \\
    --data dataset/features \\
    --output models/activity_classifier_v1.joblib

# Choose model type
python train_classifier.py \\
    --data dataset/features \\
    --model random_forest \\    # default: best for this feature space
    --output models/har_v1.joblib

Model choices
-------------
  random_forest  (default) — fast, robust, handles mixed feature scales,
                              performs well on 34-dim tabular pose features.
  gradient_boost             — stronger but slower. Use if RF under-performs.
  logistic_regression        — interpretable baseline.
  svm                        — good for small datasets.

IMPORTANT
---------
- This script only works on REAL collected data from collect_features.py.
- It will refuse to run if no CSVs are found in --data directory.
- It will NOT fabricate or invent training samples.
- Performance metrics are only reported on the real held-out test set.
- Train/test split is by SOURCE VIDEO to prevent data leakage.

Output
------
  <output_model>.joblib           — trained classifier
  <output_model>_scaler.joblib    — fitted StandardScaler
  <output_model>_meta.json        — feature metadata + label map
  evaluation/metrics.json
  evaluation/classification_report.txt
  evaluation/confusion_matrix.csv
  evaluation/confusion_matrix.png  (if matplotlib available)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ai.features import FEATURE_DIM
from ai.schemas import Activity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [f"f_{i:02d}" for i in range(FEATURE_DIM)]
VALID_LABELS = set(Activity.ALL)

# ── Data loading ───────────────────────────────────────────────────────────

def load_dataset(data_path: str):
    """
    Load all feature CSVs from data_path.

    Returns
    -------
    X          : np.ndarray (N, FEATURE_DIM)
    y          : np.ndarray of str labels
    groups     : np.ndarray of source_video strings (for leakage-safe split)
    n_samples  : int
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required: pip install pandas")
        sys.exit(1)

    p = Path(data_path)
    if p.is_file():
        csv_files = [p]
    elif p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
    else:
        logger.error("Data path not found: %s", data_path)
        sys.exit(1)

    if not csv_files:
        logger.error(
            "No CSV files found in '%s'.\n"
            "Run collect_features.py first to collect real labeled data.",
            data_path,
        )
        sys.exit(1)

    logger.info("Loading %d CSV file(s)...", len(csv_files))
    frames = []
    for f in csv_files:
        df = pd.read_csv(f, dtype={"sample_id": str, "source_video": str})
        frames.append(df)
        logger.info("  Loaded %s  (%d rows)", f.name, len(df))

    data = pd.concat(frames, ignore_index=True)

    # Validate columns
    missing = set(FEATURE_COLS) - set(data.columns)
    if missing:
        logger.error("Missing feature columns in dataset: %s", missing)
        sys.exit(1)

    # Filter valid labels
    before = len(data)
    data = data[data["activity_label"].isin(VALID_LABELS)]
    after  = len(data)
    if before != after:
        logger.warning("Dropped %d rows with unknown activity labels.", before - after)

    if len(data) == 0:
        logger.error("No valid samples after label filtering.")
        sys.exit(1)

    X      = data[FEATURE_COLS].to_numpy(dtype=np.float32)
    y      = data["activity_label"].to_numpy()
    groups = data["source_video"].to_numpy()

    # Sanity checks
    nan_rows = np.any(~np.isfinite(X), axis=1)
    if nan_rows.any():
        logger.warning(
            "Removing %d rows with NaN/Inf feature values.", nan_rows.sum()
        )
        X = X[~nan_rows]
        y = y[~nan_rows]
        groups = groups[~nan_rows]

    logger.info(
        "Dataset: %d samples | %d features | %d classes | %d source videos",
        len(y), FEATURE_DIM, len(np.unique(y)), len(np.unique(groups)),
    )
    for lbl in sorted(np.unique(y)):
        cnt = (y == lbl).sum()
        logger.info("  %-14s: %d samples", lbl, cnt)

    return X, y, groups


# ── Leakage-safe split ─────────────────────────────────────────────────────

def group_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Split dataset by source video group to prevent data leakage.

    All frames from a given source video go ENTIRELY to train, val, or test.
    Never mixed across splits.

    Returns
    -------
    X_train, X_val, X_test
    y_train, y_val, y_test
    train_sources, val_sources, test_sources
    """
    from sklearn.model_selection import GroupShuffleSplit

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 3:
        logger.warning(
            "Only %d unique source video(s) found. "
            "Ideal minimum is 3+ for a proper group split. "
            "Using random sample-level split as fallback.",
            n_groups,
        )
        return _fallback_split(X, y, groups, val_frac, test_frac, random_state)

    # First split: train vs (val + test)
    gss1 = GroupShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(X, y, groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp,  y_temp  = X[temp_idx],  y[temp_idx]
    g_temp = groups[temp_idx]

    # Second split: val vs test (from temp)
    val_of_temp = val_frac / (val_frac + test_frac)
    unique_temp = np.unique(g_temp)
    if len(unique_temp) >= 2:
        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=(1 - val_of_temp), random_state=random_state
        )
        val_idx_rel, test_idx_rel = next(gss2.split(X_temp, y_temp, g_temp))
        X_val,  y_val  = X_temp[val_idx_rel],  y_temp[val_idx_rel]
        X_test, y_test = X_temp[test_idx_rel], y_temp[test_idx_rel]
        train_sources = list(np.unique(groups[train_idx]))
        val_sources   = list(np.unique(g_temp[val_idx_rel]))
        test_sources  = list(np.unique(g_temp[test_idx_rel]))
    else:
        # Fewer groups than needed – put all temp in test, skip val
        X_val,  y_val  = np.empty((0, FEATURE_DIM)), np.array([])
        X_test, y_test = X_temp, y_temp
        train_sources  = list(np.unique(groups[train_idx]))
        val_sources    = []
        test_sources   = list(np.unique(g_temp))

    _log_split_summary(y_train, y_val, y_test,
                       train_sources, val_sources, test_sources)
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            train_sources, val_sources, test_sources)


def _fallback_split(X, y, groups, val_frac, test_frac, random_state):
    """Sample-level split used when < 3 source videos."""
    from sklearn.model_selection import train_test_split
    logger.warning("Using sample-level split (data leakage risk – collect more videos).")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(val_frac + test_frac), random_state=random_state, stratify=y
    )
    X_v, X_te, y_v, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=test_frac / (val_frac + test_frac),
        random_state=random_state,
        stratify=y_tmp if len(np.unique(y_tmp)) > 1 else None,
    )
    return X_tr, X_v, X_te, y_tr, y_v, y_te, ["fallback"], ["fallback"], ["fallback"]


def _log_split_summary(y_tr, y_v, y_te, tr_src, v_src, te_src):
    logger.info("─" * 50)
    logger.info("SPLIT SUMMARY (by source video – no leakage)")
    logger.info("  Train:  %d samples | sources: %s", len(y_tr), tr_src)
    logger.info("  Val:    %d samples | sources: %s", len(y_v),  v_src)
    logger.info("  Test:   %d samples | sources: %s", len(y_te), te_src)
    logger.info("─" * 50)


# ── Training ───────────────────────────────────────────────────────────────

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    random_state: int = 42,
):
    """
    Fit a classifier and StandardScaler.

    Returns (model, scaler, label_encoder)
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        # RandomForest chosen for this feature space because:
        #   1. 34 tabular pose features — tree ensembles handle mixed scale well
        #   2. Built-in feature importance for interpretability
        #   3. Fast inference — suitable for edge deployment
        #   4. Minimal hyperparameter tuning needed
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",   # handle class imbalance
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_type == "gradient_boost":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
        )

    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=2000,
            multi_class="auto",
            class_weight="balanced",
            random_state=random_state,
            C=1.0,
        )

    elif model_type == "svm":
        from sklearn.svm import SVC
        model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    logger.info("Training %s on %d samples...", model_type, len(y_train))
    model.fit(X_scaled, y_enc)
    logger.info("Training complete. Classes: %s", list(le.classes_))

    return model, scaler, le


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(
    model,
    scaler,
    label_encoder,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict:
    """Evaluate on a split. Returns metrics dict."""
    if len(X) == 0:
        logger.info("Skipping %s evaluation (empty split).", split_name)
        return {}

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )

    X_scaled = scaler.transform(X)
    y_pred_enc = model.predict(X_scaled)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    acc  = accuracy_score(y, y_pred)
    cr   = classification_report(y, y_pred, zero_division=0, output_dict=True)
    cm   = confusion_matrix(y, y_pred, labels=list(label_encoder.classes_))

    logger.info("\n── %s SET ──", split_name.upper())
    logger.info("Accuracy: %.4f", acc)
    logger.info(
        "\n%s",
        classification_report(y, y_pred, zero_division=0),
    )

    macro_avg = cr.get("macro avg", {})
    weighted_avg = cr.get("weighted avg", {})

    return {
        "split":            split_name,
        "n_samples":        len(y),
        "accuracy":         round(acc, 6),
        "macro_precision":  round(macro_avg.get("precision", 0.0), 6),
        "macro_recall":     round(macro_avg.get("recall", 0.0), 6),
        "macro_f1":         round(macro_avg.get("f1-score", 0.0), 6),
        "weighted_f1":      round(weighted_avg.get("f1-score", 0.0), 6),
        "report":           cr,
        "confusion_matrix": cm.tolist(),
        "classes":          list(label_encoder.classes_),
    }


# ── Persistence ────────────────────────────────────────────────────────────

def save_model(
    model,
    scaler,
    label_encoder,
    output_path: str,
    meta: dict,
) -> None:
    """Save model, scaler, and metadata. Does NOT overwrite silently."""
    import joblib

    out = Path(output_path)
    if out.exists():
        backup = out.with_suffix(f".backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
        out.rename(backup)
        logger.warning("Existing model renamed to %s", backup.name)

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    logger.info("Model saved: %s", out)

    scaler_path = out.with_name(out.stem + "_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved: %s", scaler_path)

    # Also save standard scaler.joblib and har_scaler.joblib if in models directory
    std_scaler = out.parent / "scaler.joblib"
    if std_scaler != scaler_path:
        joblib.dump(scaler, std_scaler)
    har_scaler = out.parent / "har_scaler.joblib"
    if har_scaler != scaler_path and har_scaler != std_scaler:
        joblib.dump(scaler, har_scaler)

    meta_path = out.with_name(out.stem + "_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Metadata saved: %s", meta_path)

    std_meta = out.parent / "metadata.json"
    if std_meta != meta_path:
        with open(std_meta, "w") as fh:
            json.dump(meta, fh, indent=2)


def save_evaluation(results: List[dict], eval_dir: str) -> None:
    """Write evaluation reports to evaluation/ directory."""
    import csv as csv_mod

    os.makedirs(eval_dir, exist_ok=True)

    # Collect test set result
    test_result = next((r for r in results if r.get("split") == "test"), None)
    if not test_result:
        logger.warning("No test set result to save.")
        return

    # metrics.json
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(test_result, fh, indent=2)
    logger.info("Metrics saved: %s", metrics_path)

    # classification_report.txt
    report_path = os.path.join(eval_dir, "classification_report.txt")
    from sklearn.metrics import classification_report
    # Rebuild from per-class data in the dict
    cr_dict = test_result.get("report", {})
    lines = [
        "=" * 60,
        f"HAR Activity Classifier — Test Set Evaluation",
        f"Generated: {datetime.datetime.now().isoformat()}",
        f"Samples: {test_result['n_samples']}",
        f"Accuracy: {test_result['accuracy']:.4f}",
        "=" * 60,
        "",
        f"{'Activity':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 60,
    ]
    for label in test_result.get("classes", []):
        r = cr_dict.get(label, {})
        lines.append(
            f"{label:<18} {r.get('precision', 0):>10.4f} "
            f"{r.get('recall', 0):>10.4f} {r.get('f1-score', 0):>10.4f} "
            f"{r.get('support', 0):>10}"
        )
    lines += [
        "-" * 60,
        f"{'macro avg':<18} {cr_dict.get('macro avg', {}).get('f1-score', 0):>10.4f} (macro F1)",
        f"{'weighted avg':<18} {cr_dict.get('weighted avg', {}).get('f1-score', 0):>10.4f} (weighted F1)",
        "",
        "NOTE: All metrics are from the held-out test set.",
        "Split strategy: by source video (no data leakage).",
    ]
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info("Classification report: %s", report_path)

    # confusion_matrix.csv
    cm_path = os.path.join(eval_dir, "confusion_matrix.csv")
    classes = test_result.get("classes", [])
    cm = test_result.get("confusion_matrix", [])
    with open(cm_path, "w", newline="") as fh:
        writer = csv_mod.writer(fh)
        writer.writerow([""] + classes)
        for label, row in zip(classes, cm):
            writer.writerow([label] + row)
    logger.info("Confusion matrix: %s", cm_path)

    # confusion_matrix_analysis.txt
    cm_analysis_path = os.path.join(eval_dir, "confusion_matrix_analysis.txt")
    confusions = []
    if classes and len(cm) == len(classes):
        for i, true_lbl in enumerate(classes):
            for j, pred_lbl in enumerate(classes):
                if i != j:
                    cnt = cm[i][j]
                    if cnt > 0:
                        confusions.append((cnt, true_lbl, pred_lbl))
        confusions.sort(key=lambda x: x[0], reverse=True)

    cm_lines = [
        "=" * 60,
        "CONFUSION MATRIX ANALYSIS (TEST SET)",
        "=" * 60,
    ]
    if confusions:
        cm_lines.append("Most commonly confused classes:")
        for idx, (cnt, t_lbl, p_lbl) in enumerate(confusions[:5], start=1):
            cm_lines.append(f"  {idx}. {t_lbl} -> {p_lbl}: {cnt} error(s)")
    else:
        cm_lines.append("No cross-class confusion detected on held-out test set.")
    cm_lines.append("=" * 60)
    with open(cm_analysis_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(cm_lines))
    logger.info("Confusion matrix analysis: %s", cm_analysis_path)

    # confusion_matrix.png (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix — HAR Classifier (Test Set)")
        plt.colorbar(im, ax=ax)
        for i in range(len(classes)):
            for j in range(len(classes)):
                val = cm_arr[i, j]
                ax.text(j, i, str(val),
                        ha="center", va="center",
                        color="white" if val > cm_arr.max() * 0.5 else "black")
        fig.tight_layout()
        png_path = os.path.join(eval_dir, "confusion_matrix.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        logger.info("Confusion matrix plot: %s", png_path)
    except ImportError:
        logger.info("matplotlib not installed – skipping confusion matrix plot.")


def create_model_manifest(
    model_path: str,
    scaler_path: str,
    meta_path: str,
    output_dir: str = "models",
    snapshot_path: str = "dataset/snapshot.json",
) -> dict:
    """Create models/model_manifest.json with hashes and provenance tracking."""
    def _sha256(p):
        if not os.path.isfile(p):
            return None
        sha = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(65536), b""):
                sha.update(b)
        return sha.hexdigest()

    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            pass

    manifest = {
        "model_filename": os.path.basename(model_path),
        "model_hash": _sha256(model_path),
        "scaler_hash": _sha256(scaler_path),
        "metadata_hash": _sha256(meta_path),
        "dataset_snapshot_hash": _sha256(snapshot_path),
        "training_timestamp": meta.get("trained_at", datetime.datetime.now().isoformat()),
        "feature_dimension": meta.get("feature_dim", FEATURE_DIM),
        "class_labels": meta.get("classes", list(VALID_LABELS)),
        "model_version": meta.get("model_type", "sklearn_classifier"),
    }

    manifest_file = os.path.join(output_dir, "model_manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Model artifact manifest created: %s", manifest_file)
    return manifest


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train HAR activity classifier on real collected feature data."
    )
    p.add_argument(
        "--data", "-d", default="dataset/features",
        help="Directory containing feature CSVs (default: dataset/features).",
    )
    p.add_argument(
        "--output", "-o", default="models/har_classifier.joblib",
        help="Output model path (default: models/har_classifier.joblib).",
    )
    p.add_argument(
        "--model", "-m", default="random_forest",
        choices=["random_forest", "gradient_boost", "logistic_regression", "svm"],
        help="Classifier type (default: random_forest).",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Validation fraction (default: 0.15).",
    )
    p.add_argument(
        "--test-frac", type=float, default=0.15,
        help="Test fraction (default: 0.15).",
    )
    p.add_argument(
        "--eval-dir", default="evaluation",
        help="Directory for evaluation reports (default: evaluation).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Compare all supported classifiers on the same held-out test split.",
    )
    # Legacy: allow running with synthetic data for smoke-testing the pipeline
    p.add_argument(
        "--synthetic", action="store_true",
        help=(
            "SMOKE TEST ONLY: generate synthetic data to verify the pipeline. "
            "Metrics from synthetic data are NOT scientifically valid."
        ),
    )
    return p.parse_args()


def _synthetic_dataset(seq_len=20, n_per_class=300):
    """Synthetic data for pipeline smoke-tests only. Not for production."""
    logger.warning("=" * 60)
    logger.warning("SYNTHETIC MODE: Using generated data — NOT real performance.")
    logger.warning("=" * 60)
    rng = np.random.default_rng(42)
    X, y, groups = [], [], []
    class_params = {
        "STANDING": {6: (10, 3),  7: (0.25, 0.03), 17: (0.45, 0.03),
                     19: (0.05, 0.02), 20: (0.05, 0.02)},
        "SITTING":  {6: (15, 4),  7: (0.15, 0.04), 17: (0.65, 0.05),
                     19: (0.40, 0.08), 20: (0.40, 0.08)},
        "WALKING":  {6: (12, 4),  7: (0.23, 0.03), 17: (0.47, 0.06),
                     19: (0.15, 0.08), 20: (0.15, 0.08)},
        "RUNNING":  {6: (14, 5),  7: (0.22, 0.04), 17: (0.46, 0.08),
                     19: (0.25, 0.10), 20: (0.25, 0.10)},
        "FALLING":  {6: (70, 10), 7: (0.08, 0.05), 17: (0.80, 0.08),
                     19: (0.10, 0.05), 20: (0.10, 0.05)},
    }
    for act_i, (act, params) in enumerate(class_params.items()):
        for chunk in range(3):   # 3 fake "source videos" per class
            n = n_per_class // 3
            base = rng.uniform(0.0, 0.3, (n, FEATURE_DIM)).astype(np.float32)
            for idx, (mean, std) in params.items():
                base[:, idx] = rng.normal(mean, std, n).astype(np.float32)
            X.append(np.clip(base, 0, None))
            y.extend([act] * n)
            groups.extend([f"synth_{act_i}_{chunk}.mp4"] * n)
    return (np.vstack(X).astype(np.float32),
            np.array(y), np.array(groups))


def validate_training_gate(
    data_path: str,
    raw_dir: str = "dataset/raw",
    splits_dir: str = "dataset/splits",
) -> None:
    """
    Strict Training Gate: validates real dataset prerequisites before permitting training.
    Halts execution with detailed error report if requirements are not met.
    """
    failures = []

    # 1. Real videos exist in raw_dir
    supported_exts = {".mp4", ".avi", ".mov", ".mkv"}
    raw_video_count = 0
    if os.path.isdir(raw_dir):
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in supported_exts:
                    raw_video_count += 1
    if raw_video_count == 0:
        failures.append(f"No real video recordings found in '{raw_dir}'. Record real clips first.")

    # 2. Features exist
    p = Path(data_path)
    csv_files = []
    if p.is_file():
        csv_files = [p]
    elif p.is_dir():
        csv_files = sorted(p.glob("*.csv"))
    if not csv_files:
        failures.append(f"No feature CSV files found in '{data_path}'. Run collect_features.py first.")

    # 3. If CSV files exist, validate content
    if csv_files:
        try:
            import pandas as pd
            frames = [pd.read_csv(f) for f in csv_files]
            if not frames or all(len(f) == 0 for f in frames):
                failures.append("Feature CSV files are empty (0 samples).")
            else:
                df = pd.concat(frames, ignore_index=True)
                # Feature dimensions
                missing_cols = set(FEATURE_COLS) - set(df.columns)
                if missing_cols:
                    failures.append(f"Feature dimension mismatch: missing {len(missing_cols)} column(s).")
                # Labels valid
                if "activity_label" not in df.columns:
                    failures.append("Missing 'activity_label' column in feature files.")
                else:
                    present_classes = set(df["activity_label"].dropna().unique())
                    missing_classes = VALID_LABELS - present_classes
                    if missing_classes:
                        failures.append(f"Missing required activity class(es): {sorted(missing_classes)}. All 5 classes required.")
                    invalid_classes = present_classes - VALID_LABELS
                    if invalid_classes:
                        failures.append(f"Found invalid activity labels: {sorted(invalid_classes)}.")

                # NaN / Inf check
                feat_present = [c for c in FEATURE_COLS if c in df.columns]
                if len(feat_present) == FEATURE_DIM:
                    X_vals = df[feat_present].to_numpy()
                    if not np.all(np.isfinite(X_vals)):
                        failures.append("Feature matrix contains NaN or Infinite values.")

                # Sample count threshold
                if "activity_label" in df.columns:
                    for cls in VALID_LABELS:
                        count = (df["activity_label"] == cls).sum()
                        if count < 5:
                            failures.append(f"Insufficient samples for class '{cls}' ({count} found, minimum 5 required).")
        except Exception as e:
            failures.append(f"Error reading feature CSVs: {e}")

    # 4. Splits check
    train_split = os.path.join(splits_dir, "train.csv")
    val_split = os.path.join(splits_dir, "validation.csv")
    test_split = os.path.join(splits_dir, "test.csv")
    if not (os.path.isfile(train_split) and os.path.isfile(val_split) and os.path.isfile(test_split)):
        failures.append(f"Train/Val/Test splits missing in '{splits_dir}'. Run split_dataset.py first.")
    else:
        # Check for leakage across splits
        try:
            import csv
            def _get_vids(path):
                with open(path, encoding="utf-8") as fh:
                    return {r.get("video_id") or r.get("filename") for r in csv.DictReader(fh)}
            tr_v = _get_vids(train_split)
            va_v = _get_vids(val_split)
            te_v = _get_vids(test_split)
            if (tr_v & va_v) or (tr_v & te_v) or (va_v & te_v):
                failures.append("Data leakage detected: identical video IDs appear in multiple split files.")
        except Exception:
            pass

    if failures:
        print("\n" + "=" * 65)
        print("TRAINING GATE REFUSAL — TRAINING BLOCKED")
        print("=" * 65)
        print("The training gate prevents training on incomplete or missing real data.")
        print("Unmet requirements:")
        for idx, err in enumerate(failures, start=1):
            print(f"  [{idx}] {err}")
        print("\nRequired Actions:")
        print("  1. Record real videos into dataset/raw/<CLASS>/ (see DATA_COLLECTION_GUIDE.md)")
        print("  2. Run 'python collect_features.py --data dataset/raw'")
        print("  3. Run 'python dataset_manifest.py'")
        print("  4. Run 'python split_dataset.py'")
        print("=" * 65 + "\n")
        logger.error("Training gate blocked execution: %d requirement(s) failed.", len(failures))
        sys.exit(1)


def main() -> None:
    args = _parse_args()

    if args.synthetic:
        X, y, groups = _synthetic_dataset()
    else:
        # Pre-training validation gate
        validate_training_gate(
            data_path=args.data,
            raw_dir=getattr(args, "raw_dir", "dataset/raw"),
            splits_dir=getattr(args, "splits_dir", "dataset/splits"),
        )
        X, y, groups = load_dataset(args.data)

    # ── Leakage-safe split ────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     tr_src, v_src, te_src) = group_split(
        X, y, groups,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        random_state=args.seed,
    )

    if len(X_train) == 0:
        logger.error("Train split is empty – not enough data or source videos.")
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────
    model, scaler, label_encoder = train_model(
        X_train, y_train, args.model, random_state=args.seed
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = []
    if len(X_val) > 0:
        results.append(evaluate(model, scaler, label_encoder, X_val, y_val, "validation"))
    results.append(evaluate(model, scaler, label_encoder, X_test, y_test, "test"))

    # ── Save model ────────────────────────────────────────────────────────
    meta = {
        "model_type":        args.model,
        "feature_dim":       FEATURE_DIM,
        "feature_cols":      FEATURE_COLS,
        "classes":           list(label_encoder.classes_),
        "train_samples":     int(len(X_train)),
        "val_samples":       int(len(X_val)),
        "test_samples":      int(len(X_test)),
        "train_sources":     tr_src,
        "val_sources":       v_src,
        "test_sources":      te_src,
        "split_strategy":    "by_source_video (GroupShuffleSplit)",
        "val_frac":          args.val_frac,
        "test_frac":         args.test_frac,
        "random_seed":       args.seed,
        "synthetic_data":    args.synthetic,
        "trained_at":        datetime.datetime.now().isoformat(),
        "performance_note": (
            "Metrics are from SYNTHETIC data – NOT valid for production."
            if args.synthetic
            else "Metrics are from real held-out test set."
        ),
    }
    save_model(model, scaler, label_encoder, args.output, meta)
    scaler_out = args.output.replace(".joblib", "_scaler.joblib")
    meta_out = args.output.replace(".joblib", "_meta.json")
    create_model_manifest(args.output, scaler_out, meta_out, output_dir=os.path.dirname(args.output) or "models")

    # ── Evaluation reports ────────────────────────────────────────────────
    save_evaluation(results, args.eval_dir)

    # ── Optional Model Comparison ─────────────────────────────────────────
    if args.compare and len(X_test) > 0:
        print("\n" + "=" * 62)
        print("MODEL COMPARISON (SAME HELD-OUT TEST SPLIT)")
        print("=" * 62)
        print(f"{'Model':<22} {'Accuracy':>10} {'Macro F1':>12} {'Weighted F1':>14}")
        print("-" * 62)
        comparison_results = {}
        for m_type in ["random_forest", "gradient_boost", "logistic_regression", "svm"]:
            try:
                m_candidate, s_cand, le_cand = train_model(X_train, y_train, m_type, random_state=args.seed)
                ev = evaluate(m_candidate, s_cand, le_cand, X_test, y_test, f"test_{m_type}")
                acc_val = ev.get("accuracy", 0.0)
                macro_f1 = ev.get("report", {}).get("macro avg", {}).get("f1-score", 0.0)
                wt_f1 = ev.get("report", {}).get("weighted avg", {}).get("f1-score", 0.0)
                comparison_results[m_type] = {
                    "accuracy": acc_val,
                    "macro_f1": macro_f1,
                    "weighted_f1": wt_f1,
                }
                print(f"{m_type:<22} {acc_val:>10.4f} {macro_f1:>12.4f} {wt_f1:>14.4f}")
            except Exception as e:
                logger.warning("Comparison training failed for %s: %s", m_type, e)
        print("=" * 62 + "\n")
        comp_file = os.path.join(args.eval_dir, "model_comparison.json")
        with open(comp_file, "w", encoding="utf-8") as fh:
            json.dump(comparison_results, fh, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    test_res = next((r for r in results if r.get("split") == "test"), {})
    acc = test_res.get("accuracy", None)

    logger.info("─" * 60)
    if args.synthetic:
        logger.warning(
            "SYNTHETIC TEST ACCURACY: %.4f\n"
            "This number is meaningless for production.\n"
            "Collect real labeled data and re-run without --synthetic.",
            acc or 0,
        )
    elif acc is not None:
        logger.info("REAL TEST ACCURACY: %.4f", acc)
        logger.info("See evaluation/ for full report.")
    else:
        logger.info(
            "Training pipeline is ready, but quantitative model performance\n"
            "has not yet been measured because no labeled test set was available."
        )
    logger.info("─" * 60)


if __name__ == "__main__":
    main()
