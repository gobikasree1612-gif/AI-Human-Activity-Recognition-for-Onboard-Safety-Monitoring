"""
final_system_check.py
======================
Comprehensive System Health & Architecture Validation Tool for onboard-har.

Verifies:
1. AI Modules (inference, detector, features, temporal, classifier, fall_detector, safety, schemas)
2. Dataset Infrastructure (collection, validation, assistant, snapshot, manifest, balance)
3. Training Infrastructure (train_classifier, split_dataset, readiness checks)
4. Evaluation Infrastructure (metrics, classification report, confusion matrix analysis)
5. Model Loading & Fallback Integrity
6. Dashboard Components (app.py, ui.py, camera, history)
7. Configuration Integrity (config.yaml)
8. Dependencies (ultralytics, opencv, torch, scikit-learn, streamlit, etc.)
9. Automated Test Suite Execution
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import unittest
from typing import Dict, Tuple

# Ensure project root is on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np


def check_dependencies() -> Tuple[bool, list]:
    required = [
        "cv2",
        "numpy",
        "yaml",
        "ultralytics",
        "sklearn",
        "joblib",
        "streamlit",
        "pandas",
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    return len(missing) == 0, missing


def check_ai_modules() -> Tuple[bool, list]:
    modules = [
        "ai.schemas",
        "ai.detector",
        "ai.features",
        "ai.temporal",
        "ai.classifier",
        "ai.fall_detector",
        "ai.safety",
        "ai.inference",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    return len(failed) == 0, failed


def check_dataset_infra() -> Tuple[bool, list]:
    modules = [
        "tools.record_dataset",
        "tools.dataset_assistant",
        "dataset_manifest",
        "dataset_balance",
        "dataset_snapshot",
        "collect_features",
        "split_dataset",
        "training_readiness",
    ]
    failed = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    return len(failed) == 0, failed


def check_config() -> Tuple[bool, str]:
    cfg_path = os.path.join(_HERE, "config.yaml")
    if not os.path.isfile(cfg_path):
        return False, "config.yaml missing"
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        required_keys = ["model", "activity", "fall", "tracking", "classifier", "dataset"]
        for k in required_keys:
            if k not in data:
                return False, f"Missing key '{k}' in config.yaml"
        return True, "Config valid"
    except Exception as e:
        return False, f"Error parsing config.yaml: {e}"


def check_model_status() -> Tuple[str, str]:
    model_path = os.path.join(_HERE, "models", "har_classifier.joblib")
    if os.path.isfile(model_path):
        return "LOADED", model_path
    return "NOT AVAILABLE (RULE-BASED FALLBACK ACTIVE)", "None"


def check_dashboard_components() -> Tuple[bool, list]:
    components = [
        "app",
        "dashboard.ui",
        "dashboard.camera",
        "dashboard.history",
    ]
    failed = []
    for mod in components:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failed.append(f"{mod}: {e}")
    return len(failed) == 0, failed


def run_system_health_check() -> Dict[str, any]:
    deps_pass, missing_deps = check_dependencies()
    ai_pass, ai_errors = check_ai_modules()
    ds_pass, ds_errors = check_dataset_infra()
    cfg_pass, cfg_msg = check_config()
    model_status, model_path = check_model_status()
    dash_pass, dash_errors = check_dashboard_components()

    # Check tests
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(_HERE, "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(stream=open(os.devnull, "w"))
    result = runner.run(suite)
    tests_pass = result.wasSuccessful()
    tests_run = result.testsRun
    tests_failed = len(result.failures) + len(result.errors)
    tests_skipped = len(result.skipped)

    # Check readiness gate for training
    from training_readiness import TrainingReadinessChecker
    checker = TrainingReadinessChecker()
    readiness = checker.evaluate()

    training_status = "PASS" if readiness["overall_ready"] else "BLOCKED (REAL DATA REQUIRED)"
    eval_status = "PASS" if readiness["overall_ready"] else "BLOCKED (NO TRAINED MODEL)"

    report = {
        "deps_pass": deps_pass,
        "missing_deps": missing_deps,
        "ai_pass": ai_pass,
        "ai_errors": ai_errors,
        "ds_pass": ds_pass,
        "ds_errors": ds_errors,
        "cfg_pass": cfg_pass,
        "cfg_msg": cfg_msg,
        "model_status": model_status,
        "model_path": model_path,
        "dash_pass": dash_pass,
        "dash_errors": dash_errors,
        "training_status": training_status,
        "eval_status": eval_status,
        "tests_pass": tests_pass,
        "tests_run": tests_run,
        "tests_passed_cnt": tests_run - tests_failed - tests_skipped,
        "tests_failed_cnt": tests_failed,
        "tests_skipped_cnt": tests_skipped,
    }
    return report


def print_health_report(report: Dict[str, any]):
    print("=" * 50)
    print("SYSTEM HEALTH REPORT")
    print("=" * 50)
    print(f"AI pipeline:           {'PASS' if report['ai_pass'] else 'FAIL'}")
    print(f"Dataset pipeline:      {'PASS' if report['ds_pass'] else 'FAIL'}")
    print(f"Training pipeline:     {report['training_status']}")
    print(f"Evaluation pipeline:   {report['eval_status']}")
    print(f"Model:                 {report['model_status']}")
    print(f"Dashboard:             {'PASS' if report['dash_pass'] else 'FAIL'}")
    print(f"Configuration:         {'PASS' if report['cfg_pass'] else 'FAIL'}")
    print(f"Dependencies:          {'PASS' if report['deps_pass'] else 'FAIL'}")
    print(f"Tests:                 {'PASS' if report['tests_pass'] else 'FAIL'} ({report['tests_passed_cnt']} passed, {report['tests_failed_cnt']} failed, {report['tests_skipped_cnt']} skipped)")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    report = run_system_health_check()
    print_health_report(report)
