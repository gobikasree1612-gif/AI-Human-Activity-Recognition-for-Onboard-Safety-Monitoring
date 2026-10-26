# Training Pipeline Documentation

The training system in **onboard-har** enforces strict data validation gates to prevent training on missing, corrupted, or synthetic data.

## Training Workflow

1. **Verify Training Gate:**
   ```bash
   python training_readiness.py
   ```
   If dataset readiness requirements are unmet, execution halts with detailed remediation instructions.

2. **Feature Extraction:**
   ```bash
   python collect_features.py --data dataset/raw
   ```

3. **Leakage-Safe Group Shuffle Split:**
   ```bash
   python split_dataset.py
   ```
   Ensures all frames from a single source video are assigned exclusively to train, validation, or test.

4. **Model Fitting & Benchmark Comparison:**
   ```bash
   python train_classifier.py --compare
   ```
   Trains candidate classifiers (RandomForest, GradientBoosting, LogisticRegression, SVM) on the identical training split and evaluates them on the held-out test split.

5. **Artifact Generation:**
   Saves:
   - `models/har_classifier.joblib`
   - `models/scaler.joblib`
   - `models/metadata.json`
   - `models/model_manifest.json` (SHA-256 artifact hashes)
