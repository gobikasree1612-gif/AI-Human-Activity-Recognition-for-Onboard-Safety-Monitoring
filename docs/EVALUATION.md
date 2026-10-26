# Evaluation Framework Documentation

This document describes evaluation metrics, classification reporting, and confusion matrix analysis.

## Metrics Generated

During model evaluation, `train_classifier.py` produces:
1. **`evaluation/metrics.json`**:
   - Accuracy
   - Macro Precision, Recall, F1
   - Weighted Precision, Recall, F1
   - Per-class precision, recall, F1, and support
2. **`evaluation/classification_report.txt`**: Standard text format classification breakdown.
3. **`evaluation/confusion_matrix.csv`**: Raw counts of true vs predicted labels across all 5 classes.
4. **`evaluation/confusion_matrix.png`**: High-resolution heatmap visualization.
5. **`evaluation/confusion_matrix_analysis.txt`**: Textual summary of primary class confusion pairs.

## Held-Out Test Set Rule
Evaluation MUST use the held-out test split (`dataset/splits/test.csv`). The test set is never exposed during hyperparameter selection or threshold tuning.
