"""
CrowdPulse: Continuous Regression & Risk Tier Classification Evaluation (src/evaluate.py)
Evaluates continuous crowd prediction accuracy (MAE, RMSE, R2) and
evaluates operational safety by discretizing into 3 risk tiers:
  - Green (<10): Low risk / free flow
  - Yellow (10-24): Moderate risk / monitoring required
  - Red (>=25): High risk / bottleneck formation
Generates multi-class precision, recall, macro F1-score, and confusion matrix.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.data_loader import load_processed_data
from src.feature_extractor import FEATURE_COLUMNS

TIER_LABELS: List[str] = ["Green", "Yellow", "Red"]


def discretize_counts_to_tiers(
    counts: Union[np.ndarray, List[float]],
    green_cutoff: float = 10.0,
    red_cutoff: float = 25.0,
) -> np.ndarray:
    """
    Categorizes crowd counts into three operational safety tiers:
      Green:  count < green_cutoff (default 10)
      Yellow: green_cutoff <= count < red_cutoff (default 10 to 24)
      Red:    count >= red_cutoff (default 25)

    Args:
        counts: 1D array of crowd counts.
        green_cutoff: Upper threshold for normal green status.
        red_cutoff: Lower threshold for red bottleneck status.

    Returns:
        np.ndarray of strings: ['Green', 'Yellow', 'Red', ...]
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    tiers = np.empty(len(counts_arr), dtype=object)
    tiers[counts_arr < green_cutoff] = "Green"
    tiers[(counts_arr >= green_cutoff) & (counts_arr < red_cutoff)] = "Yellow"
    tiers[counts_arr >= red_cutoff] = "Red"
    return tiers


def evaluate_continuous_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates standard continuous regression metrics.
    Clipped to 0 since human counts cannot be negative.
    """
    clipped = np.clip(y_pred, 0, None)
    mae = float(mean_absolute_error(y_true, clipped))
    rmse = float(root_mean_squared_error(y_true, clipped))
    r2 = float(r2_score(y_true, clipped))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_classification_metrics(
    y_true_tiers: np.ndarray,
    y_pred_tiers: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculates multi-class classification metrics:
    Precision, Recall, F1-score (macro and weighted), and Confusion Matrix.
    """
    if labels is None:
        labels = TIER_LABELS

    report_dict = classification_report(
        y_true_tiers,
        y_pred_tiers,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true_tiers, y_pred_tiers, labels=labels)

    return {
        "report_dict": report_dict,
        "confusion_matrix": conf_mat.tolist(),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "accuracy": float(report_dict["accuracy"]),
    }


def run_full_evaluation(
    model_path: Union[str, Path] = "models/crowdpulse_pipeline.joblib",
    test_data_path: Optional[Union[str, Path]] = None,
    green_cutoff: float = 10.0,
    red_cutoff: float = 25.0,
) -> Dict[str, Any]:
    """
    Loads serialized model and test data, produces both continuous and categorical evaluations.
    """
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found at {m_path}. Train model first.")

    pipeline = joblib.load(m_path)

    if test_data_path is not None:
        test_df = pd.read_csv(test_data_path)
    else:
        _, test_df = load_processed_data()

    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["head_count"].values

    raw_preds = pipeline.predict(X_test)
    y_pred = np.clip(raw_preds, 0, None)

    # 1. Regression Metrics
    reg_metrics = evaluate_continuous_metrics(y_test, y_pred)

    # 2. Risk Tier Classification
    y_true_tiers = discretize_counts_to_tiers(y_test, green_cutoff, red_cutoff)
    y_pred_tiers = discretize_counts_to_tiers(y_pred, green_cutoff, red_cutoff)

    clf_metrics = evaluate_classification_metrics(y_true_tiers, y_pred_tiers)

    # Print Formatted Report
    print("=" * 65)
    print("CROWDPULSE EVALUATION REPORT")
    print("=" * 65)
    print("\n[1] CONTINUOUS REGRESSION METRICS:")
    print(f"    - Mean Absolute Error (MAE):     {reg_metrics['mae']:.3f} heads/zone")
    print(f"    - Root Mean Squared Error (RMSE): {reg_metrics['rmse']:.3f} heads/zone")
    print(f"    - Coefficient of Determination (R2): {reg_metrics['r2']:.3f}")

    print("\n[2] OPERATIONAL RISK TIER CLASSIFICATION:")
    print(f"    - Thresholds: Green < {green_cutoff} | Yellow {green_cutoff}-{red_cutoff-1} | Red >= {red_cutoff}")
    print(f"    - Overall Accuracy: {clf_metrics['accuracy'] * 100:.2f}%")
    print(f"    - Macro F1-Score:   {clf_metrics['macro_f1']:.3f}")
    print(f"    - Weighted F1-Score: {clf_metrics['weighted_f1']:.3f}")

    print("\n[3] PER-CLASS CLASSIFICATION DETAILS:")
    rep_text = classification_report(y_true_tiers, y_pred_tiers, labels=TIER_LABELS, digits=3, zero_division=0)
    print(rep_text)

    print("[4] CONFUSION MATRIX (Rows: True, Columns: Predicted):")
    conf_mat_arr = np.array(clf_metrics["confusion_matrix"])
    header = f"{'':>10}" + "".join([f"{lbl:>10}" for lbl in TIER_LABELS])
    print(header)
    for idx, row in enumerate(conf_mat_arr):
        row_str = f"{TIER_LABELS[idx]:>10}" + "".join([f"{val:>10}" for val in row])
        print(row_str)
    print("=" * 65)

    return {
        "regression": reg_metrics,
        "classification": clf_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CrowdPulse model performance.")
    parser.add_argument("--model-path", type=str, default="models/crowdpulse_pipeline.joblib")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--green-cutoff", type=float, default=10.0)
    parser.add_argument("--red-cutoff", type=float, default=25.0)
    args = parser.parse_args()

    run_full_evaluation(
        model_path=args.model_path,
        test_data_path=args.test_data,
        green_cutoff=args.green_cutoff,
        red_cutoff=args.red_cutoff,
    )
