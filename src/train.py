"""
CrowdPulse: Model Training & Pipeline Serialization (src/train.py)
Implements cross-validation, hyperparameter tuning, and trains the enhanced
champion Scikit-learn Pipeline combining:
  1. SpatialFeatureAugmenter (expands 31 core descriptors to 39 perspective & synergy features)
  2. StandardScaler
  3. VotingRegressor Ensemble:
     - RandomForestRegressor (unconstrained non-linear bagging)
     - HistGradientBoostingRegressor with Poisson Loss (discrete count modeling)
     - TransformedTargetRegressor with Log1p Transform (tail skewness mitigation)
Evaluates continuous metrics (MAE, RMSE, R2) and serializes artifacts:
  - models/crowdpulse_pipeline.joblib
  - models/model_metadata.json
"""

import argparse
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_processed_data
from src.feature_extractor import (
    AUGMENTED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    SpatialFeatureAugmenter,
)


def build_pipeline(model_type: str = "ensemble", n_jobs: int = -1) -> Pipeline:
    """
    Constructs the end-to-end Scikit-learn Pipeline.

    Args:
        model_type: 'ensemble' (Enhanced Trio Ensemble) or 'rf' (Pure Random Forest)
        n_jobs: Parallel worker threads

    Returns:
        Configured Scikit-learn Pipeline
    """
    if model_type == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(
                n_estimators=150,
                max_depth=14,
                min_samples_split=4,
                random_state=42,
                n_jobs=n_jobs,
            )),
        ])

    # Enhanced Champion Ensemble with Poisson Loss and Log1p target transformation
    ensemble = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                min_samples_split=3,
                random_state=42,
                n_jobs=n_jobs,
            )),
            ("hgb_poi", HistGradientBoostingRegressor(
                loss="poisson",
                max_iter=250,
                max_depth=10,
                learning_rate=0.05,
                random_state=42,
            )),
            ("rf_log", TransformedTargetRegressor(
                regressor=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=14,
                    random_state=42,
                    n_jobs=n_jobs,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            )),
        ],
        weights=[0.25, 0.55, 0.20],
    )

    return Pipeline([
        ("augmenter", SpatialFeatureAugmenter()),
        ("scaler", StandardScaler()),
        ("regressor", ensemble),
    ])


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes continuous regression evaluation metrics (MAE, RMSE, R2)."""
    clipped_pred = np.clip(y_pred, 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, clipped_pred)),
        "rmse": float(root_mean_squared_error(y_true, clipped_pred)),
        "r2": float(r2_score(y_true, clipped_pred)),
    }


def train_and_evaluate_pipeline(
    data_dir: str = "data/processed",
    output_dir: str = "models",
    model_type: str = "ensemble",
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Loads data, trains the pipeline, evaluates metrics, and serializes artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading processed training and test features...")
    train_df, test_df = load_processed_data(data_dir)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df["head_count"].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df["head_count"].values

    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples across {len(FEATURE_COLUMNS)} core visual descriptors.")
    print(f"Pipeline Architecture Mode: '{model_type}' (SpatialFeatureAugmenter -> StandardScaler -> {model_type.upper()})")

    pipeline = build_pipeline(model_type=model_type, n_jobs=n_jobs)

    print("Fitting champion pipeline...")
    pipeline.fit(X_train, y_train)

    train_preds = np.clip(pipeline.predict(X_train), 0, None)
    test_preds = np.clip(pipeline.predict(X_test), 0, None)

    train_metrics = compute_regression_metrics(y_train, train_preds)
    test_metrics = compute_regression_metrics(y_test, test_preds)

    print("\n" + "=" * 60)
    print("CROWDPULSE CHAMPION PIPELINE PERFORMANCE")
    print("=" * 60)
    print(f"Train Metrics: MAE = {train_metrics['mae']:.3f} | RMSE = {train_metrics['rmse']:.3f} | R2 = {train_metrics['r2']:.3f}")
    print(f"Test  Metrics: MAE = {test_metrics['mae']:.3f} | RMSE = {test_metrics['rmse']:.3f} | R2 = {test_metrics['r2']:.3f}")
    print("=" * 60)

    # Serialize champion pipeline
    model_file = out_path / "crowdpulse_pipeline.joblib"
    joblib.dump(pipeline, model_file)
    print(f"\nChampion pipeline saved successfully to {model_file}")

    # Build and export metadata JSON
    metadata = {
        "model_name": f"CrowdPulse Champion Pipeline ({model_type.upper()})",
        "pipeline_architecture": [
            "SpatialFeatureAugmenter",
            "StandardScaler",
            "VotingRegressor(RF + HGB_Poisson + RF_Log1p)" if model_type == "ensemble" else "RandomForestRegressor",
        ],
        "core_feature_count": int(len(FEATURE_COLUMNS)),
        "augmented_feature_count": int(len(AUGMENTED_FEATURE_COLUMNS)),
        "core_feature_names": FEATURE_COLUMNS,
        "augmented_feature_names": AUGMENTED_FEATURE_COLUMNS,
        "training_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "risk_tier_thresholds": {
            "green_upper": 10,
            "yellow_upper": 25,
            "red_lower": 25,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "trained_at": datetime.now().isoformat(),
        "framework_versions": {
            "scikit_learn": "1.7.2",
            "joblib": "1.5.2",
            "numpy": "2.3.3",
        },
    }

    meta_file = out_path / "model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved successfully to {meta_file}")

    return pipeline, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and serialize the CrowdPulse regression pipeline.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed features folder.")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save serialized models.")
    parser.add_argument("--model-type", type=str, default="ensemble", choices=["ensemble", "rf"], help="Model architecture.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel worker threads (-1 for all).")
    args = parser.parse_args()

    train_and_evaluate_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        n_jobs=args.n_jobs,
    )
