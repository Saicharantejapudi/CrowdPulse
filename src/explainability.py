"""
CrowdPulse: Model Explainability Engine (src/explainability.py)
Implements SHAP (SHapley Additive exPlanations) TreeExplainer routines:
  - Supports both pure RandomForest and VotingRegressor ensemble pipelines.
  - Generates global beeswarm summary plot -> models/shap_summary.png
  - Generates local sample waterfall plot   -> models/shap_waterfall_sample.png
  - Provides plain-text human-interpretable feature contribution summaries.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.data_loader import load_processed_data
from src.feature_extractor import (
    AUGMENTED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
)


FEATURE_DESCRIPTIONS = {
    "canny_edge_density_ratio": "High density of Canny edges indicates dense silhouette contours",
    "canny_edge_mean_gradient": "Strong edge gradients indicate distinct human body boundaries",
    "hog_gradient_mean": "Overall gradient magnitude captures head and shoulder edges",
    "hog_gradient_std": "Variation in gradient directions signals diverse human orientations",
    "lbp_texture_entropy": "High local binary pattern entropy corresponds to cluttered pedestrian textures",
    "lbp_texture_energy": "Uniform texture energy reflects smooth, non-congested surfaces",
    "lbp_bin_9_flat_region": "Proportion of flat regions signals open ground or background pavement",
    "lbp_bin_0_spot_dark": "Dark spot texture patterns correlate with human hair and heads",
    "lbp_bin_1_edge_pattern": "Linear edge patterns align with upright torso postures",
    "intensity_mean": "Mean brightness level of the surveillance region",
    "intensity_std": "Pixel intensity spread across crowded vs. uncrowded regions",
    "intensity_skewness": "Asymmetry in lighting distribution across the zone",
    "spatial_zone_row": "Vertical position in camera perspective (perspective scaling effect)",
    "spatial_zone_col": "Horizontal camera angle",
    "perspective_edge_density": "Perspective-scaled edge density (distant vs foreground crowd density)",
    "perspective_hog_gradient": "Perspective-scaled directional contour intensity",
    "perspective_lbp_entropy": "Perspective-scaled texture complexity",
    "spatial_center_dist": "Geometric distance from camera focus center",
    "edge_intensity_ratio": "Edge prominence relative to background contrast",
    "hog_edge_synergy": "Synergistic alignment between directional gradients and edge contours",
    "lbp_entropy_energy_ratio": "Texture randomness relative to uniformity",
    "intensity_spread_ratio": "Relative variation coefficient of brightness",
}


def get_tree_explainer(pipeline: Pipeline) -> Tuple[shap.TreeExplainer, Any, List[str]]:
    """
    Extracts the tree regressor and prefix transformers from the pipeline.
    Returns:
        (explainer, transform_fn, feature_names)
    """
    regressor = pipeline.named_steps.get("regressor", pipeline)

    # If VotingRegressor, extract the constituent tree model ('rf')
    if hasattr(regressor, "named_estimators_") and "rf" in regressor.named_estimators_:
        tree_model = regressor.named_estimators_["rf"]
    else:
        tree_model = regressor

    has_augmenter = "augmenter" in pipeline.named_steps
    feat_names = AUGMENTED_FEATURE_COLUMNS if has_augmenter else FEATURE_COLUMNS

    def transform_features(X_raw: np.ndarray) -> np.ndarray:
        X_curr = np.asarray(X_raw, dtype=np.float64)
        if "augmenter" in pipeline.named_steps:
            X_curr = pipeline.named_steps["augmenter"].transform(X_curr)
        if "scaler" in pipeline.named_steps:
            X_curr = pipeline.named_steps["scaler"].transform(X_curr)
        return X_curr

    explainer = shap.TreeExplainer(tree_model)
    return explainer, transform_features, feat_names


def generate_shap_artifacts(
    pipeline: Pipeline,
    X_sample: np.ndarray,
    output_dir: Union[str, Path] = "models",
    sample_index: int = 0,
    max_display: int = 15,
) -> Tuple[Path, Path]:
    """
    Generates and saves SHAP summary beeswarm and waterfall plots to disk.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    explainer, transform_fn, feat_names = get_tree_explainer(pipeline)
    X_transformed = transform_fn(X_sample)

    print(f"Calculating SHAP values for {len(X_sample)} background samples across {len(feat_names)} features...")
    explanation = explainer(X_transformed)
    explanation.feature_names = feat_names

    # 1. Global Summary Beeswarm Plot
    summary_path = out_p / "shap_summary.png"
    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(
        explanation.values,
        features=X_transformed,
        feature_names=feat_names,
        max_display=max_display,
        show=False,
    )
    plt.title("CrowdPulse: Global SHAP Feature Importance (TreeExplainer)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    print(f"Global SHAP summary plot saved to: {summary_path}")

    # 2. Local Congestion Explanation Waterfall Plot
    waterfall_path = out_p / "shap_waterfall_sample.png"
    plt.figure(figsize=(9, 6), dpi=150)
    shap.plots.waterfall(
        explanation[sample_index],
        max_display=max_display,
        show=False,
    )
    plt.title(f"CrowdPulse: Local SHAP Explanation (Sample Zone #{sample_index})", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(waterfall_path, bbox_inches="tight")
    plt.close()
    print(f"Local SHAP waterfall plot saved to: {waterfall_path}")

    return summary_path, waterfall_path


def explain_prediction_plain_text(
    pipeline: Pipeline,
    feature_vector: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Produces human-readable, plain-text explanations of top feature contributions.
    """
    feats = np.asarray(feature_vector).reshape(1, -1)
    explainer, transform_fn, feat_names = get_tree_explainer(pipeline)
    feats_trans = transform_fn(feats)

    shap_vals = explainer(feats_trans).values[0]
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]

    explanations = []
    for idx in sorted_indices:
        fname = feat_names[idx]
        val = float(shap_vals[idx])
        direction = "increases" if val >= 0 else "decreases"
        desc = FEATURE_DESCRIPTIONS.get(
            fname,
            "Spatial visual descriptor with notable impact on localized count",
        )
        explanations.append({
            "feature": fname,
            "shap_value": round(val, 3),
            "direction": direction,
            "impact_description": f"{fname} ({val:+.2f} heads): {desc}",
        })

    return explanations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP explainability artifacts.")
    parser.add_argument("--model-path", type=str, default="models/crowdpulse_pipeline.joblib")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    m_path = Path(args.model_path)
    if not m_path.exists():
        print(f"Error: Model not found at {m_path}. Run src/train.py first.")
    else:
        pipe = joblib.load(m_path)
        _, test_df = load_processed_data()
        X_sample = test_df[FEATURE_COLUMNS].values[: args.sample_size]
        congested_mask = np.where(test_df["head_count"].values[: args.sample_size] >= 20)[0]
        sample_idx = int(congested_mask[0]) if len(congested_mask) > 0 else 0

        generate_shap_artifacts(
            pipeline=pipe,
            X_sample=X_sample,
            output_dir=args.output_dir,
            sample_index=sample_idx,
        )
