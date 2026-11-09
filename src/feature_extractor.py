"""
CrowdPulse: Feature Extraction Engine (src/feature_extractor.py)
Implements edge-compatible spatial visual descriptors:
  1. Histogram of Oriented Gradients (HOG) via skimage.feature.hog
  2. Edge density ratio & gradient magnitude via OpenCV Canny & Sobel
  3. Local Binary Patterns (LBP) texture histogram, entropy, and energy
  4. First- and second-order pixel intensity statistics (mean, std, var, skew, kurt)
  5. Spatial grid coordinate descriptors (spatial_zone_row, spatial_zone_col)
  6. SpatialFeatureAugmenter: Scikit-learn Transformer expanding to 39 perspective & synergy features
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import scipy.stats as stats
from skimage.feature import hog, local_binary_pattern
from sklearn.base import BaseEstimator, TransformerMixin


FEATURE_COLUMNS: List[str] = [
    "hog_bin_0_deg0_22",
    "hog_bin_1_deg22_45",
    "hog_bin_2_deg45_67",
    "hog_bin_3_deg67_90",
    "hog_bin_4_deg90_112",
    "hog_bin_5_deg112_135",
    "hog_bin_6_deg135_157",
    "hog_bin_7_deg157_180",
    "hog_gradient_mean",
    "hog_gradient_std",
    "canny_edge_density_ratio",
    "canny_edge_mean_gradient",
    "lbp_bin_0_spot_dark",
    "lbp_bin_1_edge_pattern",
    "lbp_bin_2_edge_pattern",
    "lbp_bin_3_corner_pattern",
    "lbp_bin_4_corner_pattern",
    "lbp_bin_5_corner_pattern",
    "lbp_bin_6_corner_pattern",
    "lbp_bin_7_texture_transition",
    "lbp_bin_8_texture_transition",
    "lbp_bin_9_flat_region",
    "lbp_texture_entropy",
    "lbp_texture_energy",
    "intensity_mean",
    "intensity_std",
    "intensity_variance",
    "intensity_skewness",
    "intensity_kurtosis",
    "spatial_zone_row",
    "spatial_zone_col",
]

# 8 High-order spatial perspective and cross-feature interaction terms
DERIVED_FEATURE_COLUMNS: List[str] = [
    "perspective_edge_density",
    "perspective_hog_gradient",
    "perspective_lbp_entropy",
    "spatial_center_dist",
    "edge_intensity_ratio",
    "hog_edge_synergy",
    "lbp_entropy_energy_ratio",
    "intensity_spread_ratio",
]

AUGMENTED_FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS


def get_feature_names() -> List[str]:
    """Returns the ordered list of 31 core feature names."""
    return list(FEATURE_COLUMNS)


def get_augmented_feature_names() -> List[str]:
    """Returns the ordered list of all 39 augmented feature names."""
    return list(AUGMENTED_FEATURE_COLUMNS)


class SpatialFeatureAugmenter(BaseEstimator, TransformerMixin):
    """
    Scikit-learn Transformer that dynamically expands the 31 raw feature descriptors
    into 39 features by computing perspective interactions, texture ratios, and geometric priors.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SpatialFeatureAugmenter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        # Indices in FEATURE_COLUMNS:
        # 8: hog_gradient_mean, 10: canny_edge_density_ratio
        # 22: lbp_texture_entropy, 23: lbp_texture_energy
        # 24: intensity_mean, 25: intensity_std
        # 29: spatial_zone_row, 30: spatial_zone_col
        row = X_arr[:, 29]
        col = X_arr[:, 30]
        hog_grad = X_arr[:, 8]
        canny_edge = X_arr[:, 10]
        lbp_ent = X_arr[:, 22]
        lbp_eng = X_arr[:, 23]
        int_mean = X_arr[:, 24]
        int_std = X_arr[:, 25]

        # Perspective & cross-feature interactions
        f1 = (row * canny_edge)[:, None]
        f2 = (row * hog_grad)[:, None]
        f3 = (row * lbp_ent)[:, None]
        f4 = np.sqrt((row - 0.5) ** 2 + (col - 0.5) ** 2)[:, None]
        f5 = (canny_edge / (int_std + 1e-4))[:, None]
        f6 = (hog_grad * canny_edge)[:, None]
        f7 = (lbp_ent / (lbp_eng + 1e-4))[:, None]
        f8 = (int_std / (int_mean + 1e-4))[:, None]

        return np.hstack([X_arr, f1, f2, f3, f4, f5, f6, f7, f8])


def extract_hog_descriptors(
    gray_patch: np.ndarray,
    orientations: int = 8,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (1, 1),
) -> Tuple[np.ndarray, float, float]:
    """Extracts Histogram of Oriented Gradients (HOG) averaged across cell blocks."""
    hog_feats = hog(
        gray_patch,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=False,
    )
    reshaped = hog_feats.reshape(-1, orientations)
    bins = reshaped.mean(axis=0)
    mean_grad = float(np.mean(bins))
    std_grad = float(np.std(bins))
    return bins, mean_grad, std_grad


def extract_canny_edge_descriptors(
    gray_patch: np.ndarray,
    threshold1: float = 30.0,
    threshold2: float = 70.0,
) -> Tuple[float, float]:
    """Extracts edge density ratio and mean edge gradient magnitude."""
    edges = cv2.Canny(gray_patch, int(threshold1), int(threshold2))
    total_pixels = gray_patch.shape[0] * gray_patch.shape[1]
    edge_count = int(np.count_nonzero(edges))
    edge_density_ratio = float(edge_count) / float(max(1, total_pixels))

    sobel_x = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    if edge_count > 0:
        edge_mean_gradient = float(np.mean(grad_mag[edges > 0]))
    else:
        edge_mean_gradient = 0.0

    return edge_density_ratio, edge_mean_gradient


def extract_lbp_descriptors(
    gray_patch: np.ndarray,
    radius: int = 1,
    n_points: int = 8,
) -> Tuple[np.ndarray, float, float]:
    """Extracts Local Binary Patterns (LBP) uniform histogram, entropy, and energy."""
    lbp = local_binary_pattern(gray_patch, P=n_points, R=radius, method="uniform")
    num_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins), density=False)
    hist = hist.astype(np.float64)
    total = np.sum(hist)
    if total > 0:
        hist = hist / total
    else:
        hist = np.ones(num_bins, dtype=np.float64) / num_bins

    entropy = float(stats.entropy(hist + 1e-12))
    energy = float(np.sum(hist**2))
    return hist, entropy, energy


def extract_intensity_statistics(
    gray_patch: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Calculates 1st- and 2nd-order pixel intensity statistics."""
    flat = gray_patch.ravel().astype(np.float64)
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    var_val = float(np.var(flat))

    if std_val > 1e-6:
        skew_val = float(stats.skew(flat))
        kurt_val = float(stats.kurtosis(flat))
    else:
        skew_val = 0.0
        kurt_val = 0.0

    return mean_val, std_val, var_val, skew_val, kurt_val


def extract_patch_features(
    image_patch: np.ndarray,
    row: int = 0,
    col: int = 0,
    grid_size: Tuple[int, int] = (3, 3),
    target_shape: Tuple[int, int] = (256, 256),
) -> Dict[str, float]:
    """
    Extracts the 31-dimensional base visual feature dictionary for an image patch.
    """
    if image_patch is None or image_patch.size == 0:
        raise ValueError("Invalid empty image patch passed to extract_patch_features.")

    if len(image_patch.shape) == 3:
        gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_patch.copy()

    if gray.shape != target_shape:
        gray = cv2.resize(gray, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

    hog_bins, hog_mean, hog_std = extract_hog_descriptors(gray)
    edge_ratio, edge_grad = extract_canny_edge_descriptors(gray)
    lbp_hist, lbp_entropy, lbp_energy = extract_lbp_descriptors(gray)
    int_mean, int_std, int_var, int_skew, int_kurt = extract_intensity_statistics(gray)

    max_row = max(1, grid_size[0] - 1)
    max_col = max(1, grid_size[1] - 1)
    norm_row = float(row) / float(max_row)
    norm_col = float(col) / float(max_col)

    feature_dict: Dict[str, float] = {
        "hog_bin_0_deg0_22": float(hog_bins[0]),
        "hog_bin_1_deg22_45": float(hog_bins[1]),
        "hog_bin_2_deg45_67": float(hog_bins[2]),
        "hog_bin_3_deg67_90": float(hog_bins[3]),
        "hog_bin_4_deg90_112": float(hog_bins[4]),
        "hog_bin_5_deg112_135": float(hog_bins[5]),
        "hog_bin_6_deg135_157": float(hog_bins[6]),
        "hog_bin_7_deg157_180": float(hog_bins[7]),
        "hog_gradient_mean": hog_mean,
        "hog_gradient_std": hog_std,
        "canny_edge_density_ratio": edge_ratio,
        "canny_edge_mean_gradient": edge_grad,
        "lbp_bin_0_spot_dark": float(lbp_hist[0]),
        "lbp_bin_1_edge_pattern": float(lbp_hist[1]),
        "lbp_bin_2_edge_pattern": float(lbp_hist[2]),
        "lbp_bin_3_corner_pattern": float(lbp_hist[3]),
        "lbp_bin_4_corner_pattern": float(lbp_hist[4]),
        "lbp_bin_5_corner_pattern": float(lbp_hist[5]),
        "lbp_bin_6_corner_pattern": float(lbp_hist[6]),
        "lbp_bin_7_texture_transition": float(lbp_hist[7]),
        "lbp_bin_8_texture_transition": float(lbp_hist[8]),
        "lbp_bin_9_flat_region": float(lbp_hist[9]),
        "lbp_texture_entropy": lbp_entropy,
        "lbp_texture_energy": lbp_energy,
        "intensity_mean": int_mean,
        "intensity_std": int_std,
        "intensity_variance": int_var,
        "intensity_skewness": int_skew,
        "intensity_kurtosis": int_kurt,
        "spatial_zone_row": norm_row,
        "spatial_zone_col": norm_col,
    }

    return feature_dict
