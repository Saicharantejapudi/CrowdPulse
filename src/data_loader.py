"""
CrowdPulse: Data Loading & Spatial Partitioning (src/data_loader.py)
Handles parsing of ShanghaiTech Part B ground-truth .mat files,
spatial partitioning of images into a 3x3 grid (Zone_0_0 to Zone_2_2),
exact ground-truth head count assignment per zone,
and dataset batch processing to tabular format.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from src.feature_extractor import FEATURE_COLUMNS, extract_patch_features


def find_raw_data_dir() -> Path:
    """
    Locates the ShanghaiTech Part B raw data directory.
    Checks data/raw, Data/raw, Data/ShanghaiTech/part_B, and data/ShanghaiTech/part_B.
    """
    candidates = [
        Path("data/raw"),
        Path("Data/raw"),
        Path("Data/ShanghaiTech/part_B"),
        Path("data/ShanghaiTech/part_B"),
        Path("../data/raw"),
        Path("../Data/ShanghaiTech/part_B"),
    ]
    for c in candidates:
        if c.exists() and (c / "train_data").exists():
            return c.resolve()
    # Fallback to current working directory relative
    return Path("data/raw").resolve()


def find_processed_data_dir() -> Path:
    """
    Locates the processed features directory containing train_features.csv and test_features.csv.
    """
    candidates = [
        Path("data/processed"),
        Path("Data/processed"),
        Path("../data/processed"),
        Path("../Data/processed"),
    ]
    for c in candidates:
        if c.exists() and (c / "train_features.csv").exists():
            return c.resolve()
    return Path("data/processed").resolve()


def parse_mat_annotations(mat_path: Union[str, Path]) -> np.ndarray:
    """
    Parses ShanghaiTech Part B MATLAB .mat ground-truth annotations.
    Points are located at image_info[0][0][0][0][0] as an (N, 2) array of (x, y) coordinates.

    Args:
        mat_path: Path to the .mat file.

    Returns:
        np.ndarray: (N, 2) array of float coordinates [x, y]. If no points, returns empty (0, 2) array.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {mat_path}")

    mat_data = sio.loadmat(str(mat_path))
    try:
        pts = mat_data["image_info"][0][0][0][0][0]
        if pts is None or pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Malformed ShanghaiTech annotation structure in {mat_path}: {exc}")


def partition_image_into_grid(
    image: np.ndarray,
    grid_size: Tuple[int, int] = (3, 3),
) -> List[Dict]:
    """
    Slices any surveillance camera frame into distinct spatial zones.
    Default (3, 3) yields 9 zones: Zone_0_0 through Zone_2_2.

    Args:
        image: Numpy image array (H, W, C) or (H, W).
        grid_size: Tuple of (n_rows, n_cols).

    Returns:
        List of dicts, each with keys:
          - 'zone_id': e.g. 'Zone_0_0'
          - 'row': row index (int)
          - 'col': column index (int)
          - 'patch': sliced image sub-array
          - 'bbox': (ymin, xmin, ymax, xmax) in pixel coordinates
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot partition empty image.")

    h, w = image.shape[:2]
    n_rows, n_cols = grid_size
    row_step = h / float(n_rows)
    col_step = w / float(n_cols)

    zones: List[Dict] = []
    for r in range(n_rows):
        y_min = int(round(r * row_step))
        y_max = int(round((r + 1) * row_step)) if r < n_rows - 1 else h
        for c in range(n_cols):
            x_min = int(round(c * col_step))
            x_max = int(round((c + 1) * col_step)) if c < n_cols - 1 else w

            patch = image[y_min:y_max, x_min:x_max]
            zone_id = f"Zone_{r}_{c}"
            zones.append({
                "zone_id": zone_id,
                "row": r,
                "col": c,
                "patch": patch,
                "bbox": (y_min, x_min, y_max, x_max),
            })

    return zones


def calculate_ground_truth_counts(
    annotations: np.ndarray,
    grid_size: Tuple[int, int] = (3, 3),
    image_shape: Tuple[int, int] = (1536, 2048),
) -> Dict[Tuple[int, int], int]:
    """
    Maps (x, y) head coordinate points to spatial grid bounding boxes and counts heads per zone.

    Args:
        annotations: (N, 2) array of [x, y] coordinates.
        grid_size: (n_rows, n_cols) grid partition.
        image_shape: (height, width) of the image frame.

    Returns:
        Dict mapping (row, col) tuple to integer head count.
    """
    n_rows, n_cols = grid_size
    h, w = image_shape[:2]

    counts: Dict[Tuple[int, int], int] = {
        (r, c): 0 for r in range(n_rows) for c in range(n_cols)
    }

    if annotations is None or len(annotations) == 0:
        return counts

    row_step = h / float(n_rows)
    col_step = w / float(n_cols)

    for x, y in annotations:
        # Clamp coordinates within valid image bounds
        r = int(y / row_step)
        c = int(x / col_step)
        r = max(0, min(n_rows - 1, r))
        c = max(0, min(n_cols - 1, c))
        counts[(r, c)] += 1

    return counts


def get_matching_mat_filename(image_filename: str) -> str:
    """
    Maps an image filename to its corresponding ground-truth .mat filename.
    Examples:
      'processed_IMG_1.jpg' -> 'GT_IMG_1.mat'
      'IMG_1.jpg' -> 'GT_IMG_1.mat'
    """
    stem = Path(image_filename).stem
    if stem.startswith("processed_"):
        stem = stem[len("processed_"):]
    return f"GT_{stem}.mat"


def process_dataset_to_dataframe(
    split: str = "train",
    raw_dir: Optional[Union[str, Path]] = None,
    grid_size: Tuple[int, int] = (3, 3),
    output_csv: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Batch processes a dataset split ('train' or 'test'), extracting features from 3x3 zones
    and computing ground-truth counts.

    Args:
        split: 'train' or 'test'
        raw_dir: Base raw directory containing 'train_data' and 'test_data'
        grid_size: Spatial grid division (default 3x3)
        output_csv: Optional path to save the resulting CSV
        max_samples: Optional limit on the number of images to process (useful for tests/quick runs)

    Returns:
        pd.DataFrame with 36 columns (5 metadata + 31 features)
    """
    if raw_dir is None:
        raw_path = find_raw_data_dir()
    else:
        raw_path = Path(raw_dir)

    split_dir = raw_path / f"{split}_data"
    images_dir = split_dir / "images"
    gt_dir = split_dir / "ground-truth"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if max_samples is not None and max_samples > 0:
        image_files = image_files[:max_samples]

    records = []
    for img_name in image_files:
        img_path = images_dir / img_name
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        mat_name = get_matching_mat_filename(img_name)
        mat_path = gt_dir / mat_name

        if mat_path.exists():
            ann = parse_mat_annotations(mat_path)
            counts_map = calculate_ground_truth_counts(ann, grid_size=grid_size, image_shape=(h, w))
        else:
            counts_map = {(r, c): 0 for r in range(grid_size[0]) for c in range(grid_size[1])}

        zones = partition_image_into_grid(image, grid_size=grid_size)
        for z in zones:
            r = z["row"]
            c = z["col"]
            zone_id = z["zone_id"]
            patch = z["patch"]
            head_count = counts_map.get((r, c), 0)

            features = extract_patch_features(patch, row=r, col=c, grid_size=grid_size)

            row_record = {
                "image_name": img_name,
                "zone_id": zone_id,
                "row": r,
                "col": c,
                "head_count": head_count,
            }
            row_record.update(features)
            records.append(row_record)

    # Maintain consistent column ordering
    meta_cols = ["image_name", "zone_id", "row", "col", "head_count"]
    final_cols = meta_cols + FEATURE_COLUMNS
    df = pd.DataFrame(records)[final_cols]

    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df


def load_processed_data(
    processed_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train_features.csv and test_features.csv from processed directory.

    Returns:
        (train_df, test_df)
    """
    if processed_dir is None:
        p_dir = find_processed_data_dir()
    else:
        p_dir = Path(processed_dir)

    train_path = p_dir / "train_features.csv"
    test_path = p_dir / "test_features.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed CSV files missing in {p_dir}. Required: train_features.csv, test_features.csv"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df
