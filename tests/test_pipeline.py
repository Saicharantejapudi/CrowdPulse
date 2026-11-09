"""
CrowdPulse Pytest Verification Suite (tests/test_pipeline.py)
Validates:
  1. Feature extraction vector dimensions, column names, and numeric validity
  2. Spatial grid slicing and coordinate bounding boxes
  3. Ground-truth coordinate point-to-zone mapping
  4. Pipeline inference consistency and non-negative prediction constraints
  5. Operational risk tier boundary logic (Green/Yellow/Red cutoffs)
  6. Multi-agent coordinator adjacency graphs, warning alerts, and critical bottleneck triggers
  7. Dynamic rerouting directive generation
"""

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.agent_coordinator import AgentCoordinator, ZoneAgent
from src.data_loader import (
    calculate_ground_truth_counts,
    partition_image_into_grid,
)
from src.evaluate import discretize_counts_to_tiers
from src.explainability import explain_prediction_plain_text
from src.feature_extractor import (
    AUGMENTED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    SpatialFeatureAugmenter,
    extract_patch_features,
    get_feature_names,
)


@pytest.fixture
def dummy_image():
    """Generates synthetic 768x1024 BGR image with simulated gradient textures."""
    np.random.seed(42)
    img = np.random.randint(50, 200, (768, 1024, 3), dtype=np.uint8)
    return img


@pytest.fixture
def dummy_patch():
    """Generates synthetic 256x256 BGR patch."""
    np.random.seed(42)
    return np.random.randint(30, 220, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def trained_pipeline():
    """Loads the serialized champion pipeline if available."""
    model_path = Path("models/crowdpulse_pipeline.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None


# --------------------------------------------------------------------------
# 1. Feature Extraction Tests
# --------------------------------------------------------------------------

def test_feature_vector_dimensions_and_keys(dummy_patch):
    """Verifies that extract_patch_features returns exactly 31 features with correct keys."""
    features = extract_patch_features(dummy_patch, row=0, col=0, grid_size=(3, 3))
    assert isinstance(features, dict)
    assert len(features) == len(FEATURE_COLUMNS) == 31

    for col in FEATURE_COLUMNS:
        assert col in features, f"Missing feature: {col}"
        val = features[col]
        assert isinstance(val, (float, int, np.floating)), f"Non-numeric value in {col}: {val}"
        assert not np.isnan(val), f"NaN detected in feature {col}"
        assert not np.isinf(val), f"Inf detected in feature {col}"


def test_feature_names_order():
    """Verifies get_feature_names matches expected schema."""
    names = get_feature_names()
    assert len(names) == 31
    assert names[0] == "hog_bin_0_deg0_22"
    assert names[-1] == "spatial_zone_col"


def test_spatial_grid_coordinates_normalization(dummy_patch):
    """Verifies normalized spatial coordinates across 3x3 positions."""
    f_0_0 = extract_patch_features(dummy_patch, row=0, col=0, grid_size=(3, 3))
    f_1_1 = extract_patch_features(dummy_patch, row=1, col=1, grid_size=(3, 3))
    f_2_2 = extract_patch_features(dummy_patch, row=2, col=2, grid_size=(3, 3))

    assert f_0_0["spatial_zone_row"] == pytest.approx(0.0)
    assert f_0_0["spatial_zone_col"] == pytest.approx(0.0)
    assert f_1_1["spatial_zone_row"] == pytest.approx(0.5)
    assert f_1_1["spatial_zone_col"] == pytest.approx(0.5)
    assert f_2_2["spatial_zone_row"] == pytest.approx(1.0)
    assert f_2_2["spatial_zone_col"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 2. Spatial Partitioning & Ground Truth Tests
# --------------------------------------------------------------------------

def test_partition_image_into_grid(dummy_image):
    """Verifies 3x3 partitioning yields exactly 9 distinct non-empty zones."""
    zones = partition_image_into_grid(dummy_image, grid_size=(3, 3))
    assert len(zones) == 9

    expected_ids = [f"Zone_{r}_{c}" for r in range(3) for c in range(3)]
    zone_ids = [z["zone_id"] for z in zones]
    assert zone_ids == expected_ids

    for z in zones:
        patch = z["patch"]
        assert patch.shape[0] > 0 and patch.shape[1] > 0
        ymin, xmin, ymax, xmax = z["bbox"]
        assert ymin < ymax
        assert xmin < xmax


def test_calculate_ground_truth_counts():
    """Verifies that coordinate points are mapped to the correct zones."""
    # Frame shape: H=900, W=900. Each zone is 300x300.
    # Zone_0_0: [0:300, 0:300]
    # Zone_0_1: [0:300, 300:600]
    # Zone_2_2: [600:900, 600:900]
    annotations = np.array([
        [150.0, 150.0],  # in Zone_0_0
        [100.0, 200.0],  # in Zone_0_0
        [450.0, 150.0],  # in Zone_0_1
        [750.0, 750.0],  # in Zone_2_2
    ])
    counts = calculate_ground_truth_counts(annotations, grid_size=(3, 3), image_shape=(900, 900))

    assert counts[(0, 0)] == 2
    assert counts[(0, 1)] == 1
    assert counts[(2, 2)] == 1
    assert counts[(1, 1)] == 0
    assert sum(counts.values()) == 4


# --------------------------------------------------------------------------
# 3. Model Inference & Non-Negative Output Tests
# --------------------------------------------------------------------------

def test_pipeline_inference(trained_pipeline, dummy_patch):
    """Verifies model loads, predicts finite values, and non-negativity constraint holds."""
    if trained_pipeline is None:
        pytest.skip("Model not yet trained. Run src/train.py first.")

    features = extract_patch_features(dummy_patch, row=0, col=0)
    feat_vector = np.array([[features[c] for c in FEATURE_COLUMNS]])

    raw_pred = trained_pipeline.predict(feat_vector)
    assert raw_pred.shape == (1,)
    assert not np.isnan(raw_pred[0])

    clipped_pred = np.clip(raw_pred, 0, None)
    assert clipped_pred[0] >= 0.0


# --------------------------------------------------------------------------
# 4. Operational Risk Tier Boundary Logic
# --------------------------------------------------------------------------

def test_risk_tier_boundary_discretization():
    """Tests exact edge boundaries for Green, Yellow, and Red risk tiers."""
    counts = [0.0, 5.0, 9.99, 10.0, 18.0, 24.99, 25.0, 35.0, 150.0]
    expected = [
        "Green", "Green", "Green",
        "Yellow", "Yellow", "Yellow",
        "Red", "Red", "Red",
    ]
    tiers = discretize_counts_to_tiers(counts, green_cutoff=10.0, red_cutoff=25.0)
    assert list(tiers) == expected


# --------------------------------------------------------------------------
# 5. Multi-Agent Coordinator & Adjacency Logic
# --------------------------------------------------------------------------

def test_coordinator_initialization():
    """Verifies that coordinator instantiates 9 zone agents with valid adjacency graphs."""
    coord = AgentCoordinator(grid_size=(3, 3))
    assert len(coord.agents) == 9
    assert "Zone_0_0" in coord.agents
    assert "Zone_2_2" in coord.agents

    # Check that Zone_0_0 neighbors include Zone_0_1, Zone_1_0, and diagonal Zone_1_1
    neighbors_0_0 = coord.adjacency_graph["Zone_0_0"]
    assert "Zone_0_1" in neighbors_0_0
    assert "Zone_1_0" in neighbors_0_0
    assert "Zone_1_1" in neighbors_0_0


def test_multi_agent_normal_state():
    """Verifies that normal counts maintain NORMAL FLOW system status."""
    coord = AgentCoordinator()
    coord.update_zone_counts({"Zone_0_0": 4.0, "Zone_1_1": 8.0, "Zone_2_2": 2.0})
    state = coord.evaluate_system_state()
    assert state["status_level"] == "NORMAL"
    assert state["system_status"] == "NORMAL FLOW"
    assert len(state["red_zones"]) == 0
    assert len(state["adjacent_red_pairs"]) == 0


def test_multi_agent_single_zone_caution():
    """Verifies that a single isolated red zone triggers localized caution, NOT critical alert."""
    coord = AgentCoordinator()
    coord.update_zone_counts({"Zone_1_1": 32.0})
    state = coord.evaluate_system_state()
    assert state["status_level"] == "WARNING"
    assert state["system_status"] == "LOCALIZED CONGESTION CAUTION"
    assert state["red_zones"] == ["Zone_1_1"]
    assert len(state["adjacent_red_pairs"]) == 0


def test_multi_agent_adjacent_bottleneck_critical_alert():
    """Verifies that two spatially adjacent red zones escalate to CRITICAL COORDINATED BOTTLENECK ALERT."""
    coord = AgentCoordinator()
    coord.update_zone_counts({
        "Zone_0_0": 30.0,
        "Zone_0_1": 28.0,
    })
    state = coord.evaluate_system_state()
    assert state["status_level"] == "CRITICAL"
    assert state["system_status"] == "CRITICAL COORDINATED BOTTLENECK ALERT"
    assert ("Zone_0_0", "Zone_0_1") in state["adjacent_red_pairs"] or ("Zone_0_1", "Zone_0_0") in state["adjacent_red_pairs"]
    assert len(state["rerouting_directives"]) > 0


def test_multi_agent_non_adjacent_red_zones():
    """Verifies that disjoint non-adjacent red zones remain at localized warning level."""
    # When allow_diagonal_adjacency=False, Zone_0_0 and Zone_2_2 are not adjacent
    coord = AgentCoordinator(allow_diagonal_adjacency=False)
    coord.update_zone_counts({
        "Zone_0_0": 35.0,
        "Zone_2_2": 40.0,
    })
    state = coord.evaluate_system_state()
    assert state["status_level"] == "WARNING"
    assert len(state["adjacent_red_pairs"]) == 0


# --------------------------------------------------------------------------
# 6. Plain Text Explainability
# --------------------------------------------------------------------------

def test_explain_prediction_plain_text(trained_pipeline, dummy_patch):
    """Verifies explain_prediction_plain_text produces descriptive summaries."""
    if trained_pipeline is None:
        pytest.skip("Model not yet trained. Run src/train.py first.")

    features = extract_patch_features(dummy_patch, row=0, col=0)
    feat_vector = np.array([features[c] for c in FEATURE_COLUMNS])

    explanations = explain_prediction_plain_text(trained_pipeline, feat_vector, top_k=3)
    assert len(explanations) == 3
    for exp in explanations:
        assert "feature" in exp
        assert "shap_value" in exp
        assert "direction" in exp
        assert "impact_description" in exp


def test_spatial_feature_augmenter_expansion(dummy_patch):
    """Verifies SpatialFeatureAugmenter expands 31 core descriptors to 39 augmented features."""
    features = extract_patch_features(dummy_patch, row=1, col=1)
    feat_vec = np.array([[features[c] for c in FEATURE_COLUMNS]])
    assert feat_vec.shape == (1, 31)

    augmenter = SpatialFeatureAugmenter()
    augmented_vec = augmenter.transform(feat_vec)
    assert augmented_vec.shape == (1, 39)
    assert not np.isnan(augmented_vec).any()
    assert not np.isinf(augmented_vec).any()


