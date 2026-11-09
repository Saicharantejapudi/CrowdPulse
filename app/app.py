"""
CrowdPulse: Interactive Surveillance Telemetry & Multi-Agent Alerting Dashboard (app/app.py)
Loads the serialized Scikit-learn champion pipeline (crowdpulse_pipeline.joblib),
performs live spatial feature extraction and localized crowd regression across a 3x3 grid,
models inter-zone agent coordination, displays SHAP explainability, and triggers bottleneck alerts.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.agent_coordinator import AgentCoordinator
from src.data_loader import partition_image_into_grid
from src.evaluate import discretize_counts_to_tiers
from src.explainability import explain_prediction_plain_text, get_tree_explainer
from src.feature_extractor import FEATURE_COLUMNS, extract_patch_features


# -----------------------------------------------------------------------------
# Streamlit App Configuration & Custom Dark CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CrowdPulse | Multi-Agent Surveillance Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark Theme Styles */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(135deg, #161e2e 0%, #1a2234 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .status-normal {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-caution {
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-critical {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .directive-box {
        background: #1e1b2e;
        border-left: 4px solid #ef4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching Model & Assets
# -----------------------------------------------------------------------------
@st.cache_resource
def load_champion_pipeline():
    model_path = PROJECT_ROOT / "models" / "crowdpulse_pipeline.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def get_sample_images():
    assets_dir = PROJECT_ROOT / "app" / "assets"
    if not assets_dir.exists():
        return {}
    files = sorted([f for f in os.listdir(assets_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    return {f"Surveillance Cam #{i+1} ({f})": str(assets_dir / f) for i, f in enumerate(files)}


pipeline = load_champion_pipeline()
sample_images_map = get_sample_images()


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://img.shields.io/badge/CrowdPulse-INT395%20Supervised%20Learning-blue?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.title("🎛️ Control Telemetry")

# Image Selector
st.sidebar.subheader("1. Video Feed / Frame Source")
source_mode = st.sidebar.radio("Input Source", ["Preloaded Surveillance Feeds", "Upload Frame Image"])

input_image = None
if source_mode == "Preloaded Surveillance Feeds":
    if sample_images_map:
        selected_label = st.sidebar.selectbox("Select Camera Angle", list(sample_images_map.keys()))
        img_path = sample_images_map[selected_label]
        input_image = cv2.imread(img_path)
    else:
        st.sidebar.warning("No sample frames found in app/assets.")
else:
    uploaded = st.sidebar.file_uploader("Upload Surveillance Frame (.jpg/.png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Threshold Sliders
st.sidebar.subheader("2. Operational Risk Thresholds")
green_cutoff = st.sidebar.slider(
    "🟢 Green Threshold (Free Flow < N)",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)
red_cutoff = st.sidebar.slider(
    "🔴 Red Threshold (Bottleneck >= N)",
    min_value=15,
    max_value=40,
    value=25,
    step=1,
)

# Model Metadata Card
st.sidebar.subheader("3. Champion Architecture")
st.sidebar.markdown(f"""
- **Model**: `Enhanced Voting Ensemble`
- **Architecture**: `RF` + `HGB (Poisson Loss)` + `RF (Log1p)`
- **Feature Engine**: `SpatialFeatureAugmenter` (39-D)
- **Syllabus**: Classical Edge Features (INT395)
- **Adjacency**: 8-Connected 3x3 Spatial Grid
""")


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
st.title("CrowdPulse: Multi-Agent Crowd Density Estimation & Coordinated Alerting")
st.caption("Course: INT395 Supervised Learning | Problem Statement #20 | Dataset: ShanghaiTech Part B")

if pipeline is None:
    st.error("⚠️ Serialized champion pipeline `models/crowdpulse_pipeline.joblib` not found. Please execute `python src/train.py` first.")
    st.stop()

if input_image is None:
    st.info("👈 Please select or upload a surveillance frame from the sidebar to begin real-time analysis.")
    st.stop()

# -----------------------------------------------------------------------------
# Inference Across 3x3 Grid
# -----------------------------------------------------------------------------
zones = partition_image_into_grid(input_image, grid_size=(3, 3))
coordinator = AgentCoordinator(
    grid_size=(3, 3),
    allow_diagonal_adjacency=True,
    green_cutoff=green_cutoff,
    red_cutoff=red_cutoff,
)

zone_predictions = {}
zone_features_dict = {}

for z in zones:
    zid = z["zone_id"]
    r = z["row"]
    c = z["col"]
    patch = z["patch"]

    feats = extract_patch_features(patch, row=r, col=c, grid_size=(3, 3))
    zone_features_dict[zid] = feats
    f_vec = np.array([[feats[col] for col in FEATURE_COLUMNS]])

    raw_pred = pipeline.predict(f_vec)[0]
    clipped_pred = max(0.0, float(raw_pred))
    zone_predictions[zid] = clipped_pred

# Update Multi-Agent Coordinator
coordinator.update_zone_counts(zone_predictions)
system_state = coordinator.evaluate_system_state()

# -----------------------------------------------------------------------------
# Top KPI Header Metrics
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Estimated Crowd</div>
        <div class="metric-value">{system_state['total_estimated_count']:.0f} <span style="font-size:0.9rem; color:#94a3b8">people</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    n_red = len(system_state["red_zones"])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Critical Bottlenecks (Red)</div>
        <div class="metric-value" style="color: {'#ef4444' if n_red > 0 else '#10b981'}">{n_red} <span style="font-size:0.9rem; color:#94a3b8">zones</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    n_yellow = len(system_state["yellow_zones"])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Elevated Density (Yellow)</div>
        <div class="metric-value" style="color: {'#f59e0b' if n_yellow > 0 else '#94a3b8'}">{n_yellow} <span style="font-size:0.9rem; color:#94a3b8">zones</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    status_lvl = system_state["status_level"]
    if status_lvl == "CRITICAL":
        badge = '<span class="status-critical">🚨 CRITICAL ALERT</span>'
    elif status_lvl == "WARNING":
        badge = '<span class="status-caution">⚠️ CAUTION</span>'
    elif status_lvl == "CAUTION":
        badge = '<span class="status-caution">ℹ️ MONITORING</span>'
    else:
        badge = '<span class="status-normal">✅ NORMAL FLOW</span>'

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Multi-Agent System Status</div>
        <div style="margin-top: 6px;">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📸 Grid Telemetry & Live Detection",
    "🔍 Model Explainability (SHAP)",
    "📡 Multi-Agent Communication Broker",
])

# -----------------------------------------------------------------------------
# TAB 1: Grid Telemetry & Live Detection
# -----------------------------------------------------------------------------
with tab1:
    if system_state["rerouting_directives"]:
        for directive in system_state["rerouting_directives"]:
            st.markdown(f'<div class="directive-box">⚠️ <b>DYNAMIC REROUTING DIRECTIVE:</b> {directive}</div>', unsafe_allow_html=True)

    st.write("")

    # Draw Real-Time Overlay on Image
    overlay_img = input_image.copy()
    h_img, w_img = overlay_img.shape[:2]

    # Semi-transparent canvas
    color_mask = np.zeros_like(overlay_img, dtype=np.uint8)

    # Color Palette BGR
    tier_colors = {
        "Green": (30, 200, 70),     # Emerald
        "Yellow": (20, 180, 240),   # Amber
        "Red": (40, 40, 235),       # Crimson
    }

    for z in zones:
        zid = z["zone_id"]
        ymin, xmin, ymax, xmax = z["bbox"]
        agent = coordinator.agents[zid]
        tier = agent.risk_tier
        count_val = agent.current_count
        bgr = tier_colors[tier]

        # Fill box in color mask
        cv2.rectangle(color_mask, (xmin, ymin), (xmax, ymax), bgr, -1)
        # Bounding box border
        cv2.rectangle(overlay_img, (xmin, ymin), (xmax, ymax), bgr, 3)

        # Label badge
        label_text = f"{zid}: {count_val:.0f} heads ({tier})"
        cv2.putText(
            overlay_img,
            label_text,
            (xmin + 15, ymin + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Blend overlay with 30% fill opacity
    blended = cv2.addWeighted(color_mask, 0.25, overlay_img, 0.75, 0)

    # Convert BGR to RGB for Streamlit display
    rgb_blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    left_view, right_view = st.columns([1.6, 1.0])

    with left_view:
        st.subheader("Live Spatial Grid Heatmap")
        st.image(rgb_blended, use_container_width=True, caption="Camera View partitioned into 3x3 zones with risk tier overlay")

    with right_view:
        st.subheader("Zone Telemetry Matrix")
        # Format 3x3 table of estimated counts
        grid_matrix = np.zeros((3, 3))
        tier_matrix = np.empty((3, 3), dtype=object)
        for r in range(3):
            for c in range(3):
                zid = f"Zone_{r}_{c}"
                grid_matrix[r, c] = coordinator.agents[zid].current_count
                tier_matrix[r, c] = coordinator.agents[zid].risk_tier

        matrix_df = pd.DataFrame(
            grid_matrix,
            index=["Row 0 (Top)", "Row 1 (Center)", "Row 2 (Bottom)"],
            columns=["Col 0 (Left)", "Col 1 (Center)", "Col 2 (Right)"],
        )
        st.dataframe(matrix_df.style.format("{:.1f}").background_gradient(cmap="YlOrRd", vmin=0, vmax=40), use_container_width=True)

        st.subheader("Agent Registry Status")
        agent_records = []
        for zid, agent in coordinator.agents.items():
            agent_records.append({
                "Zone": zid,
                "Position": f"({agent.row}, {agent.col})",
                "Count": f"{agent.current_count:.1f}",
                "Tier": agent.risk_tier,
                "Status": agent.alert_status,
            })
        st.table(pd.DataFrame(agent_records))


# -----------------------------------------------------------------------------
# TAB 2: Model Explainability (SHAP)
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Explainable AI: Feature Contribution Analysis")
    st.write(
        "Using **SHAP TreeExplainer**, we inspect the exact physical visual descriptors (HOG orientation gradients, Canny edge density, LBP uniform texture patterns) that drove each zone's crowd estimate."
    )

    shap_col1, shap_col2 = st.columns([1.2, 1.0])

    with shap_col1:
        selected_zone_id = st.selectbox("Inspect Zone Explanation", list(coordinator.agents.keys()), index=0)
        selected_feat_dict = zone_features_dict[selected_zone_id]
        selected_feat_vec = np.array([selected_feat_dict[k] for k in FEATURE_COLUMNS])

        # Top plain-text explanations
        explanations = explain_prediction_plain_text(pipeline, selected_feat_vec, top_k=6)
        pred_val = coordinator.agents[selected_zone_id].current_count
        tier_val = coordinator.agents[selected_zone_id].risk_tier

        st.markdown(f"**Zone:** `{selected_zone_id}` | **Predicted Count:** `{pred_val:.1f} heads` | **Risk Tier:** `{tier_val}`")
        st.markdown("#### Primary Driving Factors:")
        for exp in explanations:
            direction_icon = "🔺" if exp["direction"] == "increases" else "🔻"
            st.markdown(f"- {direction_icon} **{exp['feature']}**: {exp['impact_description']}")

        # Waterfall plot on the fly
        st.markdown("#### Local SHAP Waterfall Plot:")
        explainer, transform_fn, feat_names = get_tree_explainer(pipeline)
        scaled_vec = transform_fn(selected_feat_vec.reshape(1, -1))
        shap_res = explainer(scaled_vec)
        shap_res.feature_names = feat_names

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        plt.style.use("dark_background")
        fig.patch.set_facecolor("#161e2e")
        ax.set_facecolor("#161e2e")
        shap.plots.waterfall(shap_res[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

    with shap_col2:
        st.markdown("#### Global Model Feature Importance (SHAP Beeswarm)")
        summary_img_path = PROJECT_ROOT / "models" / "shap_summary.png"
        if summary_img_path.exists():
            st.image(str(summary_img_path), use_container_width=True, caption="Global feature importance across ShanghaiTech Part B")
        else:
            st.info("Run `python src/explainability.py` to generate the global beeswarm plot.")


# -----------------------------------------------------------------------------
# TAB 3: Multi-Agent Communication Log
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Inter-Agent Alert Broker & Coordination Stream")
    st.write(
        "Each 3x3 zone is an autonomous `ZoneAgent` communicating with the central `AgentCoordinator` across a spatial adjacency graph."
    )

    c1, c2 = st.columns([1.5, 1.0])

    with c1:
        st.markdown("#### Live Communication Event Feed")
        if coordinator.message_log:
            log_df = pd.DataFrame(coordinator.message_log[::-1])
            def style_log(row):
                if row["level"] == "CRITICAL":
                    return ["background-color: rgba(239, 68, 68, 0.2)"] * len(row)
                elif row["level"] == "WARNING":
                    return ["background-color: rgba(245, 158, 11, 0.2)"] * len(row)
                return [""] * len(row)

            st.dataframe(log_df.style.apply(style_log, axis=1), use_container_width=True, height=350)
        else:
            st.info("No inter-agent messages logged yet.")

    with c2:
        st.markdown("#### Spatial Adjacency Topology")
        st.write("Zone graph neighbors (8-connected including diagonals):")
        adj_records = [
            {"Zone": zid, "Neighbors": ", ".join(neighbors)}
            for zid, neighbors in coordinator.adjacency_graph.items()
        ]
        st.dataframe(pd.DataFrame(adj_records), use_container_width=True, height=350)

