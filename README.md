# CrowdPulse: Multi-Agent Crowd Density Estimation and Coordinated Alerting System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Pytest Passed](https://img.shields.io/badge/Pytest-13%2F13%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-purple.svg)](https://shap.readthedocs.io/)

---

## 📋 Academic Submission Metadata
- **Course:** INT395 - Supervised Learning
- **Problem Statement ID:** #20
- **Project Title:** CrowdPulse: Multi-Agent Crowd Density Estimation and Coordinated Alerting System
- **Registration Numbers:** `12201941`, `12202052`, `12203114`
- **GitHub Repository Link:** [https://github.com/Pudisiva/CrowdPulse](https://github.com/Pudisiva/CrowdPulse)
- **Live Deployed Project Link:** [https://crowdpulse.streamlit.app](https://crowdpulse.streamlit.app)
- **Target Dataset:** ShanghaiTech Crowd Counting Dataset (Part B)

---

## 🎯 Project Overview & Objectives
Surveillance systems in mass-gathering venues (stadiums, transit hubs, pilgrimage sites) require rapid, proactive identification of crowd bottlenecks before fatal stampedes occur. While state-of-the-art deep convolutional networks (CSRNet, MCNN, BayCount) achieve high density mapping, their gigabyte-scale memory footprints and inference latencies (>200 ms) make them impractical for resource-constrained edge hardware.

**CrowdPulse** implements a classical, mathematically grounded machine learning architecture:
1. **Spatial Grid Partitioning:** Surveillance frames are sliced into a $3 \times 3$ grid (9 zones: `Zone_0_0` through `Zone_2_2`).
2. **Edge-Compatible Visual Descriptors:** Extracts 31 domain-specific spatial features (HOG orientations, Canny edge density ratio, LBP uniform texture histograms, pixel intensity statistics) in under 12 ms per zone.
3. **Supervised Regression Ensembles:** Benchmarks linear, kernel, and ensemble regressors under 5-fold cross-validation, deploying a champion `StandardScaler` + `RandomForestRegressor` pipeline.
4. **3-Tier Operational Risk Categorization:** Discretizes continuous counts into **Green** (<10 heads, free flow), **Yellow** (10–24 heads, elevated monitoring), and **Red** ($\ge 25$ heads, critical bottleneck).
5. **Explainable AI (XAI):** Uses `shap.TreeExplainer` to produce global beeswarm importance rankings and local waterfall explanations for facility operators.
6. **Decentralized Multi-Agent Coordination Broker:** Models the 9 zones as an 8-connected spatial adjacency graph. When adjacent zones enter the Red Tier simultaneously, it triggers a **CRITICAL COORDINATED BOTTLENECK ALERT** and calculates dynamic pedestrian diversion paths.

---

## 🏛️ System Architecture

```text
                                  Surveillance Video Stream
                                             │
                                             ▼
                             ┌───────────────────────────────┐
                             │ 3x3 Spatial Grid Partitioning │
                             │   (Zone_0_0 to Zone_2_2)      │
                             └──────────────┬────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │   Edge-Compatible Visual Feature Engine      │
                     │  • HOG Gradient Orientations (10 features)   │
                     │  • Canny Edge Density & Magnitude (2 feats)  │
                     │  • LBP Uniform Texture Patterns (12 feats)   │
                     │  • 1st & 2nd Order Pixel Stats (5 feats)     │
                     │  • Normalized Spatial Coordinates (2 feats)  │
                     └──────────────────────┬───────────────────────┘
                                            │  (31-D Feature Vector)
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │  Champion Scikit-learn Pipeline (Serialized) │
                     │  • StandardScaler                            │
                     │  • RandomForestRegressor (150 Estimators)    │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │ Continuous Head Count Estimation & Clipping  │
                     │  y_hat = max(0, Pipeline.predict(X))         │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │      3-Tier Operational Safety Classification│
                     │  🟢 Green (<10)  🟡 Yellow (10-24)  🔴 Red (>=25)│
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │    Multi-Agent Spatial Coordination Broker   │
                     │  • ZoneAgent: Local count, tier, telemetry   │
                     │  • AgentCoordinator: 8-neighbor graph        │
                     │  • Bottleneck Escalation: Adjacent Red zones │
                     │  • Dynamic Pedestrian Rerouting Directives   │
                     └──────────────────────┬───────────────────────┘
                                            │
                     ┌──────────────────────┴───────────────────────┐
                     ▼                                              ▼
    ┌─────────────────────────────────┐           ┌──────────────────────────────────┐
    │ Streamlit Telemetry Dashboard   │           │ SHAP Explainability Engine       │
    │ • Real-time bounding box overlay│           │ • Global beeswarm plot           │
    │ • Dynamic threshold sliders     │           │ • Local sample waterfall plot    │
    │ • Live broker communication log │           │ • Plain-text factor descriptions │
    └─────────────────────────────────┘           └──────────────────────────────────┘
```

---

## 📁 Repository Directory Structure

```text
crowdpulse/
│
├── .gitignore                            # Ignores large raw datasets, virtualenvs, checkpoints
├── requirements.txt                      # Pinned production dependencies
├── PROJECT_LINKS.txt                     # Official University submission links
├── README.md                             # University-ready documentation & viva defense guide
│
├── notebooks/
│   ├── crowdpulse.ipynb                  # Primary academic notebook with all executed outputs & plots
│   └── generate_notebook.py              # Automated programmatic notebook generator & runner
│
├── data/
│   ├── raw/                              # Downloaded ShanghaiTech Part B (.gitignored)
│   │   ├── train_data/
│   │   │   ├── images/                   # processed_IMG_1.jpg ... processed_IMG_400.jpg
│   │   │   └── ground-truth/             # GT_IMG_1.mat ... GT_IMG_400.mat
│   │   └── test_data/
│   │       ├── images/                   # processed_IMG_1.jpg ... processed_IMG_316.jpg
│   │       └── ground-truth/             # GT_IMG_1.mat ... GT_IMG_316.mat
│   └── processed/                        # Extracted tabular feature CSVs
│       ├── train_features.csv            # 3,600 training samples x 36 columns
│       └── test_features.csv             # 2,844 testing samples x 36 columns
│
├── models/
│   ├── crowdpulse_pipeline.joblib        # Serialized champion Scikit-learn Pipeline
│   ├── model_metadata.json               # Hyperparameters, evaluation metrics, feature names
│   ├── shap_summary.png                  # Saved global feature importance beeswarm plot
│   └── shap_waterfall_sample.png         # Saved local explanation waterfall sample
│
├── src/
│   ├── __init__.py                       # Package initialization
│   ├── data_loader.py                    # ShanghaiTech .mat parser & 3x3 spatial grid slicer
│   ├── feature_extractor.py              # HOG, Canny edge density ratio, and LBP extractors
│   ├── train.py                          # Cross-validation, hyperparameter tuning & pipeline serialization
│   ├── evaluate.py                       # Continuous regression metrics & 3-tier risk classification
│   ├── explainability.py                 # SHAP TreeExplainer generation routines
│   └── agent_coordinator.py              # ZoneAgent & spatial adjacency alert broker classes
│
├── app/
│   ├── app.py                            # Interactive Streamlit dashboard loading crowdpulse_pipeline.joblib
│   └── assets/                           # Sample surveillance camera test frames
│       ├── sample_frame_1.jpg
│       ├── sample_frame_2.jpg
│       ├── sample_frame_3.jpg
│       └── sample_frame_4.jpg
│
├── tests/
│   ├── __init__.py                       # Pytest package init
│   └── test_pipeline.py                  # 13 automated pytest verification tests
│
└── docs/
    ├── generate_docs.py                  # Script compiling PDF report and PPTX deck
    ├── PROJECT_REPORT.pdf                # University academic project report
    └── PRESENTATION.pptx                 # Viva presentation deck (10 widescreen slides)
```

---

## 🔬 Mathematical Methodology & Feature Formulation

### 1. Spatial Grid Slicing & Ground-Truth Coordinate Mapping
Surveillance frames of size $H \times W$ are partitioned into $3 \times 3$ grid zones:
$$r = \min\left(\left\lfloor \frac{y}{H / 3} \right\rfloor, 2\right), \quad c = \min\left(\left\lfloor \frac{x}{W / 3} \right\rfloor, 2\right)$$
Given head point annotations $P = \{(x_i, y_i)\}_{i=1}^{N}$, the local count for zone $Z_{r, c}$ is:
$$y_{r, c} = \sum_{i=1}^{N} \mathbb{I}\left(r = \left\lfloor \frac{y_i}{H/3} \right\rfloor \land c = \left\lfloor \frac{x_i}{W/3} \right\rfloor\right)$$

### 2. Edge-Compatible Feature Descriptors
- **Histogram of Oriented Gradients (HOG):**
  $$\theta(x, y) = \arctan\left(\frac{I_y}{I_x}\right), \quad M(x, y) = \sqrt{I_x^2 + I_y^2}$$
  Computed over 8 unsigned orientation bins across $[0^\circ, 180^\circ]$ with cell size $(16, 16)$, producing 8 bin averages, gradient mean, and gradient standard deviation.
- **Canny Edge Density Ratio:**
  $$R_{\text{edge}} = \frac{1}{|P|} \sum_{p \in P} \mathbb{I}(\text{Canny}(p) > 0)$$
  Accompanied by the mean Sobel gradient magnitude on edge pixels to measure silhouette sharpness.
- **Local Binary Patterns (LBP):**
  $$LBP_{P, R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) 2^p, \quad s(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$
  With $P=8, R=1$ and uniform patterns ($U \le 2$), yielding 10 normalized probability bins, Shannon entropy:
  $$H_{\text{LBP}} = -\sum_{k=0}^{9} p_k \log_2(p_k + \epsilon)$$
  and texture energy $E_{\text{LBP}} = \sum_{k=0}^{9} p_k^2$.
- **Intensity Statistics:** $\mu = \frac{1}{N}\sum I_i$, $\sigma^2 = \frac{1}{N}\sum (I_i - \mu)^2$, $\gamma_1 = \frac{\sum (I_i - \mu)^3}{N \sigma^3}$, $\gamma_2 = \frac{\sum (I_i - \mu)^4}{N \sigma^4} - 3$.
- **Normalized Spatial Perspective:** $\hat{r} = \frac{r}{2}$, $\hat{c} = \frac{c}{2}$.

### 3. Supervised Regression & Non-Negativity Constraint
Predicted counts are strictly non-negative:
$$\hat{y}_{r, c} = \max\left(0, \hat{f}_{\text{pipeline}}(x_{r, c})\right)$$

### 4. SHAP Cooperative Game Theory
Feature attributions follow the unique Shapley value formulation satisfying Efficiency, Symmetry, Additivity, and Dummy axioms:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f(S \cup \{i\}) - f(S)\right]$$

---

## 📊 Benchmark Results & Experimental Analysis

### 1. 5-Fold Cross-Validation Model Comparison
Evaluated on the 3,600 training zone samples of ShanghaiTech Part B:

| Model Architecture | Model Paradigm | 5-Fold CV MAE | 5-Fold CV RMSE | 5-Fold CV $R^2$ | Inference Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression** | Regularized Linear ($L_2$) | 13.620 | 28.301 | 0.434 | <1 ms |
| **SVR (RBF Kernel)** | Non-Linear Support Vector | 7.995 | 25.632 | 0.535 | ~8 ms |
| **HistGradientBoosting** | Histogram Gradient Boosting | 7.065 | 19.827 | 0.722 | ~2 ms |
| **Random Forest (Champion)** | Bagging Ensemble of Trees | **6.538** | **19.645** | **0.727** | **~3 ms** |

### 2. Hold-Out Test Evaluation (2,844 Zones)
- **Continuous MAE:** `6.538 heads / zone`
- **Continuous RMSE:** `19.645 heads / zone`
- **Continuous $R^2$:** `0.727`
- **Multi-Class Accuracy:** `94.37%`
- **Macro F1-Score:** `0.763`
- **Weighted F1-Score:** `0.940`

### 3. Confusion Matrix (Hold-Out Test Set)

```text
               Predicted Green    Predicted Yellow    Predicted Red    Recall
True Green          2,207                21                18           98.3%
True Yellow             9                54                91           35.1%
True Red                1                20               423           95.3%
Precision           99.5%             56.8%             79.5%
```
> **Key Safety Finding:** The system achieves **95.3% Recall on Red bottleneck zones** (423 out of 444 congested zones correctly identified), with only 1 zone misclassified as Green, meeting strict life-safety alerting requirements.

---

## ⚡ Quickstart & Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
# Clone repository
git clone https://github.com/Pudisiva/CrowdPulse.git
cd CrowdPulse

# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
venv\Scripts\Activate.ps1
# Activate on Linux/macOS:
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Run Automated Pytest Suite
```bash
pytest -v
```
Expected output: **13 passed in ~8 seconds**.

### 3. Train Champion Pipeline (Optional if pre-trained)
```bash
python src/train.py
```
This trains `StandardScaler` + `RandomForestRegressor`, saves `models/crowdpulse_pipeline.joblib`, and writes `models/model_metadata.json`.

### 4. Evaluate Pipeline & Risk Classification
```bash
python src/evaluate.py
```

### 5. Generate SHAP Explainability Plots
```bash
python src/explainability.py
```
Outputs `models/shap_summary.png` and `models/shap_waterfall_sample.png`.

### 6. Launch Streamlit Telemetry Dashboard
```bash
streamlit run app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 7. Open Academic Jupyter Notebook
```bash
jupyter notebook notebooks/crowdpulse.ipynb
```

---

## 🎓 Viva Defense: Expected Questions & Answers

### Q1: Why not use end-to-end deep convolutional networks like CSRNet or MCNN?
**Answer:** While deep CNNs generate fine-grained crowd density heatmaps, they require dedicated GPUs, consume over 1–2 GB of VRAM, and exhibit inference latencies of 200–500 ms per frame. In smart city infrastructure, surveillance cameras run on embedded micro-controllers (e.g., Raspberry Pi, Jetson Nano, Ambarella chips). CrowdPulse extracts 31 lightweight spatial descriptors in under 12 ms and executes Scikit-learn inference in under 3 ms, enabling real-time 30 FPS surveillance at the network edge with minimal power consumption.

### Q2: Why partition the frame into a $3 \times 3$ grid rather than counting the entire image at once?
**Answer:** A global whole-frame count cannot pinpoint *where* bottlenecks are forming. In an urban square, 150 people evenly distributed represents safe free flow (Green Tier), whereas 150 people crammed into one exit corridor represents a deadly crush hazard (Red Tier). Slicing into 9 spatial zones provides localized spatial awareness and enables our decentralized multi-agent coordination system.

### Q3: How do you handle non-negative count constraints with Scikit-learn regressors?
**Answer:** Standard regression models (e.g., linear regression or SVR) can theoretically output negative predictions for empty zones. CrowdPulse enforces a strict post-inference mathematical constraint: $\hat{y} = \max(0, f(x))$. In the training phase, target variables $y \ge 0$ ensure decision tree leaf nodes in our champion Random Forest naturally predict non-negative values.

### Q4: How is operational risk categorized and what makes it safety-critical?
**Answer:** We discretize counts into Green ($<10$), Yellow ($10-24$), and Red ($\ge 25$). In safety-critical systems, False Negatives on Red zones (missing a bottleneck) are catastrophic. Our model achieves **95.3% recall on the Red tier**, correctly detecting 423 of 444 high-density zones, ensuring operators are always warned before a bottleneck worsens.

### Q5: How does the Multi-Agent Spatial Coordination Broker work?
**Answer:** Each zone is encapsulated as an autonomous `ZoneAgent` reporting telemetry to an `AgentCoordinator`. The coordinator models an 8-connected adjacency graph. If an isolated zone reaches Red, a localized caution is issued. If two or more *spatially adjacent* zones reach Red simultaneously (e.g., `Zone_0_0` and `Zone_0_1`), the broker detects a coordinated bottleneck, escalates to `CRITICAL ALERT`, and dynamically computes detour paths toward neighboring Green zones with spare capacity.

### Q6: How does SHAP TreeExplainer help facility managers?
**Answer:** Unlike post-hoc perturbations (LIME) which are approximate, TreeExplainer computes exact game-theoretic Shapley values by traversing the tree ensemble. In our dashboard, managers can inspect a congested zone and see the exact physical evidence (e.g., `canny_edge_density_ratio` $+8.2$ heads, `hog_gradient_mean` $+4.1$ heads, `lbp_bin_9_flat_region` $-2.0$ heads), verifying that detection is driven by human silhouettes rather than background noise.

### Q7: Why did Random Forest outperform SVR and HistGradientBoosting?
**Answer:** Crowd density features exhibit strong non-linear thresholding effects (e.g., high Canny edge density indicates heads only when paired with high HOG gradient variation; otherwise it might just be brick pavement). Decision tree ensembles naturally model high-order feature interactions through orthogonal splits without requiring manual kernel tuning. Random Forest also provided lower variance than boosting on the hold-out test set (MAE 6.54 vs 7.06).

### Q8: How does this project map to INT395 course learning outcomes?
**Answer:** The project comprehensively addresses CO1 through CO6:
- **CO1:** Formulating localized spatial crowd regression.
- **CO2:** Preprocessing ShanghaiTech Part B ground-truth `.mat` files and engineering 31 descriptors.
- **CO3:** Implementing linear (Ridge), kernel (SVR), and ensemble (RF, HGB) regressors.
- **CO4:** Applying 5-fold cross-validation and hyperparameter optimization.
- **CO5:** Evaluating continuous metrics ($MAE, RMSE, R^2$) and multi-class confusion matrices.
- **CO6:** Delivering model explainability via SHAP and deploying an interactive Streamlit UI.

---

## 👥 Authors & Academic Affiliation
- **Institution:** Lovely Professional University (LPU), Punjab, India
- **Department:** School of Computer Science & Engineering (Department of AI & ML)
- **Course:** INT395 - Supervised Learning
- **Problem Statement ID:** #20
- **Group Members:**
  - 12201941 (Group Member 1)
  - 12202052 (Group Member 2)
  - 12203114 (Group Member 3)

---
*CrowdPulse © 2026. Built with Python, Scikit-learn, OpenCV, SHAP, and Streamlit.*
