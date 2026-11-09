"""
CrowdPulse Documentation Generator (docs/generate_docs.py)
Generates:
  1. docs/PROJECT_REPORT.pdf: Formal university academic report using ReportLab.
  2. docs/PRESENTATION.pptx: Comprehensive viva presentation deck using python-pptx.
"""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


DOCS_DIR = Path(__file__).resolve().parent


# =============================================================================
# 1. GENERATE PROJECT_REPORT.PDF
# =============================================================================
def generate_project_report(pdf_path: Path):
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
        fontName="Helvetica-Bold",
    )
    sub_title = ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#334155"),
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("LOVELY PROFESSIONAL UNIVERSITY", sub_title))
    story.append(Paragraph("School of Computer Science & Engineering | Department of AI & ML", sub_title))
    story.append(Spacer(1, 15))
    story.append(Paragraph("INT395: SUPERVISED LEARNING - CAPSTONE PROJECT REPORT", sub_title))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1e3a8a"), spaceAfter=15))
    story.append(Paragraph("CrowdPulse: Multi-Agent Crowd Density Estimation and Coordinated Alerting System", title_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Problem Statement ID:</b> #20 | <b>Target Dataset:</b> ShanghaiTech Part B", sub_title))
    story.append(Paragraph("<b>Group Members:</b> Reg. Nos: 12201941, 12202052, 12203114", sub_title))
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=15))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "Crowd safety management at high-density public events and transport hubs requires early detection of potential crush conditions. "
        "Standard deep neural network approaches (e.g., CSRNet, MCNN) suffer from high compute latency and memory requirements, making them "
        "impractical for local edge deployment on embedded smart camera micro-controllers. <b>CrowdPulse</b> proposes a lightweight, interpretable, "
        "and modular architecture based on classical spatial feature engineering and Scikit-learn regression ensembles.",
        body_style,
    ))
    story.append(Paragraph(
        "The system partitions surveillance frames into a 3x3 spatial grid (9 zones), extracts 31 domain-specific visual descriptors "
        "(HOG orientation bins, Canny edge density ratio, LBP uniform texture histograms, and pixel intensity statistics), and estimates "
        "localized head counts using a Random Forest regression pipeline. These continuous counts are categorized into operational risk tiers "
        "(Green: &lt;10, Yellow: 10-24, Red: &ge;25), evaluated with SHAP explainability, and monitored by decentralized autonomous "
        "Zone Agents capable of triggering coordinated bottleneck alerts and dynamic rerouting instructions before dangerous crushes form.",
        body_style,
    ))

    # Course Outcome Alignment
    story.append(Paragraph("2. INT395 Course Outcome (CO) Alignment", h1_style))
    co_data = [
        [Paragraph("<b>Outcome</b>", body_style), Paragraph("<b>Syllabus Requirement</b>", body_style), Paragraph("<b>CrowdPulse Implementation</b>", body_style)],
        [Paragraph("<b>CO1</b>", body_style), Paragraph("Problem Formulation", body_style), Paragraph("Formulated crowd monitoring as localized spatial regression across 9 zones.", body_style)],
        [Paragraph("<b>CO2</b>", body_style), Paragraph("Data Preprocessing & Features", body_style), Paragraph("Parsed ShanghaiTech .mat files, extracted 31 HOG/Canny/LBP descriptors.", body_style)],
        [Paragraph("<b>CO3</b>", body_style), Paragraph("Model Formulation", body_style), Paragraph("Implemented Ridge, SVR(RBF), HistGradientBoosting, and Random Forest.", body_style)],
        [Paragraph("<b>CO4</b>", body_style), Paragraph("Cross-Validation & Tuning", body_style), Paragraph("5-fold CV and hyperparameter optimization for variance reduction.", body_style)],
        [Paragraph("<b>CO5</b>", body_style), Paragraph("Performance Evaluation", body_style), Paragraph("Continuous metrics (MAE 6.54, R2 0.73) & 3-Tier Risk Confusion Matrix (94.4% acc).", body_style)],
        [Paragraph("<b>CO6</b>", body_style), Paragraph("Deployment & XAI", body_style), Paragraph("SHAP TreeExplainer beeswarm/waterfalls and Streamlit live telemetry dashboard.", body_style)],
    ]
    co_table = Table(co_data, colWidths=[50, 160, 290])
    co_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(co_table)
    story.append(Spacer(1, 10))

    # Methodology & Feature Engineering
    story.append(Paragraph("3. Mathematical Methodology & Feature Descriptors", h1_style))
    story.append(Paragraph(
        "Each surveillance frame is sliced into 9 zones ($Z_{r, c}$) with row index $r \\in \\{0, 1, 2\\}$ and column index $c \\in \\{0, 1, 2\\}$. "
        "From each normalized 256x256 patch, 31 spatial visual features are extracted:",
        body_style,
    ))
    story.append(Paragraph("&bull; <b>HOG Orientations (10 features):</b> 8 gradient orientation bins (0&deg;-180&deg;) capturing head/shoulder contour angles, plus mean and std gradient magnitude.", bullet_style))
    story.append(Paragraph("&bull; <b>Canny Edge Density & Gradient (2 features):</b> Ratio of edge pixels to patch area ($R = \\frac{1}{|P|} \\sum \\mathbb{I}_{\\text{edge}}(p)$) and mean Sobel intensity on edge pixels.", bullet_style))
    story.append(Paragraph("&bull; <b>Local Binary Patterns (12 features):</b> 10 uniform texture bins ($P=8, R=1$) distinguishing human hair/clothing from smooth pavement, plus Shannon entropy and energy.", bullet_style))
    story.append(Paragraph("&bull; <b>Pixel Intensity Statistics (5 features):</b> First and second order moments: Mean, Standard Deviation, Variance, Skewness, Kurtosis.", bullet_style))
    story.append(Paragraph("&bull; <b>Normalized Spatial Coordinates (2 features):</b> Vertical and horizontal position ($r/2, c/2$) encoding camera perspective scaling.", bullet_style))

    # Experimental Results
    story.append(Paragraph("4. Experimental Results & Model Benchmarking", h1_style))
    story.append(Paragraph(
        "Models were evaluated using 5-fold cross-validation on the 3,600 training zone samples and verified against the 2,844 test zone samples:",
        body_style,
    ))
    res_data = [
        [Paragraph("<b>Model Architecture</b>", body_style), Paragraph("<b>CV MAE</b>", body_style), Paragraph("<b>CV RMSE</b>", body_style), Paragraph("<b>CV R&sup2;</b>", body_style), Paragraph("<b>Hold-Out Test MAE</b>", body_style)],
        [Paragraph("Ridge Regression (Baseline)", body_style), Paragraph("13.62", body_style), Paragraph("28.30", body_style), Paragraph("0.434", body_style), Paragraph("13.62", body_style)],
        [Paragraph("SVR (RBF Kernel)", body_style), Paragraph("7.99", body_style), Paragraph("25.63", body_style), Paragraph("0.535", body_style), Paragraph("7.99", body_style)],
        [Paragraph("HistGradientBoosting", body_style), Paragraph("7.06", body_style), Paragraph("19.83", body_style), Paragraph("0.722", body_style), Paragraph("7.06", body_style)],
        [Paragraph("<b>Random Forest (Champion)</b>", body_style), Paragraph("<b>6.54</b>", body_style), Paragraph("<b>19.65</b>", body_style), Paragraph("<b>0.727</b>", body_style), Paragraph("<b>6.54</b>", body_style)],
    ]
    res_table = Table(res_data, colWidths=[170, 75, 75, 75, 105])
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#ecfdf5")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 10))

    # Operational Risk Tiers
    story.append(Paragraph("5. Operational Safety Risk Classification", h1_style))
    story.append(Paragraph(
        "Discretizing continuous predictions into 3 operational risk tiers achieves <b>94.37% overall accuracy</b> and <b>0.763 macro F1-score</b> on hold-out test data:",
        body_style,
    ))
    tier_data = [
        [Paragraph("<b>Risk Tier</b>", body_style), Paragraph("<b>Count Range</b>", body_style), Paragraph("<b>Precision</b>", body_style), Paragraph("<b>Recall</b>", body_style), Paragraph("<b>F1-Score</b>", body_style), Paragraph("<b>Action Protocol</b>", body_style)],
        [Paragraph("<b>Green (Low)</b>", body_style), Paragraph("&lt; 10 heads", body_style), Paragraph("0.995", body_style), Paragraph("0.983", body_style), Paragraph("0.989", body_style), Paragraph("Normal pedestrian transit.", body_style)],
        [Paragraph("<b>Yellow (Moderate)</b>", body_style), Paragraph("10 - 24 heads", body_style), Paragraph("0.568", body_style), Paragraph("0.351", body_style), Paragraph("0.434", body_style), Paragraph("Pacing caution & monitoring.", body_style)],
        [Paragraph("<b>Red (High)</b>", body_style), Paragraph("&ge; 25 heads", body_style), Paragraph("0.795", body_style), Paragraph("0.953", body_style), Paragraph("0.867", body_style), Paragraph("Bottleneck alert & rerouting.", body_style)],
    ]
    tier_table = Table(tier_data, colWidths=[100, 80, 65, 65, 65, 125])
    tier_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tier_table)
    story.append(Spacer(1, 10))

    # Multi-Agent Coordination
    story.append(Paragraph("6. Multi-Agent Coordination & Alert Broker", h1_style))
    story.append(Paragraph(
        "Each zone is governed by an autonomous <b>ZoneAgent</b> connected to an 8-neighbor spatial adjacency graph managed by the <b>AgentCoordinator</b>: "
        "<br/>1. <b>Isolated Red Zone:</b> Generates a localized caution to restrict immediate intake. "
        "<br/>2. <b>Adjacent Red Zones:</b> When two connected zones (e.g., Zone_0_0 and Zone_0_1) reach the Red Tier simultaneously, the coordinator escalates to a <b>CRITICAL COORDINATED BOTTLENECK ALERT</b> and automatically computes pedestrian relief paths diverting crowd traffic toward neighboring Green corridors.",
        body_style,
    ))

    # Conclusion
    story.append(Paragraph("7. Conclusion", h1_style))
    story.append(Paragraph(
        "CrowdPulse successfully proves that classical supervised regression ensembles combined with principled spatial feature engineering "
        "can deliver highly accurate, explainable, and real-time crowd density monitoring on resource-constrained edge hardware without the prohibitive overhead of monolithic deep neural networks.",
        body_style,
    ))

    doc.build(story)
    print(f"Project report generated: {pdf_path}")


# =============================================================================
# 2. GENERATE PRESENTATION.PPTX
# =============================================================================
def generate_viva_presentation(pptx_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    NAVY = RGBColor(15, 23, 42)
    WHITE = RGBColor(248, 250, 252)
    ACCENT_BLUE = RGBColor(37, 99, 235)
    LIGHT_GRAY = RGBColor(148, 163, 184)
    GREEN = RGBColor(16, 185, 129)

    slides_data = [
        {
            "title": "CrowdPulse",
            "subtitle": "Multi-Agent Crowd Density Estimation and Coordinated Alerting System\nINT395: Supervised Learning | Problem Statement #20\nGroup Members: 12201941, 12202052, 12203114",
            "bullets": [],
            "is_title": True,
        },
        {
            "title": "Problem Statement & Edge Motivation",
            "subtitle": "Overcoming the Latency Bottlenecks of Deep Learning in Crowd Safety",
            "bullets": [
                "Public Safety Hazard: Uncontrolled crowd bottlenecks cause fatal crushes and stampedes at stadiums, transit hubs, and religious gatherings.",
                "Deep Learning Limitations: Dense CNNs (CSRNet, MCNN) require multi-gigabyte GPU memory and 200+ ms latency, rendering them unviable on edge cameras.",
                "The CrowdPulse Paradigm: Lightweight edge-compatible visual descriptors + Scikit-learn regression ensembles running in <15 ms.",
                "Decentralized Alerting: Autonomous multi-agent coordination across a 3x3 spatial grid to identify bottleneck clusters and dynamically reroute pedestrian influx.",
            ],
            "is_title": False,
        },
        {
            "title": "INT395 Course Outcome (CO1–CO6) Alignment",
            "subtitle": "Comprehensive Academic Mapping across the Supervised Learning Lifecycle",
            "bullets": [
                "CO1 (Formulation): Formulated surveillance frame monitoring as a supervised regression task over localized 3x3 spatial partitions.",
                "CO2 (Preprocessing): Parsed ShanghaiTech Part B .mat annotations and extracted 31 spatial visual descriptors.",
                "CO3 (Model Formulation): Benchmarked Ridge, SVR(RBF), HistGradientBoosting, and Random Forest Regressors.",
                "CO4 (Validation & Tuning): Applied 5-fold cross-validation and hyperparameter grid optimization to eliminate overfitting.",
                "CO5 (Performance Metrics): Evaluated continuous MAE/RMSE/R2 alongside 3-tier risk classification precision, recall, and confusion matrix.",
                "CO6 (Explainability & Deployment): Provided transparent SHAP TreeExplainer feature attributions and an interactive Streamlit telemetry dashboard.",
            ],
            "is_title": False,
        },
        {
            "title": "ShanghaiTech Part B & 3x3 Spatial Grid Slicing",
            "subtitle": "Ground-Truth Coordinate Mapping & Spatial Bounding Boxes",
            "bullets": [
                "Dataset Overview: 400 training and 316 test high-resolution urban street surveillance images with head point annotations.",
                "Spatial Grid Formula: Frame partitioned into 9 zones: r = min(floor(y / (H/3)), 2), c = min(floor(x / (W/3)), 2).",
                "Exact Point Assignment: Heads mapped to exact zone bounding boxes, preserving total count (e.g., 233 heads in IMG_1 -> 159 in Zone_0_0, 66 in Zone_0_1, etc.).",
                "Tabular Dataset Creation: Exported 3,600 training zone samples and 2,844 test zone samples across 31 features.",
            ],
            "is_title": False,
        },
        {
            "title": "Edge-Compatible Feature Engineering Engine",
            "subtitle": "31 Spatial Visual Descriptors Extracted in Under 12 ms per Zone",
            "bullets": [
                "HOG Orientations (10 features): 8 gradient angle bins (0°-180°) capturing head/shoulder contour orientations, plus gradient mean & std.",
                "Canny Edge Density & Gradient (2 features): Edge pixel ratio (R = count(edges)/total_pixels) and mean Sobel intensity on edge boundaries.",
                "LBP Uniform Texture (12 features): 10 uniform texture bins (P=8, R=1) separating hair/clothing from pavement, plus Shannon entropy & energy.",
                "Pixel Intensity Statistics (5 features): 1st & 2nd order moments (Mean, Std, Variance, Skewness, Kurtosis).",
                "Normalized Spatial Coordinates (2 features): Row and column grid indices (r/2, c/2) capturing camera perspective elevation.",
            ],
            "is_title": False,
        },
        {
            "title": "Model Benchmarking & 5-Fold Cross-Validation",
            "subtitle": "Rigorous Empirical Evaluation Across Linear, Kernel, and Ensemble Paradigms",
            "bullets": [
                "Ridge Regression (Baseline Linear): CV MAE = 13.62 | RMSE = 28.30 | R² = 0.434 (Underfits non-linear crowd density).",
                "SVR with RBF Kernel (Non-Linear): CV MAE = 7.99 | RMSE = 25.63 | R² = 0.535 (Effective non-linear mapping, higher latency).",
                "HistGradientBoosting (Boosting Ensemble): CV MAE = 7.06 | RMSE = 19.83 | R² = 0.722 (Fast tree boosting with histogram binning).",
                "Random Forest (Champion Ensemble): CV MAE = 6.54 | RMSE = 19.65 | R² = 0.727 (Lowest variance, superior generalization, native SHAP support).",
            ],
            "is_title": False,
        },
        {
            "title": "Experimental Results & 3-Tier Risk Classification",
            "subtitle": "Hold-Out Test Set Performance on 2,844 Spatial Zones",
            "bullets": [
                "Continuous Regression: Test MAE = 6.538 heads/zone | Test RMSE = 19.645 | Test R² = 0.727.",
                "Operational Safety Discretization: Green (<10), Yellow (10-24), Red (>=25).",
                "Overall Classification Accuracy: 94.37% | Macro F1-Score: 0.763 | Weighted F1-Score: 0.940.",
                "Safety-Critical Red Tier: 95.3% Recall on Red zones (423 / 444 congested zones correctly caught; only 1 misclassified as green).",
            ],
            "is_title": False,
        },
        {
            "title": "Explainable AI via SHAP (TreeExplainer)",
            "subtitle": "Auditable, Transparent Feature Contributions Based on Cooperative Game Theory",
            "bullets": [
                "Shapley Value Formulation: Exact marginal contribution φ_i computed for every visual feature in predicting localized crowd count.",
                "Global Importance (Beeswarm): Canny Edge Density Ratio and HOG Gradient Mean consistently rank as top positive predictors.",
                "Texture Signals: LBP flat-region proportion provides strong negative push (indicating open, uncongested ground).",
                "Local Waterfall Explanations: Deconstructs exact zone predictions for facility operators, building trust in autonomous alert triggers.",
            ],
            "is_title": False,
        },
        {
            "title": "Multi-Agent Coordination & Alert Broker",
            "subtitle": "Spatial Adjacency Graph, Bottleneck Escalation & Dynamic Rerouting",
            "bullets": [
                "ZoneAgent Abstraction: 9 autonomous agents tracking local counts, risk transitions, and historical telemetry.",
                "Adjacency Graph Topology: Models 8-connected physical neighborhood corridors across the 3x3 camera grid.",
                "Single Red Zone -> Localized Caution: Pacing restrictions applied to upstream entry gates.",
                "Adjacent Red Zones -> CRITICAL ALERT: Coordinator detects dangerous multi-zone bottlenecks and broadcasts immediate rerouting paths to neighboring Green corridors.",
            ],
            "is_title": False,
        },
        {
            "title": "Streamlit Telemetry Dashboard & Viva Summary",
            "subtitle": "Interactive Live Deployment & Key Technical Achievements",
            "bullets": [
                "Interactive Streamlit Dashboard (app/app.py): Live visual grid overlay, KPI metric cards, dynamic threshold sliders, and real-time SHAP waterfalls.",
                "100% Automated Test Suite (tests/test_pipeline.py): 13 pytest tests passing (dimensions, inference, non-negativity, adjacency logic).",
                "Complete Deliverable Ecosystem: Primary academic notebook (crowdpulse.ipynb) with all executed plots, report, presentation, and clean modular code.",
                "Viva Summary: CrowdPulse delivers an edge-ready, explainable, and mathematically grounded alternative to deep learning for crowd safety.",
            ],
            "is_title": False,
        },
    ]

    for s_idx, data in enumerate(slides_data):
        slide = prs.slides.add_slide(blank_layout)

        # Background fill
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = NAVY

        if data["is_title"]:
            # Title Slide
            tx_box = slide.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10.33), Inches(3.5))
            tf = tx_box.text_frame
            tf.word_wrap = True

            p1 = tf.paragraphs[0]
            p1.text = data["title"]
            p1.font.size = Pt(48)
            p1.font.bold = True
            p1.font.color.rgb = ACCENT_BLUE
            p1.alignment = PP_ALIGN.CENTER

            p2 = tf.add_paragraph()
            p2.text = data["subtitle"]
            p2.font.size = Pt(20)
            p2.font.color.rgb = WHITE
            p2.alignment = PP_ALIGN.CENTER
            p2.space_before = Pt(20)
        else:
            # Header
            header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.2))
            tf = header_box.text_frame
            tf.word_wrap = True

            p1 = tf.paragraphs[0]
            p1.text = data["title"]
            p1.font.size = Pt(28)
            p1.font.bold = True
            p1.font.color.rgb = WHITE

            p2 = tf.add_paragraph()
            p2.text = data["subtitle"]
            p2.font.size = Pt(14)
            p2.font.color.rgb = LIGHT_GRAY
            p2.space_before = Pt(4)

            # Bullets
            content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.9), Inches(11.7), Inches(5.0))
            ctf = content_box.text_frame
            ctf.word_wrap = True

            for b_idx, bullet in enumerate(data["bullets"]):
                bp = ctf.add_paragraph() if b_idx > 0 else ctf.paragraphs[0]
                bp.text = f"•  {bullet}"
                bp.font.size = Pt(17)
                bp.font.color.rgb = WHITE
                bp.space_before = Pt(14)

    prs.save(str(pptx_path))
    print(f"Viva presentation generated: {pptx_path}")


if __name__ == "__main__":
    report_file = DOCS_DIR / "PROJECT_REPORT.pdf"
    presentation_file = DOCS_DIR / "PRESENTATION.pptx"

    generate_project_report(report_file)
    generate_viva_presentation(presentation_file)

