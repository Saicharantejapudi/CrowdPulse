"""
CrowdPulse Live Detection FastAPI Backend (server/api.py)
Provides high-precision live human detection, adaptive high-resolution inference,
3x3 spatial grid mapping, crowd count aggregation, and multi-agent bottleneck alerting.
"""

import base64
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from ultralytics import YOLO

from src.agent_coordinator import AgentCoordinator

app = FastAPI(
    title="CrowdPulse Live Crowd Detection API",
    description="High-Precision human detection & spatial crowd telemetry using YOLO and multi-agent coordination.",
    version="2.1.0",
)

# Enable CORS for React frontend (Vite default: http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded models
MODELS = {}

def get_yolo_model(model_name: str = "yolov8m.pt") -> YOLO:
    """Loads and caches YOLO models (yolov8n, yolov8s, yolov8m, yolov8_head_detector.pt)."""
    if model_name not in MODELS:
        model_file = PROJECT_ROOT / "models" / model_name
        if not model_file.exists():
            model_file = PROJECT_ROOT / model_name
        target = str(model_file) if model_file.exists() else model_name
        print(f"Loading detection model '{target}'...")
        MODELS[model_name] = YOLO(target)
        print(f"Model '{model_name}' loaded successfully.")
    return MODELS[model_name]

# Preload lightweight default model for instant health checks without exceeding free cloud RAM
try:
    get_yolo_model("yolov8n.pt")
except Exception as e:
    print(f"Deferred model loading: {e}")

CLASSICAL_PIPELINE = None

def get_classical_pipeline():
    """Loads and caches the INT395 trained Scikit-learn crowd counting pipeline."""
    global CLASSICAL_PIPELINE
    if CLASSICAL_PIPELINE is None:
        model_file = PROJECT_ROOT / "models" / "crowdpulse_pipeline.joblib"
        if model_file.exists():
            import joblib
            print(f"Loading INT395 classical pipeline '{model_file}'...")
            CLASSICAL_PIPELINE = joblib.load(str(model_file))
            print("Classical pipeline loaded successfully.")
    return CLASSICAL_PIPELINE


PET_MODEL_INSTANCE = None

def get_pet_model():
    """Loads and caches the SkyeBrowse PET (Point-quEry Transformer) for ultra-dense crowds (1,000 - 10,000+ heads)."""
    global PET_MODEL_INSTANCE
    if PET_MODEL_INSTANCE is None:
        pet_dir = PROJECT_ROOT / "models" / "pet_crowd"
        weights_file = pet_dir / "weights" / "PET_Finetuned.safetensors"
        if not weights_file.exists():
            return None, None, None
        import sys
        if str(pet_dir) not in sys.path:
            sys.path.insert(0, str(pet_dir))
        import count
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SkyeBrowse PET Transformer on {device}...")
        model = count.build_pet(device)
        model.load_state_dict(count.load_state_dict(weights_file), strict=True)
        model.eval()
        PET_MODEL_INSTANCE = (model, device, count)
        print("SkyeBrowse PET Transformer loaded successfully.")
    return PET_MODEL_INSTANCE


def run_sliced_detection(
    image: np.ndarray,
    detector: YOLO,
    confidence: float = 0.06,
    tile_size: int = 480,
    overlap: int = 80,
    iou_thresh: float = 0.35,
) -> tuple:
    """
    Sliced-Aided Hyper Inference (SAHI style) for ultra-dense & aerial/drone crowds.
    Slices large high-res images into native-resolution patches so tiny overhead heads (5-15px)
    are not destroyed by global downsampling, runs NMS, and returns merged global boxes.
    """
    h, w = image.shape[:2]
    step = max(100, tile_size - overlap)
    all_boxes = []
    all_confs = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2 = min(h, y + tile_size)
            x2 = min(w, x + tile_size)
            patch = image[y:y2, x:x2]
            res = detector(patch, classes=[0], conf=confidence, imgsz=tile_size, verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy()
                    all_boxes.append([float(bx1 + x), float(by1 + y), float(bx2 + x), float(by2 + y)])
                    all_confs.append(float(b.conf[0].cpu().numpy()))

    if not all_boxes:
        return [], []

    nms_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in all_boxes]
    indices = cv2.dnn.NMSBoxes(nms_boxes, all_confs, score_threshold=confidence, nms_threshold=iou_thresh)

    final_boxes = []
    final_confs = []
    if len(indices) > 0:
        for idx in indices.flatten():
            final_boxes.append(all_boxes[idx])
            final_confs.append(all_confs[idx])

    return final_boxes, final_confs


def run_classical_detection(image: np.ndarray, grid_rows: int = 3, grid_cols: int = 3) -> dict:
    """
    Runs the INT395 Classical Supervised ML VotingRegressor pipeline (HOG + LBP + Canny edge density)
    across the 3x3 spatial grid.
    """
    from src.data_loader import partition_image_into_grid
    from src.feature_extractor import extract_patch_features
    import pandas as pd

    pipeline = get_classical_pipeline()
    if pipeline is None:
        raise ValueError("Classical model pipeline 'models/crowdpulse_pipeline.joblib' not found.")

    partitions = partition_image_into_grid(image, (grid_rows, grid_cols))
    feats = [extract_patch_features(p['patch'], row=p['row'], col=p['col']) for p in partitions]
    df = pd.DataFrame(feats)
    preds = pipeline.predict(df)

    zone_counts = {}
    for p, pred in zip(partitions, preds):
        zone_counts[p['zone_id']] = max(0, int(round(float(pred))))

    return zone_counts


def encode_image_base64(image: np.ndarray, quality: int = 88) -> str:
    """Encodes a BGR OpenCV image into a base64 JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image to JPEG.")
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def process_detection(
    image: np.ndarray,
    confidence: float = 0.15,
    iou: float = 0.45,
    imgsz: int = 1280,
    model_name: str = "yolov8m.pt",
    green_cutoff: float = 10.0,
    red_cutoff: float = 25.0,
    draw_boxes: bool = True,
    draw_grid: bool = True,
    draw_points: bool = True,
    detection_mode: str = "standard",  # "standard", "aerial_sliced", "classical"
) -> dict:
    """
    Executes live human detection with adaptive high-resolution inference,
    sliced native-resolution tiling for aerial/drone crowds, or classical
    INT395 feature regression. Maps persons into 3x3 zones, evaluates multi-agent
    alerts, and returns annotated imagery with telemetry data.
    """
    start_time = time.time()
    h, w = image.shape[:2]
    max_dim = max(h, w)

    # Slicing grid dimensions (3x3 default)
    grid_rows, grid_cols = 3, 3
    row_step = h / float(grid_rows)
    col_step = w / float(grid_cols)

    # Initialize zone counts & detection list
    zone_counts = {f"Zone_{r}_{c}": 0 for r in range(grid_rows) for c in range(grid_cols)}
    detections_list = []

    model_display = model_name
    resolution_display = f"{imgsz}px"

    if detection_mode == "classical":
        zone_counts = run_classical_detection(image, grid_rows, grid_cols)
        total_count = sum(zone_counts.values())
        model_display = "CrowdPulse VotingRegressor (INT395 HOG + LBP + Canny)"
        resolution_display = "3x3 Spatial Feature Augmentation"
    elif detection_mode == "aerial_sliced":
        # Sliced Aided Hyper Inference (SAHI)
        effective_model = model_name
        detector = get_yolo_model(effective_model)
        sliced_conf = confidence
        raw_boxes, raw_confs = run_sliced_detection(
            image,
            detector,
            confidence=sliced_conf,
            tile_size=480,
            overlap=80,
            iou_thresh=iou,
        )
        for idx, (box, conf) in enumerate(zip(raw_boxes, raw_confs)):
            x1, y1, x2, y2 = box
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            r = max(0, min(grid_rows - 1, int(cy / row_step)))
            c = max(0, min(grid_cols - 1, int(cx / col_step)))
            zone_id = f"Zone_{r}_{c}"
            zone_counts[zone_id] += 1
            detections_list.append({
                "id": idx + 1,
                "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                "center": [round(cx, 1), round(cy, 1)],
                "confidence": round(float(conf), 3),
                "zone_id": zone_id,
                "row": r,
                "col": c,
            })
        total_count = len(detections_list)
        model_display = f"{effective_model} (Sliced Native-Resolution SAHI)"
        resolution_display = "480px Sliced Native Tiles"
    elif detection_mode == "pet_transformer":
        pet_tuple = get_pet_model()
        if pet_tuple[0] is None:
            raise ValueError("SkyeBrowse PET weights 'models/pet_crowd/weights/PET_Finetuned.safetensors' not found.")
        pet_model, pet_device, pet_count = pet_tuple

        # Adaptive threshold: if confidence >= 0.05 use slider value, default for dense crowds is 0.35
        pet_thresh = confidence if confidence >= 0.05 else 0.35
        upscale_factor = 1.35 if max_dim < 1200 else 1.0

        points, scores = pet_count.predict_candidates_tiled(
            pet_model, image, pet_device, upscale=upscale_factor, tile=768, overlap=192
        )
        filtered_points = pet_count.filter_predictions(points, scores, threshold=pet_thresh)

        for idx, pt in enumerate(filtered_points):
            cx = float(pt[0])
            cy = float(pt[1])
            r = max(0, min(grid_rows - 1, int(cy / row_step)))
            c = max(0, min(grid_cols - 1, int(cx / col_step)))
            zone_id = f"Zone_{r}_{c}"
            zone_counts[zone_id] += 1
            box_rad = 5.0
            detections_list.append({
                "id": idx + 1,
                "box": [round(cx - box_rad, 1), round(cy - box_rad, 1), round(cx + box_rad, 1), round(cy + box_rad, 1)],
                "center": [round(cx, 1), round(cy, 1)],
                "confidence": round(float(pet_thresh), 3),
                "zone_id": zone_id,
                "row": r,
                "col": c,
            })
        total_count = len(detections_list)
        model_display = "SkyeBrowse PET Transformer (Point-quEry Crowd Specialist)"
        resolution_display = f"768px Windowed Attention (PET {total_count:,} heads)"
    else:
        # Standard Single-Pass Detection
        max_dim = max(h, w)
        effective_imgsz = imgsz
        if imgsz == 0:
            effective_imgsz = 1280 if max_dim >= 1200 else 960 if max_dim >= 800 else 640

        detector = get_yolo_model(model_name)
        results = detector(
            image,
            classes=[0],
            conf=confidence,
            iou=iou,
            imgsz=effective_imgsz,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                r = max(0, min(grid_rows - 1, int(cy / row_step)))
                c = max(0, min(grid_cols - 1, int(cx / col_step)))
                zone_id = f"Zone_{r}_{c}"
                zone_counts[zone_id] += 1
                detections_list.append({
                    "id": idx + 1,
                    "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                    "confidence": round(conf, 3),
                    "zone_id": zone_id,
                    "row": r,
                    "col": c,
                })
        total_count = len(detections_list)
        model_display = model_name
        resolution_display = f"{effective_imgsz}px"

    # Multi-Agent Coordination Broker Evaluation
    coordinator = AgentCoordinator(
        grid_size=(3, 3),
        allow_diagonal_adjacency=True,
        green_cutoff=green_cutoff,
        red_cutoff=red_cutoff,
    )
    coordinator.update_zone_counts(zone_counts)
    system_state = coordinator.evaluate_system_state()

    # Draw Visualization Overlay
    annotated = image.copy()
    color_mask = np.zeros_like(image, dtype=np.uint8)

    tier_colors_bgr = {
        "Green": (50, 205, 50),     # Lime Green
        "Yellow": (20, 190, 245),   # Amber
        "Red": (40, 40, 240),       # Crimson
    }

    # Dynamic line scaling based on image dimensions
    base_scale = max(0.6, min(2.5, max_dim / 1200.0))
    line_thick = max(1, int(round(2 * base_scale)))
    font_scale = 0.55 * base_scale

    # 1. Draw 3x3 Spatial Grid Zones
    if draw_grid:
        for r in range(grid_rows):
            y_min = int(round(r * row_step))
            y_max = int(round((r + 1) * row_step)) if r < grid_rows - 1 else h
            for c in range(grid_cols):
                x_min = int(round(c * col_step))
                x_max = int(round((c + 1) * col_step)) if c < grid_cols - 1 else w
                zid = f"Zone_{r}_{c}"
                agent = coordinator.agents[zid]
                tier = agent.risk_tier
                cnt = agent.current_count
                color = tier_colors_bgr[tier]

                # Fill zone with soft tint
                cv2.rectangle(color_mask, (x_min, y_min), (x_max, y_max), color, -1)
                # Zone boundary
                cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, line_thick)

                # Zone badge
                badge_text = f"{zid}: {cnt:.0f} heads ({tier})"
                pad = int(14 * base_scale)
                cv2.putText(
                    annotated,
                    badge_text,
                    (x_min + pad, y_min + int(32 * base_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 1.1,
                    (255, 255, 255),
                    line_thick,
                    cv2.LINE_AA,
                )

        # Blend grid overlay
        annotated = cv2.addWeighted(color_mask, 0.18, annotated, 0.82, 0)

    # 2. Draw Person Bounding Boxes & Head Points
    for det in detections_list:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        conf = det["confidence"]
        zid = det["zone_id"]
        tier = coordinator.agents[zid].risk_tier
        box_color = tier_colors_bgr[tier]

        if draw_boxes and detection_mode != "pet_transformer":
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_thick)
            if total_count <= 250:
                label = f"#{det['id']} {conf*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), box_color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        if draw_points:
            if detection_mode == "pet_transformer":
                head_x = int(round(det["center"][0]))
                head_y = int(round(det["center"][1]))
                # Render high-contrast SkyeBrowse purple dot over each detected head
                cv2.circle(annotated, (head_x, head_y), 3, (215, 30, 160), -1)
            else:
                head_x = int(round(det["center"][0]))
                head_y = int(round(y1 + (y2 - y1) * 0.12))
                pt_radius = max(3, int(round(4 * base_scale)))
                cv2.circle(annotated, (head_x, head_y), pt_radius, (0, 255, 255), -1)
                cv2.circle(annotated, (head_x, head_y), pt_radius + 1, (0, 0, 0), 1)

    elapsed_ms = round((time.time() - start_time) * 1000.0, 1)
    annotated_b64 = encode_image_base64(annotated)

    # Construct 3x3 matrix for frontend display
    grid_matrix = [[zone_counts[f"Zone_{r}_{c}"] for c in range(3)] for r in range(3)]

    # Zone summaries
    zones_summary = []
    for zid, agent in coordinator.agents.items():
        zones_summary.append({
            "zone_id": zid,
            "row": agent.row,
            "col": agent.col,
            "count": int(agent.current_count),
            "tier": agent.risk_tier,
            "status": agent.alert_status,
        })

    return {
        "total_count": total_count,
        "inference_time_ms": elapsed_ms,
        "model_used": model_display,
        "resolution_used": resolution_display,
        "detection_mode": detection_mode,
        "image_dimensions": {"width": w, "height": h},
        "zone_counts": zone_counts,
        "grid_matrix": grid_matrix,
        "zones_summary": zones_summary,
        "detections": detections_list,
        "alert": {
            "system_status": system_state["system_status"],
            "status_level": system_state["status_level"],
            "red_zones": system_state["red_zones"],
            "yellow_zones": system_state["yellow_zones"],
            "adjacent_red_pairs": system_state["adjacent_red_pairs"],
            "rerouting_directives": system_state["rerouting_directives"],
        },
        "annotated_image": annotated_b64,
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "service": "CrowdPulse Live Detection Engine",
        "models_available": ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt"],
        "device": device,
    }


@app.get("/api/samples")
def get_sample_frames():
    """Returns available sample surveillance frames for quick UI testing."""
    assets_dir = PROJECT_ROOT / "app" / "assets"
    if not assets_dir.exists():
        return {"samples": []}

    files = sorted([f for f in os.listdir(assets_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    samples = []
    for f in files:
        samples.append({
            "id": f,
            "name": f"Surveillance Feed ({f})",
            "filename": f,
        })
    return {"samples": samples}


@app.post("/api/detect/sample/{filename}")
def detect_sample(
    filename: str,
    confidence: float = Form(0.15),
    iou: float = Form(0.45),
    imgsz: int = Form(1280),
    model_name: str = Form("yolov8m.pt"),
    green_cutoff: float = Form(10.0),
    red_cutoff: float = Form(25.0),
    detection_mode: str = Form("standard"),
):
    """Executes detection on a preloaded sample frame."""
    sample_path = PROJECT_ROOT / "app" / "assets" / filename
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample frame not found: {filename}")

    image = cv2.imread(str(sample_path))
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode sample image.")

    return process_detection(
        image,
        confidence=confidence,
        iou=iou,
        imgsz=imgsz,
        model_name=model_name,
        green_cutoff=green_cutoff,
        red_cutoff=red_cutoff,
        detection_mode=detection_mode,
    )


@app.post("/api/detect")
async def detect_upload(
    file: UploadFile = File(...),
    confidence: float = Form(0.15),
    iou: float = Form(0.45),
    imgsz: int = Form(1280),
    model_name: str = Form("yolov8m.pt"),
    green_cutoff: float = Form(10.0),
    red_cutoff: float = Form(25.0),
    draw_boxes: bool = Form(True),
    draw_grid: bool = Form(True),
    draw_points: bool = Form(True),
    detection_mode: str = Form("standard"),
):
    """
    Accepts an uploaded image file, runs real-time person detection,
    and returns exact crowd counts and visual telemetry.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file or format.")

        response = process_detection(
            image,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            model_name=model_name,
            green_cutoff=green_cutoff,
            red_cutoff=red_cutoff,
            draw_boxes=draw_boxes,
            draw_grid=draw_grid,
            draw_points=draw_points,
            detection_mode=detection_mode,
        )
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


# Mount built React frontend if available
DIST_PATH = PROJECT_ROOT / "frontend" / "dist"
if DIST_PATH.exists():
    app.mount("/", StaticFiles(directory=str(DIST_PATH), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("Starting CrowdPulse Live Detection API on http://127.0.0.1:8000...")
    uvicorn.run("server.api:app", host="0.0.0.0", port=8000, reload=True)
