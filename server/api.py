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
    """Loads and caches YOLO models (yolov8n, yolov8s, yolov8m)."""
    if model_name not in MODELS:
        model_file = PROJECT_ROOT / model_name
        target = str(model_file) if model_file.exists() else model_name
        print(f"Loading detection model '{target}'...")
        MODELS[model_name] = YOLO(target)
        print(f"Model '{model_name}' loaded successfully.")
    return MODELS[model_name]

# Preload default model
get_yolo_model("yolov8m.pt")


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
) -> dict:
    """
    Executes live human detection with adaptive high-resolution inference,
    maps persons into 3x3 zones, evaluates multi-agent alerts, and returns
    annotated imagery with telemetry data.
    """
    start_time = time.time()
    h, w = image.shape[:2]

    # Adaptive image resolution: if image is large, use 1280 or higher
    max_dim = max(h, w)
    effective_imgsz = imgsz
    if imgsz == 0:  # Auto mode
        effective_imgsz = 1280 if max_dim >= 1200 else 960 if max_dim >= 800 else 640

    detector = get_yolo_model(model_name)

    # Run inference for person class (class 0 in COCO)
    results = detector(
        image,
        classes=[0],
        conf=confidence,
        iou=iou,
        imgsz=effective_imgsz,
        verbose=False,
    )
    boxes = results[0].boxes

    # Slicing grid dimensions (3x3 default)
    grid_rows, grid_cols = 3, 3
    row_step = h / float(grid_rows)
    col_step = w / float(grid_cols)

    # Initialize zone counts
    zone_counts = {f"Zone_{r}_{c}": 0 for r in range(grid_rows) for c in range(grid_cols)}
    detections_list = []

    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            # Determine spatial center of detected person
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # Map to grid zone
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

        if draw_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, line_thick)
            label = f"#{det['id']} {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (255, 255, 255), 1, cv2.LINE_AA)

        if draw_points:
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
        "model_used": model_name,
        "resolution_used": f"{effective_imgsz}px",
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
