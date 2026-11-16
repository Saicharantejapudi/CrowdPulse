import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent
PET_ROOT = REPO_ROOT / "PET"
sys.path.insert(0, str(PET_ROOT))

import util.misc as utils
from models import build_model

DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "PET_Finetuned.safetensors"
MAX_INFERENCE_SIDE = 2560

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize(rgb):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD

MODEL_ARGS = dict(
    backbone="vgg16_bn",
    position_embedding="sine",
    dec_layers=2,
    dim_feedforward=512,
    hidden_dim=256,
    dropout=0.0,
    nheads=8,
    set_cost_class=1,
    set_cost_point=0.05,
    ce_loss_coef=1.0,
    point_loss_coef=5.0,
    eos_coef=0.5,
    dataset_file="SHA",
    data_path="./data/ShanghaiTech/PartA",
)

MAC_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def select_device(requested):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state_dict(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    return checkpoint


def build_pet(device):
    args = SimpleNamespace(device=str(device), **MODEL_ARGS)
    previous_cwd = os.getcwd()
    os.chdir(PET_ROOT)
    try:
        model, _ = build_model(args)
    finally:
        os.chdir(previous_cwd)
    model.to(device)
    model.eval()
    return model


def resize_for_inference(rgb):
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= MAX_INFERENCE_SIDE:
        return rgb, 1.0
    scale = MAX_INFERENCE_SIDE / float(longest)
    resized = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
    return resized, scale


@torch.no_grad()
def predict_batch(model, crops_rgb, device, thrs):
    tensors = [normalize(crop) for crop in crops_rgb]
    samples = utils.nested_tensor_from_tensor_list(tensors).to(device)
    padded_h, padded_w = samples.tensors.shape[-2:]
    outputs = model(samples, test=True, thrs=thrs)

    points = outputs["pred_points"]
    logits = outputs["pred_logits"]
    if points.dim() == 3:
        points, logits = points[0], logits[0]
    points = points.detach().cpu().numpy()
    scores = torch.softmax(logits.detach().float().cpu(), -1)[:, 1].numpy()

    sparse_ends = np.cumsum([0] + outputs["sparse_keep_counts"].tolist())
    dense_ends = np.cumsum([0] + outputs["dense_keep_counts"].tolist()) + sparse_ends[-1]
    results = []
    for i, crop in enumerate(crops_rgb):
        idx = np.concatenate([
            np.arange(sparse_ends[i], sparse_ends[i + 1]),
            np.arange(dense_ends[i], dense_ends[i + 1]),
        ]).astype(int)
        crop_h, crop_w = crop.shape[:2]
        ys = np.clip(points[idx, 0] * float(padded_h), 0, crop_h - 1)
        xs = np.clip(points[idx, 1] * float(padded_w), 0, crop_w - 1)
        results.append((np.stack([xs, ys], axis=1), scores[idx]))
    return results


def tile_starts(extent, tile, stride, offset):
    if extent <= tile:
        return [0]
    last = extent - tile
    return sorted({0, last} | set(range(offset, last + 1, stride)))


def grid_pass(model, up_rgb, device, tile, overlap, thrs, offset):
    up_h, up_w = up_rgb.shape[:2]
    stride = max(1, tile - overlap)
    margin = overlap // 2
    ys_starts = tile_starts(up_h, tile, stride, offset)
    xs_starts = tile_starts(up_w, tile, stride, offset)

    tiles = []
    for iy, y0 in enumerate(ys_starts):
        for ix, x0 in enumerate(xs_starts):
            x1, y1 = min(x0 + tile, up_w), min(y0 + tile, up_h)
            own_x0 = 0 if ix == 0 else x0 + margin
            own_y0 = 0 if iy == 0 else y0 + margin
            own_x1 = up_w if ix == len(xs_starts) - 1 else x1 - margin
            own_y1 = up_h if iy == len(ys_starts) - 1 else y1 - margin
            tiles.append((x0, y0, x1, y1, own_x0, own_y0, own_x1, own_y1))

    collected_pts, collected_scores = [], []
    for x0, y0, x1, y1, own_x0, own_y0, own_x1, own_y1 in tiles:
        local, local_scores = predict_batch(model, [up_rgb[y0:y1, x0:x1]], device, thrs)[0]
        if local.size == 0:
            continue
        gx, gy = local[:, 0] + x0, local[:, 1] + y0
        keep = (gx >= own_x0) & (gx < own_x1) & (gy >= own_y0) & (gy < own_y1)
        collected_pts.append(np.stack([gx[keep], gy[keep]], axis=1))
        collected_scores.append(local_scores[keep])

    if not collected_pts:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(collected_pts), np.concatenate(collected_scores)


def dedup_points(points, scores, max_radius):
    if len(points) < 6:
        return points
    from scipy.spatial import cKDTree
    order = np.argsort(-scores)
    ranked = points[order]
    tree = cKDTree(ranked)
    spacing = tree.query(ranked, k=5)[0][:, 4]
    radii = np.clip(0.45 * spacing, 1.5, max_radius)
    keep = np.ones(len(ranked), dtype=bool)
    for i, neighbors in enumerate(tree.query_ball_point(ranked, radii)):
        if keep[i]:
            for j in neighbors:
                if j > i:
                    keep[j] = False
    return ranked[keep]


@torch.no_grad()
def predict_candidates_tiled(model, image_bgr, device, upscale, tile, overlap, thrs=0.1, grids=2, time_budget=None):
    import time
    started = time.monotonic()
    orig_h, orig_w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if upscale != 1.0:
        rgb = cv2.resize(rgb, (round(orig_w * upscale), round(orig_h * upscale)), interpolation=cv2.INTER_CUBIC)
    stride = max(1, tile - overlap)
    offsets = [0, stride // 2][:max(1, grids)]

    passes = []
    for pass_index, offset in enumerate(offsets):
        if pass_index > 0 and time_budget is not None and time.monotonic() - started > time_budget * 0.45:
            break
        passes.append(grid_pass(model, rgb, device, tile, overlap, thrs, offset))
    points = np.concatenate([p for p, _ in passes]) / upscale
    scores = np.concatenate([s for _, s in passes])
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32), scores
    points[:, 0] = np.clip(points[:, 0], 0, orig_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, orig_h - 1)
    return points, scores


def filter_predictions(points, scores, threshold, dedup_radius=4.0):
    keep = scores > float(threshold)
    return dedup_points(points[keep], scores[keep], dedup_radius)


def predict_points_tiled(model, image_bgr, device, upscale, tile, overlap, thrs=0.5, grids=2, dedup_radius=4.0, time_budget=None):
    points, scores = predict_candidates_tiled(
        model, image_bgr, device, upscale, tile, overlap, thrs, grids, time_budget,
    )
    return dedup_points(points, scores, dedup_radius)


@torch.no_grad()
def predict_points(model, image_bgr, device, thrs=0.5):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized, scale = resize_for_inference(rgb)
    points, _ = predict_batch(model, [resized], device, thrs)[0]
    if points.size == 0:
        return points
    points /= scale
    orig_h, orig_w = image_bgr.shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0, orig_w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, orig_h - 1)
    return points


def load_font(size):
    for path in MAC_FONTS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_overlay(image_bgr, points, count, dot_color=(90, 17, 255), logo=None):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    radius = max(2, round(max(canvas.size) * 0.0022))
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=dot_color)

    width, height = canvas.size
    bar_height = max(64, round(height * 0.085))
    footer = Image.new("RGBA", (width, height + bar_height), color=(15, 15, 18, 255))
    footer.paste(canvas, (0, 0))

    footer_draw = ImageDraw.Draw(footer)
    label = f"Estimated crowd count: {count:,}"
    stroke = max(1, round(bar_height * 0.03))
    font = load_font(max(26, round(bar_height * 0.5)))
    bbox = footer_draw.textbbox((0, 0), label, font=font, stroke_width=stroke)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    footer_draw.text(
        ((width - text_w) / 2, height + (bar_height - text_h) / 2 - bbox[1]),
        label, fill=(255, 255, 255), font=font,
        stroke_width=stroke, stroke_fill=(15, 15, 18),
    )

    if logo is not None:
        logo_height = round(bar_height * 0.55)
        logo_width = round(logo_height * logo.width / logo.height)
        bar_margin = round(bar_height * 0.35)
        if bar_margin + logo_width < (width - text_w) / 2:
            resized_logo = logo.resize((logo_width, logo_height), Image.LANCZOS)
            footer.alpha_composite(
                resized_logo,
                (bar_margin, height + (bar_height - logo_height) // 2),
            )
    return footer.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Aerial crowd counter (PET) with count overlay")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default="", help="Output image path")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--json", type=str, default="", help="Optional path to write count + points JSON")
    parser.add_argument("--plain", action="store_true", help="Disable tiling (single-pass, faster, less accurate on dense scenes)")
    parser.add_argument("--upscale", type=float, default=1.0, help="Upscale factor before tiling")
    parser.add_argument("--tile", type=int, default=768, help="Tile size in upscaled pixels")
    parser.add_argument("--overlap", type=int, default=192, help="Tile overlap in upscaled pixels")
    parser.add_argument("--thrs", type=float, default=0.35, help="Head-confidence threshold; lower recovers more dense-core heads (0.5 = PET default)")
    args = parser.parse_args()

    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    device = select_device(args.device)
    print(f"device: {device}")
    model = build_pet(device)
    model.load_state_dict(load_state_dict(args.weights), strict=True)

    if args.plain:
        points = predict_points(model, image_bgr, device, args.thrs)
    else:
        points = predict_points_tiled(model, image_bgr, device, args.upscale, args.tile, args.overlap, args.thrs)
    count = int(points.shape[0])
    print(f"estimated_count: {count}")

    output_path = Path(args.output) if args.output else REPO_ROOT / "outputs" / f"{image_path.stem}_counted.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logo_path = REPO_ROOT / "assets" / "logo_white.png"
    logo = Image.open(logo_path).convert("RGBA") if logo_path.exists() else None
    render_overlay(image_bgr, points, count, logo=logo).save(str(output_path))
    print(f"overlay_saved: {output_path}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"image": str(image_path), "count": count, "points_xy": points.tolist()}, indent=2))
        print(f"json_saved: {json_path}")


if __name__ == "__main__":
    main()
