import json
import hashlib
import html
import os
import re
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from urllib.parse import urlencode
from pathlib import Path

import cv2
import gradio as gr
import gradio.route_utils
import spaces
import torch
from PIL import Image
from pillow_heif import register_heif_opener
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

register_heif_opener()

import count
from drive_storage import save_images_async
from fallback_ticket import issue_fallback_ticket

# The crowd.skyebrowse.com proxy forwards x-forwarded-host but not
# x-forwarded-proto, so Gradio resolves an http:// root and every asset/API
# URL gets mixed-content blocked on the https page. Upgrade the scheme.
_original_get_root_url = gradio.route_utils.get_root_url


def _https_get_root_url(request, route_path, root_path):
    root_url = _original_get_root_url(request, route_path, root_path)
    if root_url.startswith("http://") and not any(
        host in root_url for host in ("localhost", "127.0.0.1", "0.0.0.0")
    ):
        root_url = "https://" + root_url[len("http://"):]
    return root_url


gradio.route_utils.get_root_url = _https_get_root_url


FRAME_ANCESTORS_POLICY = (
    "frame-ancestors https://skyebrowse.com https://*.skyebrowse.com "
    "http://localhost:8080"
)


# Gradio 6 follows the OS dark preference unless __theme is set, and the
# custom_js/head hooks are not executed by the CSR frontend, so the light
# theme must be forced server-side. Redirect only browser navigations
# (Accept: text/html) so HF health probes keep getting 200s, and use a
# relative Location because the proxied request scheme is plain http.
async def _force_light_theme(request, call_next):
    if (
        request.url.path == "/"
        and request.query_params.get("__theme") != "light"
        and "text/html" in request.headers.get("accept", "")
    ):
        from starlette.responses import RedirectResponse

        params = dict(request.query_params)
        params["__theme"] = "light"
        response = RedirectResponse(f"/?{urlencode(params)}")
    else:
        response = await call_next(request)

    response.headers["Content-Security-Policy"] = FRAME_ANCESTORS_POLICY
    return response


force_light_middleware = Middleware(BaseHTTPMiddleware, dispatch=_force_light_theme)

TILE = 768
OVERLAP = 192
UPSCALE = 1.0

_model = count.build_pet(torch.device("cpu"))
_model.load_state_dict(count.load_state_dict(count.DEFAULT_WEIGHTS), strict=True)
_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

_logo = Image.open("assets/logo_white.png").convert("RGBA")


def add_watermark(image):
    from PIL import ImageDraw

    image = image.convert("RGBA")
    width, height = image.size
    logo_width = max(140, round(width * 0.18))
    logo_height = round(logo_width * _logo.height / _logo.width)
    logo = _logo.resize((logo_width, logo_height), Image.LANCZOS)

    pad = round(logo_height * 0.45)
    margin = round(width * 0.02)
    # Offset below Gradio's "Counted overlay" label chip, which covers the corner
    top = margin + max(56, round(height * 0.04))

    pill = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(pill).rounded_rectangle(
        (margin, top, margin + logo_width + 2 * pad, top + logo_height + 2 * pad),
        radius=(logo_height + 2 * pad) // 2,
        fill=(15, 15, 18, 150),
    )
    image.alpha_composite(pill)
    image.alpha_composite(logo, (margin + pad, top + pad))
    return image.convert("RGB")


def crowd_gpu_duration(image_bgr):
    height, width = image_bgr.shape[:2]
    return 4 if max(height, width) <= 1280 and min(height, width) <= 768 else 7


@spaces.GPU(duration=crowd_gpu_duration)
@torch.inference_mode()
def predict_crowd_candidates(image_bgr):
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points, scores = count.predict_candidates_tiled(
        _model, image_bgr, device, UPSCALE, TILE, OVERLAP,
        thrs=0.1, time_budget=crowd_gpu_duration(image_bgr) - 0.5,
    )
    return points, scores, time.perf_counter() - started


def crowd_image_key(image_bgr):
    digest = hashlib.sha256(image_bgr.tobytes()).hexdigest()
    return image_bgr.shape, digest


def crowd_predictions(image_bgr, prediction_state):
    image_key = crowd_image_key(image_bgr)
    cache_hit = prediction_state is not None and prediction_state["image_key"] == image_key
    gpu_seconds = 0.0
    if not cache_hit:
        points, scores, gpu_seconds = predict_crowd_candidates(image_bgr)
        prediction_state = {"image_key": image_key, "points": points, "scores": scores}
    print(f"Crowd cache_hit={cache_hit} gpu_seconds={gpu_seconds:.3f}", flush=True)
    return prediction_state


def _count_crowd(image_path, threshold, prediction_state, save_images, allow_gpu=True):
    if image_path is None:
        return None, "Upload an image first.", None
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None, "Could not read that image.", None
    if not 0.1 <= float(threshold) <= 0.9:
        raise gr.Error("Choose a threshold between 0.1 and 0.9.")
    try:
        if not allow_gpu and (
            prediction_state is None
            or prediction_state.get("image_key") != crowd_image_key(image_bgr)
        ):
            return None, "Cached result expired. Please upload the image again.", None
        prediction_state = crowd_predictions(image_bgr, prediction_state)
        points = count.filter_predictions(
            prediction_state["points"], prediction_state["scores"], threshold,
        )
        total = int(points.shape[0])
        overlay = add_watermark(count.render_overlay(image_bgr, points, total, logo=_logo))
        if save_images:
            save_images_async(image_path, overlay, total)
        return overlay, f"{total:,} people", prediction_state
    except gr.Error as exc:
        message = html.unescape(re.sub(r"<[^>]+>", "", str(exc.message))).strip()
        return None, message, prediction_state
    except Exception as exc:
        print(f"count_crowd failed: {exc}", flush=True)
        return None, "Crowd counter failed. Please try a smaller image or try again in a minute.", None


def count_crowd(image_path, threshold, prediction_state=None):
    return _count_crowd(image_path, threshold, prediction_state, save_images=True)


def recount_crowd(image_path, threshold, prediction_state=None):
    return _count_crowd(image_path, threshold, prediction_state, save_images=False)


def count_crowd_web(image_path, threshold, prediction_state=None):
    overlay, message, state = count_crowd(image_path, threshold, prediction_state)
    return overlay, message, state, {"ticket": issue_fallback_ticket(image_path, message)}


def recount_cached_web(image_path, threshold, prediction_state=None):
    overlay, message, state = _count_crowd(
        image_path, threshold, prediction_state, save_images=False, allow_gpu=False,
    )
    return overlay, message, state, None


def count_crowd_embedded(image_path, threshold, prediction_state=None):
    overlay, message, state, receipt = count_crowd_web(image_path, threshold, prediction_state)
    return overlay, message, state, {**receipt, "image": image_path}


def recount_crowd_embedded(image_path, threshold, prediction_state, receipt):
    if receipt and receipt.get("ticket") and receipt.get("image") == image_path:
        return gr.skip(), gr.skip(), prediction_state, receipt
    overlay, message, state = recount_crowd(image_path, threshold, prediction_state)
    return overlay, message, state, {
        "ticket": issue_fallback_ticket(image_path, message), "image": image_path,
    }


fallback_bridge = Path(__file__).with_name("fallback_bridge.js").read_text()


css = """
/* Force light page background even when the viewer's OS is in dark mode */
body { background: #f9fafb !important; }
h1, h2, h3 {
    text-align: center;
    display: block;
}
h1 a {
    color: #5A11FF !important;
    text-decoration: none !important;
}
footer {
    visibility: hidden;
}
.gradio-container {
    width: 100% !important;
    min-width: 0 !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}
.gradio-container a { text-decoration: none !important; }
.inter-app-nav {
    text-align: center;
    margin-bottom: 15px;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}
.inter-app-nav-link {
    color: #2f6bff !important;
    font-weight: bold !important;
    margin: 0 12px !important;
    text-decoration: none !important;
}
#count-box .output-class, #count-box .label-content, #count-box .text, #count-box textarea, #count-box input {
    color: #5A11FF !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    text-align: center !important;
}
#count-box { color: #5A11FF !important; }
.sponsor-banner {
    text-align: center;
    margin: 8px 0 14px 0;
}
.sponsor-banner-title a {
    color: #5A11FF !important;
    text-decoration: none !important;
}
.sponsor-banner-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.sponsor-banner-button {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 10px;
    font-weight: 700;
    text-decoration: none !important;
    background: linear-gradient(90deg, #2f6bff 0%, #7d4dff 100%);
    color: #ffffff !important;
}
.toast-wrap, .toast-body, .toast-container {
    display: none !important;
}
"""

schema_data = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    "name": "Crowd Counter by SkyeBrowse",
    "operatingSystem": "Web",
    "applicationCategory": "UtilitiesApplication",
    "description": "Free AI crowd-counting tool for aerial and dense crowd photos with a per-head dot overlay.",
    "author": {
        "@type": "Organization",
        "name": "SkyeBrowse",
        "url": "https://www.skyebrowse.com"
    }
}

head_html = f"""
<script type="application/ld+json">
{json.dumps(schema_data)}
</script>
<link rel="canonical" href="https://crowd.skyebrowse.com/" />
<meta name="description" content="Count people in aerial and dense crowd photos with SkyeBrowse AI. Free crowd-counting tool with a per-head dot overlay.">
<meta property="og:title" content="Crowd Counter | Powered by SkyeBrowse">
<meta property="og:type" content="website">
<meta property="og:url" content="https://crowd.skyebrowse.com/">
<meta property="og:image" content="https://www.skyebrowse.com/logo.png">
"""

custom_js = """
() => {
    // Match the sibling Spaces: always light. Gradio 6 follows the OS
    // preference unless __theme is set explicitly, and class-stripping alone
    // no longer overrides it, so force the param on every host.
    const url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.replace(url.toString());
        return;
    }

    // Force light mode. Gradio applies dark via a `dark` class; in <gradio-app>
    // web-component embeds (e.g. app.skyebrowse.com) that class lands on the host
    // element, not <html>/<body>, so strip it from EVERY element that has it.
    function forceLight() {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
        document.documentElement.classList.remove('dark');
    }
    forceLight();
    new MutationObserver(forceLight).observe(document.documentElement, {
        attributes: true, attributeFilter: ['class'], subtree: true
    });

    // Hide ZeroGPU progress messages
    new MutationObserver(() => {
        document.querySelectorAll('.progress-text, .eta-bar, .progress-level-inner').forEach(el => {
            if (el.textContent.match(/zero\\s*gpu/i)) { el.style.visibility = 'hidden'; }
        });
    }).observe(document.body, {childList: true, subtree: true, characterData: true});

    // Rewrite external app links when hosted on *.app.skyebrowse.com
    const hostname = window.location.hostname;
    if (hostname.endsWith('app.skyebrowse.com')) {
        const origin = window.location.origin;
        const linkMap = {
            'interiorai.skyebrowse.com': origin + '/interior-ai',
            'anime.skyebrowse.com':      origin + '/anime-ai',
            '3dai.skyebrowse.com':       origin + '/3d-ai',
            'crowd.skyebrowse.com':      origin + '/crowd-counter',
            'app.skyebrowse.com':        origin,
        };
        function rewriteLinks() {
            document.querySelectorAll('a[href]').forEach(a => {
                try {
                    const url = new URL(a.href);
                    if (linkMap[url.hostname]) {
                        a.href = linkMap[url.hostname];
                    }
                } catch(e) {}
            });
        }
        rewriteLinks();
        new MutationObserver(rewriteLinks).observe(document.body, {childList: true, subtree: true});
    }
}
"""

with gr.Blocks(title="Crowd Counter | SkyeBrowse") as demo:
    gr.Markdown("# 👥 Crowd Counter by [SkyeBrowse](https://www.skyebrowse.com)")
    gr.Markdown(
        "Estimate how many people are in an aerial or dense crowd photo. Upload a photo — it "
        "counts automatically. Drag the **threshold** lower to "
        "recover more heads in tightly packed crowds (0.5 is the model default; 0.35 is a good "
        "balance)."
    )
    gr.Markdown(
        "Uploaded images and counted overlays are saved to SkyeBrowse's Google Drive."
    )
    sponsor_banner = gr.HTML(
        '<div class="sponsor-banner">'
        '<a class="sponsor-banner-button" href="https://app.skyebrowse.com" target="_blank">Try more AI + 3D modeling</a>'
        "</div>"
    )
    # Inter-App Navigation Cluster
    with gr.Row():
        gr.HTML(
            '<div class="inter-app-nav">'
            '<span>Try our other AI tools + 3D modeling: </span>'
            '<a class="inter-app-nav-link" href="https://interiorai.skyebrowse.com">🏠 Interior AI Designer</a>'
            '<a class="inter-app-nav-link" href="https://anime.skyebrowse.com">🎨 Anime AI Art</a>'
            '<a class="inter-app-nav-link" href="https://3dai.skyebrowse.com">🤖 3D AI</a>'
            '</div>'
        )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(type="filepath", label="Crowd image", sources=["upload"])
            threshold = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Head-confidence threshold (lower = more dense-core recall)")
        with gr.Column(scale=1, min_width=300):
            overlay_output = gr.Image(type="pil", label="Counted overlay", interactive=False, buttons=["download", "fullscreen"])
            count_output = gr.Textbox(label="Estimated count", elem_id="count-box", interactive=False)

    prediction_state = gr.State(value=None, time_to_live=3600)
    inputs = [image_input, threshold, prediction_state]
    outputs = [overlay_output, count_output, prediction_state]
    fallback_ticket = gr.JSON(visible=False)
    web_outputs = [*outputs, fallback_ticket]
    bridge_inputs = [image_input, fallback_ticket, threshold]
    upload_event = image_input.upload(count_crowd_embedded, inputs, web_outputs, api_name="count_crowd")
    upload_event.then(None, bridge_inputs, outputs[:2], js=fallback_bridge)
    image_input.clear(lambda: (None, None, None, ""), outputs=[prediction_state, fallback_ticket, overlay_output, count_output], queue=False, api_visibility="private")
    gr.Button(visible=False).click(count_crowd_web, inputs, web_outputs, api_name="count_crowd_web")
    gr.Button(visible=False).click(recount_cached_web, inputs, web_outputs, api_name="recount_cached_web")
    recount_event = threshold.release(recount_crowd_embedded, [*inputs, fallback_ticket], web_outputs, trigger_mode="always_last")
    recount_event.then(None, bridge_inputs, outputs[:2], js=fallback_bridge)
    examples = gr.Examples(
        examples=[
            ["examples/crowd.jpg", 0.35],
            ["examples/wc_14s.jpg", 0.35],
            ["examples/wc_51s.jpg", 0.35],
        ],
        inputs=inputs,
        outputs=outputs,
        fn=recount_crowd,
        run_on_click=False,
        cache_examples=False,
    )
    example_event = examples.load_input_event.then(recount_crowd_embedded, [*inputs, fallback_ticket], web_outputs)
    example_event.then(None, bridge_inputs, outputs[:2], js=fallback_bridge)

    with gr.Row():
        helper_text = gr.Markdown("## Tap and hold (on mobile) to save the image.", visible=True)

    gr.Markdown(
        """
        ---
        ### About Crowd Counter by SkyeBrowse
        Crowd Counter estimates crowd size from a single still using a fine-tuned Point-quEry
        Transformer, with native-resolution tiling for dense scenes and a per-head dot overlay you
        can audit. Built for drone footage, event safety, and situational awareness.

        **About SkyeBrowse:** Beyond crowd counting, we specialize in the world's fastest
        [3D drone modeling](https://www.skyebrowse.com). Our technology allows users to create
        accurate digital twins and 3D reconstructions for incident management, construction,
        and real estate.

        **Related resources:** [Drone 3D Modeling Guide](https://www.skyebrowse.com/news/posts/drone-3d-modeling)
        · [What Is Photogrammetry?](https://www.skyebrowse.com/news/posts/what-is-photogrammetry)
        """
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=css,
        head=head_html,
        js=custom_js,
        ssr_mode=False,
        app_kwargs={"middleware": [force_light_middleware]},
    )
