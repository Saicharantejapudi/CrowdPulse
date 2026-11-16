import io
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from PIL import Image


DRIVE_FOLDER_ID = os.getenv(
    "GOOGLE_DRIVE_FOLDER_ID", "1W0J0o-HI0vydlJDg1ed8yTrpFarwuH7M"
)
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
DRIVE_UPLOAD_URL = (
    "https://www.googleapis.com/upload/drive/v3/files"
    "?uploadType=multipart&supportsAllDrives=true&fields=id"
)
MAX_PENDING_SAVES = 8

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="crowd-drive")
_pending_saves = threading.BoundedSemaphore(MAX_PENDING_SAVES)
_logger = logging.getLogger(__name__)


def _service_account_info():
    value = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not value:
        return None
    return json.loads(value)


def _jpeg_bytes(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue()


def _upload_image(session, token, name, data, properties):
    metadata = {
        "name": name,
        "parents": [DRIVE_FOLDER_ID],
        "appProperties": properties,
    }
    files = {
        "metadata": (
            "metadata",
            json.dumps(metadata),
            "application/json; charset=UTF-8",
        ),
        "file": (name, data, "image/jpeg"),
    }
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(3):
        response = session.post(
            DRIVE_UPLOAD_URL,
            headers=headers,
            files=files,
            timeout=30,
        )
        if response.ok:
            return response.json()["id"]
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        if attempt < 2:
            time.sleep(2**attempt)

    response.raise_for_status()


def _save_images(service_account_info, run_id, original, overlay, total):
    try:
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=[DRIVE_SCOPE],
        )
        credentials.refresh(GoogleAuthRequest())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"{timestamp}_{run_id}"
        common = {"source": "crowd-counter", "run_id": run_id}

        with requests.Session() as session:
            _upload_image(
                session,
                credentials.token,
                f"{prefix}_original.jpg",
                original,
                {**common, "image_type": "original"},
            )
            _upload_image(
                session,
                credentials.token,
                f"{prefix}_counted_{total}.jpg",
                overlay,
                {**common, "image_type": "counted", "estimated_count": str(total)},
            )
    except Exception:
        _logger.exception("Failed to save Crowd Counter images to Google Drive")
    finally:
        _pending_saves.release()


def save_images_async(original_path, overlay, total):
    try:
        service_account_info = _service_account_info()
    except (json.JSONDecodeError, TypeError):
        _logger.exception("Invalid GOOGLE_SERVICE_ACCOUNT_JSON")
        return False

    if not service_account_info:
        return False
    if not _pending_saves.acquire(blocking=False):
        _logger.error("Skipped Crowd Counter Drive save because the queue is full")
        return False

    try:
        original = _jpeg_bytes(Image.open(original_path))
        counted = _jpeg_bytes(overlay)
    except Exception:
        _pending_saves.release()
        _logger.exception("Failed to prepare Crowd Counter images for Google Drive")
        return False

    run_id = uuid.uuid4().hex[:12]
    _executor.submit(
        _save_images,
        service_account_info,
        run_id,
        original,
        counted,
        int(total),
    )
    return True
