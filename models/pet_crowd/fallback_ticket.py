import base64
import hashlib
import hmac
import json
import os
import re
import time
from uuid import uuid4

QUOTA_ERROR = re.compile(r"^You have exceeded your (?:Pro |free )?ZeroGPU (?:quota|runs limit)\b", re.IGNORECASE)


def issue_fallback_ticket(image_path, message):
    secret = os.environ.get("LABS_FALLBACK_SIGNING_KEY")
    if not secret or not image_path or not QUOTA_ERROR.search(message.strip()):
        return None
    payload = {
        "v": 1,
        "space": "broyang/CrowdCounter",
        "nonce": uuid4().hex,
        "image": os.path.abspath(image_path),
        "exp": int(time.time()) + 3600,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    signature = hmac.new(secret.encode(), encoded, hashlib.sha256).digest()
    return encoded.decode() + "." + base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
