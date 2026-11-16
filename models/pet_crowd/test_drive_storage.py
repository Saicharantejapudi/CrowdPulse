import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import drive_storage


class DriveStorageTests(unittest.TestCase):
    def test_save_is_disabled_without_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(drive_storage.save_images_async("missing.jpg", None, 0))

    def test_upload_retries_transient_failure(self):
        failed = MagicMock(ok=False, status_code=503)
        succeeded = MagicMock(ok=True)
        succeeded.json.return_value = {"id": "file-id"}
        session = MagicMock()
        session.post.side_effect = [failed, succeeded]

        with patch("drive_storage.time.sleep"):
            file_id = drive_storage._upload_image(
                session,
                "token",
                "image.jpg",
                b"jpeg",
                {"source": "crowd-counter"},
            )

        self.assertEqual(file_id, "file-id")
        self.assertEqual(session.post.call_count, 2)

    def test_save_prepares_original_and_overlay(self):
        submitted = []
        original_executor = drive_storage._executor
        drive_storage._executor = MagicMock()
        drive_storage._executor.submit.side_effect = lambda *args: submitted.append(args)

        try:
            with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
                Image.new("RGB", (8, 8), "white").save(image_file.name)
                with patch.dict(
                    os.environ,
                    {"GOOGLE_SERVICE_ACCOUNT_JSON": '{"client_email":"bot@example.com"}'},
                    clear=True,
                ):
                    saved = drive_storage.save_images_async(
                        image_file.name,
                        Image.new("RGB", (8, 8), "black"),
                        42,
                    )
        finally:
            drive_storage._executor = original_executor
            if submitted:
                drive_storage._pending_saves.release()

        self.assertTrue(saved)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0][-1], 42)
        self.assertTrue(submitted[0][-3].startswith(b"\xff\xd8"))
        self.assertTrue(submitted[0][-2].startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
