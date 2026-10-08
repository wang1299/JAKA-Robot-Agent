import io
import json
import unittest
import sys
from pathlib import Path
from email.message import Message
from http import HTTPStatus
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_web


class MobileAudioRequestTests(unittest.TestCase):
    def make_handler(self, content_type, body, declared_length=None):
        handler = object.__new__(robot_web.RobotWebHandler)
        headers = Message()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body) if declared_length is None else declared_length)
        handler.headers = headers
        handler.rfile = io.BytesIO(body)
        return handler

    def test_accepts_browser_codec_parameter(self):
        handler = self.make_handler("audio/webm;codecs=opus", b"abc")
        self.assertEqual(handler._read_mobile_audio(), b"abc")

    def test_rejects_unsupported_media_type(self):
        handler = self.make_handler("text/plain", b"abc")
        with self.assertRaises(robot_web.AudioRequestError) as caught:
            handler._read_mobile_audio()
        self.assertEqual(caught.exception.status, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    def test_rejects_empty_audio(self):
        handler = self.make_handler("audio/webm", b"")
        with self.assertRaises(robot_web.AudioRequestError) as caught:
            handler._read_mobile_audio()
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_REQUEST)

    def test_rejects_audio_larger_than_eight_megabytes_before_reading(self):
        handler = self.make_handler(
            "audio/webm",
            b"",
            declared_length=robot_web.MAX_AUDIO_BYTES + 1,
        )
        with self.assertRaises(robot_web.AudioRequestError) as caught:
            handler._read_mobile_audio()
        self.assertEqual(caught.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


class MobileAudioDecodeTests(unittest.TestCase):
    @mock.patch("robot_web.shutil.which", return_value="ffmpeg")
    @mock.patch("robot_web.subprocess.run")
    def test_decodes_to_16khz_mono_pcm(self, run, _which):
        run.return_value = SimpleNamespace(returncode=0, stdout=b"\0\0" * 16000, stderr=b"")
        pcm, duration_ms = robot_web._decode_mobile_audio(b"encoded")
        self.assertEqual(len(pcm), 32000)
        self.assertEqual(duration_ms, 1000)
        command = run.call_args.args[0]
        self.assertIn("16000", command)
        self.assertIn("pcm_s16le", command)
        self.assertEqual(run.call_args.kwargs["input"], b"encoded")

    @mock.patch("robot_web.shutil.which", return_value="ffmpeg")
    @mock.patch("robot_web.subprocess.run")
    def test_rejects_audio_longer_than_thirty_seconds(self, run, _which):
        run.return_value = SimpleNamespace(
            returncode=0,
            stdout=b"\0\0" * 16000 * 31,
            stderr=b"",
        )
        with self.assertRaises(robot_web.AudioRequestError) as caught:
            robot_web._decode_mobile_audio(b"encoded")
        self.assertEqual(caught.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


class PwaAssetTests(unittest.TestCase):
    def test_manifest_is_valid_and_installable(self):
        manifest = json.loads(robot_web.PWA_MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["start_url"], "/")
        self.assertEqual(manifest["display"], "standalone")
        self.assertTrue(manifest["icons"])

    def test_service_worker_keeps_robot_api_network_only(self):
        source = robot_web.SERVICE_WORKER_PATH.read_text(encoding="utf-8")
        self.assertIn("url.pathname.startsWith('/api/')", source)
        self.assertIn("cache: 'no-store'", source)


if __name__ == "__main__":
    unittest.main()
