"""Deployment contract checks without weights, GPU, SSH or external network."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOME_ENV = {key: os.environ[key] for key in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH") if key in os.environ}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


deploy = module("deployment_helper", "deploy/server/model_deploy.py")
voice = module("voice_download_helper", "deploy/raspberrypi/download_voice_models.py")
smoke = module("model_smoke_helper", "tools/check_models.py")
privacy = module("deployment_privacy_helper", "tools/check_deployment_privacy.py")


class DeploymentTests(unittest.TestCase):
    def test_config_environment_wins_and_home_expands(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.env"
            path.write_text('JAKA_MODEL_HOME="$HOME/model files" # local\nJAKA_QWEN_GPU=0\n', encoding="utf-8")
            config = deploy.read_config(path, {"JAKA_QWEN_GPU": "2"})
            self.assertEqual(config["JAKA_MODEL_HOME"], str(Path.home()) + "/model files")
            self.assertEqual(config["JAKA_QWEN_GPU"], "2")

    def test_config_rejects_invalid_keys_and_blank_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.env"
            for value in ["PASSWORD=value", "JAKA_HOME=", "JAKA_HOME=two words", "export JAKA_HOME=value"]:
                with self.subTest(value=value):
                    path.write_text(value, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        deploy.read_config(path, {})

    def test_service_configuration_rejects_collisions(self):
        for config in [{"JAKA_QWEN_PORT": "8000"}, {"JAKA_QWEN_PORT": "65536"},
                       {"JAKA_MODEL_HOME": "relative"}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                deploy.settings(config)

    def test_serve_command_is_local_and_separates_gpu_namespace(self):
        models = deploy.settings({})
        command = deploy.serve_command(models["qwen"])
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertIn("--no-continuous-batching", command)
        self.assertEqual(command[-2:], ["--reasoning", "off"])
        self.assertNotEqual(models["qwen"].port, models["minicpm"].port)

    def test_client_config_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, HOME_ENV, clear=True):
            target = Path(temporary) / "client.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(deploy.main(["client-config", "--output", str(target)]), 0)
                before = target.read_bytes()
                self.assertEqual(deploy.main(["client-config", "--output", str(target)]), 1)
            self.assertEqual(before, target.read_bytes())
            config = json.loads(before)
            self.assertEqual(config["JAKA_AGENT_PROTOCOL"], "json")
            self.assertFalse(any("KEY" in key for key in config))

    def test_explicit_missing_config_fails_before_install(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(deploy, "install") as installer:
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(deploy.main(["install", "--config", str(Path(temporary) / "missing")]), 1)
            installer.assert_not_called()

    def test_serve_exports_offline_and_selected_gpu(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, HOME_ENV, clear=True):
            root = Path(temporary)
            config = root / "server.env"
            config.write_text('JAKA_MODEL_HOME="' + root.as_posix() + '"\nJAKA_QWEN_GPU=3\n', encoding="utf-8")
            model = deploy.settings(deploy.read_config(config, {}))["qwen"]
            model.weights.mkdir(parents=True)
            (model.weights / "config.json").write_text("{}", encoding="utf-8")
            (model.venv / "bin").mkdir(parents=True)
            (model.venv / "bin" / "transformers").touch()
            with patch.object(deploy.os, "execvpe") as execute:
                self.assertEqual(deploy.main(["serve", "--role", "qwen", "--config", str(config)]), 0)
            env = execute.call_args.args[2]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")


class VoiceDownloadTests(unittest.TestCase):
    def archive(self, path, name, content=b"asset", kind=None):
        with tarfile.open(path, "w:bz2") as archive:
            item = tarfile.TarInfo(name)
            item.size = len(content) if kind is None else 0
            if kind:
                item.type = kind
                item.linkname = "../../outside"
            archive.addfile(item, io.BytesIO(content) if kind is None else None)

    def test_archive_rejects_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kind in [("../outside", None), ("/outside", None),
                               ("model/link", tarfile.SYMTYPE), ("model\\outside", None)]:
                with self.subTest(name=name):
                    self.archive(root / "archive.tar.bz2", name, kind=kind)
                    with self.assertRaises(ValueError):
                        voice.safe_extract(root / "archive.tar.bz2", root / "out")
            self.assertFalse((root / "outside").exists())

    def test_archive_extracts_regular_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.archive(root / "archive.tar.bz2", "model/tokens.txt")
            voice.safe_extract(root / "archive.tar.bz2", root / "out")
            self.assertEqual((root / "out/model/tokens.txt").read_bytes(), b"asset")

    def test_partial_existing_models_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / voice.ASSETS["asr"][0]
            target.mkdir()
            (target / "tokens.txt").write_text("keep", encoding="utf-8")
            with patch.object(voice.urllib.request, "urlopen") as network:
                with self.assertRaises(ValueError):
                    voice.download("asr", base)
            network.assert_not_called()
            self.assertEqual((target / "tokens.txt").read_text(), "keep")

    def test_complete_models_skip_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / voice.ASSETS["asr"][0]
            target.mkdir()
            for name in ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]:
                (target / name).write_bytes(b"asset")
            with patch.object(voice.urllib.request, "urlopen") as network, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(voice.download("asr", base), target)
            network.assert_not_called()

    def test_download_validates_and_publishes_with_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:bz2") as tar:
                for name in ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"]:
                    item = tarfile.TarInfo("official-model/" + name)
                    item.size = 5
                    tar.addfile(item, io.BytesIO(b"asset"))
            with patch.object(voice.urllib.request, "urlopen", return_value=io.BytesIO(archive.getvalue())), \
                    contextlib.redirect_stdout(io.StringIO()):
                target = voice.download("asr", base)
            voice.validate(target, "asr")
            manifest = json.loads((target / "download-manifest.json").read_text())
            self.assertEqual(len(manifest["archive_sha256"]), 64)
            self.assertEqual(manifest["source"], voice.ASSETS["asr"][1])
            self.assertFalse(list(base.glob(".voice-download-*")))

    def test_checksum_mismatch_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with patch.object(voice.urllib.request, "urlopen", return_value=io.BytesIO(b"invalid archive")):
                with self.assertRaises(ValueError):
                    voice.download("asr", base, "0" * 64)
            self.assertFalse((base / voice.ASSETS["asr"][0]).exists())
            self.assertFalse(list(base.glob(".voice-download-*")))


class Handler(BaseHTTPRequestHandler):
    requests = []
    fail = False

    def log_message(self, *args):
        pass

    def send(self, payload, status=200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send({"data": []} if self.path == "/v1/models" else {"status": "ok"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(payload)
        if type(self).fail:
            self.send({"private_response": "must never reach console"}, 500)
            return
        content = payload["messages"][0]["content"]
        count = 0 if isinstance(content, str) else sum(item["type"] == "image_url" for item in content)
        answer = {"status": "ok"} if not count else {"color": "red"} if count == 1 else {"colors": ["red", "blue"]}
        self.send({"choices": [{"message": {"content": json.dumps(answer)}}]})


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        Handler.requests, Handler.fail = [], False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_text_single_and_double_image_contract(self):
        self.assertIn("text-json", smoke.check("qwen", self.base, "test-model", "not-needed", 5))
        self.assertIn("two-image", smoke.check("minicpm", self.base, "test-model", "not-needed", 5))
        self.assertEqual(len(Handler.requests), 3)
        self.assertFalse(any(request["stream"] for request in Handler.requests))

    def test_health_only_does_not_generate(self):
        self.assertEqual(smoke.check("qwen", self.base, "test-model", "not-needed", 5, True),
                         ["health", "model-list"])
        self.assertEqual(Handler.requests, [])

    def test_failure_does_not_print_endpoint_or_response(self):
        Handler.fail = True
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            config = Path(temporary) / "client.json"
            config.write_text(json.dumps({"JAKA_AGENT_BASE_URL": self.base,
                                          "JAKA_AGENT_MODEL": "private-weight-path"}), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(smoke.main(["--config", str(config), "--role", "qwen"]), 1)
            self.assertNotIn(self.base, output.getvalue())
            self.assertNotIn("private-weight-path", output.getvalue())
            self.assertNotIn("must never reach console", output.getvalue())

    def test_parse_reasoning_and_json_fence(self):
        response = {"choices": [{"message": {"content": '<think>analysis</think>```json\n{"status":"ok"}\n```'}}]}
        self.assertEqual(smoke.parse_answer(response), {"status": "ok"})


class DeploymentPrivacyTests(unittest.TestCase):
    def test_address_and_credential_findings_hide_values(self):
        address = ".".join(["203", "0", "113", "9"])
        value = "sensitive-example-value"
        findings = privacy.inspect_file("sample", "HOST=" + address + "\nPASSWORD=" + value)
        self.assertEqual(len(findings), 2)
        self.assertNotIn(address, str(findings))
        self.assertNotIn(value, str(findings))

    def test_loopback_and_local_key_placeholders_are_allowed(self):
        self.assertEqual(privacy.inspect_file("sample", "URL=http://127.0.0.1:8000\nJAKA_AGENT_API_KEY=not-needed"), [])


@unittest.skipUnless(sys.platform == "linux" and shutil.which("bash"), "Tunnel process test requires Linux")
class TunnelTests(unittest.TestCase):
    def test_foreground_uses_alias_and_loopback_forwards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "ssh"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n', encoding="utf-8")
            fake.chmod(0o755)
            output = root / "args"
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"], CAPTURE_ARGS=str(output),
                       JAKA_TUNNEL_CONFIG=str(ROOT / "configs/tunnel.env.example"))
            subprocess.run(["bash", str(ROOT / "tunnel.sh"), "foreground"], env=env, check=True, timeout=10)
            args = output.read_text().splitlines()
            self.assertEqual(args[-1], "jaka-model-server")
            self.assertIn("127.0.0.1:8000:127.0.0.1:8000", args)
            self.assertIn("127.0.0.1:8001:127.0.0.1:8001", args)
            self.assertIn("StrictHostKeyChecking=yes", args)
            self.assertIn("BatchMode=yes", args)


if __name__ == "__main__":
    unittest.main()
