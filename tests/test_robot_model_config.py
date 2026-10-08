import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.models.config import load_model_config


class ModelConfigTests(unittest.TestCase):
    def load(self, values):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "models.json"
            path.write_text(json.dumps(values), encoding="utf-8")
            return load_model_config(path)

    @patch.dict(os.environ, {}, clear=True)
    def test_independent_endpoints(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "src/jaka_agent/resources/model_config.json").read_text())
        self.load(config)
        self.assertEqual(os.environ["JAKA_AGENT_MODEL"], "/data2/models/Qwen3.5-9B")
        self.assertIn(":8001/", os.environ["JAKA_AGENT_BASE_URL"])
        self.assertIn(":8000/", os.environ["DASHSCOPE_BASE_URL"])
        self.assertEqual(os.environ["QWEN_VISION_MODEL"], "/data/model/MiniCPM-V-4.6")
        self.assertEqual(os.environ["JAKA_AGENT_PROTOCOL"], "json")

    @patch.dict(os.environ, {"JAKA_AGENT_MODEL": "override"}, clear=True)
    def test_environment_wins(self):
        self.load({"JAKA_AGENT_MODEL": "default"})
        self.assertEqual(os.environ["JAKA_AGENT_MODEL"], "override")

    @patch.dict(os.environ, {}, clear=True)
    def test_bad_config_is_atomic(self):
        for values in ({"PATH": "bad"}, [], {"JAKA_AGENT_MODEL": ""},
                       {"JAKA_AGENT_MODEL": "ok", "DASHSCOPE_BASE_URL": "file:///tmp"},
                       {"JAKA_AGENT_MODEL": "ok", "JAKA_AGENT_PROTOCOL": "guess"},
                       {"DASHSCOPE_BASE_URL": "http://user:secret@host/v1"}):
            with self.assertRaises(ValueError):
                self.load(values)
            self.assertNotIn("JAKA_AGENT_MODEL", os.environ)

    @patch.dict(os.environ, {"JAKA_MODEL_CONFIG": "missing-model-config.json"}, clear=True)
    def test_explicit_missing_config_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_model_config()


if __name__ == "__main__":
    unittest.main()
