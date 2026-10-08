"""Map publication integration with real filesystem operations, no GPU or model.

Archived coordinator fixtures are optional in other checkouts.  They are captured
read-only from the mapping server for this integration run, not deployed code.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "server"))
sys.path.insert(0, str(ROOT / "tools"))
import map_graph_hook as hook
import patch_map_graph_hook as patcher
import sync_scene_graph

EVIDENCE = ROOT / ".codex-tmp" / "map-update-evidence"


def example_map():
    return {
        "video_id": "integration-scene", "scene_graph_version": 1,
        "objects": {
            "1": {"ann_id": 1, "category": "Chair"},
            "2": {"ann_id": 2, "category": "Table"},
        },
        "relationships": {
            "functional": [],
            "positional": [{"head_obj": {"ann_id": 1}, "tail_obj": {"ann_id": 2},
                            "relation": "near"}],
        },
        "stats": {"num_objects": 2, "num_functional_relationships": 0,
                  "num_positional_relationships": 1},
    }


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-map-hook-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "zmq_scene_graph.json"
        self.output = self.root / "semantic_instance_graph.json"
        self.write_source(example_map())

    def write_source(self, data):
        self.source.write_text(json.dumps(data), encoding="utf-8")

    def invoke(self, **kwargs):
        return hook.update_published_graph(
            self.source, self.output, dataset_id="tests", scene_id="office", run_id="run-1",
            **kwargs,
        )

    def test_snapshot_success_idempotent_retry_and_source_hash(self):
        before = self.source.read_bytes()
        first = self.invoke(every_nth_frame=1)
        self.assertEqual(first["status"], "DONE", first)
        self.assertEqual(first["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(first["counts"]["active_entities"], 2)
        self.assertEqual(first["counts"]["active_edges"], 1)
        output_bytes = self.output.read_bytes()
        retry = self.invoke(every_nth_frame=1)
        self.assertTrue(retry["idempotent"])
        self.assertEqual(retry["revision"], first["revision"])
        self.assertEqual(output_bytes, self.output.read_bytes())
        self.assertEqual(before, self.source.read_bytes())

    def test_invalid_map_or_failed_replace_keeps_old_graph(self):
        self.assertEqual(self.invoke(every_nth_frame=1)["status"], "DONE")
        original = self.output.read_bytes()
        invalid = example_map()
        invalid["relationships"]["positional"][0]["tail_obj"]["ann_id"] = 999
        self.write_source(invalid)
        self.assertEqual(self.invoke(every_nth_frame=1)["status"], "FAILED")
        self.assertEqual(original, self.output.read_bytes())
        changed = example_map()
        changed["objects"]["1"]["category"] = "Sofa"
        self.write_source(changed)
        with patch.object(sync_scene_graph.os, "replace", side_effect=OSError("simulated disk failure")):
            result = self.invoke(every_nth_frame=1)
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(original, self.output.read_bytes())

    def test_sampled_and_unknown_coverage_preserve_unobserved_records(self):
        self.invoke(every_nth_frame=1)
        partial = example_map()
        partial["objects"].pop("2")
        partial["relationships"]["positional"] = []
        partial["stats"].update(num_objects=1, num_positional_relationships=0)
        self.write_source(partial)
        result = self.invoke(every_nth_frame=2)
        self.assertEqual(result["mode"], "partial")
        self.assertFalse(result["complete_snapshot"])
        self.assertEqual(result["counts"]["active_entities"], 2)
        unknown = self.invoke()
        self.assertEqual(unknown["mode"], "partial")

    def test_skip_or_wrong_run_never_changes_graph(self):
        with patch.object(hook, "_load_sync", side_effect=AssertionError("must not import")):
            self.assertEqual(self.invoke(skip=True)["status"], "SKIPPED")
        other = example_map()
        other["run_id"] = "another-run"
        self.write_source(other)
        self.assertEqual(self.invoke(every_nth_frame=1)["status"], "FAILED")
        self.assertFalse(self.output.exists())

    def test_input_is_frozen_before_sync(self):
        original = self.source.read_bytes()
        def sync_frozen(snapshot, output, **kwargs):
            self.source.write_text('{"newer_publication": true}', encoding="utf-8")
            self.assertEqual(Path(snapshot).read_bytes(), original)
            return sync_scene_graph.sync_graph(snapshot, output, **kwargs)
        with patch.object(hook, "_load_sync", return_value=sync_frozen):
            result = self.invoke(every_nth_frame=1)
        self.assertEqual(result["status"], "DONE", result)
        self.assertEqual(result["source_sha256"], hashlib.sha256(original).hexdigest())

    def test_real_local_and_server_published_maps(self):
        fixtures = [ROOT / "zmq_scene_graph.json", EVIDENCE / "server-published-map.json"]
        found = 0
        for index, fixture in enumerate(fixtures):
            if not fixture.exists():
                continue
            found += 1
            data = json.loads(fixture.read_text(encoding="utf-8-sig"))
            result = hook.update_published_graph(
                fixture, self.root / f"real-{index}.json", dataset_id="tests",
                scene_id=f"office-{index}", run_id=data.get("run_id") or "archived-local-map",
                every_nth_frame=1,
            )
            self.assertEqual(result["status"], "DONE", result)
            self.assertEqual(result["input_counts"]["objects"], len(data["objects"]))
            self.assertEqual(result["counts"]["active_entities"],
                             len(data["objects"]) + len(data.get("doorways", {})))
        self.assertGreater(found, 0, "at least the project map must be available")


@unittest.skipUnless((EVIDENCE / "update_map.py").exists(), "archived server coordinator unavailable")
class CoordinatorTests(unittest.TestCase):
    def test_patch_rejects_unknown_source_and_places_hook_after_publication(self):
        original = (EVIDENCE / "update_map.py").read_bytes()
        candidate = patcher.patch_bytes(original)
        with self.assertRaises(ValueError):
            patcher.patch_bytes(original + b"\n# concurrent edit\n")
        with self.assertRaises(ValueError):
            patcher.patch_bytes(candidate)
        tree = ast.parse(candidate)
        function = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                        and item.name == "process_batch")
        calls = [(node.func.id, node.lineno) for node in ast.walk(function)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        positions = {name: lineno for name, lineno in calls}
        self.assertLess(positions["atomic_publish_pair"], positions["update_published_graph"])
        self.assertLess(positions["update_published_graph"], positions["mark_done"])

    def run_coordinator(self, *, graph_should_fail):
        with tempfile.TemporaryDirectory(prefix="test-coordinator-") as temporary:
            root = Path(temporary)
            boxfusion = root / "BoxFusion"
            boxfusion.mkdir()
            batch = root / "batch"
            batch.mkdir()
            (batch / "READY").touch()
            payload = example_map()
            if graph_should_fail:
                # Public map validation permits missing semantic labels; entity
                # extraction requires a real category and reports degraded state.
                payload["objects"]["1"]["category"] = None
            demo = """import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--output-dir')
p.add_argument('--ready-file')
a, _ = p.parse_known_args()
data = json.loads(%r)
for name in ('zmq_scene_graph.json', 'zmq_scene_graph.archive.json'):
    (Path(a.output_dir) / name).write_text(json.dumps(data), encoding='utf-8')
Path(a.ready_file).touch()
""" % json.dumps(payload)
            (boxfusion / "batch_demo.py").write_text(demo, encoding="utf-8")
            (root / "batch_replay.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
            candidate = root / "update_map.py"
            candidate.write_bytes(patcher.patch_bytes((EVIDENCE / "update_map.py").read_bytes()))
            sys.path.insert(0, str(EVIDENCE))
            try:
                spec = importlib.util.spec_from_file_location("test_archived_coordinator", candidate)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            finally:
                sys.path.remove(str(EVIDENCE))
            argv = ["update_map.py", "--model-path", "unused-model", "--repo-root", str(root),
                    "--batch", str(batch), "--startup-timeout", "10", "--process-timeout", "10",
                    "--drain-timeout", "10"]
            with patch.object(sys, "argv", argv):
                args = module._parse_args()
            fake_batch = types.SimpleNamespace(frame_count=2, first_timestamp=0, last_timestamp=1,
                                               coordinate_frame_id="test", pose_warnings=[])
            stderr = io.StringIO()
            with patch.object(module, "preflight_batch", return_value=fake_batch), \
                    patch.object(module, "_run_id", return_value="integration-run"), redirect_stderr(stderr):
                result = module.process_batch(args, batch)
            self.assertEqual(result, 0)
            done = json.loads((batch / "DONE.json").read_text(encoding="utf-8"))
            self.assertEqual(done["status"], "DONE")
            expected = "FAILED" if graph_should_fail else "DONE"
            self.assertEqual(done["graph_update"]["status"], expected, done)
            self.assertEqual(done["graph_update"]["run_id"], "integration-run")
            self.assertTrue((boxfusion / "results" / "zmq_scene_graph.json").exists())
            if graph_should_fail:
                self.assertIn("WARNING: scene map published", stderr.getvalue())
            else:
                self.assertEqual(done["graph_update"]["counts"]["active_entities"], 2)

    def test_dummy_child_pipeline_publishes_map_then_graph(self):
        self.run_coordinator(graph_should_fail=False)

    def test_dummy_child_pipeline_keeps_map_done_when_graph_fails(self):
        self.run_coordinator(graph_should_fail=True)


if __name__ == "__main__":
    unittest.main()
