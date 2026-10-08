"""Software regression tests with CONSTRUCTED changes, not field accuracy tests."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.sync_scene_graph import GraphValidationError, sync_graph


def fixture():
    return {
        "objects": {
            "1": {"ann_id": 1, "category": "chair", "geometry": {"center_world": [1, 2, 0]},
                  "display_name_zh": "蓝色椅子"},
            "2": {"ann_id": 2, "category": "door", "semantic_name": "工作室前门"},
        },
        "doorways": {"doorway:2": {"portal_id": "doorway:2", "display_name_zh": "工作室前门"}},
        "relationships": {
            "positional": [{"head_obj": {"ann_id": 1}, "relation": "靠近", "tail_obj": {"ann_id": 2}}],
            "structural": [{"head_obj": {"ann_id": 2}, "relation": "part_of", "tail_portal_id": "doorway:2"}],
            "functional": [],
        },
    }


class SceneGraphSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "scene.json"
        self.output = Path(self.tmp.name) / "instances.json"

    def sync(self, data=None, **kwargs):
        if data is not None:
            self.source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        params = dict(dataset_id="regression-fixture", scene_id="office", run_id="run-A")
        params.update(kwargs)
        return sync_graph(self.source, self.output, **params)

    def graph(self):
        return json.loads(self.output.read_text(encoding="utf-8"))

    def test_initial_exact_triples_and_human_names(self):
        result = self.sync(fixture())
        self.assertEqual(result["summary"]["active_entities"], 3)
        self.assertEqual(result["summary"]["active_edges"], 2)
        self.assertEqual(set(result["summary"]["relation_frequency"]), {"靠近", "part_of"})
        self.assertIn("工作室前门", [n["label"] for n in self.graph()["entities"].values()])

    def test_idempotent_bytes_revision_and_frequencies(self):
        self.sync(fixture())
        before = self.output.read_bytes()
        result = self.sync()
        self.assertTrue(result["idempotent"])
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(self.graph()["revision"], 1)

    def test_same_run_updates_geometry_attribute_and_relation(self):
        self.sync(fixture())
        changed = fixture()
        changed["objects"]["1"]["geometry"]["center_world"] = [3, 2, 0]
        changed["objects"]["1"]["display_name_zh"] = "已移动的椅子"
        changed["relationships"]["positional"][0]["relation"] = "远离"
        result = self.sync(changed, mode="snapshot", complete_snapshot=True)
        self.assertEqual(result["delta_counts"]["entities"]["updated"], 1)
        self.assertEqual(result["delta_counts"]["edges"]["added"], 1)
        self.assertEqual(result["delta_counts"]["edges"]["retired"], 1)
        change = result["delta"]["entities"]["updated"][0]["changes"]["geometry"]
        self.assertEqual(change["before"]["center_world"], [1, 2, 0])
        self.assertEqual(change["after"]["center_world"], [3, 2, 0])

    def test_partial_and_uncertified_snapshot_do_not_retire(self):
        for mode in ("partial", "snapshot"):
            with self.subTest(mode=mode):
                self.sync(fixture())
                smaller = fixture()
                del smaller["objects"]["1"]
                smaller["relationships"]["positional"] = []
                result = self.sync(smaller, mode=mode)
                self.assertEqual(result["summary"]["active_entities"], 3)
                self.assertEqual(result["summary"]["active_edges"], 2)
                self.assertEqual(result["delta_counts"]["entities"]["retired"], 0)

    def test_partial_omitted_attributes_preserve_human_name_and_geometry(self):
        initial = fixture()
        initial["objects"]["1"]["aliases"] = ["窗边座椅"]
        self.sync(initial)
        partial = {"objects": {"1": {"ann_id": 1, "category": "chair", "color": "blue"}},
                   "relationships": {}}
        self.sync(partial)
        chair = next(node for node in self.graph()["entities"].values() if node["source_id"] == "1")
        self.assertEqual(chair["label"], "蓝色椅子")
        self.assertEqual(chair["aliases"], ["窗边座椅"])
        self.assertEqual(chair["geometry"]["center_world"], [1, 2, 0])
        self.assertEqual(chair["attributes"]["color"], "blue")

    def test_complete_snapshot_archives_missing_and_can_restore(self):
        self.sync(fixture())
        smaller = fixture()
        del smaller["objects"]["1"]
        smaller["relationships"]["positional"] = []
        result = self.sync(smaller, mode="snapshot", complete_snapshot=True)
        retired = result["delta"]["entities"]["retired"]
        self.assertEqual(len(retired), 1)
        self.assertFalse(retired[0]["physical_removal_confirmed"])
        self.assertEqual(result["summary"]["active_entities"], 2)
        self.assertEqual(self.sync(fixture())["delta_counts"]["entities"]["restored"], 1)

    def test_new_run_same_ann_id_is_new_entity_not_update(self):
        self.sync(fixture(), mode="snapshot", complete_snapshot=True)
        result = self.sync(fixture(), run_id="run-B", mode="snapshot", complete_snapshot=True)
        self.assertEqual(result["delta_counts"]["entities"]["added"], 3)
        self.assertEqual(result["delta_counts"]["entities"]["updated"], 0)
        self.assertEqual(result["delta_counts"]["entities"]["retired"], 3)
        self.assertEqual(result["summary"]["active_entities"], 3)
        self.assertEqual(result["summary"]["archived_entities"], 3)
        self.assertEqual(self.graph()["latest_complete_run_id"], "run-B")

    def test_new_object_upsert(self):
        self.sync(fixture())
        data = fixture()
        data["objects"]["3"] = {"ann_id": 3, "category": "cup"}
        result = self.sync(data)
        self.assertEqual(result["delta_counts"]["entities"]["added"], 1)
        self.assertEqual(result["summary"]["active_entities"], 4)

    def test_duplicate_relations_count_once_and_conflict_rejected(self):
        data = fixture()
        data["relationships"]["positional"] *= 2
        result = self.sync(data)
        self.assertEqual(result["input_counts"]["duplicate_relationship_records"], 1)
        self.assertEqual(result["summary"]["active_edges"], 2)
        data["relationships"]["positional"] = deepcopy(data["relationships"]["positional"])
        data["relationships"]["positional"][1] = {**data["relationships"]["positional"][1], "confidence": 0.9}
        with self.assertRaises(GraphValidationError):
            self.sync(data)

    def test_dangling_relation_keeps_previous_output(self):
        self.sync(fixture())
        before = self.output.read_bytes()
        data = fixture()
        data["relationships"]["positional"][0]["tail_obj"]["ann_id"] = 999
        with self.assertRaisesRegex(GraphValidationError, "Dangling"):
            self.sync(data)
        self.assertEqual(before, self.output.read_bytes())

    def test_wrong_stats_namespace_invalid_geometry_and_partial_flag_rejected(self):
        self.sync(fixture())
        before = self.output.read_bytes()
        with self.assertRaises(GraphValidationError):
            self.sync(fixture(), scene_id="another-scene")
        with self.assertRaises(GraphValidationError):
            self.sync(fixture(), complete_snapshot=True)
        data = fixture()
        data["stats"] = {"num_objects": 99}
        with self.assertRaises(GraphValidationError):
            self.sync(data)
        data = fixture()
        data["objects"]["1"]["geometry"]["center_world"] = [1, "invalid", 3]
        with self.assertRaises(GraphValidationError):
            self.sync(data)
        self.assertEqual(before, self.output.read_bytes())

    def test_source_overwrite_rejected(self):
        self.source.write_text(json.dumps(fixture()), encoding="utf-8")
        with self.assertRaises(GraphValidationError):
            sync_graph(self.source, self.source, dataset_id="d", scene_id="s", run_id="r")

    def test_atomic_replace_failure_keeps_previous_output(self):
        self.sync(fixture())
        before = self.output.read_bytes()
        data = fixture()
        data["objects"]["1"]["color"] = "red"
        with patch("tools.sync_scene_graph.os.replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                self.sync(data)
        self.assertEqual(before, self.output.read_bytes())
        self.assertFalse(list(Path(self.tmp.name).glob("*.lock")))
        self.assertFalse(list(Path(self.tmp.name).glob("*.tmp")))

    def test_exclusive_output_lock_prevents_lost_update(self):
        self.sync(fixture())
        before = self.output.read_bytes()
        lock = Path(str(self.output) + ".lock")
        lock.touch()
        with self.assertRaisesRegex(GraphValidationError, "Another sync"):
            self.sync()
        self.assertEqual(before, self.output.read_bytes())
        self.assertTrue(lock.exists())

    def test_actual_map_schema_integration(self):
        """Real archived input validates extraction, not recognition accuracy."""
        path = Path(__file__).resolve().parents[1] / "zmq_scene_graph.json"
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        before = path.read_bytes()
        result = sync_graph(path, self.output, dataset_id="archive-integration", scene_id="actual-map",
                            run_id="archive-1084", mode="snapshot", complete_snapshot=True)
        self.assertEqual(result["input_counts"]["objects"], len(raw["objects"]))
        self.assertEqual(result["input_counts"]["doorways"], len(raw["doorways"]))
        self.assertEqual(result["summary"]["active_entities"], len(raw["objects"]) + len(raw["doorways"]))
        self.assertEqual(before, path.read_bytes())
        graph = self.graph()
        self.assertTrue(all(edge["head"] in graph["entities"] and edge["tail"] in graph["entities"]
                            for edge in graph["edges"].values()))


if __name__ == "__main__":
    unittest.main()
