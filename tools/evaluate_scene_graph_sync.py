#!/usr/bin/env python3
"""Produce reproducible synchronization evidence; all change inputs are constructed.

Run from repository root: python -m tools.evaluate_scene_graph_sync --source
zmq_scene_graph.json --output-dir <new-directory>. This does not assess perception
accuracy and does not change the source map or any robot service.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from tools.sync_scene_graph import GraphValidationError, sync_graph


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_evaluation(source, output_dir):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evidence = {
        "purpose": "图谱同步软件验证；构造变化不代表真实环境变化试验或识别准确率测量",
        "source_path": str(source), "source_sha256": source_hash, "cases": [],
    }
    actual_output = output_dir / "actual_scene_instance_graph.json"
    start = time.perf_counter()
    actual = sync_graph(source, actual_output, dataset_id="archived-scene-evidence", scene_id="jaka-office",
                        run_id="archive-version-1084", mode="snapshot", complete_snapshot=True)
    evidence["actual_map_extraction"] = {
        "input_counts": actual["input_counts"], "summary": actual["summary"],
        "duration_ms": round((time.perf_counter() - start) * 1000, 3),
        "output": str(actual_output),
    }
    actual_bytes = actual_output.read_bytes()
    again = sync_graph(source, actual_output, dataset_id="archived-scene-evidence", scene_id="jaka-office",
                       run_id="archive-version-1084", mode="snapshot", complete_snapshot=True)
    evidence["actual_map_extraction"]["idempotent"] = again["idempotent"] and actual_bytes == actual_output.read_bytes()

    base = {
        "objects": {
            "1": {"ann_id": 1, "category": "chair", "display_name_zh": "测试椅子", "geometry": {"center_world": [1, 2, 0]}},
            "2": {"ann_id": 2, "category": "table", "geometry": {"center_world": [2, 2, 0]}},
        },
        "doorways": {}, "relationships": {"positional": [
            {"head_obj": {"ann_id": 1}, "relation": "靠近", "tail_obj": {"ann_id": 2}}],
            "structural": [], "functional": []},
    }
    output = output_dir / "constructed_instance_graph.json"

    def apply(name, data, *, mode="snapshot", complete=True, run_id="constructed-A"):
        path = output_dir / f"{len(evidence['cases']) + 1:02d}_{name}.json"
        save(path, data)
        start = time.perf_counter()
        result = sync_graph(path, output, dataset_id="constructed-software-regression", scene_id="fixture",
                            run_id=run_id, mode=mode, complete_snapshot=complete)
        evidence["cases"].append({"name": name, "input_kind": "constructed", "input": str(path),
                                  "duration_ms": round((time.perf_counter() - start) * 1000, 3),
                                  "result": result})
        return result

    initial = apply("baseline", base)
    added = deepcopy(base)
    added["objects"]["3"] = {"ann_id": 3, "category": "cup", "display_name_zh": "测试杯子"}
    r_add = apply("add_object", added)
    moved = deepcopy(added)
    moved["objects"]["1"]["geometry"]["center_world"] = [4, 2, 0]
    moved["objects"]["1"]["display_name_zh"] = "位置调整后的测试椅子"
    moved["relationships"]["positional"][0]["relation"] = "远离"
    r_move = apply("move_and_relation_change", moved)
    missing = deepcopy(moved)
    del missing["objects"]["3"]
    r_missing = apply("missing_in_complete_snapshot", missing)
    partial = {"objects": {"1": {"ann_id": 1, "category": "chair", "color": "blue"}},
               "doorways": {}, "relationships": {}}
    r_partial = apply("partial_observation", partial, mode="partial", complete=False)
    before = output.read_bytes()
    r_repeat = apply("repeat_identical_partial", partial, mode="partial", complete=False)
    repeat_preserved = before == output.read_bytes()
    r_new = apply("new_run_reused_ann_ids", base, run_id="constructed-B")
    invalid = deepcopy(base)
    invalid["relationships"]["positional"][0]["tail_obj"]["ann_id"] = 999
    invalid_path = output_dir / "08_invalid_dangling_endpoint.json"
    save(invalid_path, invalid)
    before = output.read_bytes()
    try:
        sync_graph(invalid_path, output, dataset_id="constructed-software-regression", scene_id="fixture",
                   run_id="constructed-B", mode="snapshot", complete_snapshot=True)
        invalid_rejected = False
    except GraphValidationError as exc:
        invalid_rejected = True
        evidence["invalid_input_rejection"] = {"error": str(exc), "old_output_unchanged": before == output.read_bytes()}
    checks = {
        "actual_map_idempotent": evidence["actual_map_extraction"]["idempotent"],
        "baseline_2_entities_1_edge": initial["summary"]["active_entities"] == 2 and initial["summary"]["active_edges"] == 1,
        "new_object_added_once": r_add["delta_counts"]["entities"]["added"] == 1,
        "move_and_attribute_update": r_move["delta_counts"]["entities"]["updated"] == 1,
        "changed_relation_replaces_prior": r_move["delta_counts"]["edges"]["added"] == 1 and r_move["delta_counts"]["edges"]["retired"] == 1,
        "missing_archived_not_physical_removal": r_missing["delta_counts"]["entities"]["retired"] == 1 and not r_missing["delta"]["entities"]["retired"][0]["physical_removal_confirmed"],
        "partial_preserves_unobserved_records": r_partial["summary"]["active_entities"] == 2 and r_partial["summary"]["active_edges"] == 1,
        "identical_sync_byte_idempotent": r_repeat["idempotent"] and repeat_preserved,
        "new_run_does_not_match_ann_ids": r_new["delta_counts"]["entities"]["added"] == 2 and r_new["delta_counts"]["entities"]["updated"] == 0,
        "dangling_rejected_keeps_previous_graph": invalid_rejected and before == output.read_bytes(),
        "actual_source_untouched": hashlib.sha256(source.read_bytes()).hexdigest() == source_hash,
    }
    evidence["checks"] = checks
    evidence["passed"] = all(checks.values())
    evidence["check_count"] = len(checks)
    save(output_dir / "evidence_summary.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New evidence directory; must not already exist")
    args = parser.parse_args()
    result = run_evaluation(args.source, args.output_dir)
    print(json.dumps({"passed": result["passed"], "checks": result["checks"],
                      "actual_input_counts": result["actual_map_extraction"]["input_counts"],
                      "evidence": str(args.output_dir / "evidence_summary.json")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
