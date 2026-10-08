#!/usr/bin/env python3
"""Synchronize a published BoxFusion map into a separate, auditable instance graph.

Only explicit JSON entities and relationships are extracted. Category frequency is
descriptive, never evidence for a new relation. A complete *source snapshot* is not
proof that every physical object was observed. Missing records are archived, not
declared physically removed. Identifiers are only comparable within one run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote


SCHEMA = "jaka.scene-instance-graph/v1"
FAMILIES = ("functional", "positional", "structural")
WORLD_GEOMETRY = ("center_world", "aabb_half_sizes_world", "corners_world")


class GraphValidationError(ValueError):
    """The input cannot be represented without silently losing information."""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GraphValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    raw = Path(path).read_bytes()
    try:
        data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              GraphValidationError(f"Nonfinite JSON constant: {value}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GraphValidationError(f"Invalid JSON: {exc}") from exc
    return data, hashlib.sha256(raw).hexdigest()


def _text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise GraphValidationError(f"{context} must be a nonempty string")
    return value


def _identifier(value, context):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
        raise GraphValidationError(f"{context} must be a string or integer identifier")
    return str(value)


def _mapping(value, context):
    if not isinstance(value, dict):
        raise GraphValidationError(f"{context} must be an object")
    return value


def _finite_geometry(value, context):
    if isinstance(value, list):
        for child in value:
            _finite_geometry(child, context)
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GraphValidationError(f"{context} contains invalid geometry")


def _entity_id(dataset_id, scene_id, run_id, kind, local_id):
    return "/".join(quote(str(part), safe="") for part in (
        "scene", dataset_id, scene_id, "run", run_id, kind, local_id))


def extract_scene(data, *, dataset_id, scene_id, run_id):
    """Validate and extract explicit entities/edges; no model or text inference."""
    _mapping(data, "scene")
    objects = _mapping(data.get("objects"), "objects")
    doorways = _mapping(data.get("doorways", {}), "doorways")
    relationships = _mapping(data.get("relationships"), "relationships")
    unknown = set(relationships) - set(FAMILIES)
    if unknown:
        raise GraphValidationError(f"Unsupported relationship families: {sorted(unknown)}")
    nodes, references = {}, {}
    for kind, collection, id_field in (("object", objects, "ann_id"), ("doorway", doorways, "portal_id")):
        for key, raw in collection.items():
            _mapping(raw, f"{kind} {key}")
            local_id = _identifier(raw.get(id_field), f"{kind}.{id_field}")
            if str(key) != local_id:
                raise GraphValidationError(f"{kind} key {key} differs from {id_field} {local_id}")
            category = _text(raw.get("category", "Doorway" if kind == "doorway" else None), "category")
            aliases = raw.get("aliases", [])
            if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
                raise GraphValidationError(f"{kind} {key} aliases must be a list of strings")
            geometry = _mapping(raw.get("geometry", {}), f"{kind} {key}.geometry")
            projected = {}
            for field in WORLD_GEOMETRY:
                if field in geometry and geometry[field] is not None:
                    _finite_geometry(geometry[field], f"{kind} {key}.{field}")
                    expected = 8 if field == "corners_world" else 3
                    if not isinstance(geometry[field], list) or len(geometry[field]) != expected:
                        raise GraphValidationError(f"{kind} {key}.{field} needs {expected} values")
                    if field == "corners_world" and any(not isinstance(row, list) or len(row) != 3 for row in geometry[field]):
                        raise GraphValidationError(f"{kind} {key}.corners_world must be 8 by 3")
                    projected[field] = deepcopy(geometry[field])
            entity_kind = "doorway" if kind == "doorway" else (
                "navigation_waypoint" if raw.get("type") == "navigation waypoint" else "physical_object")
            node_id = _entity_id(dataset_id, scene_id, run_id, kind, local_id)
            nodes[node_id] = {
                "id": node_id, "run_id": run_id, "source_id": local_id,
                "entity_kind": entity_kind, "category": category,
                "label": raw.get("semantic_name") or raw.get("display_name_zh") or raw.get("category_zh") or category,
                "aliases": deepcopy(aliases), "geometry": projected,
                "attributes": {key: deepcopy(value) for key, value in raw.items()
                               if key not in (id_field, "geometry", "category", "aliases")},
            }
            references[(kind, local_id)] = node_id

    def endpoint(raw, side):
        object_key, portal_key = f"{side}_obj", f"{side}_portal_id"
        if (object_key in raw) == (portal_key in raw):
            raise GraphValidationError(f"Relationship needs exactly one {object_key} or {portal_key}")
        if object_key in raw:
            item = _mapping(raw[object_key], object_key)
            ref = ("object", _identifier(item.get("ann_id"), object_key))
        else:
            ref = ("doorway", _identifier(raw[portal_key], portal_key))
        if ref not in references:
            raise GraphValidationError(f"Dangling relationship endpoint: {ref}")
        return references[ref]

    edges = {}
    input_relationship_records = 0
    for family in FAMILIES:
        records = relationships.get(family, [])
        if not isinstance(records, list):
            raise GraphValidationError(f"relationships.{family} must be a list")
        input_relationship_records += len(records)
        for raw in records:
            _mapping(raw, f"{family} relationship")
            head, tail = endpoint(raw, "head"), endpoint(raw, "tail")
            relation = _text(raw.get("relation"), "relation")
            edge_id = "edge/" + _hash([head, family, relation, tail])
            edge = {"id": edge_id, "run_id": run_id, "head": head, "relation": relation,
                    "tail": tail, "family": family,
                    "attributes": {key: deepcopy(value) for key, value in raw.items() if key not in (
                        "head_obj", "tail_obj", "head_portal_id", "tail_portal_id", "relation")}}
            if edge_id in edges and edges[edge_id] != edge:
                raise GraphValidationError(f"Conflicting attributes for duplicate relationship: {edge_id}")
            edges[edge_id] = edge
    stats = data.get("stats", {})
    _mapping(stats, "stats")
    expected = {"num_objects": len(objects), "num_doorways": len(doorways)}
    for family in FAMILIES:
        expected[f"num_{family}_relationships"] = len(relationships.get(family, []))
    for key, count in expected.items():
        if key in stats and (type(stats[key]) is not int or stats[key] != count):
            raise GraphValidationError(f"stats.{key}={stats[key]!r}, actual={count}")
    counts = {
        "objects": len(objects), "doorways": len(doorways), "entities": len(nodes),
        "physical_objects": sum(node["entity_kind"] == "physical_object" for node in nodes.values()),
        "navigation_waypoints": sum(node["entity_kind"] == "navigation_waypoint" for node in nodes.values()),
        "object_categories": len({raw["category"] for raw in objects.values()}),
        "physical_object_categories": len({node["category"] for node in nodes.values() if node["entity_kind"] == "physical_object"}),
        "input_relationship_records": input_relationship_records,
        "deduplicated_relationships": len(edges),
        "duplicate_relationship_records": input_relationship_records - len(edges),
        "relationships_by_family": dict(sorted(Counter(edge["family"] for edge in edges.values()).items())),
    }
    return nodes, edges, counts


def _payload(record):
    return {key: value for key, value in record.items() if key not in (
        "status", "first_seen_revision", "last_changed_revision", "archived_revision", "archive_reason")}


def _changes(before, after):
    return {key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}


def _merge_patch(before, after):
    """Preserve fields omitted from a partial observation, including human labels."""
    merged = deepcopy(before)
    for key, value in after.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_patch(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _active_summary(entities, edges):
    active_nodes = [node for node in entities.values() if node["status"] == "active"]
    active_edges = [edge for edge in edges.values() if edge["status"] == "active"]
    return {
        "active_entities": len(active_nodes), "archived_entities": len(entities) - len(active_nodes),
        "active_edges": len(active_edges), "archived_edges": len(edges) - len(active_edges),
        "active_entity_kinds": dict(sorted(Counter(node["entity_kind"] for node in active_nodes).items())),
        "active_edge_families": dict(sorted(Counter(edge["family"] for edge in active_edges).items())),
        "category_frequency": dict(sorted(Counter(node["category"] for node in active_nodes).items())),
        "relation_frequency": dict(sorted(Counter(edge["relation"] for edge in active_edges).items())),
        "frequency_scope": "active explicit records only; frequency is not a truth/confidence score",
    }


def _validate_graph(graph, namespace):
    _mapping(graph, "graph")
    if graph.get("schema_version") != SCHEMA or graph.get("namespace") != namespace:
        raise GraphValidationError("Existing graph schema or dataset/scene namespace differs")
    if type(graph.get("revision")) is not int or graph["revision"] < 0:
        raise GraphValidationError("Invalid graph revision")
    entities = _mapping(graph.get("entities"), "entities")
    edges = _mapping(graph.get("edges"), "edges")
    if not isinstance(graph.get("audit"), list):
        raise GraphValidationError("graph.audit must be a list")
    for kind, records in (("entity", entities), ("edge", edges)):
        for key, record in records.items():
            _mapping(record, f"{kind} {key}")
            if record.get("id") != key or record.get("status") not in ("active", "archived"):
                raise GraphValidationError(f"Invalid {kind} identity/status {key}")
            _text(record.get("run_id"), f"{kind}.run_id")
            if kind == "edge":
                if record.get("head") not in entities or record.get("tail") not in entities:
                    raise GraphValidationError(f"Dangling stored edge {key}")
                if record["status"] == "active" and any(entities[record[end]]["status"] != "active" for end in ("head", "tail")):
                    raise GraphValidationError(f"Active edge {key} has archived endpoint")


@contextmanager
def _output_lock(path):
    lock_path = Path(str(path) + ".lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise GraphValidationError(f"Another sync holds {lock_path}; inspect before removing a stale lock") from exc
    try:
        os.close(fd)
        yield
    finally:
        lock_path.unlink()


def _atomic_write(path, graph):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(graph, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sync_graph(input_path, output_path, *, dataset_id, scene_id, run_id,
               mode="partial", complete_snapshot=False):
    """Atomically upsert source records and return a machine-readable audit summary.

    ``run_id`` identifies a BoxFusion reconstruction run, not this sync invocation.
    With mode=snapshot AND complete_snapshot=True, missing records of this scene
    are archived. Other runs are superseded without ann_id matching. Without both
    arguments this is upsert-only, including when a batch observes a small area.
    The existing output remains intact on validation or write failure.
    """
    for name, value in (("dataset_id", dataset_id), ("scene_id", scene_id), ("run_id", run_id)):
        _text(value, name)
    if mode not in ("snapshot", "partial"):
        raise GraphValidationError("mode must be snapshot or partial")
    if complete_snapshot and mode != "snapshot":
        raise GraphValidationError("complete_snapshot requires mode=snapshot")
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise GraphValidationError("Output must be separate from the input scene map")
    data, source_hash = _read_json(input_path)
    nodes, extracted_edges, input_counts = extract_scene(data, dataset_id=dataset_id, scene_id=scene_id, run_id=run_id)
    namespace = {"dataset_id": dataset_id, "scene_id": scene_id}
    receipt = {"sha256": source_hash, "run_id": run_id, "mode": mode,
               "complete_snapshot": bool(complete_snapshot)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_path):
        if output_path.exists():
            graph, _ = _read_json(output_path)
            _validate_graph(graph, namespace)
        else:
            graph = {"schema_version": SCHEMA, "namespace": namespace, "revision": 0,
                     "entities": {}, "edges": {}, "audit": []}
        if graph.get("last_receipt") == receipt:
            return {"changed": False, "idempotent": True, "output": str(output_path),
                    "revision": graph["revision"], "input_sha256": source_hash,
                    "input_counts": input_counts, "summary": graph["summary"], "delta": None}
        revision = graph["revision"] + 1
        delta = {kind: {"added": [], "updated": [], "retired": [], "restored": []}
                 for kind in ("entities", "edges")}
        for kind, incoming in (("entities", nodes), ("edges", extracted_edges)):
            stored = graph[kind]
            for key, record in incoming.items():
                before = stored.get(key)
                if before is None:
                    stored[key] = {**record, "status": "active", "first_seen_revision": revision,
                                   "last_changed_revision": revision}
                    delta[kind]["added"].append(key)
                else:
                    if not (mode == "snapshot" and complete_snapshot):
                        record = _merge_patch(_payload(before), record)
                        if kind == "entities":
                            source_collection = data["doorways"] if record["entity_kind"] == "doorway" else data["objects"]
                            source_record = source_collection[record["source_id"]]
                            if "aliases" not in source_record:
                                record["aliases"] = deepcopy(before["aliases"])
                            if not any(name in source_record for name in ("semantic_name", "display_name_zh", "category_zh")):
                                if before["label"] != before["category"]:
                                    record["label"] = before["label"]
                    change = _changes(_payload(before), record)
                    if change:
                        delta[kind]["updated"].append({"id": key, "changes": change})
                    if before["status"] == "archived":
                        delta[kind]["restored"].append(key)
                    if change or before["status"] == "archived":
                        stored[key] = {**record, "status": "active", "first_seen_revision": before["first_seen_revision"],
                                       "last_changed_revision": revision}
            if mode == "snapshot" and complete_snapshot:
                for key, record in stored.items():
                    if key not in incoming and record["status"] == "active":
                        reason = ("not_observed_in_complete_source_snapshot" if record["run_id"] == run_id
                                  else "superseded_by_complete_source_snapshot")
                        record.update(status="archived", archived_revision=revision, archive_reason=reason,
                                      last_changed_revision=revision)
                        delta[kind]["retired"].append({"id": key, "reason": reason,
                                                     "physical_removal_confirmed": False})
        delta_counts = {kind: {action: len(records) for action, records in changes.items()}
                        for kind, changes in delta.items()}
        source = {"input_path": str(input_path), "input_sha256": source_hash, "run_id": run_id,
                  "scene_graph_version": data.get("scene_graph_version"), "saved_at": data.get("saved_at"),
                  "input_counts": input_counts, "mode": mode, "complete_source_snapshot": bool(complete_snapshot)}
        event = {"revision": revision, "recorded_at": datetime.now(timezone.utc).isoformat(),
                 "source": source, "delta_counts": delta_counts, "delta": delta,
                 "physical_removal_inferred": False}
        graph.update(revision=revision, latest_source=source, last_receipt=receipt,
                     summary=_active_summary(graph["entities"], graph["edges"]))
        if mode == "snapshot" and complete_snapshot:
            graph["latest_complete_run_id"] = run_id
        graph["audit"].append(event)
        _validate_graph(graph, namespace)
        _atomic_write(output_path, graph)
    return {"changed": True, "idempotent": False, "output": str(output_path), "revision": revision,
            "input_sha256": source_hash, "input_counts": input_counts, "summary": graph["summary"],
            "delta_counts": delta_counts, "delta": delta}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Published BoxFusion scene JSON")
    parser.add_argument("--output", required=True, type=Path, help="Separate scene instance graph JSON")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True, help="Stable reconstruction run ID; never reuse across rebuilds")
    parser.add_argument("--mode", choices=("snapshot", "partial"), default="partial")
    parser.add_argument("--complete-snapshot", action="store_true", help="Certify a complete source snapshot, allowing archival of missing records")
    args = parser.parse_args(argv)
    try:
        result = sync_graph(args.input, args.output, dataset_id=args.dataset_id, scene_id=args.scene_id,
                            run_id=args.run_id, mode=args.mode, complete_snapshot=args.complete_snapshot)
    except (GraphValidationError, OSError) as exc:
        parser.exit(1, f"scene graph sync failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
