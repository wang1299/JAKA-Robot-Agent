#!/usr/bin/env python3
"""Deduplicate same-category ZMQ scene-graph tracks by world-space proximity."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _finite_geometry(obj: dict) -> bool:
    geometry = obj.get("geometry", {})
    center = geometry.get("center_world") or []
    half_sizes = geometry.get("aabb_half_sizes_world") or []
    return (
        len(center) >= 3
        and len(half_sizes) >= 3
        and all(math.isfinite(float(value)) for value in center[:3] + half_sizes[:3])
    )


def _distance_xy(a: dict, b: dict) -> float:
    ac = a["geometry"]["center_world"]
    bc = b["geometry"]["center_world"]
    return math.hypot(float(ac[0]) - float(bc[0]), float(ac[1]) - float(bc[1]))


def _cluster(objects: list[dict], distance: float, z_distance: float) -> list[list[dict]]:
    """Single-link spatial clustering, applied only within one category."""
    parent = list(range(len(objects)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(objects)):
        left_z = float(objects[left]["geometry"]["center_world"][2])
        for right in range(left + 1, len(objects)):
            right_z = float(objects[right]["geometry"]["center_world"][2])
            if abs(left_z - right_z) <= z_distance and _distance_xy(objects[left], objects[right]) <= distance:
                union(left, right)

    groups: dict[int, list[dict]] = defaultdict(list)
    for index, obj in enumerate(objects):
        groups[find(index)].append(obj)
    return list(groups.values())


def _median(values: list[float]) -> float:
    return float(statistics.median(float(value) for value in values))


def _merge_cluster(members: list[dict]) -> dict:
    centers = [obj["geometry"]["center_world"] for obj in members]
    median_center = [_median([center[axis] for center in centers]) for axis in range(3)]

    def representative_score(obj: dict) -> tuple[float, int, int]:
        center = obj["geometry"]["center_world"]
        distance = math.sqrt(sum((float(center[i]) - median_center[i]) ** 2 for i in range(3)))
        last_seen = int(obj["geometry"].get("last_seen_frame", -1))
        populated = sum(value not in (None, "", [], {}) for value in obj.values())
        return distance, -last_seen, -populated

    representative = min(members, key=representative_score)
    merged = copy.deepcopy(representative)
    geometry = merged["geometry"]
    half_sizes = [obj["geometry"]["aabb_half_sizes_world"] for obj in members]
    geometry["center_world"] = median_center
    geometry["aabb_half_sizes_world"] = [
        _median([half_size[axis] for half_size in half_sizes]) for axis in range(3)
    ]
    geometry["last_seen_frame"] = max(
        int(obj["geometry"].get("last_seen_frame", -1)) for obj in members
    )
    geometry["last_seen_pts"] = max(
        int(obj["geometry"].get("last_seen_pts", -1)) for obj in members
    )

    # corners_world belongs to a particular noisy track and no longer matches the
    # robust median geometry. The downstream plot and planner use center/AABB.
    geometry.pop("corners_world", None)
    member_ids = [obj.get("ann_id") for obj in members]
    merged["dedup_count"] = len(members)
    merged["dedup_members"] = member_ids
    return merged


def deduplicate(graph: dict, distance: float, z_distance: float) -> tuple[dict, dict]:
    raw_objects = graph.get("objects", {})
    objects = list(raw_objects.values()) if isinstance(raw_objects, dict) else list(raw_objects)
    valid = [obj for obj in objects if _finite_geometry(obj)]
    invalid = [copy.deepcopy(obj) for obj in objects if not _finite_geometry(obj)]

    by_category: dict[str, list[dict]] = defaultdict(list)
    for obj in valid:
        by_category[obj.get("category") or "Unknown"].append(obj)

    merged_objects = []
    cluster_sizes = []
    for category_objects in by_category.values():
        for members in _cluster(category_objects, distance, z_distance):
            merged_objects.append(_merge_cluster(members))
            cluster_sizes.append(len(members))

    # Keep invalid records unchanged instead of silently losing source data.
    for obj in invalid:
        obj["dedup_count"] = 1
        obj["dedup_members"] = [obj.get("ann_id")]
        merged_objects.append(obj)

    merged_objects.sort(key=lambda obj: str(obj.get("ann_id", "")))
    output = copy.deepcopy(graph)
    output["objects"] = {str(obj.get("ann_id")): obj for obj in merged_objects}
    output.setdefault("stats", {})["num_objects"] = len(merged_objects)
    output["deduplication"] = {
        "method": "same category + single-link world XY distance + Z guard",
        "xy_distance_m": distance,
        "z_distance_m": z_distance,
        "source_objects": len(objects),
        "result_objects": len(merged_objects),
        "merged_tracks": len(objects) - len(merged_objects),
        "invalid_geometry_kept": len(invalid),
    }
    summary = {
        "source": len(objects),
        "result": len(merged_objects),
        "merged": len(objects) - len(merged_objects),
        "invalid": len(invalid),
        "categories": Counter(obj.get("category") or "Unknown" for obj in merged_objects),
        "largest_clusters": sorted(cluster_sizes, reverse=True)[:10],
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="zmq_scene_graph.json")
    parser.add_argument("-o", "--output", default="zmq_scene_graph_dedup.json")
    parser.add_argument("--distance", type=float, default=1.0, help="XY merge distance in metres")
    parser.add_argument("--z-distance", type=float, default=1.0, help="Maximum Z difference in metres")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without writing JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    output, summary = deduplicate(graph, args.distance, args.z_distance)

    print(
        f"{summary['source']} -> {summary['result']} objects "
        f"(merged {summary['merged']}, invalid kept {summary['invalid']})"
    )
    print("categories:", dict(summary["categories"].most_common()))
    print("largest clusters:", summary["largest_clusters"])
    if not args.dry_run:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"written: {output_path.resolve()}")


if __name__ == "__main__":
    main()
