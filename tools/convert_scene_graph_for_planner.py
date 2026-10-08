#!/usr/bin/env python3
"""Convert a ZMQ/BoxFusion scene graph to qwen_planner's object-map schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CATEGORY_ZH = {
    "Silver elevator door": "电梯门",
    "Ceiling spotlights": "顶灯",
    "Potted plant": "盆栽",
    "Blue chair": "椅子",
    "Paper wall poster": "墙面海报",
    "White utility table": "白色操作台",
    "Wooden storage cabinet": "木质储物柜",
    "Black flight case": "黑色航空箱",
    "Leather armchair": "皮质扶手椅",
}

FUNC_DESC_FALLBACK = {
    "Silver elevator door": "电梯入口门",
    "Ceiling spotlights": "室内照明灯具",
    "Potted plant": "装饰用盆栽植物",
    "Blue chair": "供人乘坐的座椅",
    "Paper wall poster": "墙面展示或宣传海报",
    "White utility table": "可放置物品的操作台",
    "Wooden storage cabinet": "存放物品的木质柜体",
    "Black flight case": "可搬运的箱体",
    "Leather armchair": "皮质单人座椅",
}


def _sort_key(obj: dict) -> tuple[int, str]:
    ann_id = obj.get("ann_id")
    try:
        return 0, f"{int(ann_id):012d}"
    except (TypeError, ValueError):
        return 1, str(ann_id)


def _normalized_ann_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _valid_vector(value, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= length
        and all(math.isfinite(float(item)) for item in value[:length])
    )


def convert(source: dict) -> tuple[dict, list[dict]]:
    raw_objects = source.get("objects", {})
    objects = list(raw_objects.values()) if isinstance(raw_objects, dict) else list(raw_objects)
    converted = []
    skipped = []

    for obj in sorted(objects, key=_sort_key):
        geometry = obj.get("geometry") or {}
        center = geometry.get("center_world")
        half_sizes = geometry.get("aabb_half_sizes_world")
        if not _valid_vector(center, 3) or not _valid_vector(half_sizes, 3):
            skipped.append({"ann_id": _normalized_ann_id(obj.get("ann_id")), "category": obj.get("category")})
            continue

        center = [round(float(value), 3) for value in center[:3]]
        size = [round(float(value) * 2.0, 3) for value in half_sizes[:3]]
        category = obj.get("category") or "Unknown"
        category_zh = obj.get("category_zh") or CATEGORY_ZH.get(category, category)
        func_desc = obj.get("func_desc") or FUNC_DESC_FALLBACK.get(category, category_zh)

        converted.append(
            {
                "ann_id": _normalized_ann_id(obj.get("ann_id")),
                "category": category,
                "category_zh": category_zh,
                "func_desc": func_desc,
                "position": f"世界系({center[0]:.1f},{center[1]:.1f},z={center[2]:.1f})",
                "box3d": {
                    "center": center,
                    "size": size,
                    "yaw": 0.0,
                },
                "floor_xy": center[:2],
                "viewpoint": None,
                "marker": None,
            }
        )

    relationships = source.get("relationships") or {}
    if isinstance(relationships, dict):
        func_relationships = relationships.get("functional") or []
        pos_relationships = relationships.get("positional") or []
    else:
        func_relationships = source.get("func_relationships") or []
        pos_relationships = source.get("pos_relationships") or []

    result = {
        "frame": "robot_map",
        "objects": converted,
        "func_relationships": func_relationships,
        "pos_relationships": pos_relationships,
    }
    return result, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    result, skipped = convert(source)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"converted: {len(result['objects'])} objects -> {output_path.resolve()}")
    if skipped:
        print(f"skipped without valid 3D geometry: {skipped}")


if __name__ == "__main__":
    main()
