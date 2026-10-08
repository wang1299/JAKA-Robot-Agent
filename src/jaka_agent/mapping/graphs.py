"""Normalize external scene graphs into the navigation object schema."""
from __future__ import annotations
import copy
import math
import re

CATEGORY_ZH_MAP = {
    "black flight case": "黑色航空箱",
    "blue chair": "蓝色椅子",
    "ceiling spotlights": "天花板射灯",
    "glass coffee table": "玻璃茶几",
    "glass_table": "玻璃桌",
    "guest drop-off point": "送客点",
    "guest pickup point": "接客点",
    "leather armchair": "皮质扶手椅",
    "ordinary office door": "普通办公室门",
    "door_panel": "门板",
    "paper wall poster": "纸质墙面海报",
    "potted plant": "盆栽",
    "shrub or bush": "灌木",
    "silver elevator door": "银色电梯门",
    "white utility table": "白色工作台",
    "window": "窗户",
    "wooden storage cabinet": "木质储物柜",
    "wooden study desk": "木质书桌",
    "chair": "椅子",
    "coffee table": "茶几",
    "desk": "办公桌",
    "dining table": "餐桌",
    "elevator display": "电梯显示屏",
    "plant": "植物",
    "sofa": "沙发",
    "table": "桌子",
    "whiteboard": "白板",
    "cabinet": "柜子",
    "case": "箱子",
    "shelf": "置物架",
    "platform trolley": "平台推车",
}


MAP_ICON_NAMES = {
    "chair", "plant", "shrub", "case", "table", "coffeetable", "desk", "cabinet",
    "door", "window", "poster", "elevator", "light", "object", "pickup", "dropoff",
}


def _normalized_ann_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _normalize_graph(data: dict, name="scene_graph.json") -> dict:
    """任意场景图 → planner 可用结构; 无二维世界坐标的对象不进入导航地图。"""
    raw = data.get("objects", [])
    objects = list(raw.values()) if isinstance(raw, dict) else list(raw or [])
    normalized = []
    for source in objects:
        if not isinstance(source, dict):
            continue
        obj = copy.deepcopy(source)
        geometry = obj.get("geometry") or {}
        center = obj.get("floor_xy") or geometry.get("center_world")
        if not isinstance(center, list) or len(center) < 2:
            continue
        try:
            x, y = float(center[0]), float(center[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        obj["ann_id"] = _normalized_ann_id(obj.get("ann_id"))
        obj["category"] = obj.get("category") or "Unknown"
        category_key = re.sub(r"\s+", " ", str(obj["category"]).strip().lower())
        obj["category_zh"] = obj.get("category_zh") or CATEGORY_ZH_MAP.get(category_key, obj["category"])
        obj["category_label_zh"] = CATEGORY_ZH_MAP.get(category_key, obj["category"])
        obj["func_desc"] = obj.get("func_desc") or ""
        obj["position"] = obj.get("position") or f"世界系({x:.1f},{y:.1f})"
        obj["floor_xy"] = [x, y]
        obj.setdefault("viewpoint", None)
        obj.setdefault("marker", None)
        if not obj.get("box3d"):
            center3 = geometry.get("center_world") or [x, y, 0.0]
            half = geometry.get("aabb_half_sizes_world") or [0.25, 0.25, 0.25]
            if len(center3) >= 3 and len(half) >= 3:
                obj["box3d"] = {
                    "center": [float(center3[0]), float(center3[1]), float(center3[2])],
                    "size": [float(half[0]) * 2, float(half[1]) * 2, float(half[2]) * 2],
                    "yaw": 0.0,
                }
        normalized.append(obj)
    if not normalized:
        raise ValueError(f"{name} 中没有带 floor_xy/center_world 的有效物体")
    portals = []
    raw_portals = data.get('doorways') or {}
    for portal in (raw_portals.values() if isinstance(raw_portals, dict) else raw_portals):
        if not isinstance(portal, dict) or portal.get('lifecycle') == 'deleted':
            continue
        center = portal.get('floor_xy') or (portal.get('geometry') or {}).get('center_world')
        if not isinstance(center, list) or len(center) < 2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in center[:2]):
            continue
        if not isinstance(portal.get('portal_id'), str):
            continue
        nav_id = portal.get('navigation_ann_id')
        navigation_enabled = (portal.get('navigation_mode') == 'approach' and type(nav_id) is int and nav_id >= 0)
        if navigation_enabled:
            existing = next((obj for obj in normalized if obj['ann_id'] == nav_id), None)
            if existing and existing.get('source_portal_id') != portal['portal_id']:
                raise ValueError('门洞导航编号与现有对象冲突')
            if not existing:
                # An explicitly configured approach target, not proof the opening
                # is traversable. Existing accessible-point planning is retained.
                target = copy.deepcopy(portal)
                target.update(ann_id=nav_id, source_portal_id=portal['portal_id'],
                    category_zh=portal.get('display_name_zh') or portal.get('semantic_name') or '门洞',
                    floor_xy=list(center[:2]), lifecycle='active',
                    func_desc='门洞附近观察目标；通行未验证，不表示可穿越或进入。')
                target.pop('doorways', None)
                target.pop('navigation_pose', None)
                target = _normalize_graph({'objects':[target]}, name)['objects'][0]
                normalized.append(target)
        portals.append({'portal_id':portal['portal_id'], 'category':portal.get('category') or 'Doorway',
            'category_zh':portal.get('display_name_zh') or portal.get('semantic_name') or portal.get('category_zh') or '门洞',
            'floor_xy':center[:2], 'anchor_ann_id':portal.get('anchor_ann_id'),
            'member_ids':portal.get('member_ids') or [], 'status':portal.get('status'),
            'navigation_ann_id':nav_id if navigation_enabled else None,
            'navigation_mode':'approach' if navigation_enabled else None,
            'traversability':portal.get('traversability') or 'not_verified', 'display_only':not navigation_enabled})
    return {
        "name": name,
        "frame": data.get("frame") or data.get("video_id") or "robot_map",
        "objects": normalized,
        "doorways": portals,
        "func_relationships": data.get("func_relationships") or (data.get("relationships") or {}).get("functional", []),
        "pos_relationships": data.get("pos_relationships") or (data.get("relationships") or {}).get("positional", []),
    }
