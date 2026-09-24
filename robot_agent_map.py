"""Semantic selection belongs to the model; filtering/counting belongs to code.

No object-name synonyms or user-phrase routing rules live here. One category
query covers every matching record, including objects not on the display page.
"""
from collections import Counter
import hashlib
import json
import math
from robot_agent import ToolInputError


FILTER_FIELDS = ("category", "category_zh", "position", "func_desc")
FILTER_SCHEMA = {"type": "array", "maxItems": 8, "default": [],
    "description": "可选属性条件数组，无需条件则省略或 []。每项含 field（category/category_zh/position/func_desc）、operator（contains/equals）、value（字面文字）。多个条件同时满足。",
    "items": {
    "type": "object", "properties": {
        "field": {"type": "string", "enum": list(FILTER_FIELDS)},
        "operator": {"type": "string", "enum": ["contains", "equals"]},
        "value": {"type": "string"}},
    "required": ["field", "operator", "value"], "additionalProperties": False}}


class MapEvidence:
    def __init__(self, graph):
        self.name = str(graph.get("name", "未命名地图"))[:200]
        self.objects, self.by_category, self.labels = {}, {}, {}
        self.excluded = 0
        for original in graph.get("objects", []):
            obj = dict(original)
            if obj.get("lifecycle", "active") != "active" or obj.get("ann_id") is None:
                self.excluded += 1
                continue
            key = str(obj["ann_id"])
            if key in self.objects:
                if self.objects[key] != obj:
                    raise ToolInputError("地图包含冲突的对象 ID，暂时无法可靠查询")
                continue
            self.objects[key] = obj
            category = str(obj.get("category") or obj.get("category_zh") or "未知类别")
            self.by_category.setdefault(category, []).append(key)
            self.labels[category] = str(obj.get("category_zh") or category)
        encoded = json.dumps(self.objects, ensure_ascii=False, sort_keys=True)
        self.version = hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def metadata(self):
        return {"map_name": self.name, "snapshot": self.version,
                "source": "保存的拟物体地图，不代表实时现场", "source_type": "saved_map",
                "total_objects": len(self.objects), "excluded_inactive_or_unidentified": self.excluded}

    def category_catalog(self, offset=0, limit=80):
        categories = list(self.by_category)
        return {"categories": [{"category": name, "label": self.labels[name]} for name in categories[offset:offset + limit]],
                "total_categories": len(categories),
                "next_offset": offset + limit if offset + limit < len(categories) else None}

    def compare_positions(self, reference_ann_id, candidate_ann_ids):
        """Geometric evidence, never an automatic choice or navigable route."""
        if (type(reference_ann_id) is not int or not isinstance(candidate_ann_ids, list)
                or not 1 <= len(candidate_ann_ids) <= 40
                or any(type(i) is not int for i in candidate_ann_ids)):
            raise ToolInputError("请提供已查询到的地图对象整数 ID")
        if reference_ann_id in candidate_ann_ids or len(set(candidate_ann_ids)) != len(candidate_ann_ids):
            raise ToolInputError("参考对象与候选不能重复")

        def point(ann_id):
            obj = self.objects.get(str(ann_id))
            if obj is None:
                raise ToolInputError("对象不在当前活动地图中，请重新查询")
            xy = obj.get("floor_xy")
            if (not isinstance(xy, (list, tuple)) or len(xy) < 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy[:2])):
                raise ToolInputError("对象缺少有效地图中心坐标，不能推断空间关系")
            return {"ann_id": ann_id, "category": obj.get("category"), "floor_xy": xy[:2]}

        ref = point(reference_ann_id)
        candidates = []
        for ann_id in candidate_ann_ids:
            row = point(ann_id)
            row["center_distance_m"] = round(math.dist(ref["floor_xy"], row["floor_xy"]), 3)
            candidates.append(row)
        return {"ok": True, **self.metadata(), "reference": ref,
                "candidates": sorted(candidates, key=lambda row: row["center_distance_m"]),
                "meaning": "同一保存地图坐标系内的平面中心直线距离（米），不是可行走路径；最近不等于相邻，不证明实时位置。不自动选择目标；若都很远、坐标系不一致或线索矛盾，应澄清。"}

    def query(self, question, categories, filters=None, offset=0):
        selected = list(dict.fromkeys(categories))
        if "*" in selected:
            if len(selected) != 1:
                raise ToolInputError("全图选择 * 不能与具体类别混用")
            selected = list(self.by_category)
        unknown = set(selected) - set(self.by_category)
        if unknown:
            raise ToolInputError("类别不在当前地图中，请使用目录里的原始 category 值；没有匹配类别时传空列表。")
        filters = filters or []
        # Defense in depth: no arbitrary field access, eval, regex or shell.
        if len(filters) > 8 or any(set(rule) != {"field", "operator", "value"}
                or rule["field"] not in FILTER_FIELDS or rule["operator"] not in ("contains", "equals")
                or not isinstance(rule["value"], str) or not rule["value"].strip() for rule in filters):
            raise ToolInputError("地图筛选条件无效")
        ids = [key for name in selected for key in self.by_category[name]]
        missing_fields = set()
        def matches(key):
            obj = self.objects[key]
            for rule in filters:
                value = str(obj.get(rule["field"]) or "").casefold()
                wanted = rule["value"].strip().casefold()
                if not value:
                    missing_fields.add(rule["field"])
                    return False
                if (value != wanted if rule["operator"] == "equals" else wanted not in value):
                    return False
            return True
        ids = [key for key in ids if matches(key)]
        if type(offset) is not int or not 0 <= offset <= len(ids):
            raise ToolInputError("详情页偏移超出查询结果范围")
        counts = Counter(str(self.objects[key].get("category") or self.objects[key].get("category_zh") or "未知类别") for key in ids)
        fields = ("ann_id", "category", "category_zh", "position", "func_desc", "floor_xy", "nav_xy")
        details, used = [], 0
        for key in ids[offset:offset + 20]:
            row = {field: (value[:300] if isinstance(value, str) else value)
                   for field, value in self.objects[key].items() if field in fields}
            cost = len(json.dumps(row, ensure_ascii=False))
            if used + cost > 10000:
                break
            details.append(row)
            used += cost
        relations = []
        if len(ids) <= 8:
            for index, key in enumerate(ids):
                for other in ids[index + 1:]:
                    try:
                        pair = self.compare_positions(int(key), [int(other)])
                        relations.append({"from_ann_id": int(key), "to_ann_id": int(other),
                                          "center_distance_m": pair["candidates"][0]["center_distance_m"]})
                    except (ToolInputError, ValueError, TypeError):
                        pass
        return {"ok": True, **self.metadata(), "question": question,
                "selection": {"categories": selected, "filters": filters},
                "count": len(ids), "groups": [{"category": name, "label": self.labels[name], "count": count}
                                              for name, count in counts.items()],
                "target_ids": ids[:500], "highlight_truncated": len(ids) > 500,
                "objects": details, "offset": offset,
                "spatial_relations": relations,
                "spatial_note": "小结果集附中心直线距离；不是导航路径，也不自动证明相邻。更多对象请调用 compare_map_positions。",
                "next_offset": offset + len(details) if offset + len(details) < len(ids) else None,
                "count_scope": "所选类别及条件下的全部地图记录，数量不受详情分页影响；不代表实时现场或未记录区域。",
                "missing_filter_fields": sorted(missing_fields)}
