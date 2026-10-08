"""Validate web task inputs and task identifiers."""
from __future__ import annotations
import re

def _step_target_ids(step: dict) -> list:
    if step.get("type") in ("cruise", "patrol"):
        return list(step.get("target_ann_ids") or [])
    target = step.get("target_ann_id")
    return [] if target is None else [target]


def _fallback_clarification(instruction: str, plan: dict, objects: list):
    """模型漏问时兜底: 同名的单目标导航必须先让用户选定 ann_id。"""
    if plan.get("clarification"):
        return plan["clarification"]
    object_by_id = {obj["ann_id"]: obj for obj in objects}
    explicit_ids = {
        int(value)
        for value in re.findall(r"(?:ann_id\s*[=:：]\s*|#|编号\s*)(\d+)", instruction, re.I)
    }
    for step in plan.get("steps", []):
        if step.get("type") not in ("navigate", "observe"):
            continue
        chosen = object_by_id.get(step.get("target_ann_id"))
        if not chosen:
            continue
        category_zh = str(chosen.get("category_zh") or "").strip().lower()
        category = str(chosen.get("category") or "").strip().lower()
        siblings = [
            obj for obj in objects
            if (category_zh and str(obj.get("category_zh") or "").strip().lower() == category_zh)
            or (category and str(obj.get("category") or "").strip().lower() == category)
        ]
        if len(siblings) < 2 or explicit_ids.intersection(obj["ann_id"] for obj in siblings):
            continue
        aliases = [value for value in (category_zh, category) if value]
        if not any(alias in instruction.lower() for alias in aliases):
            continue
        described = []
        for obj in siblings:
            fields = [str(obj.get("position") or ""), str(obj.get("func_desc") or "")]
            if obj.get("marker") is not None:
                fields.append(f"marker={obj['marker']}")
            if any(len(field) >= 2 and field in instruction for field in fields):
                described.append(obj["ann_id"])
        if len(set(described)) == 1:
            continue
        name = chosen.get("category_zh") or chosen.get("category") or "目标"
        return {
            "question": f"地图中有 {len(siblings)} 个{name}，请选择要前往哪一个。",
            "candidate_ann_ids": [obj["ann_id"] for obj in siblings],
        }
    return None


OBSERVATION_INTENT_WORDS = (
    "观察", "查看", "看一下", "看一看", "检查", "拍照", "现场情况", "现场环境", "汇报",
)


def _ensure_explicit_observation_steps(plan: dict, instruction: str) -> bool:
    """把模型遗漏的到点观察补成显式步骤，保证任务卡片与真实执行一致。"""
    if not any(word in str(instruction) for word in OBSERVATION_INTENT_WORDS):
        return False
    steps = list(plan.get("steps") or [])
    navigate_indexes = [index for index, step in enumerate(steps) if step.get("type") == "navigate"]
    if not navigate_indexes or any(step.get("type") == "observe" for step in steps):
        return False

    observe_all = any(word in str(instruction) for word in ("分别观察", "逐个观察", "依次观察", "每个"))
    target_indexes = set(navigate_indexes if observe_all else navigate_indexes[-1:])
    repaired = []
    for index, step in enumerate(steps):
        repaired.append(step)
        if index not in target_indexes:
            continue
        ann_id = step.get("target_ann_id")
        repaired.append({
            "type": "observe",
            "target_ann_id": ann_id,
            "focus": "目标当前状态及周围环境",
            "question": str(instruction),
            "fine_adjust": False,
            "capture_only": True,
        })
    plan["steps"] = repaired
    return True


class TaskSafetyError(RuntimeError):
    """An expected execution blocker, not an internal HTTP server failure."""


def _estop_flag(value):
    """Normalize documented boolean encodings; missing/invalid is not safe."""
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "0", "1"):
        return value.strip().lower() in ("true", "1")
    return None
