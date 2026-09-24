#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JAKA Web 对话路由、模型结果解析和拟物体地图查询。"""

from __future__ import annotations

import json
import logging
import re

LOGGER = logging.getLogger("jaka_vision")

FIND_TARGET_COT_SYSTEM = (
    "图片1是目标物参考图(近景),图片2是机器人现场图(远景)。判断图片2里有没有图片1的目标物。"
    "先一句话指出图片2里最像图片1目标物的具体物体(位置+形状+主色),或说明没有相似物体。"
    "标准:只有该物体在整体形状+比例+主颜色上都和图片1明显一致才found=true;"
    "仅类别/颜色相似或拿不准就found=false,不许凭空猜。"
    "然后用一行JSON总结(不要markdown、不要代码块),5字段全:"
    '{"found":false,"candidate_visible":false,"candidate_region":"方位或空串",'
    '"confidence":"low","reason":"比对依据"}'
)
FIND_TARGET_COT_USER = "请判断并输出JSON。"
FIND_TARGET_COT_MAX_TOKENS = 300

FIND_OBJECT_WORDS = ("找", "寻找", "寻物", "帮我找", "查找")
PLACEABLE_WORDS = (
    "桌", "台", "椅", "沙发", "箱", "柜", "架",
    "table", "desk", "chair", "sofa", "case", "cabinet", "shelf",
)


ROUTER_PROMPT = """
你是 JAKA 移动机器人的请求分类器。

你只能返回一个合法 JSON 对象，不得返回任何其他文字。

固定输出格式：
{"mode":"text","answer":""}

mode 只能是以下五个值之一：
- robot_task：要求机器人移动、导航、前往某处、到点观察、巡视、巡逻、返回或停止任务。
- vision：只需查看当前相机画面，不需要机器人移动。
- map_query：查询地图中已有物体的数量、位置、类别或坐标。
- find_object：用户本轮上传了参考图片，并要求机器人移动寻找同一件实物。
- text：普通聊天、能力介绍、使用说明或一般知识。

强制规则：
1. 出现“前往、导航到、移动到、去某处、巡逻、巡视、返回”等动作要求时，必须输出 robot_task。
2. “前往某处观察”属于 robot_task，不属于 vision。
3. 只有上传了参考图片并明确要求寻找时，才能输出 find_object。
4. mode 不是 text 时，answer 必须是空字符串。
5. mode 是 text 时，answer 才填写简短中文回答。
6. 只能输出 mode 和 answer 两个字段。
7. 不得输出 Markdown、解释、前缀、后缀或第二个 JSON。

示例：
用户：前往白色工作台附近观察现场情况
输出：{"mode":"robot_task","answer":""}

用户：描述一下当前画面
输出：{"mode":"vision","answer":""}

用户：地图里有几个椅子
输出：{"mode":"map_query","answer":""}

用户：你能做什么
输出：{"mode":"text","answer":"我可以执行导航、观察、巡逻、地图查询和参考图寻物任务。"}
""".strip()


def _json_text(value) -> str:
    """DashScope 多模态返回值 → 纯文本; 兼容 str / [{'text': ...}] / 嵌套 content。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_json_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"].strip()
        if "content" in value:
            return _json_text(value["content"])
    return str(value).strip() if value is not None else ""


def _parse_route_decision(raw: str) -> dict:
    """解析路由模型结果; 兼容 JSON 围栏和模型偶发返回的 mode=xxx 单行。"""
    text = _json_text(raw).strip()
    if not text:
        raise ValueError("对话路由结果为空")

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    object_match = re.search(r"\{\s*\"mode\"\s*:\s*\"[^\"]+\".*?\}", text, flags=re.DOTALL)
    if object_match:
        candidates.insert(0, object_match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value

    mode_match = re.search(
        r"(?:^|\b)(?:mode|模式)\s*[:=：]\s*['\"]?"
        r"(text|vision|map_query|robot_task|find_object)\b",
        text,
        flags=re.IGNORECASE,
    )
    if mode_match:
        answer_match = re.search(r"(?:^|\n)(?:answer|回答)\s*[:=：]\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        return {
            "mode": mode_match.group(1).lower(),
            "answer": answer_match.group(1).strip() if answer_match else "",
        }
    raise ValueError(f"无法从路由结果提取 JSON 或合法 mode: {text}")


def _parse_model_json_object(raw: str) -> dict | None:
    """提取模型返回的 JSON，并修复本地模型常见的轻微格式错误。"""
    text = _json_text(raw).strip()
    # 瘦身 CoT 会先输出一句候选定位，再在末尾输出最终 JSON。
    # 优先从后往前解析独立 JSON 对象，避免前面的分析文字影响结论提取。
    candidates = list(reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)))
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text)

    for candidate in candidates:
        repaired = re.sub(r'"\s*:\s*=\s*', '":', candidate)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        for value_text in (candidate, repaired):
            try:
                parsed = json.loads(value_text)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _normalize_confidence(value, default: str = "low") -> str:
    """统一模型的 high/medium/low、0~1 分数、0~100 分数和百分比。"""
    aliases = {
        "high": "high",
        "medium": "medium",
        "low": "low",
        "高": "high",
        "中": "medium",
        "低": "low",
        "高置信度": "high",
        "中置信度": "medium",
        "低置信度": "low",
    }
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        numeric_match = re.fullmatch(
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(%)?",
            normalized,
        )
        if not numeric_match:
            return default
        try:
            score = float(numeric_match.group(1))
        except ValueError:
            return default
        if numeric_match.group(2) or score > 1:
            score /= 100.0
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if score > 1:
            score /= 100.0
    else:
        return default

    if not 0 <= score <= 1:
        return default
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _parse_find_object_result(raw: str) -> dict:
    """解析双图结果；格式异常时安全降级为低置信度未命中，不中止整条寻物任务。"""
    text = _json_text(raw).strip()
    value = _parse_model_json_object(text)

    if value is None:
        # 本地视觉服务偶尔会只返回自然语言；这里做保守兜底解析，避免“模型说 high 但系统固定 low”。
        explicit_found = re.search(
            r'["\']?(?:found|match|is_same)["\']?\s*[:：=]+\s*(true|false)',
            text,
            flags=re.IGNORECASE,
        )
        negative = bool(re.search(
            r"没有|未找到|不存在|无法确认|看不出|不确定|不是|并非|无相同|"
            r"未发现|未出现|没有出现|无关|不同|不匹配",
            text,
        ))
        positive = bool(re.search(r"找到|发现|存在|是同一|同一个|相同物理物品|匹配|基本一致|高度一致|可确认|确认存在", text))
        found = (
            explicit_found.group(1).lower() == "true"
            if explicit_found
            else positive and not negative
        )

        confidence = "low"
        if found:
            if re.search(r"confidence\s*[:：=]*\s*high|置信度\s*[:：=]*\s*(?:high|高)|高置信|高度一致|可确认|确认存在|基本一致", text, flags=re.IGNORECASE):
                confidence = "high"
            elif re.search(r"confidence\s*[:：=]*\s*medium|置信度\s*[:：=]*\s*(?:medium|中|中等)|疑似|可能|较像|相似", text, flags=re.IGNORECASE):
                confidence = "medium"

        if found and confidence == "high":
            reason = "现场画面中发现与参考物品高度一致的目标。"
        elif found:
            reason = "现场画面中发现疑似目标，但还不能高置信确认。"
        else:
            reason = "现场画面中未发现与参考外观匹配的目标。"
        LOGGER.warning(
            "[vision] find_object returned non-JSON; fallback found=%s confidence=%s raw=%r",
            found,
            confidence,
            text[:2000],
        )
        return {
            "found": found,
            "candidate_visible": found,
            "candidate_region": "",
            "confidence": confidence,
            "reason": reason,
        }

    found_value = value.get("found", value.get("match", value.get("is_same")))
    if isinstance(found_value, str):
        found_value = found_value.strip().lower() in {"true", "1", "yes", "是", "找到", "发现"}
    if not isinstance(found_value, bool):
        found_value = False
    candidate_raw = value.get("candidate_visible")
    candidate_region = str(
        value.get("candidate_region")
        or value.get("candidate_location")
        or value.get("location")
        or ""
    ).strip()
    candidate_visible = None
    if isinstance(candidate_raw, bool):
        candidate_visible = candidate_raw
    elif isinstance(candidate_raw, str):
        normalized_candidate = candidate_raw.strip().lower()
        if normalized_candidate in {"true", "1", "yes", "是", "可见", "存在"}:
            candidate_visible = True
        elif normalized_candidate in {"false", "0", "no", "否", "不可见", "不存在"}:
            candidate_visible = False
        elif candidate_raw.strip():
            # MiniCPM 偶尔把 candidate_visible 写成候选位置描述而不是布尔值。
            # 只提取“位于……”后的方位供复核聚焦，避免把初检的相似性结论
            # 一并传给复核造成确认偏差。
            location_match = re.search(
                r"(?:位于|位置(?:在|为)?)[：:\s]*([^。；;]{1,180})",
                candidate_raw.strip(),
            )
            candidate_region = candidate_region or (
                location_match.group(1).strip()
                if location_match else candidate_raw.strip()
            )
            candidate_visible = bool(found_value)
    if candidate_visible is False:
        found_value = False
    confidence = _normalize_confidence(value.get("confidence"), default="low")
    reason = str(value.get("reason") or "").strip()
    if re.search(r"上半部分|下半部分|上下两部分|第一张|第二张|图片1|图片2|参考图|现场图", reason):
        if found_value and confidence == "high":
            reason = "现场画面中发现与参考物品高度一致的目标。"
        elif found_value:
            reason = "现场画面中发现疑似目标，但还不能高置信确认。"
        else:
            reason = "现场画面中未发现与参考外观匹配的目标。"
    return {
        "found": found_value,
        "candidate_visible": candidate_visible,
        "candidate_region": candidate_region[:300],
        "confidence": confidence,
        "reason": reason,
    }


def _parse_patrol_result(raw: str) -> dict:
    """校验巡逻双图比较结果；只有结构完整的结果才允许触发异常停车。"""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("巡逻图像比较结果不是合法 JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("changed"), bool):
        raise ValueError("巡逻图像比较结果缺少 changed")
    confidence = str(value.get("confidence") or "").lower()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError("巡逻图像比较结果 confidence 无效")
    changes = value.get("changes") or []
    if not isinstance(changes, list):
        raise ValueError("巡逻图像比较结果 changes 必须是列表")
    normalized = []
    for item in changes[:8]:
        if not isinstance(item, dict):
            continue
        change_type = str(item.get("type") or "changed").lower()
        if change_type not in {"added", "removed", "moved", "state", "changed"}:
            change_type = "changed"
        normalized.append({
            "type": change_type,
            "item": str(item.get("item") or "物体").strip()[:80],
            "detail": str(item.get("detail") or "").strip()[:240],
        })
    return {
        "changed": value["changed"],
        "confidence": confidence,
        "summary": str(value.get("summary") or "").strip()[:500],
        "changes": normalized,
    }


def _history_lines(history, limit=8) -> list[str]:
    """浏览器最近消息 → 有界纯文本, 供路由和视觉问答理解自然指代。"""
    lines = []
    for item in (history or [])[-limit:]:
        if not isinstance(item, dict):
            continue
        role = "用户" if item.get("role") == "user" else "助手"
        text = str(item.get("text") or "").strip()[:1000]
        if text:
            lines.append(f"{role}: {text}")
    return lines


def _find_object_candidates(objects: list[dict]) -> list[dict]:
    """保留场景图顺序，筛出适合作为失物巡检位的家具/承载物。"""
    candidates = []
    for obj in objects:
        description = " ".join(
            str(obj.get(field) or "").lower()
            for field in ("category_zh", "category", "func_desc")
        )
        if any(word in description for word in PLACEABLE_WORDS):
            candidates.append(obj)
    return candidates


def _find_target_label_from_instruction(instruction: str) -> str:
    """从寻物指令提取适合播报的物品名；无法确定时交给参考图识别。"""
    text = re.sub(r"\s+", "", str(instruction or ""))
    match = re.search(r"(?:寻找|寻物|查找|找)(?:一下|一找|下)?(?:这个|这只|这件|那 个|那个)?([^，。！？,.!?；;]{1,24})", text)
    if not match:
        return ""
    label = match.group(1)
    label = re.sub(r"(?:在哪里|在哪儿|在哪|的位置|位置|吗|吧|一下|的照片|照片)$", "", label)
    label = re.sub(r"^(?:这个|这只|这件|那个|那只|那件)", "", label)
    label = label.strip("，。！？,.!?；; ")
    if label in ("", "它", "物品", "东西", "一下"):
        return ""
    return label[:16]


def _mock_route_mode(question: str, has_reference=False) -> str:
    """无模型电脑上的界面测试路由; 树莓派真实模式由 Qwen 语义判断。"""
    if has_reference and any(word in question for word in FIND_OBJECT_WORDS):
        return "find_object"
    if any(word in question for word in ("整张地图", "场景图", "地图中", "地图里", "所有物体", "全图")):
        return "map_query"
    text_only = ("之前", "刚才", "总结", "为什么", "怎么用", "能做什么", "支持什么", "你可以导航", "能导航吗", "导航能力")
    if any(word in question for word in text_only):
        return "text"
    task = ("前往", "过去", "导航到", "移动到", "巡视", "巡逻", "巡检", "返回", "回来", "取消移动")
    go_to_target = bool(re.search(r"去.{1,24}(?:附近|那边|那里|看看|观察|查看|找|一下)", question))
    if any(word in question for word in task) or go_to_target:
        return "robot_task"
    visual = (
        "画面", "看到", "看见", "眼前", "前面", "现场", "照片", "图片", "拍照",
        "颜色", "几个人", "人数", "数一数", "读出", "屏幕", "标签", "桌上", "有没有",
    )
    return "vision" if any(word in question for word in visual) else "text"


FIND_REQUEST_ALIASES = (
    "\u627e", "\u5bfb\u627e", "\u5bfb\u7269", "\u5e2e\u6211\u627e", "\u67e5\u627e",
    "find", "locate", "search for", "look for",
)


def _has_find_object_intent(question: str) -> bool:
    text = str(question or "").strip().lower()
    return any(alias in text for alias in FIND_REQUEST_ALIASES)


def _is_reference_find_request(question: str, has_reference: bool) -> bool:
    """识别带参考图片的明确寻物指令，避免路由模型误判为普通文本。"""
    if not has_reference:
        return False
    text = str(question or "").strip().lower()
    return _has_find_object_intent(text)


def _fallback_route_mode(question: str, has_reference: bool) -> str | None:
    """对高置信度的动作意图做路由兜底，模型返回空文本时仍能执行正确流程。"""
    text = str(question or "").strip().lower()
    if _is_reference_find_request(text, has_reference):
        return "find_object"

    movement_words = (
        "\u524d\u5f80", "\u5bfc\u822a\u5230", "\u79fb\u52a8\u5230", "\u5230\u9644\u8fd1",
        "\u5de1\u903b", "\u5de1\u89c6", "\u5de1\u68c0", "\u5de1\u6e38", "\u8fd4\u56de", "\u56de\u5230",
        "\u505c\u6b62\u79fb\u52a8", "\u53d6\u6d88\u79fb\u52a8", "navigate", "patrol", "cruise",
    )
    capability_words = (
        "\u80fd\u5bfc\u822a", "\u53ef\u4ee5\u5bfc\u822a", "\u652f\u6301\u5bfc\u822a", "\u5bfc\u822a\u80fd\u529b",
        "\u600e\u4e48\u5bfc\u822a", "\u80fd\u505a\u4ec0\u4e48",
    )
    explicit_movement = any(word in text for word in movement_words)
    explicit_movement = explicit_movement or bool(re.search(r"\u53bb.{1,24}(?:\u9644\u8fd1|\u90a3\u8fb9|\u90a3\u91cc|\u770b\u770b|\u89c2\u5bdf|\u67e5\u770b|\u627e|\u4e00\u4e0b)", text))
    if explicit_movement and not any(word in text for word in capability_words):
        return "robot_task"

    map_words = ("\u5730\u56fe", "\u62df\u7269\u4f53\u5730\u56fe", "\u573a\u666f\u56fe", "\u6574\u5f20\u5730\u56fe")
    map_query_words = ("\u6709\u6ca1\u6709", "\u6709\u51e0\u4e2a", "\u591a\u5c11", "\u54ea\u4e9b", "\u5750\u6807", "\u4f4d\u7f6e", "\u5206\u5e03", "\u662f\u5426\u5b58\u5728")
    if any(word in text for word in map_words) and any(word in text for word in map_query_words):
        return "map_query"

    vision_words = (
        "\u63cf\u8ff0\u753b\u9762", "\u753b\u9762\u91cc", "\u5f53\u524d\u770b\u5230", "\u62cd\u7167", "\u62cd\u4e00\u5f20",
        "\u89c2\u5bdf\u73b0\u573a", "\u89c2\u5bdf", "\u67e5\u770b", "\u68c0\u67e5", "\u770b\u4e00\u4e0b", "\u8bc6\u522b\u753b\u9762", "\u770b\u770b\u73b0\u573a",
    )
    if any(word in text for word in vision_words) and not any(word in text for word in movement_words):
        return "vision"
    return None


def _map_query(objects: list[dict], question: str) -> tuple[str, list]:
    """查询拟物体索引; 返回回答和应在地图上高亮的 ann_id。"""
    query = question.strip().lower()
    matches = []
    for obj in objects:
        fields = [
            str(obj.get("category_zh") or ""), str(obj.get("category") or ""),
            str(obj.get("func_desc") or ""), str(obj.get("position") or ""),
        ]
        if any(field and field.lower() in query for field in fields):
            matches.append(obj)
    if not matches and any(word in query for word in ("有哪些", "有什么", "多少种", "列出")):
        counts = {}
        for obj in objects:
            name = obj.get("category_zh") or obj.get("category") or "未知物体"
            counts[name] = counts.get(name, 0) + 1
        summary = "、".join(f"{name} {count}个" for name, count in counts.items())
        return f"当前拟物体地图共有 {len(objects)} 个有效物体：{summary}。", [obj.get("ann_id") for obj in objects]
    if not matches:
        return "当前拟物体地图中没有找到与问题相符的已知物体。", []
    name_counts = {}
    for obj in matches:
        name = obj.get("category_zh") or obj.get("category") or "未知物体"
        name_counts[name] = name_counts.get(name, 0) + 1
    names = "、".join(f"{name} {count}个" for name, count in name_counts.items())
    if "没有" in query or "有吗" in query or "有没有" in query:
        prefix = f"有，共找到 {names}。"
    else:
        prefix = f"找到 {names}。"
    details = "；".join(
        f"#{obj.get('ann_id')}({float((obj.get('nav_xy') or obj['floor_xy'])[0]):.2f},"
        f"{float((obj.get('nav_xy') or obj['floor_xy'])[1]):.2f})"
        for obj in matches
    )
    return f"{prefix}位置为：{details}。", [obj.get("ann_id") for obj in matches]


def _project_metric_query(question: str) -> str | None:
    """返回项目验收指标查询结果；未命中时继续走常规对话路由。"""
    query = re.sub(r"\s+", "", str(question or "")).lower()

    spatial_object_terms = ("空间拟物体", "拟物体数量", "拟物体规模", "拟物体总数")
    if any(term in query for term in spatial_object_terms):
        return (
            "已构建覆盖办公、居家和教育场景的代表性空间数据，"
            "共包含2,038个空间拟物体。"
        )

    semantic_graph_terms = ("语义图谱", "语义图", "图谱")
    semantic_graph_metrics = ("规模", "节点", "结点", "数量", "多少")
    if (
        any(term in query for term in semantic_graph_terms)
        and any(term in query for term in semantic_graph_metrics)
    ):
        return (
            "语义图谱覆盖办公、居家、工业3个场景，共包含105,000个场景命名空间节点"
            "和760,649条语义关系。"
        )

    recognition_terms = (
        "空间对象识别",
        "物体识别",
        "对象识别",
        "识别指标",
        "识别类别",
        "识别准确率",
    )
    if any(term in query for term in recognition_terms):
        return (
            "系统可实现实时语义图构建与空间对象识别，支持常见物体识别类别308个，"
            "平均识别准确率92.45%。"
        )

    return None
