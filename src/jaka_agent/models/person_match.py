"""Validate person appearance comparisons and presence evidence."""
from __future__ import annotations
import jaka_agent.agent.routing as agent_routing
import json
import re

PERSON_MATCH_WEIGHTS = {
    "upper_clothing": 0.40,
    "face_hair": 0.30,
    "accessories": 0.10,
    "body_shape": 0.20,
}


def _parse_welcome_presence_result(raw: str) -> dict:
    """The independent scene-only gate fails closed on malformed/ambiguous output."""
    value = agent_routing._parse_model_json_object(raw)
    count = value.get("person_count") if isinstance(value, dict) else None
    evidence = str(value.get("visible_evidence") or "").strip() if isinstance(value, dict) else ""
    visible = bool(
        isinstance(value, dict)
        and value.get("person_visible") is True
        and type(count) is int and count >= 1
        and evidence and evidence not in {"实际可见的人体部位和衣着；无人则为空", "有人", "人"}
    )
    return {"person_visible": visible, "person_count": count if type(count) is int else 0,
            "visible_evidence": evidence[:200], "schema_valid": isinstance(value, dict)}


def _parse_person_reference_result(raw: str) -> dict:
    """从逐项外观比对计算人物置信度；不信任模型自报的 found/confidence。"""
    text = agent_routing._json_text(raw).strip()
    value = None
    candidates = [text]
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])
    for candidate in candidates:
        repaired = re.sub(r'"\s*:\s*=\s*', '":', candidate)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            parsed = json.loads(repaired)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    if value is None:
        value = agent_routing._parse_model_json_object(text)
    fallback = {
        "found": False,
        "candidate_visible": False,
        "candidate_region": "",
        "confidence": "low",
        "score": 0.0,
        "reason": "模型未返回完整的人物逐项比对，不能确认目标人物。",
        "comparisons": {},
        "contradictions": [],
        "schema_valid": False,
    }
    if not isinstance(value, dict):
        return fallback

    candidate_visible = value.get("candidate_visible")
    if isinstance(candidate_visible, str):
        normalized_visible = candidate_visible.strip().lower()
        if normalized_visible in {"true", "1", "yes", "是", "可见", "存在"}:
            candidate_visible = True
        elif normalized_visible in {"false", "0", "no", "否", "不可见", "不存在"}:
            candidate_visible = False
    region = str(value.get("candidate_region") or "").strip()[:300]
    reference_features = value.get("reference_features")
    candidate_features = value.get("candidate_features")
    comparisons = value.get("comparisons")
    contradictions_raw = value.get("contradictions")
    if not isinstance(reference_features, dict):
        reference_features = {
            key: value.get(f"reference_{key}")
            for key in PERSON_MATCH_WEIGHTS
        }
    if not isinstance(candidate_features, dict):
        candidate_features = {
            key: value.get(f"candidate_{key}")
            for key in PERSON_MATCH_WEIGHTS
        }
    if not isinstance(comparisons, dict):
        comparisons = {
            key: value.get(key, value.get(f"{key}_comparison"))
            for key in PERSON_MATCH_WEIGHTS
        }
    if isinstance(contradictions_raw, str):
        contradiction_text = contradictions_raw.strip()
        contradictions_raw = (
            []
            if not contradiction_text
            or contradiction_text.lower() in {"无", "没有", "none", "no"}
            else [contradiction_text]
        )
    if (
        not isinstance(candidate_visible, bool)
        or not isinstance(reference_features, dict)
        or not isinstance(candidate_features, dict)
        or not isinstance(comparisons, dict)
        or not isinstance(contradictions_raw, list)
    ):
        fallback["candidate_visible"] = candidate_visible is True
        fallback["candidate_region"] = region
        return fallback

    state_aliases = {
        "match": "match",
        "匹配": "match",
        "一致": "match",
        "same": "match",
        "mismatch": "mismatch",
        "不匹配": "mismatch",
        "冲突": "mismatch",
        "different": "mismatch",
        "unknown": "unknown",
        "未知": "unknown",
        "不可见": "unknown",
        "无法判断": "unknown",
        "unclear": "unknown",
    }
    normalized = {}
    descriptions_complete = True
    unknown_markers = (
        "无法核对", "无法看到", "无法判断", "看不清", "不可见", "不清楚", "未说明",
        "unknown", "unclear", "not visible",
    )
    mismatch_markers = ("不一致", "明显不同", "存在冲突", "mismatch")
    for key in PERSON_MATCH_WEIGHTS:
        raw_state = str(comparisons.get(key) or "").strip().lower()
        normalized[key] = state_aliases.get(raw_state, "")
        reference_description = str(reference_features.get(key) or "").strip()
        candidate_description = str(candidate_features.get(key) or "").strip()
        combined_description = (
            f"{reference_description} {candidate_description}"
        ).lower()
        if any(marker in combined_description for marker in unknown_markers):
            normalized[key] = "unknown"
        elif any(marker in combined_description for marker in mismatch_markers):
            normalized[key] = "mismatch"
        if not normalized[key]:
            descriptions_complete = False
        if not reference_description:
            descriptions_complete = False
        if not candidate_description:
            descriptions_complete = False

    if not descriptions_complete:
        fallback["candidate_visible"] = candidate_visible
        fallback["candidate_region"] = region
        fallback["comparisons"] = normalized
        return fallback

    # Some local-model responses copy the requested JSON example verbatim. Those
    # labels are not visual observations and must never produce a 1.0 match.
    if region in {"候选位置或无", "无"} or any(
        str(reference_features.get(key) or "").strip() == f"参考图{label}"
        or str(candidate_features.get(key) or "").strip() == f"现场{label}"
        for key, label in (("upper_clothing", "衣着"), ("face_hair", "头脸发型"),
                           ("accessories", "配饰"), ("body_shape", "体型"))
    ):
        fallback["reason"] = "模型返回了 JSON 模板占位文字，缺少真实画面证据。"
        fallback["candidate_visible"] = candidate_visible
        fallback["candidate_region"] = region
        fallback["comparisons"] = normalized
        return fallback

    score = sum(
        PERSON_MATCH_WEIGHTS[key]
        for key, state in normalized.items()
        if state == "match"
    )
    observable_weight = sum(
        PERSON_MATCH_WEIGHTS[key]
        for key, state in normalized.items()
        if state != "unknown"
    )
    mismatch_weight = sum(
        PERSON_MATCH_WEIGHTS[key]
        for key, state in normalized.items()
        if state == "mismatch"
    )
    contradictions = [
        str(item).strip()[:160]
        for item in contradictions_raw[:8]
        if str(item).strip()
    ]

    found = bool(
        candidate_visible
        and score >= 0.80
        and observable_weight >= 0.80
        and mismatch_weight == 0.0
        and not contradictions
    )
    if found:
        confidence = "high"
    elif (
        candidate_visible
        and score >= 0.55
        and mismatch_weight <= 0.20
        and not contradictions
    ):
        confidence = "medium"
    else:
        confidence = "low"

    reason = str(value.get("reason") or "").strip()[:500]
    if not reason:
        reason = (
            f"程序计分 {score:.2f}，可核对权重 {observable_weight:.2f}，"
            f"冲突权重 {mismatch_weight:.2f}。"
        )
    return {
        "found": found,
        "candidate_visible": candidate_visible,
        "candidate_region": region,
        "confidence": confidence,
        "score": round(score, 2),
        "observable_weight": round(observable_weight, 2),
        "mismatch_weight": round(mismatch_weight, 2),
        "reason": reason,
        "reference_features": reference_features,
        "candidate_features": candidate_features,
        "comparisons": normalized,
        "contradictions": contradictions,
        "schema_valid": True,
    }
