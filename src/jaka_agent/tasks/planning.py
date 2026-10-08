"""Model task planning prompts, response repair and plan validation."""
from __future__ import annotations
import jaka_agent.models.runtime as models_runtime
import jaka_agent.models.settings as models_settings
import json
import re

SYSTEM_PROMPT = """你是 JAKA 室内移动服务机器人(WATER 水滴底盘)的【任务规划大脑】。用户给你一句自然语言任务指令, \
你要结合"物体索引"(拟物体 graph)把它分解成机器人能执行的高层步骤, 输出【严格的 JSON 计划】。

# 机器人能力 (对齐真实底盘 API; 你只在语义层决策, 不要写出具体指令串)
- 单点导航 navigate: 给一个物体, 机器人自主导航+避障过去并停在其附近可达点。坐标用 floor_xy。
  路径规划 / 避障 / 乘梯 / 回充都由底层在导航过程中自动完成, 你不用管。
- 多点巡游 cruise: 只要求依次移动经过多个点时使用，不在每点停留拍照，也不做异常判断。
- 智能巡逻 patrol: 至少包含两个点位。机器人依次到点拍照，首轮建立视觉基线，后续轮次与该点上一张照片比较；发现新增、缺失、移动或状态变化时停止并报告。
- 到点观测 observe: 拍照并用视觉(读屏 / 读标签 / 读标识 / 数物品)回答问题。
  假定已由前序 navigate 到达目标附近; 可 fine_adjust=true 到达后原地微转找正观测角度。
- 返回 return: 任务结束回出发点。
- 取消 cancel: 中止当前移动 (罕见, 通常无需主动用)。
- 完成判定: 由执行器轮询底盘状态判定每步是否到达, 你不必操心等待/重试。

# 物体索引字段 (用户消息里逐条给出)
ann_id(唯一id), 名称(中文 category_zh / 英文 category), func_desc(功能), position(位置描述),
floor_xy=[x,y](地面坐标, 单位m, 部分物体可能没有), viewpoint(可选, {x,y,theta};
  theta: 0=朝东(+X), 1.571≈π/2=朝北(+Y), -1.571≈-π/2=朝南(-Y), 范围[-π,π]),
marker(可选, 机器人地图里已标好的点名; 有它时按点名导航更准)。
用【中文名】/ func_desc / position 匹配指令里提到的物体; 你只按 ann_id 选目标, 不用管坐标还是点名。

# 输出格式 (严格 JSON, 不要任何多余文字或 Markdown 标记)
{
  "understanding": "一句话任务理解",
  "steps": [
    {"type":"navigate","target_ann_id":<int>,"use_viewpoint":<true/false>,"reason":"为什么去"},
    {"type":"cruise","target_ann_ids":[<int>,<int>,...],"count":<int>,"reason":"巡视这些点"},
    {"type":"patrol","target_ann_ids":[<int>,<int>,...],"count":<int>,"reason":"巡逻并检查异常"},
    {"type":"observe","target_ann_id":<int>,"focus":"真正要看清的局部区域","question":"到了要回答什么","fine_adjust":<true/false>},
    {"type":"return"},
    {"type":"cancel"}
  ],
  "need_return": <true/false>
}
JSON 协议必须同时满足:
- 整条响应只能有一个 JSON 对象，首字符必须是 {，尾字符必须是 }。
- 不得输出“规划如下”“输出:”“模式:”、Markdown 代码围栏、注释或 JSON 后的解释。
- understanding、steps、need_return 三个顶层字段必须全部存在，不得增加其他顶层字段。
- steps 必须是 JSON 数组；布尔值只能写 true/false，整数不能加引号，不允许尾随逗号。
规则:
- 典型流程: navigate(或 cruise) 到达 → (需要看/读时加 observe) → (需要往返时才加 return, 否则到了就停下)。
- 单目标用 navigate；只要求依次经过多个目标用 cruise；用户说“巡逻/巡视、发现异常、看看有没有变化”时必须用 patrol。
- 单一目标绝对不能使用 cruise，也不能对同一目标同时生成 navigate 和 cruise。
- patrol 的 target_ann_ids 至少 2 个；首轮会自动建立基线，count 表示基线建立后的比较轮数。用户未限定轮数时填 -1，表示持续巡逻直到用户停止或发现异常。patrol 自带逐点拍照和前后对比，不要再拆成 navigate+observe。
- 用户在指令中明确写出 ann_id=19、#19、＃19 这类 ID 时，必须严格使用这些 ID，并保持用户给出的先后顺序，不得省略、重复或替换。
- use_viewpoint(写在 navigate 上) / fine_adjust(写在 observe 上) 仅在该物体【有 viewpoint】且【需正对观察】(如读屏)时为 true, 否则 false。
- 每个 observe 都必须填写 focus，用简短名词短语描述真正要进入画面的关键区域，不能照抄导航目标。例如：导航到设备机柜时，focus 可以是“压力表表盘、指针、运行指示灯”；导航到桌子时，focus 可以是“桌面及桌面上的遗留物品”。
- observe 的 question 要把【用户会关心的关键信息】一次性问全, 不止一个维度。例:"看电梯状态"应问 当前楼层 + 运行方向(上行/下行/已停靠) + 电梯门开关; "看屏幕/标签"要问清要读的具体内容; "数物品"要问清位置(桌面/地面/柜内)。别只问单点(只问"几楼"会导致答案丢掉方向/门状态)。
- need_return 默认 false，判断完整语义而非关键词：“去X/返回X/回到X”均表示到达指定地点X后停留，不追加返程。“去X，不用回来”也为false。只有用户明确另要求到达或办完后再回本次任务出发位置，才设true并在steps末尾加return。单独说“回去/回起点”但目标无法核实，应澄清，不把当前坐标冒充以前的起点。
- 特别注意：“告诉我现场情况”“汇报现场情况”只是要求把观察结果返回给用户，不是机器人返航；是否增加返程按完整需求判断，不能按某个词是否出现判断。
- 只引用索引里【真实存在】的 ann_id。

# 示例
指令: "去看一下电梯现在停在几楼"
输出:
{"understanding":"去南端电梯门旁的楼层显示屏读取当前楼层。","steps":[{"type":"navigate","target_ann_id":6,"use_viewpoint":true,"reason":"楼层显示屏在电梯门旁墙面"},{"type":"observe","target_ann_id":6,"focus":"楼层显示屏、方向箭头、指示灯","question":"电梯当前停在几楼？正在上行、下行还是已停靠？电梯门是开着还是关着？","fine_adjust":false},{"type":"return"}],"need_return":true}

指令: "去会议桌那边, 看看桌上有没有遗留物品"
输出:
{"understanding":"去南端电梯厅的会议桌查看桌面物品。","steps":[{"type":"navigate","target_ann_id":5,"use_viewpoint":false,"reason":"会议桌在南端电梯厅内"},{"type":"observe","target_ann_id":5,"focus":"桌面及桌面上的物品","question":"桌面上有哪些物品？","fine_adjust":false},{"type":"return"}],"need_return":true}

指令: "绕着电梯厅巡视一圈, 依次经过沙发、茶几和会议桌"
输出:
{"understanding":"在电梯厅内持续巡逻沙发、茶几、会议桌三个点位，并通过前后照片发现环境变化。","steps":[{"type":"patrol","target_ann_ids":[1,2,5],"count":-1,"reason":"循环检查三处点位是否出现新增、缺失、移动或状态变化"}],"need_return":false}

指令: "先去电梯门那边, 再到会议桌那边, 然后回出发点"
输出:
{"understanding":"依次前往电梯门、会议桌, 最后返回出发点。","steps":[{"type":"navigate","target_ann_id":18,"use_viewpoint":false,"reason":"先到电梯门"},{"type":"navigate","target_ann_id":5,"use_viewpoint":false,"reason":"再去会议桌"},{"type":"return"}],"need_return":true}

指令: "去电梯门那边"
输出:
{"understanding":"前往电梯门并停留在该处。","steps":[{"type":"navigate","target_ann_id":18,"use_viewpoint":false,"reason":"前往电梯入口门"}],"need_return":false}
"""


def _zh(o):
    """物体显示名: 优先中文名 category_zh, 退回英文 category, 再退'目标'。"""
    return o.get("category_zh") or o.get("category") or "目标"


def _normalized_ann_id(value):
    """兼容场景图把 ann_id 保存成 20 或字符串 "20" 的两种格式。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _index_digest(objects):
    """把物体索引压成模型只需的字段(去掉 parts/style 等噪声)。"""
    lines = []
    for o in objects:
        vp = o.get("viewpoint")
        vp_s = (f"有 viewpoint(theta={vp['theta']:.3f})" if vp else "无 viewpoint")
        mk = f" | marker={o['marker']}" if o.get("marker") else ""
        lines.append(
            f"- ann_id={o['ann_id']} | {_zh(o)} | 功能: {o.get('func_desc','')} "
            f"| 位置: {o.get('position','')} | floor_xy={o.get('floor_xy')} | {vp_s}{mk}"
        )
    return "\n".join(lines)


_PLAN_JSON_KEYS = (
    "understanding",
    "steps",
    "need_return",
    "type",
    "target_ann_id",
    "target_ann_ids",
    "use_viewpoint",
    "reason",
    "count",
    "focus",
    "question",
    "fine_adjust",
)


def _parse_plan_json(raw_text: str) -> tuple[dict, str, list[str]]:
    """解析规划 JSON；只修复已知字段的明确标点错误，不猜测计划语义。"""
    text = str(raw_text or "").strip()
    if not text:
        raise RuntimeError("模型输出为空，无法生成任务计划")

    # 兼容模型偶发的 Markdown 围栏或前后说明，但仍只解析其中唯一的 JSON 对象。
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"模型输出不包含完整 JSON 对象\n原始输出:\n{text}")
    candidate = cleaned[start:end + 1]

    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"模型计划必须是 JSON 对象，实际为 {type(parsed).__name__}")
        return parsed, candidate, []
    except json.JSONDecodeError as original_error:
        repaired = candidate
        repairs = []

        # MiniCPM 偶尔输出 `"target_ann_ids:[202]`：字段名前有起始引号，
        # 但冒号前漏掉结束引号。只允许修复计划协议中的白名单字段。
        key_pattern = "|".join(re.escape(key) for key in _PLAN_JSON_KEYS)
        repaired, count = re.subn(
            rf'([,{{]\s*)"({key_pattern})\s*:',
            r'\1"\2":',
            repaired,
        )
        if count:
            repairs.append(f"补全{count}个字段名结束引号")

        repaired, count = re.subn(r'"\s*:\s*=\s*', '":', repaired)
        if count:
            repairs.append(f"移除{count}个字段冒号后的多余等号")

        repaired, count = re.subn(r",\s*([}\]])", r"\1", repaired)
        if count:
            repairs.append(f"移除{count}个尾随逗号")

        if not repairs:
            raise RuntimeError(
                f"模型输出不是合法 JSON: {original_error}\n原始输出:\n{text}"
            ) from original_error
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as repaired_error:
            raise RuntimeError(
                f"模型输出不是合法 JSON，受限修复后仍无法解析: {repaired_error}\n"
                f"原始输出:\n{text}\n修复结果:\n{repaired}"
            ) from repaired_error
        if not isinstance(parsed, dict):
            raise RuntimeError(f"模型计划必须是 JSON 对象，实际为 {type(parsed).__name__}")
        return parsed, repaired, repairs


def _normalize_single_target_cruise(plan: dict) -> list[str]:
    """把小模型误生成的单目标 cruise 保守降级，避免非法计划进入执行器。"""
    steps = list(plan.get("steps") or [])
    navigate_ids = {
        _normalized_ann_id(step.get("target_ann_id"))
        for step in steps
        if step.get("type") == "navigate"
    }
    normalized = []
    changes = []
    for index, step in enumerate(steps):
        if step.get("type") != "cruise":
            normalized.append(step)
            continue
        raw_ids = step.get("target_ann_ids")
        if not isinstance(raw_ids, list):
            normalized.append(step)
            continue
        ids = list(dict.fromkeys(_normalized_ann_id(value) for value in raw_ids))
        if len(ids) != 1:
            step["target_ann_ids"] = ids
            normalized.append(step)
            continue
        ann_id = ids[0]
        if ann_id in navigate_ids:
            changes.append(
                f"删除step[{index}]与navigate重复的单目标cruise(ann_id={ann_id})"
            )
            continue
        replacement = {
            "type": "navigate",
            "target_ann_id": ann_id,
            "use_viewpoint": False,
            "reason": str(step.get("reason") or "前往单一目标"),
        }
        normalized.append(replacement)
        navigate_ids.add(ann_id)
        changes.append(
            f"将step[{index}]单目标cruise降级为navigate(ann_id={ann_id})"
        )
    plan["steps"] = normalized
    return changes


def plan_task(instruction: str, objects: list, model: str = models_settings.PLAN_MODEL) -> dict:
    """指令 + 物体索引 → 结构化计划 dict。"""
    user_msg = (
        f"【物体索引】\n{_index_digest(objects)}\n\n"
        f"【任务指令】{instruction}\n\n"
        f"请按指定 JSON 格式输出任务计划。"
    )
    resp = models_runtime._client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        response_format={"type": "json_object"},   # 强制 JSON
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    print(f"[plan-model] raw={text}", flush=True)
    plan, repaired_text, repairs = _parse_plan_json(text)
    if repairs:
        print(
            f"[plan-model] json_repaired={repairs} result={repaired_text}",
            flush=True,
        )
    _apply_explicit_patrol_targets(plan, instruction, objects)
    normalized = _normalize_single_target_cruise(plan)
    if normalized:
        print(f"[plan-normalized] {'; '.join(normalized)}", flush=True)
    _validate_plan(plan, objects)
    _enforce_return_policy(plan, instruction)
    return plan


def _enforce_return_policy(plan: dict, instruction: str):
    """Legacy CLI compatibility: normalize an explicit flag, never infer from words.

    Web Agent navigation uses prepare_skill directly and does not enter here.
    """
    if type(plan.get("need_return")) is not bool:
        raise ValueError("need_return 必须是明确的布尔值")
    plan["steps"] = [step for step in plan.get("steps", []) if step.get("type") != "return"]
    if plan["need_return"]:
        plan["steps"].append({"type": "return"})


def _apply_explicit_patrol_targets(plan: dict, instruction: str, objects: list):
    """用户明确点名多个 ann_id 时覆盖模型转抄结果，避免巡逻目标被省略或重复。"""
    if not any(word in str(instruction).lower() for word in ("巡逻", "巡视", "patrol")):
        return
    explicit_ids = []
    for value in re.findall(r"(?:ann_id\s*[=:：]\s*|[#＃]\s*)(\d+)", str(instruction), re.I):
        ann_id = int(value)
        if ann_id not in explicit_ids:
            explicit_ids.append(ann_id)
    if len(explicit_ids) < 2:
        return
    valid_ids = {_normalized_ann_id(obj["ann_id"]) for obj in objects}
    missing = [ann_id for ann_id in explicit_ids if ann_id not in valid_ids]
    if missing:
        raise RuntimeError(f"巡逻指令指定的 ann_id 不在当前拟物体地图中: {missing}")
    for step in plan.get("steps") or []:
        if step.get("type") in ("patrol", "cruise"):
            step["type"] = "patrol"
            step["target_ann_ids"] = explicit_ids
            step.setdefault("count", -1)
            return
    plan.setdefault("steps", []).insert(0, {
        "type": "patrol",
        "target_ann_ids": explicit_ids,
        "count": -1,
        "reason": "按用户明确指定的多个拟物体地图点位持续巡逻并检查环境变化",
    })


_STEP_TYPES = {"navigate", "cruise", "patrol", "observe", "return", "cancel"}


def _validate_plan(plan: dict, objects: list):
    """轻校验: 必备字段 + ann_id 存在 + 多点步骤至少两个目标。"""
    if not isinstance(plan, dict):
        raise RuntimeError(f"计划必须是 JSON 对象: {plan}")
    if not isinstance(plan.get("understanding"), str):
        raise RuntimeError(f"计划 understanding 必须是字符串: {plan}")
    if not isinstance(plan.get("need_return"), bool):
        raise RuntimeError(f"计划 need_return 必须是布尔值: {plan}")
    valid_ids = {_normalized_ann_id(o["ann_id"]) for o in objects}
    if "steps" not in plan or not isinstance(plan["steps"], list):
        raise RuntimeError(f"计划缺 steps: {plan}")
    for i, s in enumerate(plan["steps"]):
        if not isinstance(s, dict):
            raise RuntimeError(f"step[{i}] 必须是 JSON 对象: {s}")
        t = s.get("type")
        if t not in _STEP_TYPES:
            raise RuntimeError(f"step[{i}] type 非法: {t}")
        if t in ("navigate", "observe"):
            s["target_ann_id"] = _normalized_ann_id(s.get("target_ann_id"))
            if s.get("target_ann_id") not in valid_ids:
                raise RuntimeError(f"step[{i}] 引用了不存在的 ann_id={s.get('target_ann_id')}")
        elif t in ("cruise", "patrol"):
            raw_ids = s.get("target_ann_ids")
            if not isinstance(raw_ids, list):
                raise RuntimeError(f"step[{i}] {t} target_ann_ids 必须是数组")
            ids = [_normalized_ann_id(value) for value in raw_ids]
            s["target_ann_ids"] = ids
            if len(set(ids)) < 2:
                raise RuntimeError(f"step[{i}] {t} 需要 ≥2 个不同的 target_ann_ids")
            bad = [a for a in ids if a not in valid_ids]
            if bad:
                raise RuntimeError(f"step[{i}] {t} 引用了不存在的 ann_id={bad}")
            try:
                count = int(s.get("count", -1 if t == "patrol" else 1))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"step[{i}] {t} count 必须是整数") from exc
            if count == 0 or count < -1:
                raise RuntimeError(f"step[{i}] {t} count 只能是 -1 或正整数")
