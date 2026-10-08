"""Robot-owned skill contracts; not Codex skills or keyword intent routing.

The model selects a skill and its parameters. Only the existing task manager
may turn the validated request into a pending task; this module never executes
hardware, imports user-provided code or interprets natural-language commands.
"""
from dataclasses import dataclass
from functools import partial
import copy
import math

from robot_agent import Tool, ToolInputError, validate_arguments


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    purpose: str
    parameters: dict
    procedure: tuple
    completion: str
    failure_policy: str
    requires_reference: bool = False

    def card(self):
        return {"id": self.id, "name": self.name, "purpose": self.purpose,
                "parameters": copy.deepcopy(self.parameters), "procedure": list(self.procedure),
                "completion": self.completion, "failure_policy": self.failure_policy,
                "requires_reference": self.requires_reference, "requires_confirmation": True,
                "version": 1}


INSTRUCTION = {"type": "string", "description": "结合上下文还原的完整需求，不添加用户未授权的行动"}
POINT = {"type": "integer", "minimum": 0, "maximum": 2147483647,
         "description": "必须先从地图工具结果读取目标的ann_id。房号、room_id、门牌数字不是ann_id，不得直接填入；目标有歧义时先询问"}

SKILLS = {skill.id: skill for skill in (
    Skill("navigate", "地点导航", "按顺序到达指定地图地点；回到某个地点也是地点导航，不等于到达后再返回出发点。", {
        "instruction": INSTRUCTION,
        "target_ann_ids": {"type": "array", "minItems": 1, "maxItems": 30, "items": POINT,
                           "description": "只按访问顺序列出用户指定的真实地图ann_id。先查询地图；只有一个目的地时只填一个。任务出发位置由执行器记录，不能猜成某个地图点加入此列表；附加返程仅用return_to_start表达。不确定目的地时先询问。"},
        "return_to_start": {"type": "boolean", "description": "必须明确填写。通常false（到达最后目的地后停留），包括回到/返回某个指定地点。只有用户另外要求到达或办完后再回本次任务出发位置，才填true；单独的‘回去’目标不明时先澄清。"}},
        ("按顺序导航至地图目标", "到达后停留；仅按明确参数追加返回本次任务出发位置"),
        "各目标导航成功；只有return_to_start=true才追加一次返程。",
        "目标缺失或歧义先澄清；导航失败停止；只生成待确认计划，不自动执行。"),
    Skill("patrol", "视觉巡逻", "在多个地图点循环观察并检测物体级环境变化，不是单纯经过多个地点。", {
        "instruction": INSTRUCTION,
        "target_ann_ids": {"type": "array", "minItems": 2, "maxItems": 30, "items": POINT,
                           "description": "按访问顺序排列的不同地图 ann_id 整数数组，至少两个；先查询地图确定点位"},
        "rounds": {"type": "integer", "minimum": -1, "maximum": 100, "default": -1,
                   "description": "基线建立后的比较轮数；-1为持续巡逻，或1至100，不能为0"}},
        ("逐点导航并建立照片基线", "按指定轮数比较同点前后照片", "发现可信变化时停止并汇报"),
        "有限轮次结束，或检测到变化后停下并汇报；持续巡逻没有自动完成时刻。",
        "导航、目标观察或比较失败时停止并报告；不自行扩展巡逻区域。"),
    Skill("welcome", "迎宾接待", "用户要求接送人物时：去接人点等待参考照片中的人物，经外观比对后展示现场照片，主人在网页点击确认后引导到送客点。只去名为接客点或送客点的地点不属于迎宾，应选地点导航。", {
        "instruction": INSTRUCTION, "pickup_ann_id": POINT, "return_ann_id": POINT},
        ("前往接人点", "等待并比对参考照片中的人物", "展示现场照片等待主人网页确认；拒绝后继续等待", "确认后引导到送客点"),
        "主人通过网页确认现场照片且送客点导航成功；外观相似不能单独证明身份，不通过语音自动确认。",
        "两点必须不同且有坐标；缺人物图先请求上传；导航失败停止，检测失败可重试，用户随时可停止等待。", True),
    Skill("find_object", "参考图寻物", "用户要求寻找上传照片中的物品时，根据参考图片，在寻物执行器选出的地图候选点逐点寻找并比对。未知物品位置正是搜索的原因；无需先在地图中匹配该物品，也不要求用户提供物品名称、房间或位置线索。位置线索可选，不是启动规划的前提。仅分析图片、不要求寻找时不要选此技能。", {
        "instruction": INSTRUCTION},
        ("检查参考图片并生成候选路线", "逐点导航和视觉比对", "找到后报告位置，或报告未找到"),
        "达到现有比对标准后报告位置，或候选点检查完毕报告未找到；地图记录不等于实时发现。",
        "缺参考图先请求上传；无有效候选点不启动；失败和停止沿用寻物执行器，不无条件重新开始。", True),
    Skill("inspect_location", "到点查看", "移动到指定地图目标，拍照回答具体问题，可按用户要求返回起点。", {
        "instruction": INSTRUCTION, "target_ann_id": POINT,
        "question": {"type": "string", "description": "到达后需要依据现场照片回答的具体问题"},
        "return_to_start": {"type": "boolean", "default": False, "description": "只有用户要求任务后返回时才设为true"}},
        ("导航到目标", "拍照分析用户的问题", "汇报观察结果，按要求返回"),
        "导航与观察步骤完成，需返回时返回步骤也完成；无法看清必须保留不确定性。",
        "缺目标或目标不唯一先询问；导航失败停止，不用出发点照片冒充目标现场。"),
)}


def skill_catalog():
    return [skill.card() for skill in SKILLS.values()]


def skill_summary():
    """Short model-visible inventory; detailed contracts live in tool schemas."""
    return [{"id": s.id, "name": s.name, "purpose": s.purpose,
             "tool": "plan_" + s.id, "requires_reference": s.requires_reference}
            for s in SKILLS.values()]


def skill_resources(skill_id, graph, reference_available=False, reference_source=None):
    """Map-owned parameter bindings, not intent routing or hardcoded point IDs."""
    if skill_id not in SKILLS:
        raise ToolInputError("不存在这项技能")
    skill = SKILLS[skill_id]
    candidates = {}
    for obj in graph.get("objects", []):
        if obj.get("lifecycle", "active") != "active" or type(obj.get("ann_id")) is not int:
            continue
        bindings = obj.get("skill_defaults")
        if not isinstance(bindings, dict):
            continue
        parameter = bindings.get(skill_id)
        if not isinstance(parameter, str) or skill.parameters.get(parameter) != POINT:
            continue
        xy = obj.get("nav_xy") or obj.get("floor_xy")
        if (not isinstance(xy, (list, tuple)) or len(xy) < 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy[:2])):
            continue
        candidates.setdefault(parameter, []).append({
            "ann_id": obj["ann_id"], "name": obj.get("category_zh") or obj.get("category"),
            "floor_xy": list(xy[:2]), "floor": obj.get("floor"), "marker": obj.get("marker")})
    return {"ok": True, "skill_id": skill_id, "map_name": graph.get("name"),
            "requires_confirmation": True, "requires_reference": skill.requires_reference,
            "reference": {"available": bool(reference_available), "source": reference_source},
            "accepted_inputs": list(skill.parameters) + (["reference"] if skill.requires_reference else []),
            "purpose": skill.purpose,
            "default_candidates": candidates,
            "default_arguments": {name: rows[0]["ann_id"] for name, rows in candidates.items() if len(rows) == 1},
            "ambiguous_parameters": [name for name, rows in candidates.items() if len(rows) > 1],
            "note": "仅查询配置，不创建任务、不移动。默认点仅用于用户未另指定地点时；多候选不能擅选。历史参考图须确认仍属于同一需求。"}


def skill_tools(plan_handler):
    return {"plan_" + s.id: Tool(
        s.name + "：" + s.purpose + "步骤：" + "→".join(s.procedure) +
        "完成标准：" + s.completion + "异常：" + s.failure_policy +
        "本调用只生成待确认任务卡，不执行移动。",
        copy.deepcopy(s.parameters), partial(plan_handler, s.id),
        "正在准备" + s.name + "计划", s.name + "规划", "plan") for s in SKILLS.values()}


def prepare_skill(skill_id, arguments, graph, reference=None):
    """Validate data, not user wording. Return an existing-executor plan if needed."""
    if skill_id not in SKILLS:
        raise ToolInputError("不存在这项技能")
    skill = SKILLS[skill_id]
    args = validate_arguments(Tool("", skill.parameters, lambda: None), arguments)
    if skill.requires_reference and not reference:
        raise ToolInputError("这项技能需要参考图片，请先请用户上传；尚未生成或执行任务。")
    objects = {obj["ann_id"]: obj for obj in graph.get("objects", [])
               if obj.get("lifecycle", "active") == "active" and obj.get("ann_id") is not None}

    def point(ann_id):
        obj = objects.get(ann_id)
        if obj is None:
            raise ToolInputError("目标不在当前活动地图中，请重新查地图或询问用户。")
        xy = obj.get("nav_xy") or obj.get("floor_xy")
        if not isinstance(xy, (list, tuple)) or len(xy) < 2:
            raise ToolInputError("地图目标缺少导航坐标，不能生成移动计划。")
        try:
            if not all(math.isfinite(float(v)) and not isinstance(v, bool) for v in xy[:2]):
                raise ValueError()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ToolInputError("地图目标的导航坐标无效。") from exc

    plan = None
    if skill_id == "navigate":
        ids = args["target_ann_ids"]
        for ann_id in ids:
            point(ann_id)
        names = [str(objects[i].get("category_zh") or objects[i].get("category") or i) for i in ids]
        ending = "完成后返回本次任务出发位置。" if args["return_to_start"] else "到达后停留，不追加返程。"
        plan = {"understanding": "前往" + " → ".join(names) + "；" + ending,
                "need_return": args["return_to_start"],
                "steps": [{"type": "navigate", "target_ann_id": i} for i in ids]}
    elif skill_id == "patrol":
        ids = args["target_ann_ids"]
        if len(set(ids)) != len(ids) or args["rounds"] == 0:
            raise ToolInputError("巡逻点不能重复，轮数只能为-1或正整数。")
        for ann_id in ids:
            point(ann_id)
        plan = {"understanding": args["instruction"], "need_return": False,
                "steps": [{"type": "patrol", "target_ann_ids": ids, "count": args["rounds"]}]}
    elif skill_id == "inspect_location":
        point(args["target_ann_id"])
        plan = {"understanding": args["instruction"], "need_return": args["return_to_start"], "steps": [
            {"type": "navigate", "target_ann_id": args["target_ann_id"]},
            {"type": "observe", "target_ann_id": args["target_ann_id"], "question": args["question"]}]}
    elif skill_id == "welcome":
        if args["pickup_ann_id"] == args["return_ann_id"]:
            raise ToolInputError("接人点和返回点不能相同。")
        point(args["pickup_ann_id"])
        point(args["return_ann_id"])
    return skill, args, plan
