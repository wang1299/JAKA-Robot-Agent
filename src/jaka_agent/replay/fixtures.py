"""Explicit synthetic fixtures, never presented as model or robot measurements."""
import io
import json

CASES = {
    "find_object": {"name": "参考图寻物", "question": "请帮我寻找参考图中的红色杯子。",
                    "skill": "find_object", "reference": "object",
                    "description": "依次检查候选桌面；观察结果决定继续搜索或结束。"},
    "welcome": {"name": "迎宾与人工确认", "question": "请接一下参考图中的访客，并带到送客点。",
                "skill": "welcome", "reference": "person",
                "description": "到接人点等待，展示候选人物；主人确认照片后才继续引导。"},
    "blocked": {"name": "导航失败反馈", "question": "去会议桌查看桌面上的物品。",
                "skill": "inspect_location", "reference": None,
                "description": "导航返回不可达，执行器停止；Agent 根据失败状态汇报。"},
    "missing_reference": {"name": "缺少参考图", "question": "请寻找我还没有上传照片的物品。",
                          "skill": "find_object", "reference": None,
                          "description": "资源检查发现缺少照片，Agent 请求补充，机器人不移动。"},
}
FEEDBACK_QUESTION = "请根据机器人实际任务状态汇报这次执行结果。"
GRAPH = {"name": "replay-scene.json", "objects": [
    {"ann_id": 11, "category": "Table", "category_zh": "窗边桌", "semantic_name": "窗边桌",
     "position": "演示区域西侧", "floor_xy": [2.0, 1.0]},
    {"ann_id": 22, "category": "Table", "category_zh": "会议桌", "semantic_name": "会议桌",
     "position": "演示区域东侧", "floor_xy": [5.0, 1.0]},
    {"ann_id": 31, "category": "Door", "category_zh": "接人点", "semantic_name": "接人点",
     "position": "演示区域入口", "floor_xy": [1.0, 4.0], "skill_defaults": {"welcome": "pickup_ann_id"}},
    {"ann_id": 32, "category": "Door", "category_zh": "送客点", "semantic_name": "送客点",
     "position": "演示区域出口", "floor_xy": [6.0, 4.0], "skill_defaults": {"welcome": "return_ann_id"}},
]}


def picture(kind="object", target=22, reference=False):
    """Small geometric fixtures with an embedded synthetic-source label."""
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (480, 320), "#e8edf1")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), "SYNTHETIC REPLAY / NOT A CAMERA PHOTO", fill="#304252")
    if not reference:
        draw.rectangle((0, 220, 480, 320), fill="#c9d4de")
        draw.rectangle((75, 177, 405, 199), fill="#987557")
        draw.rectangle((95, 199, 110, 285), fill="#76543a")
        draw.rectangle((370, 199, 385, 285), fill="#76543a")
    if kind == "person":
        draw.ellipse((208, 65, 262, 119), fill="#d2a77f")
        draw.rounded_rectangle((190, 119, 280, 228), radius=16, fill="#3862a7")
        draw.rectangle((200, 225, 225, 290), fill="#334155")
        draw.rectangle((246, 225, 272, 290), fill="#334155")
    else:
        color = "#d94444" if reference or target == 22 else "#477db2"
        draw.rounded_rectangle((215, 105, 273, 177), radius=8, fill=color)
        draw.arc((256, 115, 294, 160), -90, 90, fill=color, width=8)
        draw.text((16, 294), "REFERENCE" if reference else f"OBSERVATION AT POINT {target}", fill="#304252")
    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=88)
    return output.getvalue()


def call(tool_name, **arguments):
    return json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False)


def scripted_completion(messages, tools):
    """Replay fixed decisions for exact sample requests; never classify free text."""
    names = {tool["function"]["name"] for tool in tools}
    if names == {"set_scene_requirement"}:
        return call("set_scene_requirement", scope="none", reason="预设案例使用任务到点观察或状态记录")
    # Tool responses use role=tool; the last user message is the actual request.
    question = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    results = []
    for message in messages:
        if message["role"] == "tool":
            text = message["content"].split("：", 1)[-1]
            results.append(json.loads(text))
    if question == FEEDBACK_QUESTION:
        return (call("get_robot_status") if not results else
                call("finish_response", intent="task_status", text="依据任务执行记录汇报。"))
    case = next((value for value in CASES.values() if value["question"] == question), None)
    if case is None:
        return call("finish_response", intent="answer", text="离线回放支持页面中的固定案例；自由对话可启用模型决策模式。")
    if not results:
        return call("get_skill_context", skill_id=case["skill"])
    if case["reference"] is None and case["skill"] == "find_object":
        return call("ask_user", goal=question, question="请先上传要寻找物品的参考照片。",
                    skill_id="find_object", missing_inputs=["reference"])
    if case["skill"] == "find_object":
        return call("plan_find_object", instruction=question)
    if case["skill"] == "welcome":
        defaults = results[0].get("default_arguments", {})
        return call("plan_welcome", instruction=question, **defaults)
    if len(results) == 1:
        return call("resolve_map_target", name="会议桌")
    targets = results[-1].get("target_ids", [])
    if len(targets) != 1:
        return call("ask_user", goal=question, question="请明确要查看的会议桌。",
                    skill_id="inspect_location", missing_inputs=[])
    return call("plan_inspect_location", instruction=question, target_ann_id=int(targets[0]),
                question="桌面上有哪些物品？", return_to_start=False)
