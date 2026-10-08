"""Coordinate agent tools, conversation context and pending task responses."""
from __future__ import annotations
import jaka_agent.agent.map_evidence as agent_map_evidence
import jaka_agent.agent.routing as agent_routing
import jaka_agent.agent.runner as agent_runner
import jaka_agent.agent.skills as agent_skills
import jaka_agent.diagnostics as diagnostics
import jaka_agent.web.settings as web_settings
import copy
import json
import math
import os
import re
import time
import uuid

class AgentServiceMixin:
    def agent_complete(self, messages, tools):
        """Adapt native/JSON serving protocols; the runner validates both alike."""
        from jaka_agent.models.settings import PLAN_MODEL
        from jaka_agent.models.runtime import _client

        # Optional decision-model endpoint. Vision continues using VISION_MODEL
        # and the existing endpoint; no implicit cloud fallback or credential reuse.
        agent_base = os.getenv("JAKA_AGENT_BASE_URL", "").strip()
        if agent_base:
            from openai import OpenAI
            client = OpenAI(base_url=agent_base, api_key=os.getenv("JAKA_AGENT_API_KEY", "EMPTY"))
        else:
            client = _client()
        protocol = os.getenv("JAKA_AGENT_PROTOCOL", "native").strip()
        if protocol not in ("native", "json"):
            raise ValueError("JAKA_AGENT_PROTOCOL must be native or json")
        if protocol == "json":
            from jaka_agent.agent.runner import json_transport_messages
            response = client.with_options(timeout=45, max_retries=0).chat.completions.create(
                model=os.getenv("JAKA_AGENT_MODEL", "").strip() or PLAN_MODEL,
                messages=json_transport_messages(messages, tools),
                response_format={"type": "json_object"}, temperature=0, max_tokens=1200,
            )
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                diagnostics.LOGGER.warning("[agent-protocol] JSON response exceeded output limit")
                return "{truncated response"
            # The runner enforces the allowlist, argument types and final intent.
            # Do not silently convert unstructured text into finish_response.
            return agent_routing._json_text(choice.message.content)
        response = client.with_options(timeout=45, max_retries=0).chat.completions.create(
            model=os.getenv("JAKA_AGENT_MODEL", "").strip() or PLAN_MODEL,
            messages=messages, tools=tools,
            tool_choice="required" if agent_base and any(t.get("function", {}).get("name") == "finish_response" for t in tools) else "auto",
            temperature=0, max_tokens=700,
            **({"parallel_tool_calls": False} if agent_base else {}),
        )
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            if len(message.tool_calls) != 1:
                return "<tool_call>invalid parallel calls"
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, ValueError):
                return "{invalid native tool arguments"
            return json.dumps({"tool": call.function.name, "arguments": arguments}, ensure_ascii=False)
        return agent_routing._json_text(message.content)

    def agent_vision(self, path, question, history):
        if self.mock:
            return "模拟画面中有桌椅。此结果仅用于测试，不代表真实现场。"
        from jaka_agent.models.settings import VISION_MODEL
        from jaka_agent.models.vision import VISUAL_EVIDENCE_RULES, format_visual_answer, _image_data_url
        from jaka_agent.models.runtime import _client

        with self.infer_lock:
            response = _client().with_options(timeout=45, max_retries=0).chat.completions.create(
                model=VISION_MODEL, temperature=0, max_tokens=512,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(str(path), VISION_MODEL)}},
                    {"type": "text", "text": VISUAL_EVIDENCE_RULES + "\n只输出给用户看的自然语言，不输出JSON。\n"
                     + "最近对话（只用于理解指代）：\n" + "\n".join(agent_routing._history_lines(history))
                     + "\n问题：" + question},
                ]}],
            )
        return format_visual_answer(agent_routing._json_text(response.choices[0].message.content))

    def conversation_task(self, conversation_id):
        task = self.tasks.snapshot()
        if task and task.get('conversation_id') == conversation_id:
            return task
        # Unowned compatibility tasks are never injected into a named conversation.
        store = getattr(self, 'conversation_store', None)
        return store.latest_task(conversation_id) if store and conversation_id else None

    def plan_for_conversation(self, conversation_id, callback, turn_id=None):
        store = getattr(self, 'conversation_store', None)
        if not store:
            return callback()
        store.conversation(conversation_id)
        with self.tasks.lock:
            prior = self.tasks.snapshot()
            if prior and prior.get('status') in ('planned', 'needs_clarification') and prior.get('conversation_id') != conversation_id:
                raise agent_runner.ToolInputError('机器人有其他会话的待确认任务，请先处理该任务')
            task = callback()
            if prior and prior.get('status') in ('planned', 'needs_clarification'):
                prior.update(status='superseded', finished_at=time.time())
                store.task(prior)
            return self.tasks.bind_conversation(task['id'], conversation_id, turn_id)

    def require_task_owner(self, task_id, conversation_id):
        if not getattr(self, 'conversation_store', None):
            return
        task = self.tasks.snapshot()
        if not task or task.get('id') != task_id or task.get('conversation_id') != conversation_id:
            raise ValueError('任务不属于当前会话，或已失效；请重新打开所属会话')

    def agent_chat(self, payload, emit=None):
        """Bounded, per-conversation evidence memory; no inferred action dispatch."""
        agent_runner.clean_history(payload.get("history", []))
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("问题必须是 1～4000 字符的文字")
        session_id = payload.get("conversation_id")
        store = getattr(self, 'conversation_store', None)
        if store:
            from jaka_agent.storage.memory import valid_id
            valid_id(session_id)
            store.conversation(session_id)
            user_id = valid_id(payload.get('user_message_id') or uuid.uuid4().hex)
            assistant_id = valid_id(payload.get('assistant_message_id') or uuid.uuid4().hex)
        if session_id is not None and (not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id)):
            raise ValueError("会话标识无效")
        if not self.agent_lock.acquire(blocking=False):
            raise RuntimeError("小卡正在处理另一条对话，请稍后再试")
        try:
            sessions = getattr(self, "agent_sessions", {})
            if store:
                store.validate_turn_ids(session_id, user_id, assistant_id)
            self.agent_sessions = sessions
            now = time.time()
            for key in list(sessions):
                if now - sessions[key]["updated_at"] > 1800:
                    del sessions[key]
            initial_history = []
            for item in payload.get("history", [])[-16:]:
                if (isinstance(item, dict) and item.get("role") in ("user", "assistant")
                        and isinstance(item.get("text"), str) and item["text"].strip()):
                    initial_history.append({"role": item["role"], "text": item["text"][:2000],
                        "capture_id": item.get("capture_id") if isinstance(item.get("capture_id"), str)
                        and web_settings.CAPTURE_ID_RE.fullmatch(item["capture_id"]) else None})
            memory = (store.load_memory(session_id) if store else
                      copy.deepcopy(sessions.get(session_id, {"history": initial_history, "evidence": []})))
            memory.setdefault("pending_request", None)
            memory.setdefault("focus", None)
            memory.setdefault("last_scene", None)
            memory.setdefault("planned_task", None)
            memory.setdefault("reference", None)
            selection = payload.get("target_selection")
            if selection is not None:
                offer = memory.get("target_offer") or {}
                if (not isinstance(selection, dict) or type(selection.get("ann_id")) is not int
                        or selection.get("snapshot") != offer.get("snapshot")
                        or selection["ann_id"] not in offer.get("ids", [])
                        or agent_map_evidence.MapEvidence(self.graph_snapshot()).version != offer.get("snapshot")):
                    raise ValueError("目标选项已失效，请重新查询地图并选择")
            memory["confirmed_target"] = selection
            memory["turn_goal"] = question
            if store:
                user_message = {'id': user_id, 'role': 'user', 'text': question, 'createdAt': time.time()*1000}
                reference = payload.get('reference')
                if reference:
                    if not isinstance(reference, dict):
                        raise ValueError('参考图片元信息无效')
                    url = reference.get('image_url', '')
                    if url != f"/references/{reference.get('reference_id')}{reference.get('suffix')}" or not store.owns_media(session_id, url):
                        raise ValueError('参考图片未绑定当前会话，请重新上传')
                    user_message.update(referenceImage=url, referenceStoredImage=url)
                store.start_turn(session_id, user_message, assistant_id)
                store.event(session_id, 'user_message', user_message)
                if getattr(self, 'media_archive', None):
                    self.media_archive.prepare_turn(session_id, user_id, question)
            turn = {**payload, "history": memory["history"]}
            if store:
                turn['user_message_id'] = user_id
            def save_exchange(answer):
                memory["history"] = (memory["history"] + [
                    {"role": "user", "text": question[:2000]},
                    {"role": "assistant", "text": answer["text"][:2000], "capture_id": answer.get("capture_id")}
                ])[-16:]
                memory["last_exchange"] = {"request": question[:2000], "reply": answer["text"][:2000],
                                           "incomplete": bool(answer.get("incomplete"))}
                memory["updated_at"] = time.time()
                memory.pop("turn_goal", None)
                if store:
                    message = {'id': assistant_id, 'role': 'assistant', 'text': answer['text'],
                               'createdAt': time.time()*1000, 'state': 'done',
                               'incomplete': bool(answer.get('incomplete'))}
                    for source, dest in {'capture_id':'captureId', 'image_url':'image', 'image_source':'imageSource',
                            'tool_trace':'toolTrace','execution':'execution','response_kind':'responseKind',
                            'target_choices':'targetChoices','target_choice_snapshot':'targetChoiceSnapshot', 'task':'task'}.items():
                        if source in answer:
                            message[dest] = answer[source]
                    if answer.get('task'):
                        message['taskId'] = answer['task']['id']
                    store.message(session_id, message)
                    store.event(session_id, 'assistant_message', message)
                    store.save_memory(session_id, memory)
                if session_id:
                    if session_id not in sessions and len(sessions) >= 64:
                        del sessions[min(sessions, key=lambda key: sessions[key]["updated_at"])]
                    sessions[session_id] = memory
            try:
                result = self._agent_chat_turn(turn, emit, memory)
            except Exception:
                memory["pending_request"] = {"goal": memory["turn_goal"][:1200], "status": "interrupted"}
                save_exchange({"text": "本轮处理未完成。", "incomplete": True})
                raise
            if result.get("requires_clarification"):
                memory["pending_request"] = result["pending_request"]
            elif result.get("requires_confirmation"):
                task = result["task"]
                # Do not interpret a generated plan as a completed physical task.
                memory["planned_task"] = {"id": task.get("id"), "status": task.get("status"),
                    "goal": memory["turn_goal"][:1200], "recorded_at": time.time(), "skill": task.get("skill")}
                memory["pending_request"] = {"goal": memory["turn_goal"][:1200],
                    "status": "needs_clarification" if task.get("status") == "needs_clarification" else "awaiting_confirmation",
                    "task_id": task.get("id"), "clarification": task.get("clarification")}
            elif result.get("incomplete"):
                memory["pending_request"] = {"goal": memory["turn_goal"][:1200], "status": "interrupted"}
            elif result.get("tool_trace"):
                # Tool failures remain unresolved even when a polite explanation
                # was generated. Successful new queries replace the old focus.
                memory["pending_request"] = (None if result["tool_trace"][-1]["ok"] else
                    {"goal": memory["turn_goal"][:1200], "status": "tool_failed"})
            save_exchange(result)
            return result
        finally:
            self.agent_lock.release()

    def _agent_chat_turn(self, payload, emit, memory):
        question = payload.get("question")
        history = payload.get("history", [])
        session_id = payload.get('conversation_id')
        store = getattr(self, 'conversation_store', None)
        agent_runner.clean_history(history)  # Validate before touching hardware or contacting the model.
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("问题必须是 1～4000 字符的文字")
        reference = payload.get("reference")
        reference_source = "current_turn" if reference is not None else None
        if reference is None and memory.get("reference"):
            saved_reference = memory["reference"]
            if self.reference_path(saved_reference["reference_id"], saved_reference["suffix"]).is_file():
                reference = saved_reference
                reference_source = "previous_turn"
            else:
                memory["reference"] = None
        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError("参考图片元信息无效")
            rid, suffix = reference.get("reference_id"), reference.get("suffix")
            if not isinstance(rid, str) or not isinstance(suffix, str):
                raise ValueError("参考图片元信息无效")
            reference_path = self.reference_path(rid, suffix)
            if not reference_path.exists():
                raise ValueError("参考图片不存在或已过期，请重新上传")
            reference = {"reference_id": rid, "suffix": suffix,
                         "image_url": f"/references/{rid}{suffix}",
                         "content_type": web_settings.REFERENCE_IMAGE_TYPES[suffix][0]}
            memory["reference"] = copy.deepcopy(reference)
        last_scene = memory.get("last_scene") or {}
        previous_capture = last_scene.get("capture_id")
        if not (isinstance(previous_capture, str) and web_settings.CAPTURE_ID_RE.fullmatch(previous_capture)
                and self.capture_path(previous_capture).exists()):
            previous_capture = None
        for turn in reversed(history[-16:]):
            if previous_capture:
                break
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            capture_id = turn.get("capture_id")
            if isinstance(capture_id, str) and web_settings.CAPTURE_ID_RE.fullmatch(capture_id):
                if self.capture_path(capture_id).exists():
                    previous_capture = capture_id
                    break
        turn_graph = self.graph_snapshot()
        map_evidence = agent_map_evidence.MapEvidence(turn_graph)
        # Only this turn's verified exact-name lookup may supersede a broad
        # category query; never infer uniqueness from an arbitrary chosen ID.
        resolved_this_turn = set()
        target_offer_this_turn = {}

        def resolve_map_target(name):
            result = map_evidence.resolve_target(name)
            if result['unique']:
                resolved_this_turn.update(result['target_ids'])
            return result

        def get_skill_context(skill_id):
            result = agent_skills.skill_resources(skill_id, turn_graph, bool(reference), reference_source)
            result["snapshot"] = map_evidence.version
            return result

        new_scene_this_turn = None

        def scene_result(capture_id, path, scene_question, fresh):
            source_note = ("这是本轮原地新拍摄的画面；历史对话中的观察不能作为本图事实。" if fresh
                           else "这是历史照片，只能回答拍摄时可见内容，不能证明当前现场或当前位置。")
            visual_question = (source_note + "\n本轮用户完整需求：" + question
                               + "\n本次图片分析问题：" + scene_question)
            try:
                observation = self.agent_vision(path, visual_question, history)
            except Exception as exc:
                diagnostics.LOGGER.warning("[vision] scene analysis failed type=%s", type(exc).__name__)
                raise agent_runner.ToolInputError("照片已采集，但本次视觉分析失败，请检查模型连接后重试。") from exc
            if fresh and getattr(self, 'media_archive', None):
                self.media_archive.record(f'/captures/{capture_id}.jpg', observation=observation)
            return {"ok": True, "capture_id": capture_id, "image_url": f"/captures/{capture_id}.jpg",
                    "image_source": "现场拍摄" if fresh else "历史照片（不是实时画面）",
                    "source_type": "scene_photo", "captured_this_turn": fresh,
                    "observed_at": path.stat().st_mtime,
                    "observation": observation}

        def inspect_previous_scene(question):
            if not previous_capture or not self.capture_path(previous_capture).exists():
                return {"ok": False, "error": "本会话的历史照片不存在或已过期，没有重新拍照。"}
            path = self.capture_path(previous_capture)
            if new_scene_this_turn is None:
                memory["last_scene"] = {**(memory.get('last_scene') or {}), "capture_id": previous_capture, "observed_at": path.stat().st_mtime}
            return scene_result(previous_capture, path, question, False)

        def observe_scene(question):
            nonlocal new_scene_this_turn
            if self.tasks.is_busy() or self.mapping.is_busy():
                return {"ok": False, "error": "机器人正在执行任务，请查询任务状态或停止任务后再拍照。"}
            if getattr(self, 'media_archive', None) and session_id and payload.get('user_message_id'):
                capture_id, path = self.capture(media_scope={'conversation_id':session_id,'turn_id':payload['user_message_id']},
                                                metadata={'question':question})
            else:
                capture_id, path = self.capture()
            new_scene_this_turn = capture_id
            memory["last_scene"] = {"capture_id": capture_id, "observed_at": path.stat().st_mtime}
            return scene_result(capture_id, path, question, True)

        def inspect_reference(question):
            if not reference:
                return {"ok": False, "error": "本轮没有上传图片，请用户上传。"}
            return {"ok": True, "image_url": reference["image_url"], "image_source": "用户上传的图片", "source_type": "uploaded_image",
                    "observation": self.agent_vision(reference_path, question, history)}

        def query_project_info():
            # Reference material, not a keyword shortcut or current map inventory.
            return {"ok": True, "source": "项目历史指标资料，非当前部署的实时测量或承诺",
                    "records": [agent_routing._project_metric_query(q) for q in
                                ("空间拟物体数量", "语义图谱规模", "空间对象识别指标")]}

        def get_robot_status():
            task = self.conversation_task(session_id) if store else self.tasks.snapshot()
            fields = ("id", "instruction", "status", "current_step", "error", "result_text", "observations", "skill", "started_at", "finished_at")
            robot = self.tasks.robot_status()
            # Geometry is evidence, not an intent router or an automatic movement decision.
            scene = memory.get('last_scene') or {}
            target = map_evidence.objects.get(str(scene.get('target_ann_id'))) or {}
            location_evidence = {'source_type': 'live_pose_compared_with_saved_map',
                                 'target_ann_id': scene.get('target_ann_id'), 'distance_m': None,
                                 'note': '历史到达记录不是当前位置；距离只表示当前定位与地图目标的距离，不证明目标可见。'}
            xy = target.get('nav_xy') or target.get('floor_xy')
            pose = robot.get('pose') or {}
            if robot.get('online') and xy and len(xy) >= 2:
                try:
                    distance = math.hypot(float(pose['x'])-float(xy[0]), float(pose['y'])-float(xy[1]))
                    if math.isfinite(distance):
                        location_evidence['distance_m'] = round(distance, 2)
                        location_evidence['target_map_xy'] = xy[:2]
                except (KeyError, TypeError, ValueError):
                    pass
            task_data = {k: task[k] for k in fields if k in task} if task else None
            if task_data and isinstance(task_data.get("observations"), list):
                task_data["observations"] = task_data["observations"][-4:]
            if task and (memory.get("planned_task") or {}).get("id") == task.get("id"):
                memory["planned_task"].update(status=task.get("status"), recorded_at=time.time())
            return {"ok": True, "source_type": "robot_status", "observed_at": time.time(),
                    "robot": {k: v for k, v in robot.items() if k != "track"},
                    "task": task_data,
                    "last_observed_target_location": location_evidence,
                    "mapping_busy": self.mapping.is_busy(),
                    "robot_has_pending_plan": bool((self.tasks.snapshot() or {}).get('status') in ('planned', 'needs_clarification')),
                    "robot_busy": self.tasks.is_busy() if store else bool(task and task.get('status') in ('running','canceling'))}

        def plan_robot_skill(skill_id, **arguments):
            if agent_map_evidence.MapEvidence(self.graph_snapshot()).version != map_evidence.version:
                raise agent_runner.ToolInputError("本轮查询后地图已更新，请重新查询并规划，不能沿用旧目标。")
            targets = (arguments.get("target_ann_ids", []) if skill_id == "navigate" else
                       [arguments.get("target_ann_id")] if skill_id == "inspect_location" else [])
            for target in targets:
                selected = memory.get("confirmed_target")
                if selected and target != selected["ann_id"]:
                    return {"ok": False, "error": "请使用用户在候选按钮中选定的目标，不要替换为其他对象"}
                obj = map_evidence.objects.get(str(target)) or {}
                scope = (memory.get("target_queries") or {}).get(obj.get("category"), {})
                candidates = [map_evidence.objects[key] for key in scope.get("ids", [])
                              if key in map_evidence.objects
                              and map_evidence.objects[key].get("category") == obj.get("category")]
                if len(candidates) > 1 and not selected and str(target) not in resolved_this_turn and scope.get("snapshot") == map_evidence.version:
                    choices = [{k: obj.get(k) for k in ("ann_id", "category", "category_zh", "position", "floor_xy")}
                               for obj in candidates[:20]]
                    memory["target_offer"] = {"ids": [c["ann_id"] for c in choices], "snapshot": map_evidence.version}
                    target_offer_this_turn.update(memory["target_offer"])
                    return {"ok": False, "retry_after_lookup": True, "error": "此前查询范围包含多个同类目标，尚未创建任务。用户已给完整名称时先调用resolve_map_target核对，唯一匹配后再提交计划。只有名称仍不唯一或用户未明确目标时，才调用ask_user请用户选择。不要把历史宽泛范围当成本轮歧义。",
                            "target_choices": choices, "target_choice_snapshot": map_evidence.version}
            task = self.plan_for_conversation(session_id, lambda: self.tasks.plan_skill(skill_id, reference=reference, **arguments),
                                              payload.get('user_message_id'))
            return {"ok": True, "task": task, "requires_confirmation": True}

        def ask_user(goal, question, skill_id, missing_inputs):
            # Validate declared missing resources; do not classify user wording.
            if skill_id != "none":
                resources = get_skill_context(skill_id)
                unsupported = sorted(set(missing_inputs) - set(resources["accepted_inputs"]))
                if unsupported:
                    return {"ok": False, "error": "这些不是该技能所需的输入，不能把可选线索当成必填条件。依据技能契约重新判断；资源满足时生成待确认计划。确有指代或用户偏好歧义可用空missing_inputs说明，不自动执行。",
                            "unsupported_missing_inputs": unsupported, "resources": resources}
                available = set(resources["default_arguments"])
                if reference_source == "current_turn" and reference:
                    available.add("reference")
                if available.intersection(missing_inputs):
                    return {"ok": False, "error": "先依据已核验资源重新判断。不要索要本轮已上传的照片或唯一默认点；用户另指定目标、历史照片指代不明或多候选仍可澄清。",
                            "resources": resources}
            elif "reference" in missing_inputs and reference_source == "current_turn" and reference:
                return {"ok": False, "error": "本轮参考图已上传成功，不需要重新上传；可查看参考图或继续规划。"}
            result = {"ok": True, "clarification": question, "pending_request": {
                "goal": goal, "clarification": question, "status": "needs_clarification"}}
            offer = target_offer_this_turn
            if offer.get('snapshot') == map_evidence.version and skill_id != 'none':
                result['target_choices'] = [{k:map_evidence.objects[str(key)].get(k)
                    for k in ('ann_id','category','category_zh','position','floor_xy')}
                    for key in offer.get('ids', []) if str(key) in map_evidence.objects]
                result['target_choice_snapshot'] = map_evidence.version
            return result

        text_arg = {"type": "string"}
        catalog = map_evidence.category_catalog()
        history_tool = agent_runner.Tool("分析本轮开始前本会话最近一张历史现场照片；即使本轮又拍了新照片，该工具仍分析旧图。不启动相机、不代表实时画面。", {"question": text_arg},
                            inspect_previous_scene, "正在查看之前的照片", "历史照片分析", "historical_observe")
        category_arg = {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 200,
            "description": "选择所有语义相关类别的原始 category 值组成字符串数组；全图传 [\"*\"]，*不能与具体类别混用；没有相关类别传 []。当前目录：" +
                json.dumps(catalog["categories"], ensure_ascii=False)}
        if catalog["next_offset"] is None:
            category_arg["items"]["enum"] = ["*", *map_evidence.by_category]
        else:
            category_arg["description"] += "目录未完整，可用 list_map_categories 继续读取。"
        tools = {
            "resolve_map_target": agent_runner.Tool("按用户给出的地点名称或别名查找，优先精确匹配，无精确结果时按名称包含查找。保留用户给出的完整名称；只有简称时也可查询。唯一结果可生成待确认计划，多结果需消歧。此工具不依赖room_name/room_type是否填写，不执行移动。", {
                "name": {"type":"string","description":"本轮明确目标的完整名称，保留房间号、前门/后门等限定；不臆造。"}},
                resolve_map_target, "正在核对目标名称", "目标名称核对"),
            "get_skill_context": agent_runner.Tool("准备技能或询问缺少资料前，读取技能所需的参考图状态及地图配置的默认点位。默认点来自当前地图而非固定编号；用户指定地点优先，多候选必须消歧。仅查询不执行。", {
                "skill_id": {"type": "string", "enum": [s["id"] for s in agent_skills.skill_summary()]}},
                get_skill_context, "正在核对任务所需资料和默认点位", "技能资源查询", "prepare"),
            "observe_scene": agent_runner.Tool("拍摄并分析原地当前的新照片，不移动。用于明确请求查看当前现场，或本会话已有明确观察对象的实时追问。如果没有历史/参考图且所指对象不明，先用ask_user澄清，不能靠随意拍照猜测所指对象。历史照片的问题使用历史分析工具。", {"question": text_arg}, observe_scene, "正在查看当前画面", "现场观察", "observe"),
            "inspect_reference": agent_runner.Tool("查看用户上传的图片", {"question": text_arg}, inspect_reference, "正在查看上传的图片", "图片分析"),
            "query_map": agent_runner.Tool("查询已保存的环境记录/拟物体地图。按语义选择全部相关类别和筛选条件，工具一次返回完整计数、分类数量和位置；不需要提供对象 ID，不需再调用统计工具。", {
                "question": {"type": "string", "description": "结合上下文还原的完整查询需求"},
                "categories": category_arg, "filters": agent_map_evidence.FILTER_SCHEMA,
                "count_unit": {"type":"string","enum":["objects","rooms"],"default":"objects",
                    "description":"问物体/门数量选objects；问多少间房间/工作室选rooms，按room_id去重，配合room_type或room_name筛选。"},
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "default": 0}}, map_evidence.query, "正在查询地图", "地图查询"),
            "query_project_info": agent_runner.Tool("查阅项目历史验收指标，不是当前地图或实时状态", {}, query_project_info, "正在查阅项目资料", "项目资料"),
            "compare_map_positions": agent_runner.Tool("核对地图中参考物与候选物的距离。先查询双方真实 ID；用于旁边、附近等空间线索消歧，不能仅凭位置描述相似选目标。不自动导航。", {
                "reference_ann_id": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "candidate_ann_ids": {"type": "array", "minItems": 1, "maxItems": 40,
                    "items": {"type": "integer", "minimum": 0, "maximum": 2147483647}}},
                map_evidence.compare_positions, "正在核对地图空间关系", "地图空间关系"),
            "get_robot_status": agent_runner.Tool("查询机器人状态与任务结果", {}, get_robot_status, "正在读取机器人状态", "机器人状态查询"),
            "ask_user": agent_runner.Tool("仅当缺少必要信息时澄清；保留未完成需求等待回答。明确的只读请求直接调用对应工具，不再确认。", {
                "goal": {"type": "string", "description": "尚未完成的完整需求"}, "question": text_arg,
                "skill_id": {"type": "string", "enum": ["none", *[s["id"] for s in agent_skills.skill_summary()]], "description": "当前准备的技能；不是技能需求才填none。先核对技能资源。"},
                "missing_inputs": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 80},
                    "description": "确实缺少的资源参数名，如reference、pickup_ann_id、return_ann_id。已有本轮参考图和唯一默认点不能称为缺失；用户偏好、目标歧义或明确另指定地点时用空数组，并在question说明真正需要确认的内容。"}}, ask_user, "有一处信息需要确认", "需求澄清"),
        }
        tools.update(agent_skills.skill_tools(plan_robot_skill))
        if store:
            tools['recall_conversation'] = agent_runner.Tool('检索本会话的历史对话、工具结果和任务经历，不访问其他会话。历史不是实时现场；可用返回的照片标识继续分析。', {
                'query': {'type':'string','default':'','maxLength':200,'description':'查询词；空字符串读取最近记录'},
                'offset': {'type':'integer','minimum':0,'maximum':1000000,'default':0}},
                lambda query='', offset=0: store.recall(session_id, query, offset), '正在查阅本会话记忆', '会话记忆查询')
            def inspect_memory_image(capture_id, question):
                url = f'/captures/{capture_id}.jpg'
                if not web_settings.CAPTURE_ID_RE.fullmatch(capture_id) or not store.owns_media(session_id, url):
                    raise agent_runner.ToolInputError('照片不属于当前会话，或标识无效')
                path = self.capture_path(capture_id)
                if not path.is_file():
                    raise agent_runner.ToolInputError('历史照片已不可用，不能假装已查看；需要新照片时明确说明')
                if new_scene_this_turn is None:
                    memory['last_scene'] = {'capture_id': capture_id, 'observed_at': path.stat().st_mtime}
                return scene_result(capture_id, path, question, False)
            tools['inspect_memory_image'] = agent_runner.Tool('分析本会话历史任务/对话照片，不启动相机，不代表当前实时画面。照片标识从本会话记忆获得。', {
                'capture_id': {'type':'string','maxLength':32}, 'question':text_arg},
                inspect_memory_image, '正在分析本会话历史照片', '历史任务照片分析', 'historical_observe')
        if previous_capture:
            tools["inspect_previous_scene"] = history_tool
        if catalog["next_offset"] is not None:
            tools["list_map_categories"] = agent_runner.Tool("分页查看地图其余类别；仅目录很大时需要。", {
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "default": 0}},
                map_evidence.category_catalog, "正在读取地图类别", "地图类别目录")
        if not reference:
            # Offer only currently available resources; never imply that an
            # absent upload can answer questions about a saved environment.
            del tools["inspect_reference"]
        def remember(name, arguments, result):
            if store:
                # Avoid recursively embedding previously recalled records into new records.
                saved_result = ({'record_sequences': [r['seq'] for r in result.get('items', [])]}
                                if name == 'recall_conversation' else result)
                store.event(session_id, 'tool_result', {'tool':name, 'arguments':arguments, 'result':saved_result})
            diagnostics.LOGGER.info("[agent-tool] tool=%s ok=%s task_id=%s task_status=%s",
                        name, result.get("ok", True), (result.get("task") or {}).get("id"),
                        (result.get("task") or {}).get("status"))
            memory["turn_goal"] = str(arguments.get("instruction") or arguments.get("goal") or arguments.get("question") or question)
            if not result.get("ok", True):
                return
            if name == "query_map":
                scopes = memory.setdefault("target_queries", {})
                for category in result.get("selection", {}).get("categories", []):
                    scopes[category] = {"ids": [key for key in result.get("target_ids", [])
                        if map_evidence.objects[key].get("category") == category], "snapshot": result.get("snapshot")}
            # Keep small, server-produced evidence, not copied client claims or a
            # second full map. Memory is historical, never a live sensor cache.
            keys = ("source", "source_type", "snapshot", "map_name", "capture_id", "image_source", "observed_at",
                    "observation", "count", "count_unit", "object_count", "rooms", "groups", "selection", "count_scope", "missing_filter_fields")
            evidence = {key: result[key] for key in keys if key in result}
            if name == "get_robot_status":
                task = result.get("task") or {}
                evidence["task"] = {key: task[key] for key in ("id", "status", "result_text", "error") if key in task}
            if evidence and len(json.dumps(evidence, ensure_ascii=False)) <= 6000:
                memory["evidence"] = (memory["evidence"] + [{"tool": name, "recorded_at": time.time(), **evidence}])[-4:]
                memory["focus"] = {"source_type": result.get("source_type"), "question": memory["turn_goal"][:1200],
                    "snapshot": result.get("snapshot"), "selection": result.get("selection"),
                    "capture_id": result.get("capture_id"), "observed_at": result.get("observed_at")}

        if self.mock:
            return {"text": "当前为模拟模式，没有连接模型或操作真实机器人。", "tool_trace": [], "target_ids": []}
        from jaka_agent.storage.memory import compact_record
        return agent_runner.AgentRunner(self.agent_complete, tools, native=True, require_final_tool=True,
                           require_scene_contract=True,
                           task_snapshot=(lambda: self.conversation_task(session_id)) if store else self.tasks.snapshot).run(question, history, {
            "has_uploaded_image": reference_source == "current_turn", "has_reference_image": bool(reference),
            "reference_source": reference_source, "has_previous_scene_image": bool(previous_capture),
            "skills": agent_skills.skill_summary(),
            "available_map": {"name": map_evidence.name, "snapshot": map_evidence.version}, "now": time.time(),
            "pending_request": memory.get("pending_request"), "focus": memory.get("focus"),
            "user_selected_target": memory.get("confirmed_target"),
            "planned_task_not_execution_proof": memory.get("planned_task"),
            "previous_evidence_not_live": [compact_record(item, 900) for item in memory["evidence"][-3:]],
            "historical_dialogue_summary": memory.get('history_summary', []),
            "task_experiences_not_live": memory.get('task_experiences', []),
            "last_scene_not_live": memory.get('last_scene'),
        }, emit, record=remember)
