"""Visual inference for web observations, references and patrol comparisons."""
from __future__ import annotations
import jaka_agent.agent.routing as agent_routing
import jaka_agent.diagnostics as diagnostics
import jaka_agent.models.person_match as models_person_match
import jaka_agent.models.vision as models_vision
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import base64
import os
import re
import time
from pathlib import Path

class ModelsPerceptionMixin:
    def route(self, question: str, history=None, has_reference=False) -> dict:
        """判断本轮是否需要新照片；纯文字问题在同一次模型调用中直接回答。"""
        metric_answer = agent_routing._project_metric_query(question)
        if metric_answer:
            diagnostics.LOGGER.info("[route] deterministic mode=project_metric question=%r", question)
            return {"mode": "text", "answer": metric_answer}
        if agent_routing._has_find_object_intent(question) and not has_reference:
            diagnostics.LOGGER.info("[route] find request is missing reference image question=%r", question)
            return {"mode": "text", "answer": "请先上传要寻找物品的参考照片，我才能开始地图巡检。"}
        if agent_routing._is_reference_find_request(question, has_reference):
            diagnostics.LOGGER.info(
                "[route] forcing find_object for reference request question=%r has_reference=%s",
                question,
                has_reference,
            )
            return {"mode": "find_object", "answer": ""}
        fallback_mode = agent_routing._fallback_route_mode(question, has_reference)
        if fallback_mode in ("robot_task", "vision"):
            diagnostics.LOGGER.info(
                "[route] deterministic mode=%s question=%r has_reference=%s",
                fallback_mode,
                question,
                has_reference,
            )
            return {"mode": fallback_mode, "answer": ""}
        if fallback_mode == "map_query":
            diagnostics.LOGGER.info("[route] deterministic mode=map_query question=%r", question)
            answer, target_ids = agent_routing._map_query(self.graph_snapshot()["objects"], question)
            return {"mode": "map_query", "answer": answer, "target_ids": target_ids}
        if self.mock:
            if not has_reference and any(word in question for word in agent_routing.FIND_OBJECT_WORDS):
                return {"mode": "text", "answer": "请先上传要寻找物品的参考照片，我才能开始地图巡检。"}
            mode = agent_routing._mock_route_mode(question, has_reference=has_reference)
            if mode == "map_query":
                answer, target_ids = agent_routing._map_query(self.graph_snapshot()["objects"], question)
                return {"mode": mode, "answer": answer, "target_ids": target_ids}
            if mode != "text":
                return {"mode": mode, "answer": ""}
            if "导航" in question or "指令" in question:
                answer = (
                    "底层 `qwen_planner.py` 支持按场景图导航、巡游、观察和返回。"
                    "移动任务会先由 小卡 生成计划并显示在地图上，确认后才驱动底盘。"
                )
            else:
                answer = "这是一个不依赖当前相机画面的文字问题，因此本轮不会启动相机。"
            return {"mode": "text", "answer": answer}

        from jaka_agent.models.settings import PLAN_MODEL
        from jaka_agent.models.runtime import _client

        context = "\n".join(agent_routing._history_lines(history)) or "(无历史)"
        user_message = (
            f"最近对话:\n{context}\n\n本轮是否有参考图: {'是' if has_reference else '否'}"
            f"\n当前问题: {question}"
        )
        response = _client().chat.completions.create(
            model=PLAN_MODEL,
            messages=[
                {"role": "system", "content": agent_routing.ROUTER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        diagnostics.LOGGER.info("[route-model] raw question=%r response=%r", question, raw)
        try:
            decision = agent_routing._parse_route_decision(raw)
        except ValueError as exc:
            raise RuntimeError(f"对话路由结果不是合法 JSON: {raw}") from exc
        if not isinstance(decision, dict):
            raise RuntimeError(f"对话路由结果不是对象: {raw}")
        mode = str(decision.get("mode") or "text")
        fallback_mode = agent_routing._fallback_route_mode(question, has_reference)
        if mode == "text" and fallback_mode:
            diagnostics.LOGGER.warning(
                "[route] corrected model mode text -> %s question=%r raw=%r",
                fallback_mode,
                question,
                raw,
            )
            mode = fallback_mode
        if mode not in ("text", "vision", "map_query", "robot_task", "find_object"):
            raise RuntimeError(f"未知对话模式: {mode}")
        if mode == "find_object" and not has_reference:
            return {"mode": "text", "answer": "请先上传要寻找物品的参考照片，我才能开始地图巡检。"}
        if mode == "map_query":
            answer, target_ids = agent_routing._map_query(self.graph_snapshot()["objects"], question)
            return {"mode": mode, "answer": answer, "target_ids": target_ids}
        answer = str(decision.get("answer") or "").strip()
        if mode == "text" and not answer:
            answer = "我暂时没有理解这句话。你可以直接说要去哪里、观察什么，或上传参考图片后说“帮我找这个物品”。"
        return {"mode": mode, "answer": answer}

    def infer(self, capture_id: str, question: str, history=None) -> str:
        """对已拍图片做视觉问答; 带最近对话以理解"它/那里/再看"等自然指代。"""
        path = self.capture_path(capture_id)
        if not path.exists():
            raise FileNotFoundError("拍摄图片不存在或已过期")
        if self.mock:
            return (
                "画面是一个室内办公区域，可以看到门、墙面窗户、桌椅和右侧的设备。"
                "当前环境光线较暗，画面带有轻微紫红色偏色，细小物体不容易准确辨认。"
            )
        with self.infer_lock:
            from jaka_agent.models.settings import VISION_MODEL
            from jaka_agent.models.vision import VISUAL_EVIDENCE_RULES, format_visual_answer, _image_data_url
            from jaka_agent.models.runtime import _client

            turns = agent_routing._history_lines(history)
            if turns:
                model_question = (
                    "以下是同一会话最近的对话，仅用于理解当前问题里的自然指代；"
                    "事实判断仍以这次新拍摄的图片为准。\n"
                    + "\n".join(turns)
                    + f"\n当前用户问题: {question}"
                )
            else:
                model_question = question
            image_url = _image_data_url(str(path), VISION_MODEL)
            response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": VISUAL_EVIDENCE_RULES + "\n只输出给用户看的自然语言，不输出JSON。\n" + model_question},
                    ],
                }],
                max_tokens=512,
                temperature=0.0,
            )
            return format_visual_answer(agent_routing._json_text(response.choices[0].message.content))

    def compare_reference(self, reference: dict, scene_path: Path) -> dict:
        """将参考图和当前现场图一起交给视觉模型，返回受限的命中结论。"""
        if self.mock:
            return {"found": False, "confidence": "low", "reason": "模拟模式不进行真实物品比对。"}
        reference_id = str(reference.get("reference_id") or "")
        suffix = str(reference.get("suffix") or "").lower()
        reference_path = self.reference_path(reference_id, suffix)
        if not reference_path.exists():
            raise FileNotFoundError("参考图片不存在或已过期")
        if not scene_path.exists():
            raise FileNotFoundError("现场拍摄图片不存在")

        with self.infer_lock:
            from jaka_agent.models.settings import BASE_URL, VISION_MODEL
            from jaka_agent.models.runtime import _client
            from jaka_agent.models.vision import _image_data_url

            # 所有模型输入统一预处理；两张图仍作为两个独立 image_url 发送。
            reference_url = _image_data_url(str(reference_path), VISION_MODEL)
            scene_url = _image_data_url(str(scene_path), VISION_MODEL)
            diagnostics.LOGGER.info(
                "[vision] compare_reference request model=%r base_url=%r input_mode=multi-image "
                "reasoning=slim-cot guidance=reference-image-only verification=disabled "
                "reference_bytes=%d scene_bytes=%d",
                VISION_MODEL,
                BASE_URL,
                reference_path.stat().st_size,
                scene_path.stat().st_size,
            )

            # 与 robot_client.py 已验证的瘦身 CoT 保持一致：
            # system 要求先做一句候选 grounding，再输出最终 JSON；
            # user 中两张图片仍独立输入，并全部放在文本之前。
            response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": agent_routing.FIND_TARGET_COT_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": reference_url}},
                            {"type": "image_url", "image_url": {"url": scene_url}},
                            {"type": "text", "text": agent_routing.FIND_TARGET_COT_USER},
                        ],
                    },
                ],
                max_tokens=agent_routing.FIND_TARGET_COT_MAX_TOKENS,
                temperature=0.0,
            )
        raw = agent_routing._json_text(response.choices[0].message.content).strip()
        comparison = agent_routing._parse_find_object_result(raw)
        diagnostics.LOGGER.info(
            "[vision] compare_reference response scene=%s found=%s confidence=%r raw=%r",
            scene_path,
            bool(comparison.get("found")),
            comparison.get("confidence"),
            raw[:2000],
        )
        return comparison

    def _welcome_scene_presence(self, scene_path: Path) -> dict:
        """Independent Qwen check of only the live scene, before reference comparison."""
        from openai import OpenAI

        base_url = os.getenv("JAKA_AGENT_BASE_URL", "").strip()
        model = os.getenv("JAKA_AGENT_MODEL", "").strip()
        if not base_url or not model:
            raise RuntimeError("迎宾现场人物检测需要配置 Qwen 图像服务")
        image_url = "data:image/jpeg;base64," + base64.b64encode(scene_path.read_bytes()).decode("ascii")
        tasks_runtime.check_cancelled()
        client = OpenAI(base_url=base_url, api_key=os.getenv("JAKA_AGENT_API_KEY", "EMPTY"))
        response = client.with_options(timeout=35, max_retries=0).chat.completions.create(
            model=model, temperature=0, max_tokens=180,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": web_settings.WELCOME_PRESENCE_PROMPT},
            ]}],
        )
        tasks_runtime.check_cancelled()
        raw = agent_routing._json_text(response.choices[0].message.content)
        result = models_person_match._parse_welcome_presence_result(raw)
        diagnostics.LOGGER.info("[welcome] scene-only person gate image=%s visible=%s count=%s evidence=%r raw=%r",
                    scene_path, result["person_visible"], result["person_count"],
                    result["visible_evidence"], raw[:500])
        return result

    def _retry_welcome_person_comparison(self, reference_path: Path, scene_path: Path) -> tuple[dict, str]:
        """One independent two-image retry when MiniCPM returned an invalid schema."""
        from openai import OpenAI

        base_url = os.getenv("JAKA_AGENT_BASE_URL", "").strip()
        model = os.getenv("JAKA_AGENT_MODEL", "").strip()
        if not base_url or not model:
            raise RuntimeError("迎宾复核需要配置 Qwen 图像服务")
        def image_url(path):
            return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        tasks_runtime.check_cancelled()
        client = OpenAI(base_url=base_url, api_key=os.getenv("JAKA_AGENT_API_KEY", "EMPTY"))
        response = client.with_options(timeout=65, max_retries=0).chat.completions.create(
            model=model, temperature=0, max_tokens=850,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(reference_path)}},
                {"type": "image_url", "image_url": {"url": image_url(scene_path)}},
                {"type": "text", "text": web_settings.WELCOME_RETRY_PROMPT},
            ]}],
        )
        tasks_runtime.check_cancelled()
        raw = agent_routing._json_text(response.choices[0].message.content).strip()
        return models_person_match._parse_person_reference_result(raw), raw

    def compare_person_reference(self, reference: dict, scene_path: Path) -> dict:
        """以两张独立图片比对迎宾人物外观，不向模型提供地点或身份文字先验。"""
        if self.mock:
            return {
                "found": True,
                "confidence": "high",
                "candidate_region": "模拟画面中央",
                "reason": "模拟模式固定命中目标人物。",
            }
        reference_id = str(reference.get("reference_id") or "")
        suffix = str(reference.get("suffix") or "").lower()
        reference_path = self.reference_path(reference_id, suffix)
        if not reference_path.exists():
            raise FileNotFoundError("迎宾人物参考图片不存在或已过期")
        if not scene_path.exists():
            raise FileNotFoundError("迎宾现场抓拍图片不存在")

        with self.infer_lock:
            presence = self._welcome_scene_presence(scene_path)
            if not presence["person_visible"]:
                return {
                    "found": False, "candidate_visible": False, "confidence": "low",
                    "score": 0.0, "schema_valid": presence["schema_valid"],
                    "reason": "现场单图未确认有人，不进行身份比对。",
                    "presence_gate": presence,
                }
            from jaka_agent.models.settings import BASE_URL, VISION_MODEL
            from jaka_agent.models.runtime import _client
            from jaka_agent.models.vision import _image_data_url

            reference_url = _image_data_url(str(reference_path), VISION_MODEL)
            scene_url = _image_data_url(str(scene_path), VISION_MODEL)
            diagnostics.LOGGER.info(
                "[vision] compare_person_reference request model=%r base_url=%r "
                "input_mode=multi-image reasoning=two-stage-slim-cot interval=%gs "
                "reference_bytes=%d scene_bytes=%d",
                VISION_MODEL,
                BASE_URL,
                web_settings.WELCOME_SNAPSHOT_INTERVAL_SECONDS,
                reference_path.stat().st_size,
                scene_path.stat().st_size,
            )
            response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": web_settings.WELCOME_PERSON_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": reference_url}},
                            {"type": "image_url", "image_url": {"url": scene_url}},
                            {"type": "text", "text": web_settings.WELCOME_PERSON_USER},
                        ],
                    },
                ],
                max_tokens=640,
                temperature=0.0,
            )
            reasoning_raw = agent_routing._json_text(response.choices[0].message.content).strip()
            format_prompt = (
                "下面是上一阶段根据两张图片得到的观察摘要：\n"
                "<<<观察摘要开始>>>\n"
                f"{reasoning_raw[:4000]}\n"
                "<<<观察摘要结束>>>\n"
                "请把它转换为以下扁平JSON结构。没有明确证据的项目必须填“无法核对”并设为unknown：\n"
                "候选位置和参考图、现场各项描述必须填写摘要中的实际可见内容，不能复制下面的模板文字。\n"
                f"{web_settings.WELCOME_PERSON_JSON_SCHEMA}"
            )
            formatted_response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": web_settings.WELCOME_PERSON_JSON_SYSTEM},
                    {"role": "user", "content": format_prompt},
                ],
                max_tokens=640,
                temperature=0.0,
            )
        raw = agent_routing._json_text(formatted_response.choices[0].message.content).strip()
        comparison = models_person_match._parse_person_reference_result(raw)
        if not comparison.get("schema_valid"):
            diagnostics.LOGGER.warning("[welcome] invalid MiniCPM comparison schema; retrying same two photos once scene=%s raw=%r",
                           scene_path, raw[:1000])
            try:
                retry, retry_raw = self._retry_welcome_person_comparison(reference_path, scene_path)
            except tasks_runtime.TaskCancelled:
                raise
            except Exception as exc:
                diagnostics.LOGGER.warning("[welcome] Qwen comparison retry failed scene=%s error=%s", scene_path, exc)
                retry, retry_raw = None, ""
            if retry is not None:
                diagnostics.LOGGER.info("[welcome] Qwen comparison retry scene=%s schema_valid=%s found=%s score=%.2f raw=%r",
                            scene_path, retry.get("schema_valid"), retry.get("found"),
                            float(retry.get("score") or 0), retry_raw[:1500])
                if retry.get("schema_valid"):
                    comparison = retry
                    raw = retry_raw
                    comparison["comparison_source"] = "qwen_retry"
        comparison["presence_gate"] = presence
        comparison["reasoning_summary"] = reasoning_raw[:2000]
        diagnostics.LOGGER.info(
            "[vision] compare_person_reference response scene=%s found=%s confidence=%r "
            "score=%.2f schema_valid=%s comparisons=%r reasoning=%r raw=%r",
            scene_path,
            bool(comparison.get("found")),
            comparison.get("confidence"),
            float(comparison.get("score") or 0.0),
            bool(comparison.get("schema_valid")),
            comparison.get("comparisons"),
            reasoning_raw[:2000],
            raw[:2000],
        )
        return comparison

    def compare_patrol_images(self, before_path: Path, after_path: Path, location_desc: str) -> dict:
        """比较同一巡逻点的前后画面，忽略轻微视角和光照变化，只报告物体级异常。"""
        if self.mock:
            return {"changed": False, "confidence": "high", "summary": "模拟比较未发现异常。", "changes": []}
        if not before_path.exists() or not after_path.exists():
            raise FileNotFoundError("巡逻基线图片或当前图片不存在")

        with self.infer_lock:
            from jaka_agent.models.settings import VISION_MODEL
            from jaka_agent.models.runtime import _client
            from jaka_agent.models.vision import _image_data_url, _is_minicpm_model

            is_minicpm = _is_minicpm_model(VISION_MODEL)
            before_url = _image_data_url(str(before_path), VISION_MODEL)
            after_url = _image_data_url(str(after_path), VISION_MODEL)
            patrol_prompt = (
                f"你是机器人巡逻的严格变化检测器。位置：{location_desc}。"
                "图片1是上次基线，图片2是本次画面。忽略轻微相机位置、透视、曝光、阴影和人员姿态差异；"
                "只把明确的物品新增、缺失、移动，门窗或设备状态改变、明显环境异常判为 changed=true。"
                "不确定时必须 changed=false 或 confidence=low。只输出 JSON："
                '{"changed":true,"confidence":"high|medium|low","summary":"中文结论",'
                '"changes":[{"type":"added|removed|moved|state|changed","item":"物品","detail":"变化说明"}]}'
            )
            if is_minicpm:
                response = _client().chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": before_url}},
                            {"type": "image_url", "image_url": {"url": after_url}},
                            {"type": "text", "text": patrol_prompt},
                        ],
                    }],
                    max_tokens=384,
                    temperature=0.0,
                )
            else:
                response = models_vision._vision_completion(
                    _client(),
                    model=VISION_MODEL,
                    messages=[
                        {"role": "system", "content": patrol_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": before_url}},
                                {"type": "image_url", "image_url": {"url": after_url}},
                                {"type": "text", "text": "请比较图片1和图片2。"},
                            ],
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
        return agent_routing._parse_patrol_result(response.choices[0].message.content)

    def find_target_label(self, instruction: str, reference: dict) -> str:
        """优先使用用户文字；指令没有物品名时让视觉模型识别参考图。"""
        from_text = agent_routing._find_target_label_from_instruction(instruction)
        if from_text:
            return from_text
        if self.mock:
            return "物品"
        try:
            reference_id = str(reference.get("reference_id") or "")
            suffix = str(reference.get("suffix") or "").lower()
            reference_path = self.reference_path(reference_id, suffix)
            if not reference_path.exists():
                return "物品"
            with self.infer_lock:
                from jaka_agent.models.settings import VISION_MODEL
                from jaka_agent.models.runtime import _client
                from jaka_agent.models.vision import _image_data_url, _is_minicpm_model

                image_url = _image_data_url(str(reference_path), VISION_MODEL)
                label_prompt = (
                    "识别这张参考图中的主要待寻找物，只输出一个简短中文物品名，"
                    f"不要标点、解释或 Markdown。用户寻物指令：{instruction or '未说明'}"
                )
                if _is_minicpm_model(VISION_MODEL):
                    request_kwargs = {
                        "model": VISION_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": label_prompt},
                            ],
                        }],
                        "max_tokens": 32,
                        "temperature": 0.0,
                    }
                else:
                    request_kwargs = {
                        "model": VISION_MODEL,
                        "messages": [
                            {"role": "system", "content": label_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_url}},
                                    {"type": "text", "text": "请识别参考物品。"},
                                ],
                            },
                        ],
                        "temperature": 0.0,
                    }
                response = None
                for attempt in range(2):
                    try:
                        response = _client().chat.completions.create(**request_kwargs)
                        break
                    except Exception as exc:
                        http_response = getattr(exc, "response", None)
                        status = (
                            getattr(exc, "status_code", None)
                            or getattr(http_response, "status_code", None)
                        )
                        if status != 500 or attempt > 0:
                            raise
                        diagnostics.LOGGER.warning(
                            "[find] reference label first vision request returned HTTP 500; retrying once"
                        )
                        time.sleep(0.5)
            label = re.sub(r"\s+", "", agent_routing._json_text(response.choices[0].message.content)).strip("，。！？,.!?；; ")
            return label[:16] or "物品"
        except Exception as exc:
            diagnostics._log_exception(
                "[find] reference label inference failed",
                exc,
                reference_id=reference.get("reference_id"),
            )
            return "物品"
