"""Build pending task cards from validated skill arguments and model plans."""
from __future__ import annotations
import json
import re
import jaka_agent.agent.map_evidence as agent_map_evidence
import jaka_agent.agent.routing as agent_routing
import jaka_agent.agent.skills as agent_skills
import jaka_agent.diagnostics as diagnostics
import jaka_agent.tasks.validation as tasks_validation
import jaka_agent.web.settings as web_settings
import copy
import math
import time
import uuid

class TaskCardsMixin:
    def _mock_plan(self, instruction: str, objects: list) -> dict:
        """仅 --mock 使用的界面测试计划; 真机模式始终由 Qwen plan_task 生成。"""
        valid_ids = {obj["ann_id"] for obj in objects}
        explicit_ids = [
            int(value)
            for value in re.findall(r"(?:ann_id\s*[=:：]\s*|#|编号\s*)(\d+)", instruction, re.I)
            if int(value) in valid_ids
        ]
        mentioned = list(dict.fromkeys(explicit_ids))
        if not mentioned:
            for obj in objects:
                names = [obj.get("category_zh", ""), obj.get("category", ""), str(obj.get("ann_id"))]
                if any(name and name in instruction for name in names):
                    mentioned.append(obj["ann_id"])
        if not mentioned:
            mentioned = [objects[0]["ann_id"]]
        observe = any(word in instruction for word in ("看", "观察", "状态", "检查", "寻找", "找"))
        patrol = any(word in instruction.lower() for word in ("巡视", "巡逻", "patrol")) and len(mentioned) >= 2
        need_return = False  # Mock text is not an authorization to append travel.
        steps = []
        if patrol:
            steps.append({
                "type": "patrol", "target_ann_ids": mentioned, "count": -1,
                "reason": "循环拍照并对比各点位是否出现环境变化",
            })
        else:
            for ann_id in mentioned:
                steps.append({"type": "navigate", "target_ann_id": ann_id, "use_viewpoint": False, "reason": "前往目标"})
                if observe:
                    steps.append({"type": "observe", "target_ann_id": ann_id, "question": instruction, "fine_adjust": False})
        if need_return:
            steps.append({"type": "return"})
        return {"understanding": instruction, "steps": steps, "need_return": need_return}

    def plan(self, instruction: str) -> dict:
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")
        graph = self.web.graph_snapshot()
        objects = graph["objects"]
        if self.web.mock:
            plan = self._mock_plan(instruction, objects)
        else:
            from jaka_agent.tasks.planning import plan_task

            plan = plan_task(instruction, objects)
        diagnostics.LOGGER.info(
            "[plan] generated instruction=%r map=%r plan=%s",
            instruction,
            graph.get("name"),
            json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
        )
        if tasks_validation._ensure_explicit_observation_steps(plan, instruction):
            diagnostics.LOGGER.warning(
                "[plan] model omitted observe; repaired explicit observation step instruction=%r plan=%s",
                instruction,
                json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            )
        clarification = tasks_validation._fallback_clarification(instruction, plan, objects)
        if clarification:
            candidate_ids = list(dict.fromkeys(clarification.get("candidate_ann_ids") or []))
            candidates = []
            for ann_id in candidate_ids:
                obj = next((value for value in objects if value["ann_id"] == ann_id), None)
                if not obj:
                    continue
                xy = obj.get("nav_xy") or obj["floor_xy"]
                candidates.append({
                    "ann_id": ann_id,
                    "name": obj.get("category_zh") or obj.get("category") or f"目标{ann_id}",
                    "position": obj.get("position") or "",
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                })
            if len(candidates) < 2:
                raise RuntimeError("目标澄清候选不足，请重新描述目标")
            task = {
                "id": uuid.uuid4().hex,
                "instruction": instruction,
                "understanding": plan.get("understanding") or clarification.get("question"),
                "status": "needs_clarification",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "current_step": None,
                "steps": [],
                "plan": plan,
                "route_points": [],
                "observations": [],
                "logs": [],
                "error": None,
                "cancel_requested": False,
                "map_name": graph["name"],
                "clarification": {
                    "question": str(clarification.get("question") or "请选择具体目标"),
                    "candidate_ann_ids": [value["ann_id"] for value in candidates],
                },
                "candidates": candidates,
            }
            with self.lock:
                self.task = task
            return copy.deepcopy(task)
        return self._publish_plan(instruction, graph, plan)

    def _publish_plan(self, instruction, graph, plan):
        """Shared pending-task builder; never starts the robot."""
        objects = graph["objects"]
        if plan.get("need_return") and not any(step.get("type") == "return" for step in plan.get("steps", [])):
            plan["steps"].append({"type": "return"})
        if not plan.get("steps"):
            raise RuntimeError("小卡 没有生成可执行步骤")
        names = {obj["ann_id"]: obj.get("category_zh") or obj.get("category") for obj in objects}
        steps = []
        route_points = []
        patrol_step = next((step for step in plan.get("steps", []) if step.get("type") == "patrol"), None)
        for index, step in enumerate(plan.get("steps", [])):
            item = copy.deepcopy(step)
            item.update({
                "index": index,
                "status": "pending",
                "target_names": [names.get(ann_id, str(ann_id)) for ann_id in tasks_validation._step_target_ids(step)],
            })
            steps.append(item)
            if step.get("type") not in ("navigate", "cruise", "patrol"):
                continue
            for visit_order, ann_id in enumerate(tasks_validation._step_target_ids(step)):
                obj = next((value for value in objects if value["ann_id"] == ann_id), None)
                if obj:
                    display_xy = obj.get("nav_xy") or obj["floor_xy"]
                    route_points.append({
                        "ann_id": ann_id,
                        "step_index": index,
                        "visit_order": visit_order,
                        "x": display_xy[0],
                        "y": display_xy[1],
                    })
        task = {
            "id": uuid.uuid4().hex,
            "kind": "patrol" if patrol_step else "robot_task",
            "instruction": instruction,
            "understanding": plan.get("understanding") or instruction,
            "status": "planned",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "current_target_ann_id": None,
            "steps": steps,
            "plan": plan,
            "route_points": route_points,
            "observations": [],
            "logs": [],
            "error": None,
            "cancel_requested": False,
            "map_name": graph["name"],
            "video": None,
        }
        if patrol_step:
            task.update({
                "patrol_round": 0,
                "patrol_count": int(patrol_step.get("count", -1)),
                "patrol_baselines": {},
                "patrol_comparisons": [],
                "patrol_status_by_ann": {
                    str(ann_id): "pending" for ann_id in tasks_validation._step_target_ids(patrol_step)
                },
                "anomaly": None,
                "result_text": "",
            })
        with self.lock:
            self.task = task
        return copy.deepcopy(task)

    def plan_skill(self, skill_id, instruction, reference=None, expected_map_snapshot=None, **parameters):
        """One adapter for robot skill proposals; confirmation still uses execute()."""
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")
            graph = self.web.graph_snapshot()
            map_snapshot = agent_map_evidence.MapEvidence(graph).version
            if expected_map_snapshot is not None and expected_map_snapshot != map_snapshot:
                raise ValueError("地图已更新，请刷新地图后重新选择巡逻点")
            skill, args, plan = agent_skills.prepare_skill(skill_id, {"instruction": instruction, **parameters}, graph, reference)
            if skill.requires_reference:
                if not isinstance(reference, dict):
                    raise ValueError("参考图片无效")
                path = self.web.reference_path(reference.get("reference_id"), reference.get("suffix"))
                if not path.is_file():
                    raise ValueError("参考图片不存在或已过期，请重新上传")
            if skill_id == "welcome":
                self.plan_welcome(args["instruction"], reference, args["pickup_ann_id"], args["return_ann_id"], announce=False)
            elif skill_id == "find_object":
                self.plan_find_object(args["instruction"], reference, announce=False)
            else:
                self._publish_plan(args["instruction"], graph, plan)
            self.task["skill"] = {"id": skill.id, "name": skill.name, "version": 1,
                                  "completion": skill.completion, "failure_policy": skill.failure_policy,
                                  "map_snapshot": map_snapshot}
            return copy.deepcopy(self.task)

    def plan_find_object(self, instruction: str, reference: dict, *, announce=True) -> dict:
        """为上传参考图生成确定性的逐点寻物任务，不依赖通用任务规划模型。"""
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")
        graph = self.web.graph_snapshot()
        candidates = agent_routing._find_object_candidates(graph["objects"])
        if not candidates:
            raise RuntimeError("当前地图没有适合放置物品的可巡检点位")
        target_label = self.web.find_target_label(instruction, reference)
        start_xy = None
        try:
            status = self.robot_status()
            if status.get("online") and status.get("pose"):
                start_xy = (float(status["pose"]["x"]), float(status["pose"]["y"]))
        except Exception:
            pass
        if start_xy is not None:
            ordered = []
            remaining = list(candidates)
            cursor = start_xy
            while remaining:
                nearest = min(
                    remaining,
                    key=lambda obj: math.hypot(
                        float((obj.get("nav_xy") or obj["floor_xy"])[0]) - cursor[0],
                        float((obj.get("nav_xy") or obj["floor_xy"])[1]) - cursor[1],
                    ),
                )
                remaining.remove(nearest)
                ordered.append(nearest)
                point = nearest.get("nav_xy") or nearest["floor_xy"]
                cursor = (float(point[0]), float(point[1]))
            candidates = ordered
        steps = []
        route_points = []
        route_distance = 0.0
        route_cursor = start_xy
        for index, obj in enumerate(candidates):
            ann_id = obj["ann_id"]
            name = obj.get("category_zh") or obj.get("category") or f"目标{ann_id}"
            xy = obj.get("nav_xy") or obj["floor_xy"]
            steps.append({
                "type": "find_object",
                "target_ann_id": ann_id,
                "index": index,
                "status": "pending",
                "target_names": [name],
                "reason": "在可放置物品的点位寻找参考物",
            })
            route_points.append({
                "ann_id": ann_id, "step_index": index, "visit_order": index,
                "x": xy[0], "y": xy[1], "verdict": "candidate",
            })
            if route_cursor is not None:
                route_distance += math.hypot(float(xy[0]) - route_cursor[0], float(xy[1]) - route_cursor[1])
            route_cursor = (float(xy[0]), float(xy[1]))
        task = {
            "id": uuid.uuid4().hex,
            "kind": "find_object",
            "instruction": instruction,
            "understanding": (
                f"先用拟物体地图筛选 {len(candidates)} 个可放置物品的候选点，机器人沿路线前往时"
                f"每 {int(web_settings.FIND_SNAPSHOT_INTERVAL_SECONDS)} 秒抓拍一次，"
                "单帧高置信匹配后立即停止。"
            ),
            "status": "planned",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "current_target_ann_id": None,
            "steps": steps,
            "plan": {"steps": copy.deepcopy(steps), "need_return": False},
            "route_points": route_points,
            "observations": [],
            "logs": [],
            "error": None,
            "cancel_requested": False,
            "map_name": graph["name"],
            "reference": copy.deepcopy(reference),
            "target_label": target_label,
            "candidate_ann_ids": [obj["ann_id"] for obj in candidates],
            "checked_ann_ids": [],
            "candidate_status": {str(obj["ann_id"]): "candidate" for obj in candidates},
            "planned_route_m": round(route_distance, 2) if start_xy is not None else None,
            "search_attempts": [],
            "snapshot_count": 0,
            "snapshot_interval_seconds": web_settings.FIND_SNAPSHOT_INTERVAL_SECONDS,
            "snapshot_confirmation_frames": web_settings.FIND_SNAPSHOT_CONFIRM_FRAMES,
            "current_search_stage": None,
            "match": None,
            "result_text": "",
            "video": None,
        }
        with self.lock:
            self.task = task
        if announce:
            self.web.speak(f"确认要开始寻找这个{target_label}吗？")
        return copy.deepcopy(task)

    def plan_welcome(
        self,
        instruction: str,
        reference: dict,
        pickup_ann_id: int,
        return_ann_id: int,
        *, announce=True,
    ) -> dict:
        """生成“前往接人点等待目标人物，确认后带回指定点”的迎宾任务。"""
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")

        graph = self.web.graph_snapshot()
        objects = {int(obj["ann_id"]): obj for obj in graph["objects"]}
        if pickup_ann_id not in objects:
            raise ValueError(f"接人点 #{pickup_ann_id} 不在当前地图中")
        if return_ann_id not in objects:
            raise ValueError(f"返回点 #{return_ann_id} 不在当前地图中")
        if pickup_ann_id == return_ann_id:
            raise ValueError("接人点和返回点不能是同一个地图目标")

        def point_value(ann_id):
            obj = objects[ann_id]
            xy = obj.get("nav_xy") or obj.get("floor_xy")
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                raise ValueError(f"地图目标 #{ann_id} 没有可用导航坐标")
            return obj, (float(xy[0]), float(xy[1]))

        pickup, pickup_xy = point_value(pickup_ann_id)
        return_point, return_xy = point_value(return_ann_id)
        pickup_name = pickup.get("category_zh") or pickup.get("category") or f"目标{pickup_ann_id}"
        return_name = (
            return_point.get("category_zh")
            or return_point.get("category")
            or f"目标{return_ann_id}"
        )
        steps = [
            {
                "type": "navigate",
                "target_ann_id": pickup_ann_id,
                "index": 0,
                "status": "pending",
                "target_names": [pickup_name],
                "reason": "前往接人点并正对所选地图目标",
            },
            {
                "type": "wait_guest",
                "target_ann_id": pickup_ann_id,
                "index": 1,
                "status": "pending",
                "target_names": [pickup_name],
                "reason": (
                    f"每 {web_settings.WELCOME_SNAPSHOT_INTERVAL_SECONDS:g} 秒抓拍识别目标人物，"
                    "发现疑似目标后展示现场照片，等待主人在网页确认"
                ),
            },
            {
                "type": "navigate",
                "target_ann_id": return_ann_id,
                "index": 2,
                "status": "pending",
                "target_names": [return_name],
                "reason": "主人在网页确认后带领客人前往送客点",
            },
        ]
        task = {
            "id": uuid.uuid4().hex,
            "kind": "welcome",
            "instruction": instruction,
            "understanding": (
                f"前往{pickup_name}等待参考照片中的人物；发现疑似目标后展示现场照片，"
                f"等待主人在网页确认后带领客人前往{return_name}；拒绝后继续等待。"
            ),
            "status": "planned",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "current_target_ann_id": None,
            "current_stage": "等待执行",
            "steps": steps,
            "plan": {"steps": copy.deepcopy(steps), "need_return": False},
            "route_points": [
                {
                    "ann_id": pickup_ann_id,
                    "step_index": 0,
                    "visit_order": 0,
                    "x": pickup_xy[0],
                    "y": pickup_xy[1],
                    "role": "pickup",
                },
                {
                    "ann_id": return_ann_id,
                    "step_index": 2,
                    "visit_order": 1,
                    "x": return_xy[0],
                    "y": return_xy[1],
                    "role": "return",
                },
            ],
            "observations": [],
            "logs": [],
            "error": None,
            "cancel_requested": False,
            "map_name": graph["name"],
            "reference": copy.deepcopy(reference),
            "pickup_ann_id": pickup_ann_id,
            "pickup_name": pickup_name,
            "return_ann_id": return_ann_id,
            "return_name": return_name,
            "snapshot_interval_seconds": web_settings.WELCOME_SNAPSHOT_INTERVAL_SECONDS,
            "snapshot_count": 0,
            "rejected_count": 0,
            "unclear_count": 0,
            "last_heard_text": "",
            "result_text": "",
            "video": None,
        }
        with self.lock:
            self.task = task
        if announce:
            self.web.speak(f"迎宾路线已设置，接人点是{pickup_name}，返回点是{return_name}。")
        return copy.deepcopy(task)
