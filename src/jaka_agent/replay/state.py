"""Replay adapters; production tool validation, ownership and confirmation remain active."""
from __future__ import annotations

from collections import deque
import copy
import threading
import time
import uuid

from jaka_agent.agent.service import AgentServiceMixin
from jaka_agent.mapping.graphs import _normalize_graph
from jaka_agent.replay.fixtures import CASES, FEEDBACK_QUESTION, GRAPH, picture, scripted_completion
from jaka_agent.storage.memory import valid_id
from jaka_agent.tasks.manager import RobotTaskManager
from jaka_agent.tasks.runtime import check_cancelled
from jaka_agent.web.state import RobotWebState


class ReplayTaskManager(RobotTaskManager):
    def execute(self, task_id):
        with self.lock:
            task = super().execute(task_id)
            self.web.record("operator", "confirm", "已确认任务卡，允许回放执行")
            return task

    def cancel(self, task_id):
        task = super().cancel(task_id)
        self.web.record("operator", "cancel", "请求停止当前任务")
        return task

    def confirm_welcome_guest(self, task_id, confirmation_id, accepted, conversation_id):
        task = super().confirm_welcome_guest(task_id, confirmation_id, accepted, conversation_id)
        self.web.record("operator", "guest_confirmation", "已确认访客照片" if accepted else "拒绝当前照片，继续等待")
        return task

    def _run(self):
        try:
            super()._run()
        finally:
            task = self.snapshot() or {}
            labels = {"succeeded": "已完成", "aborted": "导航中止", "canceled": "已取消", "error": "执行错误",
                      "not_found": "未找到", "inconclusive": "无法确认"}
            self.web.record("robot", "task_result", "执行结束：" + labels.get(task.get("status"), str(task.get("status"))),
                            status=task.get("status"), detail=task.get("result_text") or task.get("error"))

    def _mock_move(self, target_xy):
        check_cancelled()
        self.web.record("robot", "navigate", "开始前往地图目标", target_xy=list(target_xy))
        if self.web.current_case() == "blocked":
            self._update(error="回放导航接口返回目标不可达。")
            self.web.record("robot", "navigation_failed", "目标不可达，停止后续观察", status="aborted")
            return "aborted"
        start = self.mock_pose[:2]
        for step in range(1, 9):
            check_cancelled()
            self.mock_pose[:2] = [start[i] + (target_xy[i] - start[i]) * step / 8 for i in range(2)]
            self.track.append(self.mock_pose[:])
            self.track = self.track[-500:]
            time.sleep(self.web.replay_tick)
        check_cancelled()
        self.web.record("robot", "arrived", "已到达目标，准备获取观察证据", pose=self.mock_pose[:])
        return "succeeded"

    def _mock_observation(self, step):
        capture_id, path = self._capture(ann_id=step.get("target_ann_id"))
        observation = {"ann_id": step.get("target_ann_id"), "image_url": f"/captures/{capture_id}.jpg",
                       "text": self.web.agent_vision(path, step.get("question", ""), []),
                       "source_type": "synthetic_replay", "captured_at": time.time()}
        with self.lock:
            self.task["observations"].append(observation)
        return observation

    def _run_find_object_mock(self):
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        for index, step in enumerate(self.task["steps"]):
            check_cancelled()
            ann_id = step["target_ann_id"]
            self._update(current_step=index, current_target_ann_id=ann_id)
            self._update_step(index, status="running", started_at=time.time())
            self._set_find_verdict(ann_id, "active")
            status = self._mock_move(objects[ann_id]["floor_xy"])
            if status != "succeeded":
                self._update_step(index, status=status)
                self._update(status="aborted", finished_at=time.time())
                return
            capture_id, path = self._capture(ann_id=ann_id)
            comparison = self.web.compare_reference(self.task["reference"], path)
            check_cancelled()
            observation = self._find_object_match_value(ann_id, f"/captures/{capture_id}.jpg", comparison, objects)
            observation.update(source_type="synthetic_replay", captured_at=time.time())
            with self.lock:
                self.task["observations"].append(observation)
                self.task["checked_ann_ids"].append(ann_id)
            self._set_find_verdict(ann_id, observation["verdict"])
            self._update_step(index, status="succeeded", finished_at=time.time())
            self.web.record("robot", "comparison", "当前点发现匹配目标" if comparison["found"] else "当前点未匹配，继续搜索",
                            found=comparison["found"], ann_id=ann_id, source_type="synthetic_replay")
            if comparison["found"]:
                self._finish_find_object_match(index, observation)
                return
        self._finish_find_object_no_match()

    def _run_welcome_mock(self):
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        pickup, destination = self.task["pickup_ann_id"], self.task["return_ann_id"]
        self._update(current_step=0, current_target_ann_id=pickup, current_stage="前往接人点")
        self._update_step(0, status="running", started_at=time.time())
        self._mock_move(objects[pickup]["floor_xy"])
        self._update_step(0, status="succeeded", finished_at=time.time())
        self._update(current_step=1, current_stage="观察候选访客，等待主人确认")
        self._update_step(1, status="running", started_at=time.time())
        while True:
            check_cancelled()
            capture_id, _ = self._capture(ann_id=pickup)
            observation = {"ann_id": pickup, "name": self.task["pickup_name"],
                           "image_url": f"/captures/{capture_id}.jpg", "found": True,
                           "text": "预设候选人物，需主人确认后继续。", "captured_at": time.time(),
                           "source_type": "synthetic_replay"}
            with self.lock:
                self.task["observations"].append(observation)
                self.task["snapshot_count"] = int(self.task.get("snapshot_count", 0)) + 1
            self.web.record("robot", "guest_candidate", "展示候选人物照片，等待主人确认")
            if self._wait_welcome_confirmation(observation):
                break
            check_cancelled()
            if not self._wait_cancellable(0.3):
                check_cancelled()
        self._update(current_step=2, current_target_ann_id=destination, current_stage="引导至送客点")
        self._update_step(2, status="running", started_at=time.time())
        self._mock_move(objects[destination]["floor_xy"])
        self._update_step(2, status="succeeded", finished_at=time.time())
        self._update(status="succeeded", current_step=None, current_target_ann_id=None,
                     result_text="已在主人确认后，将候选访客引导到送客点。", finished_at=time.time())


class ReplayWebState(RobotWebState):
    replay = True

    def __init__(self, decision_model=False, tick=0.08):
        self.decision_model = decision_model
        self.replay_tick = max(0, float(tick))
        self._replay_lock = threading.RLock()
        self._events = deque(maxlen=600)
        self._sequence = 0
        self._cases = {}
        self._fixture_references = set()
        super().__init__(mock=True)
        self.tasks = ReplayTaskManager(self)

    def close(self):
        super().close()
        if self.tasks.thread is None or not self.tasks.thread.is_alive():
            self.conversation_store.runtime_lock.close()

    def _load_initial_graph(self):
        return _normalize_graph(copy.deepcopy(GRAPH), GRAPH["name"])

    def _load_slam_image(self):
        pass  # Replay uses its own small semantic map, never the site's SLAM image.

    def current_case(self):
        task = self.tasks.snapshot() or {}
        return self._cases.get(task.get("conversation_id"), "find_object")

    def record(self, actor, kind, title, conversation_id=None, **evidence):
        task = self.tasks.snapshot() or {}
        cid = conversation_id or task.get("conversation_id")
        if not cid:
            return
        with self._replay_lock:
            self._sequence += 1
            self._events.append({"seq": self._sequence, "actor": actor, "kind": kind, "title": title,
                                 "conversation_id": cid, "task_id": task.get("id") if task.get("conversation_id") == cid else None,
                                 "time": time.time(), "evidence": evidence})

    def record_agent_tool(self, name, arguments, result, conversation_id):
        titles = {"get_skill_context": "核对技能所需的参考图和地图资源", "resolve_map_target": "核对目标名称和地图编号",
                  "plan_find_object": "生成候选搜索计划", "plan_welcome": "生成接人和送客计划",
                  "plan_inspect_location": "生成到点观察计划", "get_robot_status": "读取机器人实际执行状态",
                  "ask_user": "缺少必要资料，请求补充"}
        task = result.get("task") or {}
        self.record("agent", "tool", titles.get(name, "调用任务工具"), conversation_id,
                    tool=name, ok=result.get("ok", True), target_ids=result.get("target_ids", []),
                    default_arguments=result.get("default_arguments", {}), task_status=task.get("status"))

    def agent_complete(self, messages, tools):
        if self.decision_model:
            return AgentServiceMixin.agent_complete(self, messages, tools)
        return scripted_completion(messages, tools)

    def agent_vision(self, path, question, history):
        target = self._target_at_pose()
        return f"合成回放观察：点位 #{target} 的预设画面，仅作为任务反馈样例。"

    def _target_at_pose(self):
        pose = self.tasks.mock_pose
        return min(self.graph["objects"], key=lambda obj: sum((obj["floor_xy"][i] - pose[i]) ** 2 for i in (0, 1)))["ann_id"]

    def capture(self, media_scope=None, kind="observations", metadata=None, **_unused):
        check_cancelled()
        capture_id = uuid.uuid4().hex
        target = (metadata or {}).get("ann_id", self._target_at_pose())
        path = self.new_capture_path(capture_id, media_scope, kind, metadata)
        case = CASES.get(self.current_case(), CASES["find_object"])
        path.write_bytes(picture(case.get("reference") or "object", target=target))
        self.media_archive.record(f"/captures/{capture_id}.jpg", source_type="synthetic_replay", captured_at=time.time())
        self.record("robot", "observation", "获取到点观察样例", ann_id=target,
                    image_url=f"/captures/{capture_id}.jpg", source_type="synthetic_replay")
        return capture_id, path

    def compare_reference(self, reference, scene_path):
        check_cancelled()
        if reference.get("reference_id") not in self._fixture_references:
            return {"found": None, "confidence": "low", "reason": "离线回放仅支持随案例提供的参考图。"}
        found = self._target_at_pose() == 22
        return {"found": found, "confidence": "high", "reason": "来自预设回放标注，不是视觉模型推理。"}

    def find_target_label(self, instruction, reference):
        return "回放参考目标"

    def begin_replay(self, case_id, conversation_id):
        valid_id(conversation_id)
        if case_id not in CASES:
            raise ValueError("请选择已有的回放案例")
        if self.tasks.is_busy():
            raise RuntimeError("请等待当前任务结束或先停止任务")
        case = CASES[case_id]
        self.conversation_store.create(conversation_id, case["name"])
        turn_id = uuid.uuid4().hex
        reference = None
        if case["reference"]:
            reference = self.save_reference_upload("replay-reference.jpg", picture(case["reference"], reference=True),
                        media_scope={"conversation_id": conversation_id, "turn_id": turn_id})
            self._fixture_references.add(reference["reference_id"])
        with self._replay_lock:
            self._cases[conversation_id] = case_id
        self.record("operator", "request", case["question"], conversation_id)
        result = self.agent_chat({"question": case["question"], "conversation_id": conversation_id,
                                  "user_message_id": turn_id, "reference": reference})
        self.record("agent", "response", result["text"], conversation_id,
                    requires_confirmation=result.get("requires_confirmation", False))
        return {"result": result, "reference": reference}

    def replay_feedback(self, conversation_id):
        valid_id(conversation_id)
        task = self.conversation_task(conversation_id)
        if not task:
            raise ValueError("本会话还没有任务结果")
        result = self.agent_chat({"question": FEEDBACK_QUESTION, "conversation_id": conversation_id})
        self.record("agent", "feedback", result["text"], conversation_id,
                    execution=result.get("execution"))
        return result

    def replay_snapshot(self, conversation_id):
        if conversation_id:
            valid_id(conversation_id)
        with self._replay_lock:
            events = [copy.deepcopy(item) for item in self._events if item["conversation_id"] == conversation_id]
        return {"mode": "model" if self.decision_model else "scripted",
                "case_id": self._cases.get(conversation_id),
                "observation_source": "synthetic_replay",
                "cases": [{"id": key, **value} for key, value in CASES.items()],
                "graph": self.graph_snapshot(), "robot": self.tasks.robot_status(),
                "task": self.conversation_task(conversation_id) if conversation_id else None, "events": events}
