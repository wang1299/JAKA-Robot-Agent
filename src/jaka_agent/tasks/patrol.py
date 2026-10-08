"""Patrol baselines, repeat visits and change reporting."""
from __future__ import annotations
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import copy
import math
import time
import uuid

class TasksPatrolMixin:
    def _patrol_spec(self):
        """返回巡逻步骤序号、目标列表和轮数；网页巡逻只接受一个完整多点步骤。"""
        for index, step in enumerate(self.task.get("steps") or []):
            if step.get("type") == "patrol":
                target_ids = list(dict.fromkeys(step.get("target_ann_ids") or []))
                if len(target_ids) < 2:
                    raise RuntimeError("巡逻至少需要两个不同点位")
                count = int(step.get("count", -1))
                return index, target_ids, count
        raise RuntimeError("巡逻任务缺少 patrol 步骤")

    def _patrol_location(self, ann_id, objects):
        obj = objects[ann_id]
        return obj.get("category_zh") or obj.get("category") or f"目标{ann_id}"

    def _patrol_capture_value(self, ann_id, capture_id, round_no, objects):
        return {
            "ann_id": ann_id,
            "name": self._patrol_location(ann_id, objects),
            "capture_id": capture_id,
            "image_url": f"/captures/{capture_id}.jpg",
            "round": round_no,
            "captured_at": time.time(),
        }

    def _set_patrol_baseline(self, ann_id, value):
        tasks_runtime.check_cancelled()
        with self.lock:
            self.task.setdefault("patrol_baselines", {})[str(ann_id)] = value
            self.task.setdefault("patrol_status_by_ann", {})[str(ann_id)] = "baseline"
            if value.get("capture_pose"):
                self.task.setdefault("patrol_observation_poses", {}).setdefault(
                    str(ann_id), copy.deepcopy(value["capture_pose"]),
                )
            self._persist_task()

    @staticmethod
    def _read_patrol_pose(driver):
        tasks_runtime.check_cancelled()
        pose = tuple(driver.get_pose())
        if len(pose) != 3:
            raise RuntimeError("巡逻无法读取有效的机器人位姿，已停止观察比较")
        x, y, theta = map(float, pose)
        if not all(math.isfinite(v) for v in (x, y, theta)):
            raise RuntimeError("巡逻机器人位姿无效，已停止观察比较")
        return {"x": x, "y": y, "theta": (theta + math.pi) % (2 * math.pi) - math.pi}

    @staticmethod
    def _check_patrol_pose(actual, expected, ann_id):
        distance = math.hypot(actual["x"] - expected["x"], actual["y"] - expected["y"])
        angle = abs((actual["theta"] - expected["theta"] + math.pi) % (2 * math.pi) - math.pi)
        if distance > web_settings.PATROL_POSE_DISTANCE_TOLERANCE_M or angle > web_settings.PATROL_POSE_THETA_TOLERANCE_RAD:
            raise RuntimeError(
                f"巡逻点 #{ann_id} 观察位姿不一致：位置误差 {distance:.3f}m、"
                f"朝向误差 {math.degrees(angle):.1f}°；已停止巡逻，未进行异常比较"
            )

    def _return_to_patrol_pose(self, driver, ann_id, pose):
        """只导航到首轮保存的绝对位姿，不重新选点或面向物体中心。"""
        tasks_runtime.check_cancelled()
        if not isinstance(pose, dict) or any(
            not isinstance(pose.get(k), (int, float)) or not math.isfinite(pose[k])
            for k in ("x", "y", "theta")
        ):
            raise RuntimeError(f"巡逻点 #{ann_id} 缺少有效首轮观察位姿，请重新开始巡逻")
        driver.move_location(
            pose["x"], pose["y"], pose["theta"],
            distance_tolerance=web_settings.PATROL_POSE_DISTANCE_TOLERANCE_M,
            theta_tolerance=web_settings.PATROL_POSE_THETA_TOLERANCE_RAD,
        )
        status = driver.wait_until_settled()
        tasks_runtime.check_cancelled()
        if status == "succeeded":
            self._check_patrol_pose(self._read_patrol_pose(driver), pose, ann_id)
        with self.lock:
            self.task.setdefault("logs", []).append({
                "step_type": "navigate", "target_ann_id": ann_id,
                "detail": f"回到首轮观察位姿 ({pose['x']:.3f},{pose['y']:.3f},θ={pose['theta']:.4f})",
                "status": status,
            })
        return status

    def _capture_patrol_at_pose(self, driver, ann_id, round_no, objects, expected=None):
        """直接采集固定视角照片；不再让观察模型改变位置或朝向。"""
        tasks_runtime.check_cancelled()
        before = self._read_patrol_pose(driver)
        if expected is not None:
            self._check_patrol_pose(before, expected, ann_id)
        capture_id = uuid.uuid4().hex
        path = self._capture_path(capture_id, ann_id=ann_id, round=round_no)
        captured_at = time.time()
        driver.capture(str(path))
        tasks_runtime.check_cancelled()
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"巡逻点 #{ann_id} 未生成有效观察画面")
        after = self._read_patrol_pose(driver)
        self._check_patrol_pose(after, before, ann_id)
        if expected is not None:
            self._check_patrol_pose(after, expected, ann_id)
        self._record_capture(capture_id, captured_at=captured_at, capture_pose=before)
        current = self._patrol_capture_value(ann_id, capture_id, round_no, objects)
        current.update(capture_pose=before, captured_at=captured_at)
        return current, path

    def _record_patrol_comparison(self, baseline, current, comparison):
        tasks_runtime.check_cancelled()
        abnormal = bool(comparison.get("changed")) and comparison.get("confidence") in ("high", "medium")
        value = {
            "ann_id": current["ann_id"],
            "name": current["name"],
            "round": current["round"],
            "before_image_url": baseline["image_url"],
            "after_image_url": current["image_url"],
            "changed": bool(comparison.get("changed")),
            "abnormal": abnormal,
            "confidence": str(comparison.get("confidence") or "low"),
            "summary": str(comparison.get("summary") or "未发现明确变化。"),
            "changes": copy.deepcopy(comparison.get("changes") or []),
            "compared_at": time.time(),
        }
        with self.lock:
            self.task.setdefault("patrol_comparisons", []).append(value)
            self.task["patrol_comparisons"] = self.task["patrol_comparisons"][-30:]
            self.task.setdefault("patrol_status_by_ann", {})[str(current["ann_id"])] = (
                "anomaly" if abnormal else "normal"
            )
            if not abnormal:
                self.task.setdefault("patrol_baselines", {})[str(current["ann_id"])] = current
        return value

    def _finish_patrol_anomaly(self, step_index, value):
        tasks_runtime.check_cancelled()
        changes = "；".join(
            f"{item.get('item') or '物体'}：{item.get('detail') or item.get('type') or '发生变化'}"
            for item in value.get("changes") or []
        )
        result = f"巡逻已在{value['name']}（#{value['ann_id']}）停止：{value['summary']}"
        if changes:
            result += f" 变化明细：{changes}。"
        self._update_step(step_index, status="anomaly", finished_at=time.time())
        self._update(
            status="anomaly", anomaly=value, result_text=result,
            current_step=None, current_target_ann_id=value["ann_id"], finished_at=time.time(),
        )
        self.web.speak("发现异常")

    def _finish_patrol_normal(self, step_index, rounds):
        tasks_runtime.check_cancelled()
        result = f"已完成 {rounds} 轮多点巡逻，没有发现可信环境变化。"
        self._update_step(step_index, status="succeeded", finished_at=time.time())
        self._update(
            status="succeeded", result_text=result,
            current_step=None, current_target_ann_id=None, finished_at=time.time(),
        )

    def _run_patrol_mock(self):
        """模拟两轮多点巡逻；第二轮第二个点产生一次异常，便于离线验收完整界面。"""
        step_index, target_ids, count = self._patrol_spec()
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        max_rounds = count + 1 if count > 0 else 2
        self._update_step(step_index, status="running", started_at=time.time())
        for round_no in range(1, max_rounds + 1):
            self._update(patrol_round=round_no, current_step=step_index)
            for visit_index, ann_id in enumerate(target_ids):
                if self._canceled():
                    self._update_step(step_index, status="canceled", finished_at=time.time())
                    self._update(status="canceled", current_target_ann_id=None, finished_at=time.time())
                    return
                self._update(current_target_ann_id=ann_id)
                self.task["patrol_status_by_ann"][str(ann_id)] = "active"
                status = self._mock_move(objects[ann_id].get("nav_xy") or objects[ann_id]["floor_xy"])
                if status != "succeeded":
                    self._update_step(step_index, status=status, finished_at=time.time())
                    self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
                    return
                if round_no == 1:
                    self.web.speak(f"到达{self._patrol_location(ann_id, objects)}")
                capture_id, _ = self._capture(ann_id=ann_id, round=round_no)
                current = self._patrol_capture_value(ann_id, capture_id, round_no, objects)
                baseline = self.task.get("patrol_baselines", {}).get(str(ann_id))
                if baseline is None:
                    self._set_patrol_baseline(ann_id, current)
                    continue
                comparison = tasks_runtime.guarded_call(self.web.compare_patrol_images,
                    self.web.capture_path(baseline["capture_id"]),
                    self.web.capture_path(current["capture_id"]),
                    current["name"],
                )
                if count < 0 and round_no == 2 and visit_index == 1:
                    comparison = {
                        "changed": True, "confidence": "high",
                        "summary": "模拟发现点位内新增了一个遗留物品。",
                        "changes": [{"type": "added", "item": "纸箱", "detail": "基线图中没有，本次画面中出现"}],
                    }
                value = self._record_patrol_comparison(baseline, current, comparison)
                if value["abnormal"]:
                    self._finish_patrol_anomaly(step_index, value)
                    return
                self.web.speak("无异常")
        self._finish_patrol_normal(step_index, max_rounds)

    def _run_patrol_hardware(self):
        """逐点导航并维护每个点的视觉基线；发现可信变化后立即停止巡逻。"""
        from jaka_agent.tasks.executor import PlanExecutor

        step_index, target_ids, count = self._patrol_spec()
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        driver = self._task_camera_driver()
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)
        self._update_step(step_index, status="running", started_at=time.time())
        round_no = 0
        while count < 0 or round_no < count + 1:
            round_no += 1
            self._update(patrol_round=round_no, current_step=step_index)
            for ann_id in target_ids:
                if self._canceled():
                    self._update_step(step_index, status="canceled", finished_at=time.time())
                    self._update(status="canceled", current_target_ann_id=None, finished_at=time.time())
                    return
                self._update(current_target_ann_id=ann_id)
                with self.lock:
                    self.task["patrol_status_by_ann"][str(ann_id)] = "active"
                before_log = len(executor.log)
                fixed_pose = None
                if round_no == 1:
                    status = executor._do_navigate({
                        "type": "navigate", "target_ann_id": ann_id,
                        "use_viewpoint": bool(objects[ann_id].get("viewpoint")),
                        "_announce_arrival": False,
                    })
                else:
                    fixed_pose = copy.deepcopy(self.task.get("patrol_observation_poses", {}).get(str(ann_id)))
                    status = self._return_to_patrol_pose(driver, ann_id, fixed_pose)
                self._append_executor_logs(executor, before_log)
                if status != "succeeded":
                    self._update_step(step_index, status=status, finished_at=time.time())
                    if status in ("failed", "timeout"):
                        try:
                            driver.cancel_move()
                        except Exception:
                            pass
                    self._update(
                        status="canceled" if status == "canceled" else "aborted",
                        error=f"巡逻点 #{ann_id} 导航状态: {status}", finished_at=time.time(),
                    )
                    return

                if round_no == 1:
                    self.web.speak(f"到达{self._patrol_location(ann_id, objects)}")
                    before_observe = len(executor.log)
                    observe_status = executor._do_observe({
                        "type": "observe", "target_ann_id": ann_id,
                        "question": "确认巡逻目标在画面中，并拍摄适合与历史基线比较的现场环境。",
                        "fine_adjust": True, "_announce_observation": False,
                    })
                    self._append_executor_logs(executor, before_observe)
                    if observe_status != "succeeded":
                        self._update_step(step_index, status=observe_status, finished_at=time.time())
                        self._update(
                            status="error", current_target_ann_id=ann_id,
                            error=f"巡逻点 #{ann_id} 到点后未找到目标，已停止巡逻。",
                            finished_at=time.time(),
                        )
                        return
                current, capture_path = self._capture_patrol_at_pose(
                    driver, ann_id, round_no, objects, expected=fixed_pose,
                )
                baseline = self.task.get("patrol_baselines", {}).get(str(ann_id))
                if baseline is None:
                    self._set_patrol_baseline(ann_id, current)
                    continue
                comparison = tasks_runtime.guarded_call(self.web.compare_patrol_images,
                    self.web.capture_path(baseline["capture_id"]), capture_path, current["name"],
                )
                value = self._record_patrol_comparison(baseline, current, comparison)
                if value["abnormal"]:
                    self._finish_patrol_anomaly(step_index, value)
                    return
                self.web.speak("无异常")
        self._finish_patrol_normal(step_index, round_no)
