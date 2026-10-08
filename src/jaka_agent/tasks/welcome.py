"""Visitor waiting, owner photo confirmation and escort workflow."""
from __future__ import annotations
import jaka_agent.diagnostics as diagnostics
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import copy
import re
import shutil
import time
import uuid

class TasksWelcomeMixin:
    @staticmethod
    def _welcome_voice_intent(text: str) -> str:
        """将一次迎宾回答分为肯定、否定或未听清；否定词必须优先于“是”。"""
        normalized = re.sub(r"[\s，。！？、,.!?；;：:]+", "", str(text or "").lower())
        if not normalized:
            return "unclear"
        if any(word in normalized for word in web_settings.WELCOME_NEGATIVE_WORDS):
            return "negative"
        if any(word in normalized for word in web_settings.WELCOME_AFFIRMATIVE_WORDS):
            return "affirmative"
        return "unclear"

    def _wait_cancellable(self, seconds: float) -> bool:
        """等待指定时间；任务被取消时提前返回 False。"""
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if self._canceled():
                return False
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        return not self._canceled()

    def confirm_welcome_guest(self, task_id, confirmation_id, accepted, conversation_id):
        """Only the owning conversation may decide on the currently displayed photo."""
        if type(accepted) is not bool:
            raise ValueError("请明确确认或拒绝本次人物识别")
        with self.lock:
            task = self.task or {}
            pending = task.get("guest_confirmation") or {}
            if task.get("id") != task_id or task.get("kind") != "welcome":
                raise ValueError("迎宾任务不存在或已被替换")
            if task.get("conversation_id") != conversation_id:
                raise ValueError("请在发起迎宾任务的会话中确认")
            if (task.get("status") != "running" or task.get("cancel_requested")
                    or task.get("current_step") != 1):
                raise ValueError("当前任务不再等待人物确认")
            if not confirmation_id or pending.get("id") != confirmation_id or pending.get("status") != "pending":
                raise ValueError("这张照片的确认已处理或已失效，请刷新任务")
            pending.update(status="accepted" if accepted else "rejected", decided_at=time.time())
            task["current_stage"] = "主人已确认，准备前往送客点" if accepted else "主人拒绝本次识别，继续等待目标人物"
            self._persist_task()
            diagnostics.LOGGER.info("[welcome] web confirmation task_id=%s accepted=%s", task_id, accepted)
            return self.snapshot()

    def _wait_welcome_confirmation(self, observation):
        """Pause detection and motion until a photo-specific web decision or cancellation."""
        tasks_runtime.check_cancelled()
        if not observation.get("image_url"):
            raise RuntimeError("缺少现场照片，不能请求迎宾确认")
        confirmation_id = uuid.uuid4().hex
        with self.lock:
            if self._canceled():
                return False
            task_id = self.task["id"]
            self._update(guest_confirmation={
                "id": confirmation_id, "status": "pending",
                "image_url": observation["image_url"],
                "captured_at": observation.get("captured_at"), "requested_at": time.time(),
            }, current_stage="发现疑似目标客人，等待主人在网页确认")
        self.web.speak("小卡识别到目标客人，等待主人确认中")
        while True:
            with self.lock:
                task = self.task or {}
                pending = task.get("guest_confirmation") or {}
                if (task.get("id") != task_id or task.get("cancel_requested")
                        or task.get("status") != "running" or pending.get("id") != confirmation_id):
                    return False
                if pending.get("status") == "accepted":
                    self._update_step(1, status="succeeded", finished_at=time.time())
                    return True
                if pending.get("status") == "rejected":
                    self._update(rejected_count=int(task.get("rejected_count") or 0) + 1)
                    return False
            if not self._wait_cancellable(0.2):
                return False

    def _capture_welcome_snapshot(self, driver, objects):
        """迎宾点抓拍一帧并计分；高置信命中后发布照片供主人确认。"""
        task = self.snapshot() or {}
        ann_id = int(task["pickup_ann_id"])
        capture_id = uuid.uuid4().hex
        capture_path = None
        task_id = task.get("id")
        try:
            if driver is None:
                capture_id, capture_path = self._capture('snapshots', ann_id=ann_id, stage='迎宾点人物检测')
            else:
                capture_path = self._capture_path(capture_id, 'snapshots', ann_id=ann_id, stage='迎宾点人物检测')
                driver.capture(str(capture_path))
        except tasks_runtime.TaskCancelled:
            raise
        except Exception as exc:
            diagnostics._log_exception(
                "[welcome] camera capture failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                capture_path=str(capture_path),
            )
            raise RuntimeError(f"迎宾抓拍失败: {exc}") from exc

        self._record_capture(capture_id, captured_at=time.time())
        try:
            comparison = tasks_runtime.guarded_call(self.web.compare_person_reference, task.get("reference") or {}, capture_path)
        except tasks_runtime.TaskCancelled:
            raise
        except Exception as exc:
            diagnostics._log_exception(
                "[welcome] person comparison failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                capture_path=str(capture_path),
            )
            raise RuntimeError(f"迎宾人物识别失败: {exc}") from exc

        obj = objects[ann_id]
        xy = obj.get("nav_xy") or obj["floor_xy"]
        matched = (
            bool(comparison.get("found"))
            and str(comparison.get("confidence") or "").lower() == "high"
        )
        observation = {
            "ann_id": ann_id,
            "name": obj.get("category_zh") or obj.get("category") or f"目标{ann_id}",
            "x": float(xy[0]),
            "y": float(xy[1]),
            "image_url": f"/captures/{capture_id}.jpg",
            "text": (
                "现场发现与参考图外观相似的人物，请主人核对照片并确认。"
                if matched
                else "本次抓拍未确认发现目标人物。"
            ),
            "found": bool(comparison.get("found")),
            "confidence": str(comparison.get("confidence") or "low"),
            "match_score": float(comparison.get("score") or 0.0),
            "match_breakdown": copy.deepcopy(comparison.get("comparisons") or {}),
            "contradictions": copy.deepcopy(comparison.get("contradictions") or []),
            "candidate_region": str(comparison.get("candidate_region") or ""),
            "model_reason": str(comparison.get("reason") or "")[:500],
            "captured_at": time.time(),
            "stage": "迎宾点人物检测",
        }
        self._record_capture(capture_id, analysis=comparison)
        with self.lock:
            if self.task:
                tasks_runtime.check_cancelled()
                self.task["snapshot_count"] = int(self.task.get("snapshot_count") or 0) + 1
                if matched:
                    self.task.setdefault("observations", []).append(observation)
                    self.task["observations"] = self.task["observations"][-12:]
        diagnostics.LOGGER.info(
            "[welcome] person probe task_id=%r ann_id=%r found=%s confidence=%r "
            "score=%.2f schema_valid=%s publish=%s image=%s",
            task_id,
            ann_id,
            bool(comparison.get("found")),
            comparison.get("confidence"),
            float(comparison.get("score") or 0.0),
            bool(comparison.get("schema_valid")),
            matched,
            capture_path,
        )
        return matched, observation, comparison

    def _run_welcome_mock(self):
        objects = {int(obj["ann_id"]): obj for obj in self._objects()}
        pickup_ann_id = int(self.task["pickup_ann_id"])
        return_ann_id = int(self.task["return_ann_id"])
        self._update(current_step=0, current_target_ann_id=pickup_ann_id, current_stage="前往接人点")
        self._update_step(0, status="running", started_at=time.time())
        pickup_xy = objects[pickup_ann_id].get("nav_xy") or objects[pickup_ann_id]["floor_xy"]
        status = self._mock_move(pickup_xy)
        self._update_step(0, status=status, finished_at=time.time())
        if status != "succeeded":
            self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
            return

        self._update(current_step=1, current_stage="已到接人点，等待目标人物")
        self._update_step(1, status="running", started_at=time.time())
        accepted = False
        while not self._canceled() and not accepted:
            if not self._wait_cancellable(web_settings.WELCOME_SNAPSHOT_INTERVAL_SECONDS):
                break
            source = web_settings.HERE / "shot.jpg"
            if not source.exists():
                reference = self.task.get("reference") or {}
                source = self.web.reference_path(reference["reference_id"], reference["suffix"])
            capture_id = uuid.uuid4().hex
            capture_path = self._capture_path(capture_id, 'snapshots', stage='模拟迎宾检测')
            shutil.copyfile(source, capture_path)
            self._record_capture(capture_id)
            observation = {
                "ann_id": pickup_ann_id,
                "name": self.task.get("pickup_name"),
                "image_url": f"/captures/{capture_id}.jpg",
                "text": "模拟发现目标人物，等待主人在网页确认。",
                "found": True,
                "confidence": "high",
                "stage": "迎宾点人物检测",
                "captured_at": time.time(),
            }
            with self.lock:
                self.task["snapshot_count"] = int(self.task.get("snapshot_count") or 0) + 1
                self.task["observations"].append(observation)
            accepted = self._wait_welcome_confirmation(observation)
        if self._canceled():
            self._update_step(1, status="canceled", finished_at=time.time())
            self._update(status="canceled", finished_at=time.time())
            return

        self.web.speak("好的，请跟我来。")
        self._update(current_step=2, current_target_ann_id=return_ann_id, current_stage="带领客人返回")
        self._update_step(2, status="running", started_at=time.time())
        return_xy = objects[return_ann_id].get("nav_xy") or objects[return_ann_id]["floor_xy"]
        status = self._mock_move(return_xy)
        self._update_step(2, status=status, finished_at=time.time())
        if status == "succeeded":
            result = (
                f"已在{self.task.get('pickup_name')}接到目标人物，"
                f"并带领其返回{self.task.get('return_name')}。"
            )
            self._update(
                status="succeeded",
                current_step=None,
                current_target_ann_id=None,
                current_stage="迎宾任务完成",
                result_text=result,
                finished_at=time.time(),
            )
        else:
            self._update(
                status="canceled" if status == "canceled" else "aborted",
                error=status,
                finished_at=time.time(),
            )

    def _run_welcome_navigation(self, executor, step_index, ann_id, stage):
        """执行迎宾任务的一段导航，并同步任务步骤和执行日志。"""
        self._update(
            current_step=step_index,
            current_target_ann_id=ann_id,
            current_stage=stage,
        )
        self._update_step(step_index, status="running", started_at=time.time())
        before = len(executor.log)
        status = executor._do_navigate({
            "type": "navigate",
            "target_ann_id": ann_id,
            "use_viewpoint": False,
        })
        self._append_executor_logs(executor, before)
        self._update_step(step_index, status=status, finished_at=time.time())
        return status

    def _run_welcome_hardware(self):
        """前往接人点识别人像；主人在网页确认现场照片后前往送客点。"""
        from jaka_agent.tasks.executor import PlanExecutor

        objects = {int(obj["ann_id"]): obj for obj in self._objects()}
        pickup_ann_id = int(self.task["pickup_ann_id"])
        return_ann_id = int(self.task["return_ann_id"])
        driver = self._task_camera_driver()
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)

        status = self._run_welcome_navigation(
            executor, 0, pickup_ann_id, "前往接人点"
        )
        if status != "succeeded":
            if status in ("failed", "timeout"):
                try:
                    driver.cancel_move()
                except Exception:
                    pass
            self._update(
                status="canceled" if status == "canceled" else "aborted",
                error=status,
                finished_at=time.time(),
            )
            return
        if self._canceled():
            self._update(status="canceled", finished_at=time.time())
            return

        self._update(
            current_step=1,
            current_target_ann_id=pickup_ann_id,
            current_stage="已到接人点，等待目标人物",
        )
        self._update_step(1, status="running", started_at=time.time())
        consecutive_errors = 0
        accepted = False
        if not self._wait_cancellable(web_settings.WELCOME_SNAPSHOT_INTERVAL_SECONDS):
            self._update_step(1, status="canceled", finished_at=time.time())
            self._update(status="canceled", finished_at=time.time())
            return
        while not self._canceled() and not accepted:
            cycle_started = time.monotonic()
            try:
                matched, _observation, _comparison = self._capture_welcome_snapshot(
                    driver, objects
                )
                consecutive_errors = 0
                self._update(last_detection_error=None)
            except Exception as exc:
                consecutive_errors += 1
                self._update(
                    current_stage="人物识别暂时失败，正在继续重试",
                    last_detection_error=str(exc),
                )
                diagnostics.LOGGER.warning(
                    "[welcome] person probe retry task_id=%r errors=%d error=%s",
                    (self.snapshot() or {}).get("id"),
                    consecutive_errors,
                    exc,
                )
                matched = False

            if self._canceled():
                break
            if matched:
                accepted = self._wait_welcome_confirmation(_observation)
                if accepted:
                    break

            remaining = web_settings.WELCOME_SNAPSHOT_INTERVAL_SECONDS - (
                time.monotonic() - cycle_started
            )
            if remaining > 0 and not self._wait_cancellable(remaining):
                break

        if self._canceled():
            self._update_step(1, status="canceled", finished_at=time.time())
            self._update(
                status="canceled",
                current_stage="迎宾任务已取消",
                finished_at=time.time(),
            )
            return

        self.web.speak("好的，请跟我来。")
        status = self._run_welcome_navigation(
            executor, 2, return_ann_id, "带领客人返回"
        )
        if status != "succeeded":
            if status in ("failed", "timeout"):
                try:
                    driver.cancel_move()
                except Exception:
                    pass
            self._update(
                status="canceled" if status == "canceled" else "aborted",
                error=status,
                finished_at=time.time(),
            )
            return

        result = (
            f"已在{self.task.get('pickup_name')}接到目标人物，"
            f"并带领其返回{self.task.get('return_name')}。"
        )
        self._update(
            status="succeeded",
            current_step=None,
            current_target_ann_id=None,
            current_stage="迎宾任务完成",
            result_text=result,
            finished_at=time.time(),
        )
        self.web.speak("已到达，请慢走。")
