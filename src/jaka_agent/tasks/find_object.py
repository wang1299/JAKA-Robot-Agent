"""Reference-image search, candidate visits and observation recovery."""
from __future__ import annotations
import jaka_agent.diagnostics as diagnostics
import jaka_agent.mapping.graphs as mapping_graphs
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import copy
import re
import threading
import time
import uuid
from contextvars import copy_context

class TasksFindObjectMixin:
    @staticmethod
    def _public_find_observation(comparison):
        """将模型内部比对说明转换成用户能直接理解的结果，不暴露拼图布局。"""
        if comparison.get("found") is True:
            return "已在当前现场确认找到与参考图外观一致的目标物品。"
        if comparison.get("found") is not False:
            return "当前照片识别结果格式错误，无法判断是否找到参考图中的物品。"
        return "未在当前现场发现与参考图外观匹配的目标物品。"

    def _find_object_match_value(self, ann_id, image_url, comparison, objects):
        obj = objects[ann_id]
        xy = obj.get("nav_xy") or obj["floor_xy"]
        name = obj.get("category_zh") or obj.get("category") or f"目标{ann_id}"
        return {
            "ann_id": ann_id,
            "name": name,
            "position": obj.get("position") or "",
            "x": float(xy[0]),
            "y": float(xy[1]),
            "image_url": image_url,
            "text": self._public_find_observation(comparison),
            "model_reason": str(comparison.get("reason") or "").strip()[:500],
            "confidence": str(comparison.get("confidence") or ""),
            "found": comparison.get("found") if type(comparison.get("found")) is bool else None,
            "verdict": ("match" if comparison.get("found") is True else
                        "miss" if comparison.get("found") is False else "uncertain"),
        }

    def _set_find_verdict(self, ann_id, verdict):
        """把寻物核验结论同步到任务步骤、候选状态和地图路线。"""
        with self.lock:
            if not self.task:
                return
            self.task.setdefault("candidate_status", {})[str(ann_id)] = verdict
            for point in self.task.get("route_points") or []:
                if str(point.get("ann_id")) == str(ann_id):
                    point["verdict"] = verdict

    def _announce_find_observation(self, observation, comparison):
        tasks_runtime.check_cancelled()
        location = str(observation.get("name") or "当前点位")
        location = mapping_graphs.CATEGORY_ZH_MAP.get(location.strip().lower(), location)
        if re.search(r'[A-Za-z]', location) or not re.search(r'[\u4e00-\u9fff]', location):
            location = '当前物体'
        where = f"前往{location}途中" if observation.get("in_transit") else f"到达{location}"
        if comparison.get("found") is True:
            self.web.speak(f"{where}，找到参考图中的物体。")
        elif comparison.get("found") is False:
            self.web.speak(f"{where}，未找到参考图中的物体。")
        else:
            self.web.speak(f"{where}，暂时无法确认参考图中的物体。")

    def _finish_find_object_match(self, index, match):
        tasks_runtime.check_cancelled()
        if match.get("in_transit"):
            result_text = (
                f"已在前往{match['name']}途中找到目标（抓拍坐标 "
                f"{match['x']:.2f}, {match['y']:.2f}）。"
            )
        else:
            result_text = (
                f"已找到目标，位于{match['name']}（#{match['ann_id']}，"
                f"坐标 {match['x']:.2f}, {match['y']:.2f}；{match['position']}）。"
            )
        with self.lock:
            for later in self.task["steps"][index + 1:]:
                if later.get("status") == "pending":
                    later.update({"status": "skipped", "finished_at": time.time()})
                    ann_id = later.get("target_ann_id")
                    self.task.setdefault("candidate_status", {})[str(ann_id)] = "skipped"
                    for point in self.task.get("route_points") or []:
                        if str(point.get("ann_id")) == str(ann_id):
                            point["verdict"] = "skipped"
            self.task.update({
                "match": match,
                "result_text": result_text,
                "status": "succeeded",
                "current_step": None,
                "current_target_ann_id": None,
                "finished_at": time.time(),
            })

    def _finish_find_object_no_match(self):
        with self.lock:
            tasks_runtime.check_cancelled()
            checked = len(self.task.get("checked_ann_ids") or [])
            uncertain = sum(value == "uncertain" for value in self.task.get("candidate_status", {}).values())
            if uncertain:
                result_text = (
                    f"已检查 {checked} 个候选点，其中 {uncertain} 个点的识别结果无效；"
                    "其余点未找到目标，但不能确认目标不存在。"
                )
            else:
                result_text = f"已完成 {checked} 个候选点的行进中抓拍与到点核验，仍未发现可信匹配。"
            self.task.update({
                "result_text": result_text,
                "status": "inconclusive" if uncertain else "not_found",
                "current_step": None,
                "current_target_ann_id": None,
                "current_search_stage": None,
                "finished_at": time.time(),
            })

    def _run_find_object_mock(self):
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        for index, step in enumerate(self.task["steps"]):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            ann_id = step["target_ann_id"]
            diagnostics.LOGGER.info(
                "[find] candidate start task_id=%r index=%d/%d ann_id=%r name=%r",
                (self.snapshot() or {}).get("id"), index + 1, len(self.task["steps"]),
                ann_id, step.get("target_names"),
            )
            self._update(current_step=index, current_target_ann_id=ann_id)
            self._set_find_verdict(ann_id, "active")
            self._update_step(index, status="running", started_at=time.time())
            stop_event = threading.Event()
            probe_result = {}
            probe_thread = self._start_reference_snapshot_loop(
                None, ann_id, objects, stop_event, probe_result, stage="行进中抓拍检测",
            )
            try:
                status = self._mock_move(objects[ann_id]["floor_xy"])
            finally:
                stop_event.set()
                probe_thread.join(timeout=30.0)
                if probe_thread.is_alive():
                    probe_result.setdefault("error", "行进中抓拍线程未能在 30 秒内停止")
            if status != "succeeded":
                diagnostics.LOGGER.error(
                    "[find] navigation failed task_id=%r ann_id=%r status=%r",
                    (self.snapshot() or {}).get("id"), ann_id, status,
                )
                self._update_step(index, status=status, finished_at=time.time())
                self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
                return
            if probe_result.get("error"):
                raise RuntimeError(str(probe_result["error"]))
            match = probe_result.get("match")
            if match is not None:
                with self.lock:
                    self.task["checked_ann_ids"].append(ann_id)
                self._set_find_verdict(ann_id, "match")
                self._update_step(index, status="succeeded", finished_at=time.time())
                self._announce_find_observation(match, probe_result.get("comparison") or match)
                self._finish_find_object_match(index, match)
                return
            if self._canceled():
                self._update_step(index, status="canceled", finished_at=time.time())
                self._update(status="canceled", current_step=None, current_target_ann_id=None, finished_at=time.time())
                return
            match, observation, canceled = self._search_reference_at_point(None, ann_id, objects)
            if canceled:
                self._update_step(index, status="canceled", finished_at=time.time())
                self._update(status="canceled", current_step=None, current_target_ann_id=None, finished_at=time.time())
                return
            if observation is None:
                raise RuntimeError(f"在候选点 #{ann_id} 没有生成有效寻物画面")
            with self.lock:
                self.task["checked_ann_ids"].append(ann_id)
            self._set_find_verdict(ann_id, observation["verdict"])
            self._update_step(index, status="uncertain" if observation["verdict"] == "uncertain" else "succeeded", finished_at=time.time())
            self._announce_find_observation(observation, observation)
            if match is not None:
                self._finish_find_object_match(index, match)
                return
        self._finish_find_object_no_match()

    def _capture_reference_snapshot(self, driver, reference, ann_id, objects, stage, publish_mode="none", **motion):
        """抓拍一帧并与参考图比对；publish_mode 控制是否进入前端任务流。

        - match: 单帧命中时立即入前端
        - always: 不论命中与否都入前端（用于到点后的正式核验）
        - none: 暂不写前端（用于行进中累计连续命中）
        """
        self._update(current_search_stage=stage)
        capture_id = uuid.uuid4().hex
        kind = 'observations' if publish_mode == 'always' else 'snapshots'
        capture_path = None
        task_id = (self.snapshot() or {}).get("id")
        captured_pose = None
        try:
            if driver is None:
                capture_id, capture_path = self._capture(kind, ann_id=ann_id, stage=stage, **motion)
            else:
                capture_path = self._capture_path(capture_id, kind, ann_id=ann_id, stage=stage, **motion)
                driver.capture(str(capture_path))
                if str(stage).startswith("行进中"):
                    try:
                        captured_pose = tuple(map(float, driver.get_pose()))
                    except Exception as exc:
                        diagnostics.LOGGER.warning(
                            "[snapshot] failed to read capture pose task_id=%r ann_id=%r error=%s",
                            task_id, ann_id, exc,
                        )
        except tasks_runtime.TaskCancelled:
            raise
        except Exception as exc:
            diagnostics._log_exception(
                "[snapshot] camera capture failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                stage=stage,
                capture_path=str(capture_path),
            )
            raise RuntimeError(f"reference snapshot camera capture failed at ann_id={ann_id}: {exc}") from exc
        self._record_capture(capture_id, captured_at=time.time(), capture_pose=captured_pose)
        # 视觉比对只接收参考图本身，不传物品名称或地图位置，避免模型被
        # “水杯”“白板附近”等文字先验诱导成类别匹配或定向猜测。
        model_reference = dict(reference or {})
        try:
            try:
                comparison = tasks_runtime.guarded_call(self.web.compare_reference, model_reference, capture_path)
            except ValueError as exc:
                if "寻物识别结果格式异常" not in str(exc):
                    raise
                diagnostics.LOGGER.warning("[snapshot] invalid reference result; retrying same photos task_id=%r ann_id=%r", task_id, ann_id)
                try:
                    comparison = tasks_runtime.guarded_call(self.web.compare_reference, model_reference, capture_path)
                except ValueError as retry_exc:
                    if "寻物识别结果格式异常" not in str(retry_exc):
                        raise
                    diagnostics.LOGGER.warning("[snapshot] invalid reference result after retry; verdict unknown task_id=%r ann_id=%r", task_id, ann_id)
                    comparison = {
                        "found": None, "confidence": "low", "schema_valid": False,
                        "reason": "同一组照片的两次识别结果均格式错误，无法判断是否找到。",
                    }
        except tasks_runtime.TaskCancelled:
            raise
        except Exception as exc:
            diagnostics._log_exception(
                "[snapshot] reference comparison failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                stage=stage,
                capture_path=str(capture_path),
                reference_id=(reference or {}).get("reference_id"),
            )
            raise RuntimeError(f"reference snapshot vision comparison failed at ann_id={ann_id}: {exc}") from exc
        observation = self._find_object_match_value(
            ann_id, f"/captures/{capture_id}.jpg", comparison, objects,
        )
        if captured_pose is not None:
            pose_x, pose_y, pose_theta = captured_pose
            observation.update({
                "in_transit": True,
                "capture_pose": {"x": pose_x, "y": pose_y, "theta": pose_theta},
                "x": pose_x,
                "y": pose_y,
                "position": f"行进中抓拍坐标({pose_x:.2f},{pose_y:.2f})",
            })
        matched = observation["verdict"] == "match"
        self._record_capture(capture_id, analysis=comparison, matched=matched)
        attempt = {
            **copy.deepcopy(observation),
            "stage": stage,
            "captured_at": time.time(),
            **motion,
        }
        with self.lock:
            if self.task:
                self.task["snapshot_count"] = int(self.task.get("snapshot_count") or 0) + 1
                should_publish = publish_mode == "always" or (publish_mode == "match" and matched)
                tasks_runtime.check_cancelled()
                if should_publish:
                    self.task.setdefault("search_attempts", []).append(attempt)
                    self.task["search_attempts"] = self.task["search_attempts"][-12:]
        diagnostics.LOGGER.info(
            "[snapshot] reference probe task_id=%r ann_id=%r stage=%r found=%s confidence=%r publish=%s image=%s",
            task_id, ann_id, stage, bool(comparison.get("found")),
            comparison.get("confidence"), bool(publish_mode == "always" or (publish_mode == "match" and matched)), capture_path,
        )
        return {
            "capture_id": capture_id,
            "capture_path": capture_path,
            "observation": observation,
            "comparison": comparison,
            "matched": matched,
            "attempt": attempt,
        }

    def _start_reference_snapshot_loop(self, driver, ann_id, objects, stop_event, result, stage="行进中抓拍检测"):
        """后台周期抓拍；明确命中即发布，到点交接保留在途结果，取消仍丢弃。"""
        reference = copy.deepcopy(self.task.get("reference") or {})
        task_id = (self.snapshot() or {}).get("id")

        def worker():
            consecutive_errors = 0
            # 线程由移动命令发送后的回调启动；再等满一个采样周期，让首帧确实来自途中。
            if stop_event.wait(web_settings.FIND_SNAPSHOT_INTERVAL_SECONDS):
                return
            while not stop_event.is_set() and not self._canceled():
                cycle_started = time.monotonic()
                try:
                    probe = self._capture_reference_snapshot(
                        driver, reference, ann_id, objects, stage, publish_mode="none",
                    )
                except tasks_runtime.TaskCancelled:
                    break
                except Exception as exc:
                    if stop_event.is_set() or self._canceled():
                        break
                    consecutive_errors += 1
                    diagnostics.LOGGER.warning(
                        "[snapshot] reference probe failed task_id=%r ann_id=%r errors=%d/%d error=%s",
                        task_id, ann_id, consecutive_errors, web_settings.FIND_SNAPSHOT_MAX_ERRORS, exc,
                    )
                    if consecutive_errors >= web_settings.FIND_SNAPSHOT_MAX_ERRORS:
                        result["error"] = (
                            f"连续 {consecutive_errors} 次抓拍识别失败，已停止本次寻物：{exc}"
                        )
                        stop_event.set()
                        try:
                            if driver is not None:
                                driver.cancel_move()
                        except Exception:
                            pass
                        break
                else:
                    consecutive_errors = 0
                    # 到点 stop_event 只停止新增采集，不否决已经在处理的命中。
                    # 取消、失败或任务替换则不能发布结果或触发后续动作。
                    if self._canceled():
                        break
                    with self.lock:
                        if (not self.task or self.task.get("id") != task_id
                                or self.task.get("status") != "running"):
                            break
                    if stop_event.is_set() and not probe.get("matched"):
                        break
                    if probe.get("matched"):
                        with self.lock:
                            tasks_runtime.check_cancelled()
                            if (self._canceled() or not self.task
                                    or self.task.get("id") != task_id
                                    or self.task.get("status") != "running"):
                                break
                            if self.task:
                                self.task.setdefault("search_attempts", []).append(probe["attempt"])
                                self.task["search_attempts"] = self.task["search_attempts"][-12:]
                        diagnostics.LOGGER.info(
                            "[snapshot] single-frame match confirmed task_id=%r ann_id=%r "
                            "publish=True image=%s",
                            task_id, ann_id, probe["capture_path"],
                        )
                        already_stopped = stop_event.is_set()
                        result["match"] = probe["observation"]
                        result["comparison"] = probe["comparison"]
                        stop_event.set()
                        try:
                            if driver is not None and not already_stopped:
                                driver.cancel_move()
                        except Exception:
                            pass
                        break
                remaining = web_settings.FIND_SNAPSHOT_INTERVAL_SECONDS - (time.monotonic() - cycle_started)
                if remaining > 0:
                    stop_event.wait(remaining)

        context = copy_context()
        def run_worker():
            try:
                context.run(worker)
            except tasks_runtime.TaskCancelled:
                pass
        thread = threading.Thread(target=run_worker, daemon=True, name=f"FindSnapshot-{ann_id}")
        with self.lock:
            tasks_runtime.check_cancelled()
            self._capture_workers.append((stop_event, thread))
            thread.start()
        return thread

    def _search_reference_at_point(self, driver, ann_id, objects):
        """到达目标点后做一次正式核验；由 observations 在前端只展示一次。"""
        if self._canceled():
            return None, None, True
        probe = self._capture_reference_snapshot(
            driver, self.task.get("reference") or {}, ann_id, objects,
            "到达点目标核验", publish_mode="always",
        )
        tasks_runtime.check_cancelled()
        observation = probe["observation"]
        comparison = probe["comparison"]
        matched = bool(probe["matched"])
        return observation if matched else None, observation, False

    def _run_find_object_hardware(self):
        """真机按拟物体地图候选点逐点导航，每个候选点只拍一张核验参考物。"""
        from jaka_agent.tasks.executor import PlanExecutor

        objects = {obj["ann_id"]: obj for obj in self._objects()}
        driver = self._task_camera_driver()
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)
        for index, step in enumerate(self.task["steps"]):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            ann_id = step["target_ann_id"]
            self._update(current_step=index, current_target_ann_id=ann_id)
            self._set_find_verdict(ann_id, "active")
            self._update_step(index, status="running", started_at=time.time())
            before = len(executor.log)
            stop_event = threading.Event()
            probe_result = {}
            probe_thread = None

            def start_transit_snapshots_after_move_command():
                nonlocal probe_thread
                self._resume_video_recording()
                self._update(current_search_stage="行进中抓拍检测")
                probe_thread = self._start_reference_snapshot_loop(
                    driver, ann_id, objects, stop_event, probe_result,
                    stage="行进中抓拍检测",
                )

            def stop_transit_snapshots_before_facing():
                stop_event.set()
                if probe_thread is not None:
                    probe_thread.join(timeout=30.0)
                if probe_thread is not None and probe_thread.is_alive():
                    raise RuntimeError("行进中抓拍线程未能在 30 秒内停止，已取消到点朝向修正")
                if not self._pause_video_recording():
                    raise RuntimeError("任务录像线程未能停止相机抓帧，已取消到点朝向修正")
                self._update(current_search_stage="到达目标点，正在正对目标")

            try:
                status = executor._do_navigate({
                    "type": "navigate", "target_ann_id": ann_id, "use_viewpoint": False,
                    "_announce_arrival": False,
                    "_on_motion_started": start_transit_snapshots_after_move_command,
                    "_on_translation_arrived": stop_transit_snapshots_before_facing,
                })
            finally:
                stop_event.set()
                if probe_thread is not None:
                    probe_thread.join(timeout=30.0)
                if probe_thread is not None and probe_thread.is_alive():
                    probe_result.setdefault("error", "行进中抓拍线程未能在 30 秒内停止")
            with self.lock:
                self.task["logs"].extend([
                    {
                        "step_type": item.step_type,
                        "target_ann_id": item.target_ann_id,
                        "detail": item.detail,
                        "observation": item.observation,
                        "status": item.status,
                    }
                    for item in executor.log[before:]
                ])
            tasks_runtime.check_cancelled()
            if probe_result.get("error"):
                self._update_step(index, status="failed", finished_at=time.time())
                self._update(
                    status="aborted", error=str(probe_result["error"]),
                    current_step=None, current_target_ann_id=None, current_search_stage=None,
                    finished_at=time.time(),
                )
                return
            match = probe_result.get("match")
            if match is not None:
                with self.lock:
                    self.task["checked_ann_ids"].append(ann_id)
                self._set_find_verdict(ann_id, "match")
                self._update_step(index, status="succeeded", finished_at=time.time())
                self._announce_find_observation(match, probe_result.get("comparison") or match)
                self._finish_find_object_match(index, match)
                return
            if status != "succeeded":
                self._update_step(index, status=status, finished_at=time.time())
                if status in ("failed", "timeout"):
                    try:
                        driver.cancel_move()
                    except Exception:
                        pass
                self._update(status="canceled" if status == "canceled" else "aborted", error=status, finished_at=time.time())
                return
            diagnostics.LOGGER.info("[find] navigation succeeded task_id=%r ann_id=%r", (self.snapshot() or {}).get("id"), ann_id)
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            match, observation, canceled = self._search_reference_at_point(driver, ann_id, objects)
            if canceled:
                self._update(status="canceled", finished_at=time.time())
                return
            if observation is None:
                raise RuntimeError(f"在候选点 #{ann_id} 没有生成有效寻物画面")
            with self.lock:
                self.task["checked_ann_ids"].append(ann_id)
            self._set_find_verdict(ann_id, observation["verdict"])
            self._update_step(index, status="uncertain" if observation["verdict"] == "uncertain" else "succeeded", finished_at=time.time())
            self._announce_find_observation(observation, observation)
            if match is not None:
                self._finish_find_object_match(index, match)
                return
        self._update(current_search_stage=None)
        self._finish_find_object_no_match()
