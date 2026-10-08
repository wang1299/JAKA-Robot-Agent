"""Task state transitions, user confirmation, cancellation and worker lifecycle."""
from __future__ import annotations
import jaka_agent.agent.map_evidence as agent_map_evidence
import jaka_agent.diagnostics as diagnostics
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.tasks.validation as tasks_validation
import jaka_agent.tasks.video as tasks_video
import jaka_agent.web.settings as web_settings
import copy
import json
import math
import os
import shutil
import threading
import time
import uuid
from jaka_agent.tasks.cards import TaskCardsMixin
from jaka_agent.tasks.find_object import TasksFindObjectMixin
from jaka_agent.tasks.patrol import TasksPatrolMixin
from jaka_agent.tasks.welcome import TasksWelcomeMixin

class RobotTaskManager(TaskCardsMixin, TasksFindObjectMixin, TasksPatrolMixin, TasksWelcomeMixin):
    def __init__(self, web_state):
        self.web = web_state
        self.lock = threading.RLock()
        self.task = None
        self.thread = None
        self.driver = None
        self.hardware_driver = None
        self._capture_workers = []
        self.video_recorder = None
        self.mock_pose = [0.0, 0.0, 0.0]
        self.track = []
        self.track_started_at = time.time()
        self.saved_tracks = self._load_saved_tracks()

    def _load_saved_tracks(self):
        try:
            value = json.loads(web_settings.TRACK_HISTORY_PATH.read_text(encoding="utf-8"))
            tracks = value.get("tracks", []) if isinstance(value, dict) else value
            if not isinstance(tracks, list):
                return []
            return [track for track in tracks if isinstance(track, dict) and isinstance(track.get("points"), list)][:20]
        except FileNotFoundError:
            return []
        except Exception as exc:
            print(f"[track] 忽略无效轨迹文件: {exc}")
            return []

    def _save_tracks_file(self):
        temp = web_settings.TRACK_HISTORY_PATH.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"tracks": self.saved_tracks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(web_settings.TRACK_HISTORY_PATH)

    @staticmethod
    def _clean_track(points):
        cleaned = []
        for point in points:
            try:
                current = [float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)]
            except (TypeError, ValueError, IndexError):
                continue
            if not all(math.isfinite(value) for value in current):
                continue
            if cleaned:
                distance = math.hypot(current[0] - cleaned[-1][0], current[1] - cleaned[-1][1])
                angle = abs((current[2] - cleaned[-1][2] + math.pi) % (2 * math.pi) - math.pi)
                if distance < 0.02 and angle < math.radians(3):
                    continue
            cleaned.append(current)
        return cleaned

    def tracks_snapshot(self):
        with self.lock:
            return {
                "tracks": copy.deepcopy(self.saved_tracks),
                "current_points": len(self._clean_track(self.track)),
            }

    def save_current_track(self, name=""):
        with self.lock:
            points = self._clean_track(self.track)
            if len(points) < 2:
                raise ValueError("当前轨迹有效点不足，机器人移动后再保存")
            distance = sum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(points, points[1:])
            )
            created_at = time.time()
            item = {
                "id": uuid.uuid4().hex,
                "name": str(name or time.strftime("轨迹 %m-%d %H:%M", time.localtime(created_at)))[:40],
                "created_at": created_at,
                "map_name": self.web.graph_snapshot().get("name"),
                "point_count": len(points),
                "distance_m": round(distance, 3),
                "points": points,
            }
            self.saved_tracks.insert(0, item)
            self.saved_tracks = self.saved_tracks[:20]
            self._save_tracks_file()
            self.track = []
            self.track_started_at = time.time()
            return copy.deepcopy(item)

    def delete_saved_track(self, track_id):
        with self.lock:
            before = len(self.saved_tracks)
            self.saved_tracks = [track for track in self.saved_tracks if str(track.get("id")) != str(track_id)]
            if len(self.saved_tracks) == before:
                raise ValueError("轨迹不存在")
            self._save_tracks_file()
            return self.tracks_snapshot()

    def clear_current_track(self):
        with self.lock:
            self.track = []
            self.track_started_at = time.time()
            return self.tracks_snapshot()

    def snapshot(self):
        with self.lock:
            task = copy.deepcopy(self.task)
            recorder = self.video_recorder
        if task is not None and recorder is not None:
            task["video"] = recorder.snapshot()
        return task

    def _update(self, **values):
        with self.lock:
            if self.task is not None:
                if self.task.get("cancel_requested"):
                    # Only worker cleanup may acknowledge completed cancellation.
                    values.pop("finished_at", None)
                    values["status"] = "canceling"
                self.task.update(values)
                self._persist_task()

    def _update_step(self, index, **values):
        with self.lock:
            if self.task is not None and 0 <= index < len(self.task["steps"]):
                self.task["steps"][index].update(values)
                self._persist_task()

    def _persist_task(self):
        store = getattr(self.web, 'conversation_store', None)
        if store and self.task and self.task.get('conversation_id'):
            store.task(copy.deepcopy(self.task))
            archive = getattr(self.web, 'media_archive', None)
            if archive:
                archive.task_record(self.task)

    def bind_conversation(self, task_id, conversation_id, turn_id=None):
        with self.lock:
            if not self.task or self.task['id'] != task_id:
                raise ValueError('任务已经变化，请重新规划')
            if self.task.get('conversation_id') not in (None, conversation_id):
                raise ValueError('任务不能转移到其他会话')
            if turn_id and self.task.get('turn_id') not in (None, turn_id):
                raise ValueError('任务不能转移到其他轮对话')
            self.task['conversation_id'] = conversation_id
            self.task['turn_id'] = turn_id or self.task.get('turn_id') or task_id
            self._persist_task()
            return copy.deepcopy(self.task)

    def _media_scope(self):
        tasks_runtime.check_cancelled()
        task = self.snapshot() or {}
        if task.get('conversation_id') and task.get('turn_id'):
            return {key:task[key] for key in ('conversation_id','turn_id')} | {'task_id':task['id']}
        return None

    def _capture_path(self, capture_id, kind='observations', **metadata):
        if getattr(self.web, 'media_archive', None):
            return self.web.new_capture_path(capture_id, self._media_scope(), kind, metadata)
        return self.web.capture_path(capture_id)

    def _capture(self, kind='observations', **metadata):
        if getattr(self.web, 'media_archive', None):
            return self.web.capture(media_scope=self._media_scope(), kind=kind, metadata=metadata)
        return self.web.capture()

    def _record_capture(self, capture_id, **metadata):
        archive = getattr(self.web, 'media_archive', None)
        if archive:
            archive.record(f'/captures/{capture_id}.jpg', **metadata)

    def _objects(self):
        return self.web.graph_snapshot()["objects"]

    def execute(self, task_id: str):
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if not self.task or self.task["id"] != task_id:
                raise ValueError("任务不存在或已被替换")
            if self.task["status"] != "planned":
                raise RuntimeError(f"任务当前状态不能执行: {self.task['status']}")
            if self.is_busy():
                raise RuntimeError("上一任务仍在收尾，请稍后再执行")
            if self.task.get("map_name") != self.web.graph_snapshot().get("name"):
                raise RuntimeError("规划后地图已切换，请重新生成任务计划")
            skill_snapshot = (self.task.get("skill") or {}).get("map_snapshot")
            if skill_snapshot and skill_snapshot != agent_map_evidence.MapEvidence(self.web.graph_snapshot()).version:
                raise RuntimeError("规划后地图内容已变化，请重新生成技能计划")
            # Fail closed before spawning any execution/video/camera thread.
            status = self.robot_status()
            if status.get("online") is not True:
                raise tasks_validation.TaskSafetyError("底盘离线或无法读取状态，未启动任务")
            if status.get("estop_state") is True:
                raise tasks_validation.TaskSafetyError("底盘处于急停状态，未启动任务。请现场确认安全并检查急停装置；系统不会自动解除急停")
            if status.get("estop_state") is not False:
                raise tasks_validation.TaskSafetyError("无法确认底盘急停状态，未启动任务，请先检查底盘连接")
            self.task.update({"status": "running", "started_at": time.time()})
            self._persist_task()
            self.thread = threading.Thread(target=self._run, daemon=True, name="RobotWebTask")
            self.thread.start()
        return self.snapshot()

    def cancel(self, task_id: str):
        with self.lock:
            if not self.task or self.task["id"] != task_id:
                raise ValueError("任务不存在")
            if self.task["status"] == "planned":
                self.task.update({"status": "canceled", "finished_at": time.time()})
                self._persist_task()
                return self.snapshot()
            if self.task["status"] not in ("running", "canceling"):
                return self.snapshot()
            self.task.update({"status": "canceling", "cancel_requested": True})
            self._persist_task()
            driver = self.driver
            for stop_event, _ in getattr(self, "_capture_workers", []):
                stop_event.set()
            if self.video_recorder is not None:
                self.video_recorder.request_stop()
        if driver is not None:
            try:
                driver.cancel_move()
            except Exception as exc:
                self._update(error=f"取消指令失败: {exc}")
        return self.snapshot()

    def _canceled(self):
        with self.lock:
            return bool(self.task and self.task.get("cancel_requested"))

    def _task_cancel_check(self):
        task_id = (self.snapshot() or {}).get("id")
        def canceled():
            with self.lock:
                return (not self.task or self.task.get("id") != task_id
                        or bool(self.task.get("cancel_requested")))
        return canceled

    def _task_camera_driver(self):
        return tasks_runtime.TaskDriver(self._hardware_camera_driver(), self._task_cancel_check(), self.lock)

    def is_busy(self):
        with self.lock:
            thread = getattr(self, "thread", None)
            return bool((self.task and self.task.get("status") in ("running", "canceling"))
                        or (thread is not None and thread.is_alive()))

    def _start_video_recording(self, driver):
        if self.web.mock or not web_settings.TASK_VIDEO_ENABLED or driver is None:
            return None
        if not hasattr(driver, "grab_color_frame"):
            return None
        task = self.snapshot() or {}
        archive = getattr(self.web, 'media_archive', None)
        recorder = tasks_video.TaskVideoRecorder(task.get("id") or uuid.uuid4().hex, driver,
                                     archive=archive, media_scope=self._media_scope() if archive else None)
        with self.lock:
            tasks_runtime.check_cancelled()
            self.video_recorder = recorder
            if self.task is not None:
                self.task["video"] = recorder.snapshot()
        recorder.start()
        with self.lock:
            if self.task is not None:
                self.task["video"] = recorder.snapshot()
        return recorder

    def _stop_video_recording(self):
        with self.lock:
            recorder = self.video_recorder
            self.video_recorder = None
        if recorder is None:
            return
        recorder.stop()
        with self.lock:
            if self.task is not None:
                self.task["video"] = recorder.snapshot()

    def _pause_video_recording(self):
        with self.lock:
            recorder = self.video_recorder
        if recorder is None:
            return True
        paused = recorder.pause(timeout=5.0)
        if not paused:
            diagnostics.LOGGER.error("[video] recorder did not pause before arrival orientation")
        return paused

    def _resume_video_recording(self):
        with self.lock:
            recorder = self.video_recorder
        if recorder is not None:
            recorder.resume()

    def _run(self):
        with tasks_runtime.cancellation_scope(self._task_cancel_check()):
            self._run_scoped()

    def _run_scoped(self):
        task_at_start = self.snapshot() or {}
        diagnostics.LOGGER.info(
            "[task] started task_id=%r kind=%r steps=%d mode=%s",
            task_at_start.get("id"),
            task_at_start.get("kind"),
            len(task_at_start.get("steps") or []),
            "mock" if self.web.mock else "hardware",
        )
        try:
            if self.web.mock:
                self._run_mock()
            else:
                self._run_hardware()
        except tasks_runtime.TaskCancelled:
            diagnostics.LOGGER.info("[task] canceled; discarded outstanding result task_id=%s", task_at_start.get("id"))
        except Exception as exc:
            task = self.snapshot() or {}
            stage = task.get("current_search_stage") or task.get("current_stage")
            diagnostics._log_exception(
                "[task] execution failed",
                exc,
                task_id=task.get("id"),
                kind=task.get("kind"),
                step=task.get("current_step"),
                target_ann_id=task.get("current_target_ann_id"),
                stage=stage,
            )
            public_error = str(exc)
            if stage and stage not in public_error:
                public_error = f"{stage}: {public_error}"
            self._update(
                status="error",
                error=public_error,
                current_stage=stage,
                finished_at=time.time(),
            )
        else:
            finished = self.snapshot() or {}
            diagnostics.LOGGER.info(
                "[task] finished task_id=%r status=%r error=%r",
                finished.get("id"), finished.get("status"), finished.get("error"),
            )
        finally:
            # Remain canceling until every producer has stopped using the camera.
            for stop_event, _ in self._capture_workers:
                stop_event.set()
            for _, thread in self._capture_workers:
                thread.join()
            self._capture_workers.clear()
            self._stop_video_recording()
            with self.lock:
                self.driver = None
                if self.task and self.task.get("cancel_requested"):
                    self.task.update(status="canceled", finished_at=time.time(),
                                     current_stage="任务已取消", current_search_stage=None)
                self._persist_task()

    def _hardware_camera_driver(self):
        """Web 服务内复用相机管线，避免任务结束时释放 Orbbec SDK 导致进程崩溃。"""
        from jaka_agent.hardware.navigation import CameraBackedDriver

        with self.lock:
            if self.hardware_driver is None:
                self.hardware_driver = CameraBackedDriver(
                    camera_factory=self._camera_factory,
                    keep_camera_open=True,
                    warmup_frames=max(0, int(os.getenv("JAKA_WEB_WARMUP_FRAMES", "30"))),
                )
            return self.hardware_driver

    def close(self):
        """只在整个 Web 服务退出时释放常驻相机。"""
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self.lock:
            driver = self.hardware_driver
            self.hardware_driver = None
            self.driver = None
        if driver is not None and hasattr(driver, "close_camera"):
            try:
                driver.close_camera(force=True)
            except Exception as exc:
                print(f"[task] 关闭相机失败: {exc}")

    def _mock_move(self, target_xy):
        start = self.mock_pose[:2]
        for index in range(1, 11):
            if self._canceled():
                return "canceled"
            ratio = index / 10
            self.mock_pose[0] = start[0] + (target_xy[0] - start[0]) * ratio
            self.mock_pose[1] = start[1] + (target_xy[1] - start[1]) * ratio
            self.track.append(self.mock_pose[:])
            self.track = self.track[-500:]
            time.sleep(0.12)
        return "succeeded"

    def _mock_observation(self, step):
        source = web_settings.HERE / "shot.jpg"
        capture_id = uuid.uuid4().hex
        if source.exists():
            shutil.copyfile(source, self._capture_path(capture_id, ann_id=step.get('target_ann_id')))
            self._record_capture(capture_id)
            image_url = f"/captures/{capture_id}.jpg"
        else:
            image_url = None
        observation = {
            "ann_id": step.get("target_ann_id"),
            "text": "模拟观察完成：已到达目标附近并拍摄现场画面。",
            "image_url": image_url,
        }
        with self.lock:
            self.task["observations"].append(observation)
        return observation

    def _run_mock(self):
        if self.task.get("kind") == "welcome":
            self._run_welcome_mock()
            return
        if self.task.get("kind") == "find_object":
            self._run_find_object_mock()
            return
        if self.task.get("kind") == "patrol":
            self._run_patrol_mock()
            return
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        start_pose = self.mock_pose[:]
        for index, step in enumerate(self.task["steps"]):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            self._update(current_step=index)
            self._update_step(index, status="running", started_at=time.time())
            status = "succeeded"
            if step["type"] in ("navigate", "cruise"):
                for ann_id in tasks_validation._step_target_ids(step):
                    status = self._mock_move(objects[ann_id]["floor_xy"])
                    if status != "succeeded":
                        break
            elif step["type"] == "observe":
                self._mock_observation(step)
            elif step["type"] == "return":
                status = self._mock_move(start_pose[:2])
            elif step["type"] == "cancel":
                status = "canceled"
            self._update_step(index, status=status, finished_at=time.time())
            if status != "succeeded":
                self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
                return
        self._update(status="succeeded", current_step=None, finished_at=time.time())

    def _camera_factory(self):
        try:
            from jaka_agent.hardware.camera import ColorDepthCamera

            cam = ColorDepthCamera(serial=web_settings.HEAD_CAMERA_SN, strict_serial=True)
        except Exception as depth_error:
            print(f"[task] 头部相机深度不可用({depth_error}), 回退同一头部相机的纯彩色流")
            from jaka_agent.hardware.camera import ColorCamera

            cam = ColorCamera(serial=web_settings.HEAD_CAMERA_SN, strict_serial=True)
        return cam

    def _save_observation(self, step, log):
        ann_id = step.get("target_ann_id")
        source = web_settings.HERE / f"obs_ann{ann_id}.png"
        image_url = None
        if source.exists():
            capture_id = uuid.uuid4().hex
            import cv2

            image = cv2.imread(str(source))
            if image is not None:
                cv2.imwrite(str(self._capture_path(capture_id, ann_id=ann_id)), image)
                self._record_capture(capture_id, observation=log.observation or '')
                image_url = f"/captures/{capture_id}.jpg"
        value = {"ann_id": ann_id, "text": log.observation or "", "image_url": image_url,
                 "observed_at": time.time(), "task_id": self.task.get('id'),
                 "source_type": "task_observation_not_live"}
        with self.lock:
            self.task["observations"].append(value)
            self._persist_task()
        return value

    def _append_executor_logs(self, executor, start_index):
        with self.lock:
            self.task["logs"].extend([
                {
                    "step_type": item.step_type,
                    "target_ann_id": item.target_ann_id,
                    "detail": item.detail,
                    "observation": item.observation,
                    "status": item.status,
                }
                for item in executor.log[start_index:]
            ])

    def _run_hardware(self):
        if self.task.get("kind") == "welcome":
            self._run_welcome_hardware()
            return
        if self.task.get("kind") == "find_object":
            self._run_find_object_hardware()
            return
        if self.task.get("kind") == "patrol":
            self._run_patrol_hardware()
            return
        from jaka_agent.hardware.navigation import JakaTCPDriver
        from jaka_agent.tasks.executor import PlanExecutor

        plan = self.task["plan"]
        need_camera = any(step.get("type") == "observe" for step in plan.get("steps", []))
        driver = (self._task_camera_driver() if (need_camera or web_settings.TASK_VIDEO_ENABLED)
                  else tasks_runtime.TaskDriver(JakaTCPDriver(), self._task_cancel_check(), self.lock))
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)
        plan_steps = plan.get("steps", [])
        for index, step in enumerate(plan_steps):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            self._update(current_step=index)
            self._update_step(index, status="running", started_at=time.time())
            before = len(executor.log)
            step_type = step["type"]
            if step_type == "navigate":
                status = executor._do_navigate(step)
            elif step_type == "cruise":
                status = executor._do_cruise(step)
            elif step_type == "observe":
                status = executor._do_observe(step)
            elif step_type == "return":
                status = executor._do_return()
            elif step_type == "cancel":
                executor._do_cancel()
                status = "canceled"
            else:
                status = "failed"
            tasks_runtime.check_cancelled()
            new_logs = executor.log[before:]
            with self.lock:
                self.task["logs"].extend([
                    {
                        "step_type": item.step_type,
                        "target_ann_id": item.target_ann_id,
                        "detail": item.detail,
                        "observation": item.observation,
                        "status": item.status,
                    }
                    for item in new_logs
                ])
            observe_logs = [item for item in new_logs if item.step_type == "observe"]
            if observe_logs:
                fallback_ann_id = step.get("target_ann_id")
                if fallback_ann_id is None and step_type == "cruise":
                    ids = list(step.get("target_ann_ids") or [])
                    fallback_ann_id = ids[-1] if ids else None
                observation_step = step if step_type == "observe" else {
                    "target_ann_id": fallback_ann_id,
                }
                self._save_observation(observation_step, observe_logs[-1])
            self._update_step(index, status=status, finished_at=time.time())
            if status != "succeeded":
                if status in ("failed", "timeout"):
                    try:
                        driver.cancel_move()
                    except Exception:
                        pass
                final = "canceled" if status == "canceled" else "aborted"
                error = status
                if status == "not_found" and observe_logs:
                    error = observe_logs[-1].observation or "到点后未找到目标"
                self._update(status=final, error=error, finished_at=time.time())
                return
        self._update(status="succeeded", current_step=None, finished_at=time.time())

    def robot_status(self) -> dict:
        if self.web.mock:
            return {
                "online": True,
                "pose": {"x": self.mock_pose[0], "y": self.mock_pose[1], "theta": self.mock_pose[2]},
                "move_status": self.task.get("status") if self.task else "idle",
                "estop_state": False,
                "track": copy.deepcopy(self.track),
            }
        try:
            from jaka_agent.hardware.navigation import JakaTCPDriver

            status = JakaTCPDriver().robot_status()
            pose = status.get("current_pose") or {}
            point = [pose.get("x", 0.0), pose.get("y", 0.0), pose.get("theta", 0.0)]
            with self.lock:
                cleaned = self._clean_track([self.track[-1], point] if self.track else [point])
                if cleaned and (not self.track or len(cleaned) > 1):
                    self.track.append(cleaned[-1])
                self.track = self.track[-500:]
                track = copy.deepcopy(self.track)
            return {
                "online": True,
                "pose": {"x": point[0], "y": point[1], "theta": point[2]},
                "move_status": status.get("move_status"),
                "estop_state": tasks_validation._estop_flag(status.get("estop_state")),
                "battery": status.get("battery") or status.get("battery_percent"),
                "track": track,
            }
        except Exception as exc:
            return {"online": False, "error": str(exc), "track": copy.deepcopy(self.track)}
