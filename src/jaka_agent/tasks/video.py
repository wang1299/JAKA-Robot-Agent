"""Record task capture streams as replayable videos."""
from __future__ import annotations
import jaka_agent.diagnostics as diagnostics
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import threading
import time
import uuid

class TaskVideoRecorder:
    """任务旁路录像器：失败只记日志，不影响导航、观察和寻物主流程。"""

    def __init__(self, task_id: str, driver, fps: float = web_settings.TASK_VIDEO_FPS, max_width: int = web_settings.TASK_VIDEO_MAX_WIDTH, archive=None, media_scope=None):
        self.task_id = task_id
        self.driver = driver
        self.fps = float(fps)
        self.max_width = int(max_width)
        self.video_id = uuid.uuid4().hex
        self.path = web_settings.VIDEO_DIR / f"{self.video_id}.mp4"
        self.url = f"/videos/{self.video_id}.mp4"
        self.archive = archive
        if archive and media_scope:
            self.path = archive.allocate(self.url, media_scope['conversation_id'], media_scope['turn_id'],
                                         'videos', task_id, {'fps':self.fps})
        self.started_at = None
        self.finished_at = None
        self.frame_count = 0
        self.error = None
        self._stop = threading.Event()
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"TaskVideo-{self.video_id[:8]}")
        self._thread.start()

    def request_stop(self):
        self._stop.set()
        self._pause_requested.clear()

    def stop(self, timeout=None):
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def pause(self, timeout=5.0):
        """等待录像线程退出相机抓帧区，供到点转向和正式拍照独占相机。"""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        if not self._pause_requested.is_set():
            self._paused.clear()
        self._pause_requested.set()
        return self._paused.wait(timeout=max(0.1, float(timeout)))

    def resume(self, timeout=2.0):
        self._pause_requested.clear()
        deadline = time.monotonic() + max(0.1, float(timeout))
        while self._paused.is_set() and self._thread is not None and self._thread.is_alive():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)

    def snapshot(self):
        return {
            "id": self.video_id,
            "url": self.url,
            "path": str(self.path),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "frame_count": self.frame_count,
            "error": self.error,
        }

    def _run(self):
        writer = None
        try:
            import cv2

            interval = 1.0 / max(0.5, self.fps)
            while not self._stop.is_set():
                if self._pause_requested.is_set():
                    self._paused.set()
                    while self._pause_requested.is_set() and not self._stop.is_set():
                        self._stop.wait(0.05)
                    self._paused.clear()
                    continue
                started = time.time()
                if not hasattr(self.driver, "grab_color_frame"):
                    self.error = "driver has no grab_color_frame"
                    return
                frame = self.driver.grab_color_frame()
                if self._stop.is_set():
                    break
                if frame is None:
                    time.sleep(interval)
                    continue
                height, width = frame.shape[:2]
                if width > self.max_width:
                    scale = self.max_width / float(width)
                    width = self.max_width
                    height = max(1, round(height * scale))
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                    if not writer.isOpened():
                        self.error = "cv2.VideoWriter open failed"
                        return
                    diagnostics.LOGGER.info("[video] recording started task_id=%r path=%s fps=%.1f size=%dx%d",
                                self.task_id, self.path, self.fps, width, height)
                writer.write(frame)
                self.frame_count += 1
                delay = interval - (time.time() - started)
                if delay > 0:
                    self._stop.wait(delay)
        except Exception as exc:
            if not isinstance(exc, tasks_runtime.TaskCancelled):
                self.error = str(exc)
                diagnostics._log_exception("[video] recording failed", exc, include_traceback=True, task_id=self.task_id)
        finally:
            self.finished_at = time.time()
            self._paused.set()
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            diagnostics.LOGGER.info("[video] recording stopped task_id=%r path=%s frames=%d error=%r",
                        self.task_id, self.path, self.frame_count, self.error)
            if self.archive:
                try:
                    self.archive.record(self.url, status='failed' if self.error else 'finished',
                                        started_at=self.started_at,finished_at=self.finished_at,
                                        frame_count=self.frame_count,error=self.error)
                except Exception:
                    diagnostics.LOGGER.exception('[media] video index update failed')
