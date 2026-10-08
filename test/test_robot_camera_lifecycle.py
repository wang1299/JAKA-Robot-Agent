"""Hardware-free cancellation races and shared-camera regressions."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import robot_web
from qwen_planner import CameraBackedDriver
from robot_runtime import (CancellationClient, SharedMappingCamera, TaskCancelled,
                           TaskDriver, cancellation_scope, guarded_call)


class CameraLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cv2 = SimpleNamespace(imwrite=Mock(return_value=True), VideoWriter=Mock())
        self.cv_patch = patch.dict(sys.modules, {'cv2': self.cv2})
        self.cv_patch.start()
        self.web = SimpleNamespace(mock=False)
        with patch.object(robot_web.RobotTaskManager, '_load_saved_tracks', return_value=[]):
            self.manager = robot_web.RobotTaskManager(self.web)
        self.web.tasks = self.manager
        self.manager.task = {'id': 'first', 'status': 'running', 'cancel_requested': False,
                             'steps': [], 'observations': [], 'snapshot_count': 0}
        self.camera = Mock()
        self.camera.grab.return_value = (np.zeros((8, 12, 3), np.uint8), np.ones((8, 12), np.uint16))
        self.factory = Mock(return_value=self.camera)
        self.driver = CameraBackedDriver(camera_factory=self.factory, keep_camera_open=True)
        self.manager.hardware_driver = self.driver

    def tearDown(self):
        self.cv_patch.stop()
        self.temp.cleanup()

    def path(self, name='frame.jpg'):
        return str(Path(self.temp.name) / name)

    def test_task_then_chat_and_mapping_share_one_camera(self):
        task_driver = self.manager._task_camera_driver()
        task_driver.capture(self.path('task.jpg'))
        self.manager.task.update(cancel_requested=True, status='canceled')
        web = object.__new__(robot_web.RobotWebState)
        web.mock, web.camera_lock, web.tasks = False, threading.Lock(), self.manager
        web.capture_path = lambda _: Path(self.path('chat.jpg'))
        _, path = web.capture()
        self.assertEqual(str(path), self.path('chat.jpg'))
        self.assertEqual(self.cv2.imwrite.call_count, 2)
        mapping = SharedMappingCamera(self.driver, lambda: False)
        color, depth = mapping.read(timeout_ms=200, tries=12)
        mapping.close()
        self.assertEqual(color.shape[:2], depth.shape)
        self.factory.assert_called_once()
        self.camera.close.assert_not_called()
        self.manager.close()
        self.camera.close.assert_called_once()

    def test_cancel_blocks_old_capture_and_motion_but_not_new_chat(self):
        raw = Mock()
        view = TaskDriver(raw, self.manager._task_cancel_check(), self.manager.lock)
        self.manager.task['cancel_requested'] = True
        for call in (lambda: view.capture('no.jpg'), lambda: view.move_location(1, 2, 0),
                     lambda: view.grab_color_frame()):
            with self.assertRaises(TaskCancelled):
                call()
        raw.capture.assert_not_called()
        raw.move_location.assert_not_called()
        raw.grab_color_frame.assert_not_called()

    def test_new_task_does_not_reactivate_old_driver(self):
        old = self.manager._task_camera_driver()
        self.manager.task = {'id': 'second', 'cancel_requested': False}
        with self.assertRaises(TaskCancelled):
            old.capture(self.path())
        self.factory.assert_not_called()

    def test_cancel_while_waiting_camera_lock_does_not_grab(self):
        view = self.manager._task_camera_driver()
        entered = threading.Event()
        outcomes = []
        def worker():
            entered.set()
            try:
                view.capture(self.path())
            except TaskCancelled:
                outcomes.append('canceled')
        with self.driver._camera_lock:
            t = threading.Thread(target=worker)
            t.start()
            self.assertTrue(entered.wait(2))
            self.manager.task['cancel_requested'] = True
        t.join(2)
        self.assertFalse(t.is_alive())
        self.assertEqual(outcomes, ['canceled'])
        self.factory.assert_not_called()

    def test_cancel_during_capture_discards_file(self):
        def frame():
            self.manager.task['cancel_requested'] = True
            return np.zeros((8, 12, 3), np.uint8), np.ones((8, 12), np.uint16)
        self.camera.grab.side_effect = frame
        with self.assertRaises(TaskCancelled):
            self.manager._task_camera_driver().capture(self.path())
        self.assertFalse(Path(self.path()).exists())
        self.camera.close.assert_not_called()

    def test_cancel_during_warmup_keeps_connection_and_stops_reading(self):
        self.driver._warmup_remaining = 30
        self.camera.grab_color.side_effect = lambda **_: self.manager.task.update(cancel_requested=True)
        with self.assertRaises(TaskCancelled):
            self.manager._task_camera_driver().capture(self.path())
        self.camera.grab_color.assert_called_once()
        self.camera.grab.assert_not_called()
        self.camera.close.assert_not_called()
        self.assertIs(self.driver._cam, self.camera)

    def test_inflight_task_cancellation_waits_and_never_publishes_match(self):
        entered, release = threading.Event(), threading.Event()
        self.web.capture_path = lambda _: Path(self.path())
        def infer(*_):
            entered.set()
            if not release.wait(3):
                raise AssertionError('test did not release inference')
            return {'found': True, 'confidence': 'high'}
        self.web.compare_reference = infer
        self.manager._find_object_match_value = Mock()
        def work():
            driver = self.manager._task_camera_driver()
            self.manager._capture_reference_snapshot(driver, {}, 1, {}, '到达点', 'always')
            driver.move_location(1, 2, 0)
        self.manager._run_hardware = work
        self.manager.thread = threading.Thread(target=self.manager._run)
        self.manager.thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.manager.cancel('first')['status'], 'canceling')
            self.assertTrue(self.manager.is_busy())
        finally:
            release.set()
            self.manager.thread.join(3)
        self.assertFalse(self.manager.thread.is_alive())
        self.assertEqual(self.manager.task['status'], 'canceled')
        self.manager._find_object_match_value.assert_not_called()
        self.camera.close.assert_not_called()

    def test_cancel_stops_recording_immediately_and_retains_connection(self):
        recorder = Mock()
        self.manager.video_recorder = recorder
        self.manager.driver = Mock()
        event = threading.Event()
        self.manager._capture_workers = [(event, Mock())]
        result = self.manager.cancel('first')
        self.assertEqual(result['status'], 'canceling')
        self.assertTrue(event.is_set())
        recorder.request_stop.assert_called_once()
        self.manager.driver.cancel_move.assert_called_once()
        self.camera.close.assert_not_called()

    def test_cleanup_must_finish_before_canceled_is_published(self):
        recorder = Mock()
        def stopped():
            self.assertEqual(self.manager.task['status'], 'canceling')
        recorder.stop.side_effect = stopped
        recorder.snapshot.return_value = {}
        self.manager.video_recorder = recorder
        def work():
            self.manager.task['cancel_requested'] = True
            self.manager._update(status='canceled', finished_at=1)
        self.manager._run_hardware = work
        self.manager._run()
        self.assertEqual(self.manager.task['status'], 'canceled')
        recorder.stop.assert_called_once()
        self.assertIs(self.manager.hardware_driver, self.driver)

    def test_canceled_match_is_not_published(self):
        self.web.capture_path = lambda _: Path(self.path())
        self.web.compare_reference = Mock()
        def infer(*args):
            self.manager.task['cancel_requested'] = True
            return {'found': True, 'confidence': 'high'}
        self.web.compare_reference.side_effect = infer
        self.manager._find_object_match_value = Mock()
        with cancellation_scope(self.manager._task_cancel_check()), self.assertRaises(TaskCancelled):
            self.manager._capture_reference_snapshot(self.manager._task_camera_driver(), {}, 1, {}, '到达点', 'always')
        self.manager._find_object_match_value.assert_not_called()
        self.assertEqual(self.manager.task['snapshot_count'], 0)
        self.assertEqual(self.manager.task['observations'], [])

    def test_cancel_between_capture_and_inference_sends_no_request(self):
        self.web.capture_path = lambda _: Path(self.path())
        self.web.compare_reference = Mock()
        capture_driver = Mock()
        capture_driver.capture.side_effect = lambda *_: self.manager.task.update(cancel_requested=True)
        with cancellation_scope(self.manager._task_cancel_check()), self.assertRaises(TaskCancelled):
            self.manager._capture_reference_snapshot(capture_driver, {}, 1, {}, '到达点')
        self.web.compare_reference.assert_not_called()

    def test_rgb_only_camera_does_not_supply_stale_depth_to_mapping(self):
        cam = SimpleNamespace(grab_color=Mock())
        driver = CameraBackedDriver(camera=cam)
        driver.last_depth = np.ones((8, 12), np.uint16)
        with self.assertRaisesRegex(RuntimeError, '无深度流'):
            driver.grab_rgbd_frame()
        cam.grab_color.assert_not_called()

    def test_video_discards_inflight_frame_after_stop(self):
        recorder = robot_web.TaskVideoRecorder('task', Mock())
        def grab():
            recorder.request_stop()
            return np.zeros((8, 12, 3), np.uint8)
        recorder.driver.grab_color_frame.side_effect = grab
        with patch('cv2.VideoWriter') as writer:
            recorder._run()
        writer.assert_not_called()
        self.assertEqual(recorder.frame_count, 0)
        self.assertIsNotNone(recorder.finished_at)


class RequestCancellationTests(unittest.TestCase):
    def client(self):
        raw = Mock()
        raw.with_options.return_value = raw
        return raw, CancellationClient(raw)

    def test_pending_request_is_not_sent_after_cancel(self):
        raw, client = self.client()
        with cancellation_scope(lambda: True), self.assertRaises(TaskCancelled):
            client.with_options(timeout=3).chat.completions.create(model='test')
        raw.chat.completions.create.assert_not_called()

    def test_inflight_result_is_discarded_and_not_retried(self):
        raw, client = self.client()
        canceled = threading.Event()
        raw.chat.completions.create.side_effect = lambda **_: canceled.set() or 'late result'
        with cancellation_scope(canceled.is_set), self.assertRaises(TaskCancelled):
            client.chat.completions.create(model='test')
        raw.chat.completions.create.assert_called_once()
        raw.with_options.assert_called_with(max_retries=0, timeout=45)

    def test_non_task_request_is_unaffected(self):
        raw, client = self.client()
        raw.chat.completions.create.return_value = 'answer'
        self.assertEqual(client.chat.completions.create(model='test'), 'answer')
        raw.with_options.assert_not_called()

    def test_cancellation_scope_does_not_poison_later_chat(self):
        with cancellation_scope(lambda: True), self.assertRaises(TaskCancelled):
            guarded_call(lambda: 'old')
        self.assertEqual(guarded_call(lambda: 'chat'), 'chat')


if __name__ == '__main__':
    unittest.main()
