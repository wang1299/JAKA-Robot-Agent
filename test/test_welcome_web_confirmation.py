"""Welcome confirmation contract; all cameras, speech and navigation are mocked."""
import sys
import threading
import time
import unittest
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_web


class WelcomeConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.web = SimpleNamespace(speak=Mock(), ask_and_listen=Mock())
        with patch.object(robot_web.RobotTaskManager, '_load_saved_tracks', return_value=[]):
            self.manager = robot_web.RobotTaskManager(self.web)
        self.manager.task = {
            'id': 'task-A', 'kind': 'welcome', 'conversation_id': 'A',
            'status': 'running', 'current_step': 1, 'cancel_requested': False,
            'pickup_ann_id': 6, 'return_ann_id': 7, 'pickup_name': '接客点', 'return_name': '送客点',
            'steps': [{'status': 'pending'} for _ in range(3)],
        }
        self.observation = {'image_url': '/captures/guest.jpg', 'captured_at': 123}
        self.threads = []

    def tearDown(self):
        self.manager.task['cancel_requested'] = True
        for thread in self.threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def start(self, callback):
        thread = threading.Thread(target=callback, daemon=True)
        self.threads.append(thread)
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pending = self.manager.snapshot().get('guest_confirmation')
            if pending:
                return thread, pending['id']
            time.sleep(.01)
        self.fail('confirmation was not published')

    def decide(self, confirmation, accepted=True, **overrides):
        args = dict(task_id='task-A', confirmation_id=confirmation, accepted=accepted, conversation_id='A')
        args.update(overrides)
        return self.manager.confirm_welcome_guest(**args)

    def test_wait_publishes_photo_and_does_not_use_microphone(self):
        answers = []
        thread, cid = self.start(lambda: answers.append(self.manager._wait_welcome_confirmation(self.observation)))
        self.assertTrue(thread.is_alive())
        self.assertEqual(answers, [])
        self.assertEqual(self.manager.snapshot()['guest_confirmation']['image_url'], self.observation['image_url'])
        self.decide(cid)
        thread.join(2)
        self.assertEqual(answers, [True])
        self.web.speak.assert_called_once_with('小卡识别到目标客人，等待主人确认中')
        self.web.ask_and_listen.assert_not_called()
        with self.assertRaises(ValueError):
            self.decide(cid)

    def test_reject_resumes_detection_without_accepting(self):
        answers = []
        thread, cid = self.start(lambda: answers.append(self.manager._wait_welcome_confirmation(self.observation)))
        self.decide(cid, False)
        thread.join(2)
        self.assertEqual(answers, [False])
        self.assertEqual(self.manager.task['rejected_count'], 1)
        self.assertNotEqual(self.manager.task['steps'][1]['status'], 'succeeded')

    def test_wrong_conversation_stale_photo_and_task_are_rejected(self):
        _, cid = self.start(lambda: self.manager._wait_welcome_confirmation(self.observation))
        for override in ({'conversation_id': 'B'}, {'confirmation_id': 'stale'},
                         {'task_id': 'old'}, {'accepted': 'true'}):
            with self.assertRaises(ValueError):
                self.decide(cid, **override)
        self.assertEqual(self.manager.task['guest_confirmation']['status'], 'pending')

    def test_cancellation_invalidates_button(self):
        answers = []
        thread, cid = self.start(lambda: answers.append(self.manager._wait_welcome_confirmation(self.observation)))
        self.manager.cancel('task-A')
        with self.assertRaises(ValueError):
            self.decide(cid)
        thread.join(2)
        self.assertEqual(answers, [False])

    def test_hardware_flow_does_not_navigate_to_dropoff_before_click(self):
        m = self.manager
        m._objects = Mock(return_value=[{'ann_id': 6}, {'ann_id': 7}])
        m._task_camera_driver = Mock(return_value=Mock())
        m._start_video_recording = Mock()
        m._run_welcome_navigation = Mock(return_value='succeeded')
        m._capture_welcome_snapshot = Mock(return_value=(True, self.observation, {}))
        with patch('qwen_planner.PlanExecutor'), patch.object(robot_web, 'WELCOME_SNAPSHOT_INTERVAL_SECONDS', .01):
            thread, cid = self.start(m._run_welcome_hardware)
            self.assertEqual(m._run_welcome_navigation.call_count, 1)
            time.sleep(.1)
            self.assertEqual(m._capture_welcome_snapshot.call_count, 1)
            self.decide(cid)
            thread.join(2)
        self.assertEqual(m._run_welcome_navigation.call_count, 2)
        self.assertEqual(m._run_welcome_navigation.call_args.args[2], 7)
        self.assertEqual(m.task['status'], 'succeeded')
        self.web.ask_and_listen.assert_not_called()

    def test_http_confirmation_and_duplicate_rejection(self):
        self.web.tasks = self.manager
        server = robot_web.RobotWebServer(('127.0.0.1', 0), robot_web.RobotWebHandler, self.web)
        service = threading.Thread(target=server.serve_forever, daemon=True)
        service.start()
        try:
            worker, cid = self.start(lambda: self.manager._wait_welcome_confirmation(self.observation))
            payload = {'task_id': 'task-A', 'conversation_id': 'A', 'confirmation_id': cid, 'accepted': True}
            request = Request(f'http://127.0.0.1:{server.server_port}/api/task/welcome-confirm',
                              data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(json.load(response)['task']['guest_confirmation']['status'], 'accepted')
            with self.assertRaises(HTTPError):
                urlopen(request, timeout=2)
            worker.join(2)
        finally:
            server.shutdown()
            server.server_close()
            service.join(2)


if __name__ == '__main__':
    unittest.main()
