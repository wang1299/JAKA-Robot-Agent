"""Patrol announcement policy; all robot, camera and model calls are mocked."""
import os as std_os
import jaka_agent.models.vision as ja_models_vision
import jaka_agent.tasks.events as ja_tasks_events
import jaka_agent.tasks.executor as ja_tasks_executor
import jaka_agent.tasks.manager as ja_tasks_manager
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class PatrolSpeechTests(unittest.TestCase):
    def manager(self, comparisons, count=2):
        web = SimpleNamespace(speak=Mock(), compare_patrol_images=Mock(side_effect=comparisons),
                              capture_path=lambda cid: Path(cid))
        with patch.object(ja_tasks_manager.RobotTaskManager, '_load_saved_tracks', return_value=[]):
            manager = ja_tasks_manager.RobotTaskManager(web)
        manager.task = {'id': 'patrol', 'kind': 'patrol', 'status': 'running',
                        'steps': [{'type': 'patrol', 'target_ann_ids': [1, 2], 'count': count}],
                        'patrol_baselines': {}, 'patrol_status_by_ann': {}, 'logs': []}
        manager._objects = Mock(return_value=[
            {'ann_id': 1, 'category_zh': '前门', 'floor_xy': [0, 1]},
            {'ann_id': 2, 'category_zh': '木质储物柜', 'floor_xy': [2, 3]}])
        manager._mock_move = Mock(return_value='succeeded')
        manager._capture = Mock(return_value=('image', Path('image.jpg')))
        manager._capture_path = Mock(return_value=Path('image.jpg'))
        manager._record_capture = Mock()
        manager._task_camera_driver = Mock(return_value=Mock())
        manager._start_video_recording = Mock()
        return manager, web

    def run_patrol(self, manager, hardware):
        if not hardware:
            manager._run_patrol_mock()
            return
        executor = Mock(log=[])
        executor._do_navigate.return_value = 'succeeded'
        executor._do_observe.return_value = 'succeeded'
        driver = manager._task_camera_driver.return_value
        driver.get_pose.return_value = (0.0, 0.0, 0.0)
        driver.wait_until_settled.return_value = 'succeeded'
        def capture_at_pose(driver, aid, round_no, objects, expected=None):
            current = manager._patrol_capture_value(aid, 'image', round_no, objects)
            current['capture_pose'] = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
            return current, Path('image.jpg')
        with patch.object(ja_tasks_executor, 'PlanExecutor', return_value=executor), \
                patch.object(manager, '_capture_patrol_at_pose', side_effect=capture_at_pose), \
                patch.object(Path, 'exists', return_value=True), \
                patch.dict(sys.modules, {'cv2': SimpleNamespace(imread=Mock(return_value=object()), imwrite=Mock())}):
            manager._run_patrol_hardware()
        for call in executor._do_navigate.call_args_list:
            self.assertIs(call.args[0]['_announce_arrival'], False)
        for call in executor._do_observe.call_args_list:
            self.assertIs(call.args[0]['_announce_observation'], False)

    def test_first_round_arrivals_later_rounds_only_results(self):
        for hardware in (False, True):
            with self.subTest(hardware=hardware):
                manager, web = self.manager([{'changed': False, 'confidence': 'high'}] * 4)
                self.run_patrol(manager, hardware)
                self.assertEqual([c.args[0] for c in web.speak.call_args_list],
                                 ['到达前门', '到达木质储物柜', '无异常', '无异常', '无异常', '无异常'])
                self.assertEqual(manager.task['status'], 'succeeded')

    def test_anomaly_short_speech_but_details_remain_on_web(self):
        for hardware in (False, True):
            with self.subTest(hardware=hardware):
                manager, web = self.manager([
                    {'changed': False, 'confidence': 'high'},
                    {'changed': True, 'confidence': 'high', 'summary': '新增纸箱',
                     'changes': [{'item': '纸箱', 'detail': '门边新增'}]}])
                self.run_patrol(manager, hardware)
                self.assertEqual([c.args[0] for c in web.speak.call_args_list],
                                 ['到达前门', '到达木质储物柜', '无异常', '发现异常'])
                self.assertEqual(manager.task['status'], 'anomaly')
                self.assertIn('门边新增', manager.task['result_text'])

    def test_comparison_failure_never_announces_normal(self):
        manager, web = self.manager([RuntimeError('model unavailable')])
        with self.assertRaises(RuntimeError):
            self.run_patrol(manager, True)
        self.assertEqual([c.args[0] for c in web.speak.call_args_list], ['到达前门', '到达木质储物柜'])

    def test_failed_navigation_does_not_announce_arrival(self):
        manager, web = self.manager([])
        manager._mock_move.return_value = 'failed'
        self.run_patrol(manager, False)
        web.speak.assert_not_called()


class ExecutorSpeechTests(unittest.TestCase):
    def executor(self, explicit_pose=False):
        obj = {'ann_id': 1, 'category_zh': '前门', 'category': 'door', 'floor_xy': [1, 2]}
        if explicit_pose:
            obj['navigation_pose'] = {'x': 1, 'y': 2, 'theta': 0}
        driver = Mock()
        driver.wait_until_settled.return_value = 'succeeded'
        ex = ja_tasks_executor.PlanExecutor([obj], driver, mark_start=False)
        ex._find_accessible_target = Mock(return_value=(1, 2, 0, 'mock'))
        ex._face_object_center = Mock(return_value='succeeded')
        return ex

    def test_arrival_suppression_is_opt_in_for_both_navigation_paths(self):
        for explicit in (False, True):
            ex = self.executor(explicit)
            with patch.object(ja_tasks_events, 'announce') as say:
                ex._do_navigate({'target_ann_id': 1, '_announce_arrival': False})
                say.assert_not_called()
                ex._do_navigate({'target_ann_id': 1})
                say.assert_called_once()

    def test_observation_suppression_keeps_failures_and_default_speech(self):
        for answer, quiet, expected in [('桌面干净', True, 0), ('桌面干净', False, 1),
                                         ('搜索失败：未找到', True, 1)]:
            ex = self.executor()
            with patch.object(std_os.path, 'exists', return_value=True), \
                    patch.object(ja_models_vision, 'read_image', return_value=answer), \
                    patch.object(ja_tasks_events, 'announce') as say:
                ex._do_observe({'target_ann_id': 1, '_announce_observation': not quiet})
                self.assertEqual(say.call_count, expected)


if __name__ == '__main__':
    unittest.main()
