"""Task evidence, geometric disambiguation and fail-closed execution; no hardware."""
import jaka_agent.tasks.manager as ja_tasks_manager
import jaka_agent.tasks.validation as ja_tasks_validation
import jaka_agent.web.handler as ja_web_handler
import jaka_agent.web.server as ja_web_server
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.agent.runner import AgentRunner, Tool, ToolInputError
from jaka_agent.agent.map_evidence import MapEvidence
from tests.test_robot_skills import web_state


def call(name, **args):
    return json.dumps({'tool': name, 'arguments': args}, ensure_ascii=False)


class ExecutionContractTests(unittest.TestCase):
    def runner(self, outputs, tools=None, task=None):
        return AgentRunner(Mock(side_effect=outputs), tools or {}, native=True,
                           require_final_tool=True, task_snapshot=lambda: task)

    def test_plain_navigation_claim_is_not_displayed(self):
        result = self.runner(['我正在导航中'] * 3).run('去看看')
        self.assertTrue(result['incomplete'])
        self.assertNotIn('正在导航', result['text'])
        self.assertEqual(result['execution']['state'], 'none')

    def test_action_final_repairs_into_real_pending_plan(self):
        task = {'id': 'p', 'status': 'planned'}
        handler = Mock(return_value={'task': task})
        result = self.runner([call('finish_response', intent='action_request', text='现在过去'),
                              call('plan_inspect_location')],
                             {'plan_inspect_location': Tool('计划', {}, handler, effect='plan')}, task).run('去看看')
        handler.assert_called_once()
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(result['execution']['state'], 'planned')

    def test_status_text_cannot_override_server_state(self):
        for task in (None, {'id': 'p', 'status': 'planned'}, {'id': 'p', 'status': 'failed'}):
            result = self.runner([call('finish_response', intent='task_status', text='已完成导航')], task=task).run('怎么样了')
            self.assertNotIn('已完成导航', result['text'])
            self.assertEqual(result['text'], result['execution']['label'])

    def test_status_tool_does_not_create_confirmation_card(self):
        task = {'id': 'p', 'status': 'running'}
        result = self.runner([call('get_robot_status'), call('finish_response', intent='task_status', text='状态')],
                             {'get_robot_status': Tool('状态', {}, lambda: {'task': task})}, task).run('状态')
        self.assertNotIn('requires_confirmation', result)
        self.assertEqual(result['execution']['state'], 'running')

    def test_normal_answer_is_one_model_call(self):
        runner = self.runner([call('finish_response', intent='answer', text='你好')])
        result = runner.run('你好')
        self.assertEqual(result['text'], '你好')
        self.assertEqual(result['tool_trace'], [])
        runner.complete.assert_called_once()

    def test_estop_normalization(self):
        for value in (True, 1, 'true', '1'):
            self.assertIs(ja_tasks_validation._estop_flag(value), True)
        for value in (False, 0, 'false', '0'):
            self.assertIs(ja_tasks_validation._estop_flag(value), False)
        for value in (None, '', [], 'unknown', 2):
            self.assertIsNone(ja_tasks_validation._estop_flag(value))

    def test_unsafe_execution_never_starts_thread_or_changes_task(self):
        for status in ({'online': False}, {'online': True, 'estop_state': True}, {'online': True}):
            web = web_state()
            task = web.tasks.plan_skill('inspect_location', '去看', target_ann_id=7, question='情况')
            web.tasks.robot_status.return_value = status
            with patch('threading.Thread') as thread, self.assertRaises(RuntimeError):
                ja_tasks_manager.RobotTaskManager.execute(web.tasks, task['id'])
            thread.assert_not_called()
            self.assertEqual(web.tasks.task['status'], 'planned')
            web.capture.assert_not_called()

    def test_safe_execution_starts_only_after_explicit_execute(self):
        web = web_state()
        task = web.tasks.plan_skill('inspect_location', '去看', target_ann_id=7, question='情况')
        web.tasks.robot_status.return_value = {'online': True, 'estop_state': False}
        with patch('threading.Thread') as thread:
            ja_tasks_manager.RobotTaskManager.execute(web.tasks, task['id'])
        thread.return_value.start.assert_called_once()

    def test_http_estop_returns_conflict_and_keeps_pending_card(self):
        web = web_state()
        task = web.tasks.plan_skill('inspect_location', '去看', target_ann_id=7, question='情况')
        web.tasks.robot_status.return_value = {'online': True, 'estop_state': True}
        web.tasks.execute = lambda task_id: ja_tasks_manager.RobotTaskManager.execute(web.tasks, task_id)
        server = ja_web_server.RobotWebServer(('127.0.0.1', 0), ja_web_handler.RobotWebHandler, web)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            request = Request(f'http://127.0.0.1:{server.server_port}/api/task/execute',
                data=json.dumps({'task_id': task['id']}).encode(), headers={'Content-Type': 'application/json'})
            with self.assertRaises(HTTPError) as caught:
                urlopen(request, timeout=3)
            self.assertEqual(caught.exception.code, 409)
            self.assertIn('急停', caught.exception.read().decode())
            self.assertEqual(web.tasks.task['status'], 'planned')
            self.assertFalse(hasattr(web.tasks, 'thread'))
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


class SpatialEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.map = MapEvidence({'objects': [
            {'ann_id': 1, 'category': 'Box', 'floor_xy': [0, 0]},
            {'ann_id': 2, 'category': 'Seat', 'floor_xy': [3, 4], 'position': '靠墙'},
            {'ann_id': 3, 'category': 'Seat', 'floor_xy': [0, 1], 'position': '靠墙'}]})

    def test_distance_is_geometry_not_matching_wall_description(self):
        result = self.map.compare_positions(1, [2, 3])
        self.assertEqual([r['ann_id'] for r in result['candidates']], [3, 2])
        self.assertEqual([r['center_distance_m'] for r in result['candidates']], [1, 5])
        self.assertNotIn('task', result)
        self.assertNotIn('selected_target', result)

    def test_invalid_or_missing_coordinates_never_inferred(self):
        for value in (None, [True, 1], [float('nan'), 1], ['3', 1]):
            self.map.objects['2']['floor_xy'] = value
            with self.assertRaises(ToolInputError):
                self.map.compare_positions(1, [2])

    def test_invalid_ids_rejected(self):
        for reference, candidates in ((True, [2]), (1, [True]), (1, [1]), (1, [2, 2]), (1, [99])):
            with self.assertRaises(ToolInputError):
                self.map.compare_positions(reference, candidates)


class AmbiguousTargetTests(unittest.TestCase):
    def setUp(self):
        self.web = web_state()
        self.web.graph_snapshot.return_value['objects'].append(
            {'ann_id': 8, 'category': 'Table', 'category_zh': '桌子', 'floor_xy': [2, 2], 'position': '靠墙'})

    def offer(self):
        from tests.test_robot_agent import model_fixture
        self.web.agent_complete = model_fixture([
            call('query_map', question='目标', categories=['Table']),
            call('query_map', question='参考物', categories=['Door']),
            call('plan_inspect_location', instruction='去看看', target_ann_id=7, question='情况'),
            call('ask_user', goal='去看看', question='请选择具体目标', skill_id='inspect_location', missing_inputs=[])])
        return self.web.agent_chat({'question': '去看看', 'conversation_id': 'target-choice'})

    def test_ambiguous_target_does_not_publish_task(self):
        result = self.offer()
        self.assertTrue(result['requires_clarification'])
        self.assertEqual([c['ann_id'] for c in result['target_choices']], [7, 8])
        self.assertIsNone(self.web.tasks.task)

    def test_user_choice_allows_plan_but_never_executes(self):
        offer = self.offer()
        from tests.test_robot_agent import model_fixture
        self.web.agent_complete = model_fixture([call('plan_inspect_location', instruction='去选择的点', target_ann_id=8, question='情况')])
        result = self.web.agent_chat({'question': '选这个', 'conversation_id': 'target-choice',
            'target_selection': {'ann_id': 8, 'snapshot': offer['target_choice_snapshot']}})
        self.assertEqual(result['task']['steps'][0]['target_ann_id'], 8)
        self.assertTrue(result['requires_confirmation'])
        self.web.tasks.execute.assert_not_called()

    def test_stale_or_unoffered_choice_rejected(self):
        offer = self.offer()
        for selection in ({'ann_id': 19, 'snapshot': offer['target_choice_snapshot']},
                          {'ann_id': 7, 'snapshot': 'stale'}):
            with self.assertRaises(ValueError):
                self.web.agent_chat({'question': '选这个', 'conversation_id': 'target-choice', 'target_selection': selection})
        self.web.graph_snapshot.return_value['objects'][0]['floor_xy'] = [10, 10]
        with self.assertRaises(ValueError):
            self.web.agent_chat({'question': '选这个', 'conversation_id': 'target-choice',
                'target_selection': {'ann_id': 7, 'snapshot': offer['target_choice_snapshot']}})


if __name__ == '__main__':
    unittest.main()
