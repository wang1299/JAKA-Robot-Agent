"""Skill contracts and real task-card adapters; no model or hardware required."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent import AgentRunner, ProtocolError, ToolInputError, validate_arguments
from robot_skills import SKILLS, prepare_skill, skill_catalog, skill_tools
import robot_web
from test_robot_agent import decision_calls, model_fixture


def reply(text):
    return json.dumps({'tool': 'finish_response', 'arguments': {'intent': 'answer', 'text': text}}, ensure_ascii=False)


def graph():
    return {'name': 'skill-test', 'objects': [
        {'ann_id': 7, 'category': 'Table', 'category_zh': '桌子', 'floor_xy': [1, 2], 'position': '大厅'},
        {'ann_id': 19, 'category': 'Door', 'category_zh': '门', 'floor_xy': [3, 4], 'position': '门口'}]}


def web_state():
    web = object.__new__(robot_web.RobotWebState)
    web.mock = False
    web.agent_lock = threading.Lock()
    web.mapping = SimpleNamespace(is_busy=Mock(return_value=False))
    web.graph_snapshot = Mock(return_value=graph())
    web.reference_path = Mock(return_value=Path(__file__))
    web.find_target_label = Mock(return_value='杯子')
    web.speak = Mock()
    web.capture = Mock(side_effect=AssertionError('No camera during planning'))
    manager = object.__new__(robot_web.RobotTaskManager)
    manager.web = web
    manager.lock = threading.RLock()
    manager.task = None
    manager.video_recorder = None
    manager.robot_status = Mock(return_value={'online': False})
    manager.execute = Mock(side_effect=AssertionError('No execution in tests'))
    web.tasks = manager
    return web


REFERENCE = {'reference_id': 'a' * 32, 'suffix': '.jpg',
             'image_url': '/references/' + 'a' * 32 + '.jpg', 'content_type': 'image/jpeg'}


class SkillContractTests(unittest.TestCase):
    def test_registry_is_discoverable_and_every_contract_has_boundaries(self):
        cards = skill_catalog()
        self.assertEqual({c['id'] for c in cards}, {'navigate', 'patrol', 'welcome', 'find_object', 'inspect_location'})
        for card in cards:
            self.assertTrue(card['requires_confirmation'])
            self.assertTrue(card['procedure'] and card['completion'] and card['failure_policy'])
        cards[0]['parameters'].clear()
        self.assertTrue(SKILLS['patrol'].parameters)

    def test_generated_tools_have_no_execute_or_keyword_dispatch(self):
        handler = Mock(return_value={'task': {'id': 'pending', 'status': 'planned'}})
        tools = skill_tools(handler)
        complete = Mock(return_value=json.dumps({'tool': 'plan_patrol', 'arguments': {
            'instruction': '按我们刚才商量的做', 'target_ann_ids': [7, 19]}}))
        result = AgentRunner(complete, tools, native=True).run('就这样安排吧')
        self.assertTrue(result['requires_confirmation'])
        handler.assert_called_once_with('patrol', instruction='按我们刚才商量的做', target_ann_ids=[7, 19], rounds=-1)
        self.assertFalse(any('execute' in name for name in tools))

    def test_integer_array_bounds_are_enforced(self):
        tool = skill_tools(Mock())['plan_patrol']
        for ids in ([True, 19], ['7', '19'], [7], [7, -1], list(range(31))):
            with self.subTest(ids=ids), self.assertRaises(ProtocolError):
                validate_arguments(tool, {'instruction': '检查', 'target_ann_ids': ids})

    def test_reject_unknown_skill_and_unknown_parameters(self):
        with self.assertRaises(ToolInputError):
            prepare_skill('shell', {}, graph())
        with self.assertRaises(ProtocolError):
            prepare_skill('patrol', {'instruction': '检查', 'target_ann_ids': [7, 19], 'execute': True}, graph())

    def test_patrol_duplicate_rounds_and_unknown_point_rejected(self):
        for args in ({'target_ann_ids': [7, 7]}, {'target_ann_ids': [7, 99]},
                     {'target_ann_ids': [7, 19], 'rounds': 0}):
            with self.assertRaises(ToolInputError):
                prepare_skill('patrol', {'instruction': '检查', **args}, graph())

    def test_inactive_missing_and_nonfinite_coordinates_rejected(self):
        for update in ({'lifecycle': 'deleted'}, {'floor_xy': None}, {'floor_xy': [float('nan'), 2]}, {'floor_xy': [True, 2]}):
            data = graph()
            data['objects'][0].update(update)
            with self.assertRaises(ToolInputError):
                prepare_skill('inspect_location', {'instruction': '去看看', 'target_ann_id': 7, 'question': '情况如何'}, data)

    def test_welcome_requires_reference_and_distinct_points(self):
        params = {'instruction': '接人', 'pickup_ann_id': 7, 'return_ann_id': 19}
        with self.assertRaises(ToolInputError):
            prepare_skill('welcome', params, graph())
        with self.assertRaises(ToolInputError):
            prepare_skill('welcome', {**params, 'return_ann_id': 7}, graph(), REFERENCE)


class SkillAdapterTests(unittest.TestCase):
    def setUp(self):
        self.web = web_state()
        self.manager = self.web.tasks

    def tearDown(self):
        self.manager.execute.assert_not_called()
        self.web.capture.assert_not_called()
        self.web.speak.assert_not_called()

    def test_patrol_builds_existing_executor_card_without_another_llm(self):
        result = self.manager.plan_skill('patrol', '轮流看看有没有变化', target_ann_ids=[19, 7], rounds=2)
        self.assertEqual(result['kind'], 'patrol')
        self.assertEqual(result['status'], 'planned')
        self.assertEqual(result['patrol_count'], 2)
        self.assertEqual(result['plan']['steps'][0]['target_ann_ids'], [19, 7])
        self.assertEqual([p['ann_id'] for p in result['route_points']], [19, 7])
        self.assertEqual(result['skill']['id'], 'patrol')
        self.assertIsNone(result['started_at'])

    def test_inspection_preserves_question_and_optional_return(self):
        for should_return in (False, True):
            result = self.manager.plan_skill('inspect_location', '去看看那里', target_ann_id=19,
                                            question='门是否打开', return_to_start=should_return)
            self.assertEqual([s['type'] for s in result['plan']['steps']],
                             ['navigate', 'observe'] + (['return'] if should_return else []))
            self.assertEqual(result['plan']['steps'][1]['question'], '门是否打开')
            self.assertEqual(result['skill']['id'], 'inspect_location')

    def test_welcome_reuses_existing_workflow_without_speaking_before_confirmation(self):
        task = self.manager.plan_skill('welcome', '接到客人后带回来', reference=REFERENCE,
                                      pickup_ann_id=19, return_ann_id=7)
        self.assertEqual(task['kind'], 'welcome')
        self.assertEqual([s['type'] for s in task['steps']], ['navigate', 'wait_guest', 'navigate'])
        self.assertEqual(task['pickup_ann_id'], 19)
        self.assertEqual(task['return_ann_id'], 7)
        self.assertEqual(task['skill']['name'], '迎宾接待')

    def test_find_reuses_candidates_and_reference_comparison_workflow(self):
        task = self.manager.plan_skill('find_object', '找到照片里的东西', reference=REFERENCE)
        self.assertEqual(task['kind'], 'find_object')
        self.assertEqual(task['candidate_ann_ids'], [7])
        self.assertEqual(task['reference'], REFERENCE)
        self.assertEqual(task['skill']['id'], 'find_object')

    def test_bad_request_preserves_previous_task(self):
        self.manager.task = {'id': 'old', 'status': 'planned'}
        with self.assertRaises(ToolInputError):
            self.manager.plan_skill('patrol', '看看', target_ann_ids=[7, 1234])
        self.assertEqual(self.manager.task, {'id': 'old', 'status': 'planned'})

    def test_legacy_generic_plan_still_builds_same_pending_card(self):
        self.web.mock = True
        with patch.object(self.manager, '_mock_plan', return_value={
            'understanding': '普通导航', 'need_return': False,
            'steps': [{'type': 'navigate', 'target_ann_id': 7}]}):
            task = self.manager.plan('普通导航')
        self.assertEqual(task['kind'], 'robot_task')
        self.assertEqual(task['status'], 'planned')
        self.assertEqual(task['steps'][0]['target_ann_id'], 7)

    def test_conflicting_map_does_not_replace_existing_task(self):
        self.manager.task = {'id': 'old', 'status': 'planned'}
        self.web.graph_snapshot.return_value['objects'].append({'ann_id': 7, 'category': 'Conflict'})
        with self.assertRaises(ToolInputError):
            self.manager.plan_skill('patrol', '检查', target_ann_ids=[7, 19])
        self.assertEqual(self.manager.task['id'], 'old')

    def test_expired_reference_cannot_create_task(self):
        self.web.reference_path.return_value = Path(__file__).with_name('missing-skill-reference.jpg')
        with self.assertRaises(ValueError):
            self.manager.plan_skill('welcome', '接人', reference=REFERENCE, pickup_ann_id=19, return_ann_id=7)
        self.assertIsNone(self.manager.task)

    def test_active_task_or_mapping_blocks_planning(self):
        for status in ('running', 'canceling'):
            self.manager.task = {'id': 'busy', 'status': status}
            with self.assertRaises(RuntimeError):
                self.manager.plan_skill('patrol', '检查', target_ann_ids=[7, 19])
        self.manager.task = None
        self.web.mapping.is_busy.return_value = True
        with self.assertRaises(RuntimeError):
            self.manager.plan_skill('patrol', '检查', target_ann_ids=[7, 19])

    def test_same_filename_map_update_requires_replanning(self):
        task = self.manager.plan_skill('patrol', '检查', target_ann_ids=[7, 19])
        self.web.graph_snapshot.return_value['objects'][0]['floor_xy'] = [8, 9]
        with patch('robot_web.threading.Thread') as thread, self.assertRaises(RuntimeError):
            robot_web.RobotTaskManager.execute(self.manager, task['id'])
        thread.assert_not_called()
        self.assertEqual(self.manager.task['status'], 'planned')

    def test_agent_can_select_welcome_and_remember_reference_across_clarification(self):
        self.web.agent_complete = model_fixture([json.dumps({'tool': 'ask_user', 'arguments': {
            'goal': '接参考图中的客人', 'question': '接人点和返回点是哪里？',
            'skill_id': 'welcome', 'missing_inputs': ['pickup_ann_id', 'return_ann_id']}})])
        self.web.agent_chat({'question': '接一下他', 'conversation_id': 'skill-session', 'reference': REFERENCE})
        self.web.agent_complete = model_fixture([json.dumps({'tool': 'plan_welcome', 'arguments': {
            'instruction': '去门口接他回大厅', 'pickup_ann_id': 19, 'return_ann_id': 7}})])
        result = self.web.agent_chat({'question': '门口接，大厅等他', 'conversation_id': 'skill-session'})
        self.assertEqual(result['task']['skill']['id'], 'welcome')
        self.assertTrue(result['requires_confirmation'])
        memory = self.web.agent_sessions['skill-session']
        self.assertEqual(memory['planned_task']['skill']['id'], 'welcome')
        self.assertEqual(memory['pending_request']['status'], 'awaiting_confirmation')
        prompt = decision_calls(self.web.agent_complete)[-1].args[0][0]['content']
        self.assertIn('previous_turn', prompt)
        self.assertIn('接参考图中的客人', prompt)

    def test_capabilities_come_from_registry_without_starting_tasks(self):
        self.web.agent_complete = model_fixture([reply('我可以迎宾、巡逻、寻物和到点查看。')])
        self.web.agent_chat({'question': '有什么本事'})
        prompt = decision_calls(self.web.agent_complete)[-1].args[0][0]['content']
        for s in SKILLS.values():
            self.assertIn(s.name, prompt)
        self.assertIsNone(self.manager.task)

    def test_reference_is_session_isolated(self):
        self.web.agent_complete = model_fixture([reply('已收到'), reply('已收到')])
        self.web.agent_chat({'question': '参考图', 'reference': REFERENCE, 'conversation_id': 'session-one'})
        self.web.agent_chat({'question': '你好', 'conversation_id': 'session-two'})
        self.assertIsNone(self.web.agent_sessions['session-two']['reference'])

    def test_missing_skill_reference_returns_failure_without_overwriting_task(self):
        self.manager.task = {'id': 'old', 'status': 'planned'}
        self.web.agent_complete = model_fixture([json.dumps({'tool': 'plan_find_object',
            'arguments': {'instruction': '去找它'}}), reply('请先上传参考图片。')])
        result = self.web.agent_chat({'question': '帮我找一下', 'conversation_id': 'missing-reference'})
        self.assertFalse(result['tool_trace'][0]['ok'])
        self.assertNotIn('task', result)
        self.assertEqual(self.manager.task['id'], 'old')

    def test_reference_expiry_is_not_silently_reused(self):
        self.web.agent_complete = model_fixture([reply('已收到')])
        self.web.agent_chat({'question': '参考图', 'reference': REFERENCE, 'conversation_id': 'expiry-session'})
        self.web.reference_path.return_value = Path(__file__).with_name('missing-skill-reference.jpg')
        self.web.agent_complete = model_fixture([json.dumps({'tool': 'plan_find_object',
            'arguments': {'instruction': '找图中的物品'}}), reply('参考图已过期，请重新上传。')])
        self.web.agent_chat({'question': '开始准备吧', 'conversation_id': 'expiry-session'})
        self.assertIsNone(self.web.agent_sessions['expiry-session']['reference'])
        self.assertIsNone(self.manager.task)

    def test_http_inventory_and_legacy_welcome_panel_use_skill_contract(self):
        server = robot_web.RobotWebServer(('127.0.0.1', 0), robot_web.RobotWebHandler, self.web)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/api/skills', timeout=3) as response:
                self.assertEqual(len(json.load(response)['skills']), 5)
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
            payload = {'instruction': '接人', 'reference': REFERENCE, 'pickup_ann_id': 19, 'return_ann_id': 7}
            request = Request(base + '/api/task/welcome/plan', data=json.dumps(payload).encode(),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=3) as response:
                self.assertEqual(json.load(response)['task']['skill']['id'], 'welcome')
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
