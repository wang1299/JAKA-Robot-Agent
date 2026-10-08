"""Resource grounding and task attribution; no camera, speech, motion or model."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_skills import skill_resources
from robot_agent import AgentRunner, Tool
from test_robot_skills import web_state, REFERENCE


def call(tool, **arguments):
    return json.dumps({'tool': tool, 'arguments': arguments}, ensure_ascii=False)


def configured_graph():
    # Arbitrary names/IDs: production logic must not depend on 6/7 or 905/906.
    return {'name': 'other-map', 'objects': [
        {'ann_id': 31, 'category': 'Reception A', 'floor_xy': [1, 2],
         'skill_defaults': {'welcome': 'pickup_ann_id'}},
        {'ann_id': 48, 'category': 'Destination B', 'floor_xy': [3, 4],
         'skill_defaults': {'welcome': 'return_ann_id'}}]}


class ResourceTests(unittest.TestCase):
    def test_unique_defaults_come_from_map(self):
        data = skill_resources('welcome', configured_graph(), True, 'current_turn')
        self.assertEqual(data['default_arguments'], {'pickup_ann_id': 31, 'return_ann_id': 48})
        self.assertTrue(data['reference']['available'])
        self.assertTrue(data['requires_confirmation'])

    def test_duplicates_never_choose_first(self):
        graph = configured_graph()
        other = copy.deepcopy(graph['objects'][0]); other['ann_id'] = 99
        graph['objects'].append(other)
        data = skill_resources('welcome', graph)
        self.assertNotIn('pickup_ann_id', data['default_arguments'])
        self.assertEqual(data['ambiguous_parameters'], ['pickup_ann_id'])
        self.assertEqual(len(data['default_candidates']['pickup_ann_id']), 2)

    def test_inactive_or_invalid_default_not_usable(self):
        for update in ({'lifecycle': 'deleted'}, {'floor_xy': [True, 0]}, {'floor_xy': [float('nan'), 1]},
                       {'skill_defaults': {'welcome': 'instruction'}}, {'skill_defaults': []}):
            graph = configured_graph(); graph['objects'][0].update(update)
            self.assertNotIn('pickup_ann_id', skill_resources('welcome', graph)['default_arguments'])

    def test_names_are_not_implicit_bindings(self):
        graph = configured_graph(); graph['objects'][0].pop('skill_defaults')
        graph['objects'][0]['category'] = 'guest pickup point'
        self.assertNotIn('pickup_ann_id', skill_resources('welcome', graph)['default_arguments'])


class GroundedWelcomeTests(unittest.TestCase):
    def setUp(self):
        self.web = web_state()
        self.web.graph_snapshot.return_value = configured_graph()
        self.web.tasks.task = {'id': 'old-find', 'status': 'succeeded', 'kind': 'find_object'}

    def tearDown(self):
        self.web.capture.assert_not_called()
        self.web.tasks.execute.assert_not_called()
        self.web.speak.assert_not_called()

    def run_turn(self, actions, reference=REFERENCE, **payload):
        from test_robot_agent import model_fixture
        self.web.agent_complete = model_fixture(actions)
        return self.web.agent_chat({'question': '帮我接个人', 'reference': reference, **payload})

    def plan(self, pickup=31, dropoff=48):
        return call('plan_welcome', instruction='接客人送到目的地', pickup_ann_id=pickup, return_ann_id=dropoff)

    def test_resource_query_then_plan_uses_uploaded_reference(self):
        result = self.run_turn([call('get_skill_context', skill_id='welcome'), self.plan()])
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(result['task']['reference'], REFERENCE)
        self.assertEqual(result['task']['pickup_ann_id'], 31)
        self.assertEqual(result['execution']['task_id'], result['task']['id'])
        self.assertEqual(result['execution']['state'], 'planned')

    def test_false_missing_claim_is_repaired_not_shown(self):
        result = self.run_turn([call('ask_user', goal='接人', question='请上传照片并告诉我地点',
                                    skill_id='welcome', missing_inputs=['reference', 'pickup_ann_id', 'return_ann_id']),
                                self.plan()])
        self.assertFalse(result['tool_trace'][0]['ok'])
        self.assertTrue(result['requires_confirmation'])
        self.assertNotIn('请上传', result['text'])

    def test_genuinely_missing_photo_only_asks_photo(self):
        result = self.run_turn([call('get_skill_context', skill_id='welcome'),
            call('ask_user', goal='接客人', question='请上传客人的参考照片。', skill_id='welcome', missing_inputs=['reference'])], reference=None)
        self.assertTrue(result['requires_clarification'])
        self.assertEqual(result['execution']['state'], 'none')
        self.assertIsNone(result['execution']['task_id'])
        self.assertEqual(self.web.tasks.task['id'], 'old-find')

    def test_missing_default_can_be_asked_without_losing_photo(self):
        self.web.graph_snapshot.return_value['objects'][1].pop('skill_defaults')
        result = self.run_turn([call('ask_user', goal='接人', question='接到后送到哪里？', skill_id='welcome', missing_inputs=['return_ann_id'])], conversation_id='followup')
        self.assertTrue(result['requires_clarification'])
        result = self.run_turn([self.plan()], reference=None, conversation_id='followup')
        self.assertEqual(result['task']['reference'], REFERENCE)

    def test_ambiguous_points_allow_clarification(self):
        other=copy.deepcopy(self.web.graph_snapshot.return_value['objects'][0]);other['ann_id']=99
        self.web.graph_snapshot.return_value['objects'].append(other)
        result=self.run_turn([call('get_skill_context',skill_id='welcome'),call('ask_user',goal='接人',question='两个接人点选哪个？',skill_id='welcome',missing_inputs=['pickup_ann_id'])])
        self.assertTrue(result['requires_clarification'])
        self.assertEqual(self.web.tasks.task['id'],'old-find')

    def test_explicit_alternative_not_overridden(self):
        result=self.run_turn([self.plan(pickup=48,dropoff=31)])
        self.assertEqual(result['task']['pickup_ann_id'],48)
        self.assertEqual(result['task']['return_ann_id'],31)

    def test_previous_photo_may_need_reconfirmation_for_new_person(self):
        self.run_turn([call('ask_user',goal='接人',question='请确认需求',skill_id='welcome',missing_inputs=[])],conversation_id='new-person')
        result=self.run_turn([call('ask_user',goal='接另一个人',question='请上传另一位客人的照片',skill_id='welcome',missing_inputs=['reference'])],reference=None,conversation_id='new-person')
        self.assertTrue(result['requires_clarification'])


class TaskAttributionTests(unittest.TestCase):
    def test_preparing_action_cannot_end_as_plain_answer(self):
        task = {'id': 'new', 'status': 'planned'}
        runner = AgentRunner(Mock(side_effect=[call('resources'),
            call('finish_response', intent='answer', text='我准备好了'), call('plan')]),
            {'resources': Tool('准备资源', {}, lambda: {'ok': True}, effect='prepare'),
             'plan': Tool('规划', {}, lambda: {'task': task}, effect='plan')},
            native=True, require_final_tool=True)
        result = runner.run('接人')
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(result['task']['id'], 'new')

    def test_configured_decision_endpoint_requires_structured_response(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        web = web_state()
        client = Mock()
        client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='hello', tool_calls=None))])
        with patch.dict(os.environ, {'JAKA_AGENT_BASE_URL': 'http://127.0.0.1:8001/v1'}, clear=True), patch.dict(sys.modules, {
            'openai': SimpleNamespace(OpenAI=Mock(return_value=client)),
            'qwen_planner': SimpleNamespace(PLAN_MODEL='model', _client=Mock())}):
            web.agent_complete([], [{'function': {'name': 'finish_response'}}])
        self.assertEqual(client.with_options.return_value.chat.completions.create.call_args.kwargs['tool_choice'], 'required')

    def test_old_completed_task_not_attached_to_new_answer(self):
        task={'id':'old','status':'succeeded'}
        runner=AgentRunner(Mock(return_value=call('finish_response',intent='answer',text='你好')),{},native=True,require_final_tool=True,task_snapshot=lambda:task)
        result=runner.run('你好')
        self.assertEqual(result['execution']['state'],'none')
        self.assertEqual(result['execution']['scope'],'current_turn')

    def test_explicit_status_query_still_reports_real_task(self):
        task={'id':'old','status':'succeeded'}
        runner=AgentRunner(Mock(side_effect=[call('get_robot_status'),call('finish_response',intent='task_status',text='状态')]),
            {'get_robot_status':Tool('状态',{},lambda:{'task':task})},native=True,require_final_tool=True,task_snapshot=lambda:task)
        result=runner.run('上个任务完成了吗')
        self.assertEqual(result['execution']['task_id'],'old')
        self.assertEqual(result['execution']['scope'],'task')


if __name__ == '__main__':
    unittest.main()
