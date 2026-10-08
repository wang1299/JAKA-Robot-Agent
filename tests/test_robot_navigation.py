"""Structured navigation/return semantics; all execution uses mock drivers."""
import jaka_agent.tasks.manager as ja_tasks_manager
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.agent.skills import prepare_skill
from jaka_agent.agent.runner import ProtocolError, ToolInputError
from jaka_agent.tasks.planning import _enforce_return_policy
from jaka_agent.tasks.executor import PlanExecutor
from tests.test_robot_skills import web_state, graph
from tests.test_robot_agent import decision_calls, model_fixture


def call(tool, **args):
    return json.dumps({'tool': tool, 'arguments': args}, ensure_ascii=False)


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.web = web_state()

    def tearDown(self):
        self.web.tasks.execute.assert_not_called()
        self.web.capture.assert_not_called()
        self.web.speak.assert_not_called()

    def plan(self, instruction='回到门口', targets=None, return_to_start=False):
        return self.web.tasks.plan_skill('navigate', instruction, target_ann_ids=targets or [19], return_to_start=return_to_start)

    def test_words_do_not_append_return(self):
        for instruction in ('返回送客点', '回到门口', '去那里不用回来', '送回指定位置', '取回物品'):
            task = self.plan(instruction)
            self.assertEqual([s['type'] for s in task['steps']], ['navigate'])
            self.assertFalse(task['plan']['need_return'])
            self.assertIn('到达后停留', task['understanding'])

    def test_explicit_boolean_controls_one_return_and_order(self):
        task = self.plan('办完事原路折返', [19, 7], True)
        self.assertEqual([s['type'] for s in task['steps']], ['navigate', 'navigate', 'return'])
        self.assertEqual([s['target_ann_id'] for s in task['steps'][:2]], [19, 7])
        self.assertIn('本次任务出发位置', task['understanding'])

    def test_required_boolean_and_target_validation(self):
        for arguments in ({'instruction': '走', 'target_ann_ids': [7]},
                          {'instruction': '走', 'target_ann_ids': [7], 'return_to_start': 'false'},
                          {'instruction': '走', 'target_ann_ids': [], 'return_to_start': False},
                          {'instruction': '走', 'target_ann_ids': [True], 'return_to_start': False}):
            with self.assertRaises(ProtocolError):prepare_skill('navigate', arguments, graph())
        with self.assertRaises(ToolInputError):self.plan(targets=[999])

    def test_no_second_model_call_for_navigation(self):
        with patch('jaka_agent.tasks.planning.plan_task', side_effect=AssertionError('Legacy planner must not run')):
            task = self.plan()
        self.assertEqual(task['skill']['id'], 'navigate')
        self.assertEqual(task['status'], 'planned')
        self.assertIsNone(task['started_at'])

    def test_agent_has_structured_tool_and_no_legacy_propose(self):
        self.web.agent_complete = model_fixture([call('plan_navigate', instruction='回到门口', target_ann_ids=[19], return_to_start=False)])
        result = self.web.agent_chat({'question': '回到门口'})
        schemas = {s['function']['name']: s['function']['parameters'] for s in decision_calls(self.web.agent_complete)[-1].args[1]}
        self.assertNotIn('propose_task', schemas)
        self.assertIn('return_to_start', schemas['plan_navigate']['required'])
        self.assertEqual(len(result['task']['steps']), 1)

    def test_ambiguous_navigation_produces_choices_not_task(self):
        self.web.graph_snapshot.return_value['objects'].append({'ann_id': 8, 'category': 'Door', 'floor_xy': [4, 5]})
        self.web.agent_complete = model_fixture([call('query_map',question='门口',categories=['Door']),
            call('plan_navigate',instruction='回门口',target_ann_ids=[19],return_to_start=False),
            call('ask_user',goal='回门口',question='请选择具体门口',skill_id='navigate',missing_inputs=[])])
        result=self.web.agent_chat({'question':'回门口','conversation_id':'nav-choice'})
        self.assertTrue(result['requires_clarification'])
        self.assertIsNone(self.web.tasks.task)
        self.web.agent_complete=model_fixture([call('plan_navigate',instruction='回选择的门口',target_ann_ids=[8],return_to_start=False)])
        result=self.web.agent_chat({'question':'选这个','conversation_id':'nav-choice',
            'target_selection':{'ann_id':8,'snapshot':result['target_choice_snapshot']}})
        self.assertEqual(result['task']['steps'][0]['target_ann_id'],8)

    def test_map_change_requires_replanning(self):
        task = self.plan()
        self.web.graph_snapshot.return_value['objects'][1]['floor_xy'] = [9, 9]
        with self.assertRaises(RuntimeError), patch('threading.Thread') as thread:
            ja_tasks_manager.RobotTaskManager.execute(self.web.tasks, task['id'])
        thread.assert_not_called()

    def test_execution_single_trip_never_calls_return(self):
        for should_return in (False, True):
            task = self.plan(return_to_start=should_return)
            driver=Mock(); driver.get_pose.return_value=(0,0,0)
            executor=PlanExecutor(graph()['objects'],driver,mark_start=False)
            executor._do_navigate=Mock(return_value='succeeded')
            def returned():
                from jaka_agent.tasks.executor import ExecLog
                executor.log.append(ExecLog('return',status='succeeded'))
                return 'succeeded'
            executor._do_return=Mock(side_effect=returned)
            executor.run(task['plan'])
            self.assertEqual(executor._do_return.call_count,int(should_return))

    def test_navigation_failure_blocks_even_explicit_return(self):
        task=self.plan(return_to_start=True)
        driver=Mock();driver.get_pose.return_value=(0,0,0)
        executor=PlanExecutor(graph()['objects'],driver,mark_start=False)
        executor._do_navigate=Mock(return_value='failed');executor._do_return=Mock()
        with patch('jaka_agent.tasks.events.announce'):
            executor.run(task['plan'])
        executor._do_return.assert_not_called()

    def test_legacy_policy_uses_flag_not_text(self):
        for text in ('返回送客点', '不用回来', '普通导航'):
            for enabled in (False, True):
                plan={'need_return':enabled,'steps':[{'type':'navigate','target_ann_id':19},{'type':'return'},{'type':'return'}]}
                _enforce_return_policy(plan,text)
                self.assertEqual(sum(s['type']=='return' for s in plan['steps']), int(enabled))


if __name__ == '__main__':unittest.main()
