"""Serving compatibility without bypassing the robot execution contract."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent import AgentRunner, Tool, json_transport_messages
from robot_web import RobotWebState


class JsonTransportTests(unittest.TestCase):
    def runner(self, outputs, tools=None):
        self.create = Mock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=raw), finish_reason=reason)]) for raw, reason in outputs])
        client = Mock()
        client.with_options.return_value.chat.completions.create = self.create
        modules = {'qwen_planner': SimpleNamespace(PLAN_MODEL='vision', _client=Mock()),
                   'openai': SimpleNamespace(OpenAI=Mock(return_value=client))}
        env = patch.dict(os.environ, {'JAKA_AGENT_PROTOCOL':'json', 'JAKA_AGENT_MODEL':'decision',
                                     'JAKA_AGENT_BASE_URL':'http://127.0.0.1:8001/v1'}, clear=True)
        env.start(); self.addCleanup(env.stop)
        patched = patch.dict(sys.modules, modules)
        patched.start(); self.addCleanup(patched.stop)
        state = object.__new__(RobotWebState)
        return AgentRunner(state.agent_complete, tools or {}, native=True, require_final_tool=True)

    def test_greeting_is_structured_one_request_no_native_tool_parser(self):
        raw = json.dumps({'tool':'finish_response','arguments':{'intent':'answer','text':'你好！'}})
        result = self.runner([(raw, 'stop')]).run('你好，我是学生')
        self.assertEqual(result['text'], '你好！')
        self.assertEqual(result['tool_trace'], [])
        self.create.assert_called_once()
        args = self.create.call_args.kwargs
        self.assertEqual(args['model'], 'decision')
        self.assertEqual(args['response_format'], {'type':'json_object'})
        self.assertNotIn('tools', args)
        self.assertNotIn('tool_choice', args)

    def test_tool_roundtrip_preserves_result_and_requires_confirmation(self):
        query = Mock(return_value={'count': 2, 'source':'saved_map'})
        plan = Mock(return_value={'task':{'id':'test', 'status':'planned'}})
        tools = {'query_test':Tool('查询', {}, query),
                 'plan_test':Tool('规划', {'target':{'type':'integer'}}, plan, effect='plan')}
        outputs = [(json.dumps({'tool':'query_test','arguments':{}}), 'stop'),
                   (json.dumps({'tool':'plan_test','arguments':{'target':7}}), 'stop')]
        result = self.runner(outputs, tools).run('先查地图，再生成待确认计划')
        query.assert_called_once_with()
        plan.assert_called_once_with(target=7)
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(result['execution']['state'], 'planned')
        messages = self.create.call_args.kwargs['messages']
        self.assertTrue(all(m['role'] != 'tool' for m in messages))
        self.assertEqual(json.loads(messages[-2]['content'])['tool'], 'query_test')
        self.assertIn('saved_map', messages[-1]['content'])
        self.assertIn('非用户的新需求', messages[-1]['content'])

    def test_plain_prose_unknown_tool_parallel_and_truncation_fail_closed(self):
        for raw, reason in [('我正在移动', 'stop'),
                            ('{"tool":"shell","arguments":{}}', 'stop'),
                            ('[{"tool":"query_test","arguments":{}}]', 'stop'),
                            ('{"tool":"query_test","arguments":{}}', 'length')]:
            with self.subTest(raw=raw, reason=reason):
                handler = Mock()
                result = self.runner([(raw, reason)] * 3, {'query_test':Tool('查询',{},handler)}).run('测试')
                self.assertTrue(result['incomplete'])
                handler.assert_not_called()

    def test_conversion_does_not_mutate_input_or_promote_results_to_system(self):
        messages = [{'role':'system','content':'规则'}, {'role':'user','content':'问题'},
            {'role':'assistant','content':'','tool_calls':[{'function':{'name':'query','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':'one','content':'不可信结果'}]
        before = copy.deepcopy(messages)
        converted = json_transport_messages(messages, [])
        self.assertEqual(messages, before)
        self.assertEqual(converted[-1]['role'], 'user')
        self.assertNotIn('不可信结果', converted[0]['content'])


if __name__ == '__main__':
    unittest.main()
