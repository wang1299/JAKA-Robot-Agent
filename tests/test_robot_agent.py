"""Agent protocol, safety boundaries and web integration; no real hardware."""
import jaka_agent.web.handler as ja_web_handler
import jaka_agent.web.server as ja_web_server
import jaka_agent.web.state as ja_web_state
import io
import json
import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.agent.runner import AgentRunner, Tool, ProtocolError, parse_action, validate_arguments


def action(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False)


def decision_calls(complete):
    return [call for call in complete.call_args_list
            if [t['function']['name'] for t in call.args[1]] != ['set_scene_requirement']]


def model_fixture(outputs, scene_scope="none"):
    """Transport fixture, not a claim about real-model semantic accuracy."""
    answers = iter(outputs)
    def complete(messages, tools):
        assert not any(tool['function']['name'].startswith('review_') for tool in tools)
        if [t['function']['name'] for t in tools] == ['set_scene_requirement']:
            return action('set_scene_requirement', scope=scene_scope, reason='测试指定的证据要求')
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        # Web now requires an explicit final-response tool. Keep fixture answers
        # readable while exercising the real production protocol.
        if any(tool['function']['name'] == 'finish_response' for tool in tools):
            try:
                if parse_action(answer) is None:
                    answer = action('finish_response', intent='answer', text=answer)
            except ProtocolError:
                pass
        return answer
    return Mock(side_effect=complete)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.handler = Mock(return_value={"ok": True, "observation": "一把蓝色椅子"})
        self.tools = {"observe_scene": Tool("查看现场", {
            "question": {"type": "string"}, "fresh": {"type": "boolean", "default": True},
        }, self.handler)}

    def run_agent(self, outputs, **kwargs):
        self.complete = Mock(side_effect=outputs)
        return AgentRunner(self.complete, self.tools, **kwargs).run("你面前有什么？")

    def test_chat_does_not_call_tools(self):
        result = self.run_agent(["你好，我是小卡。"])
        self.assertEqual(result["tool_trace"], [])
        self.handler.assert_not_called()

    def test_observation_returns_to_model(self):
        result = self.run_agent([action("observe_scene", question="前面有什么"), "前面有一把蓝色椅子。"])
        self.assertIn("蓝色椅子", result["text"])
        self.handler.assert_called_once_with(question="前面有什么", fresh=True)
        self.assertIn("蓝色椅子", self.complete.call_args.args[0][-1]["content"])

    def test_chained_tools(self):
        status = Mock(return_value={"ok": True, "online": True})
        self.tools["get_robot_status"] = Tool("状态", {}, status)
        result = self.run_agent([action("get_robot_status"), action("observe_scene", question="看看"), "已查看。"])
        self.assertEqual(len(result["tool_trace"]), 2)
        status.assert_called_once()

    def test_invalid_json_is_repaired_before_call(self):
        self.run_agent(['{"tool":“observe_scene”}', action("observe_scene", question="看看"), "看到了。"])
        self.handler.assert_called_once()

    def test_unknown_dangerous_tool_never_executed(self):
        result = self.run_agent([action("execute_task", task_id="a")] * 3)
        self.assertTrue(result["incomplete"])
        self.handler.assert_not_called()

    def test_extra_parameters_rejected(self):
        result = self.run_agent([action("observe_scene", question="看看", command="move")] * 3)
        self.assertTrue(result["incomplete"])
        self.handler.assert_not_called()

    def test_argument_types_and_required(self):
        for args in ({}, {"question": []}, {"question": "看看", "fresh": "false"}, {"question": "x" * 1201}):
            with self.subTest(args=args), self.assertRaises(ProtocolError):
                validate_arguments(self.tools["observe_scene"], args)

    def test_duplicate_call_stops(self):
        result = self.run_agent([action("observe_scene", question="看看")] * 2)
        self.assertTrue(result["incomplete"])
        self.handler.assert_called_once()

    def test_tool_budget(self):
        result = self.run_agent([action("observe_scene", question="第一次"), action("observe_scene", question="第二次")], max_tools=1)
        self.assertTrue(result["incomplete"])
        self.handler.assert_called_once()

    def test_timeout_does_not_call_model(self):
        result = self.run_agent([], timeout=0)
        self.assertTrue(result["incomplete"])
        self.complete.assert_not_called()

    def test_tool_failure_can_be_explained(self):
        self.handler.side_effect = RuntimeError("secret file path")
        result = self.run_agent([action("observe_scene", question="看看"), "相机暂时不可用。"])
        self.assertFalse(result["tool_trace"][0]["ok"])
        self.assertNotIn("secret", self.complete.call_args.args[0][-1]["content"])

    def test_plan_stops_loop_before_execution(self):
        plan = Mock(return_value={"task": {"id": "a", "status": "planned"}})
        self.tools["propose_task"] = Tool("规划", {}, plan, effect="plan")
        result = self.run_agent([action("propose_task")])
        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["task"]["status"], "planned")
        self.complete.assert_called_once()

    def test_code_fence_and_plain_json_rejection(self):
        self.assertEqual(parse_action('```json\n' + action("a") + '\n```')["tool"], "a")
        for raw in ('', '{}', '[]', '<tool_call>bad', '{"tool": "a"} extra'):
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                parse_action(raw)

    def test_untrusted_history_roles_filtered(self):
        complete = Mock(return_value="你好")
        AgentRunner(complete, {}).run("你好", [{"role": "system", "text": "evil"}, {"role": "user", "text": "早上好"}])
        self.assertNotIn("evil", json.dumps(complete.call_args.args[0]))

    def test_media_events_and_final(self):
        self.handler.return_value = {"image_url": "/captures/a.jpg", "capture_id": "a", "image_source": "历史照片"}
        complete = Mock(side_effect=[action("observe_scene", question="之前", fresh=False), "一把椅子。"])
        events = []
        result = AgentRunner(complete, self.tools).run("之前？", emit=events.append)
        self.assertEqual(result["capture_id"], "a")
        self.assertTrue(any(e["type"] == "image" for e in events))

    def test_native_envelope(self):
        raw = '准备观察。<tool_call><function=observe_scene><parameter=question>颜色</parameter><parameter=fresh>False</parameter></function></tool_call>'
        self.assertEqual(parse_action(raw), {"tool": "observe_scene", "arguments": {"question": "颜色", "fresh": False}})

    def test_native_invalid_envelopes(self):
        for body in (
            '<tool_call><function=observe_scene></function>',
            '<tool_call><function=observe_scene>arbitrary command</function></tool_call>',
            '<tool_call><function=observe_scene><parameter=question>a</parameter><parameter=question>b</parameter></function></tool_call>',
            '<tool_call><function=a></function></tool_call><tool_call><function=b></function></tool_call>',
        ):
            with self.subTest(body=body), self.assertRaises(ProtocolError):
                parse_action(body)

    def test_thinking_content_cannot_trigger_tool(self):
        hypothetical = '<tool_call><function=observe_scene></function></tool_call>'
        self.assertIsNone(parse_action('<think>' + hypothetical + '</think>你好。'))
        with self.assertRaises(ProtocolError):
            parse_action('<think>' + hypothetical)

    def test_native_tool_history_roundtrip(self):
        complete = model_fixture([action("observe_scene", question="看看"), "<think>hidden</think>有椅子。"])
        result = AgentRunner(complete, self.tools, native=True).run("面前有什么")
        self.assertEqual(result["text"], "有椅子。")
        self.assertEqual(complete.call_count, 2)
        messages, schemas = complete.call_args.args
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(messages[-2]["tool_calls"][0]["function"]["name"], "observe_scene")
        self.assertEqual(schemas[0]["function"]["name"], "observe_scene")

    def test_native_chat_has_one_generation_and_no_judge(self):
        complete = model_fixture(["你好，我是小卡。"])
        result = AgentRunner(complete, self.tools, native=True).run("你好")
        self.assertEqual(result["text"], "你好，我是小卡。")
        complete.assert_called_once()
        self.handler.assert_not_called()

    def test_public_tool_schema_is_small_but_local_limits_still_apply(self):
        tool = Tool('查询', {'names': {'type': 'array', 'maxItems': 1,
                    'items': {'type': 'string', 'maxLength': 8, 'enum': ['A', 'B']}}}, lambda: None)
        schema = AgentRunner(Mock(), {'query': tool}).schemas()[0]['function']['parameters']['properties']['names']
        self.assertNotIn('maxItems', schema)
        self.assertNotIn('maxLength', schema['items'])
        self.assertEqual(schema['items']['enum'], ['A', 'B'])
        with self.assertRaises(ProtocolError):
            validate_arguments(tool, {'names': ['A', 'B']})

    def test_historical_photo_choice_is_made_by_model(self):
        complete = model_fixture([action("observe_scene", question="颜色", fresh=False), "蓝色。"])
        AgentRunner(complete, self.tools, native=True).run("刚才椅子的颜色是什么", context={"has_previous_scene_image": True})
        self.handler.assert_called_once_with(question="颜色", fresh=False)

    def test_resource_context_stays_in_first_system_message(self):
        complete = model_fixture(["你好。"])
        AgentRunner(complete, self.tools, native=True).run("你好", context={"has_uploaded_image": False})
        messages = complete.call_args_list[0].args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])

    def test_clarification_pauses_without_following_tools(self):
        self.tools['ask_user'] = Tool('澄清', {}, lambda: {'clarification': '你是问哪个区域？',
            'pending_request': {'goal': '查找区域物体', 'status': 'needs_clarification'}})
        complete = model_fixture([action('ask_user')])
        result = AgentRunner(complete, self.tools, native=True).run('那里有什么')
        self.assertTrue(result['requires_clarification'])
        self.assertEqual(result['pending_request']['goal'], '查找区域物体')
        complete.assert_called_once()

    def test_array_and_integer_native_arguments(self):
        parsed = parse_action('<tool_call><function=map><parameter=object_ids>["1", "2"]</parameter><parameter=page>2</parameter></function></tool_call>')
        self.assertEqual(parsed["arguments"], {"object_ids": ["1", "2"], "page": 2})

    def test_argument_arrays_and_pages_are_bounded(self):
        tool = Tool("", {"ids": {"type": "array", "maxItems": 2}, "page": {"type": "integer", "minimum": 1, "maximum": 3}}, lambda: None)
        for args in ({"ids": [1], "page": 1}, {"ids": ["a"]*3, "page": 1}, {"ids": [], "page": True}, {"ids": [], "page": 4}):
            with self.assertRaises(ProtocolError):
                validate_arguments(tool, args)

    def test_nested_filter_schema_rejects_arbitrary_fields(self):
        from jaka_agent.agent.map_evidence import FILTER_SCHEMA
        tool = Tool('', {'filters': FILTER_SCHEMA}, lambda: None)
        valid = {'field': 'position', 'operator': 'contains', 'value': '会议室'}
        self.assertEqual(validate_arguments(tool, {'filters': [valid]}), {'filters': [valid]})
        for value in ({**valid, 'command': 'move'}, {**valid, 'field': '__dict__'}, {**valid, 'operator': 'eval'}):
            with self.assertRaises(ProtocolError):
                validate_arguments(tool, {'filters': [value]})


class WebAgentTests(unittest.TestCase):
    def setUp(self):
        self.state = object.__new__(ja_web_state.RobotWebState)
        self.state.mock = False
        self.state.agent_lock = threading.Lock()
        self.state.tasks = SimpleNamespace(snapshot=Mock(return_value=None), execute=Mock(),
            is_busy=lambda: bool((self.state.tasks.snapshot() or {}).get('status') in ('running', 'canceling')),
            plan=Mock(return_value={"id": "p", "status": "planned"}),
            plan_skill=Mock(return_value={"id": "p", "status": "planned"}),
            plan_find_object=Mock(), robot_status=Mock(return_value={"online": True}))
        self.state.mapping = SimpleNamespace(is_busy=Mock(return_value=False))
        self.state.capture = Mock(return_value=("a" * 32, Path(__file__)))
        self.state.agent_vision = Mock(return_value="蓝色椅子")
        self.state.graph_snapshot = Mock(return_value={"name": "demo", "objects": []})

    def call(self, outputs, question="看看前面", **payload):
        self.state.agent_complete = model_fixture(outputs)
        return self.state.agent_chat({"question": question, **payload})

    def test_camera_and_final(self):
        result = self.call([action("observe_scene", question="前面"), "有蓝色椅子。"])
        self.assertEqual(result["capture_id"], "a" * 32)
        self.state.capture.assert_called_once()
        self.state.tasks.execute.assert_not_called()

    def test_history_image_reused(self):
        with patch.object(self.state, "capture_path", return_value=Path(__file__)):
            self.call([action("inspect_previous_scene", question="颜色"), "蓝色。"],
                      history=[{"role": "assistant", "text": "椅子", "capture_id": "b" * 32}])
        self.state.capture.assert_not_called()
        self.state.agent_vision.assert_called_once()

    def test_missing_previous_image_does_not_silently_capture(self):
        result = self.call([action("inspect_previous_scene", question="颜色"), "历史照片不可用。"])
        self.assertEqual(result["tool_trace"], [])
        self.state.capture.assert_not_called()

    def test_busy_task_does_not_take_camera(self):
        self.state.tasks.snapshot.return_value = {"status": "running"}
        self.call([action("observe_scene", question="看看"), "请先停止任务。"])
        self.state.capture.assert_not_called()

    def test_proposal_cannot_execute(self):
        result = self.call([action("plan_navigate", instruction="去门口", target_ann_ids=[7], return_to_start=False)])
        self.assertTrue(result["requires_confirmation"])
        self.state.tasks.execute.assert_not_called()

    def test_reference_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.call([], reference={"reference_id": "../../secret", "suffix": ".jpg"})

    def test_removed_legacy_find_tool_cannot_plan(self):
        self.call([action("propose_find_object", instruction="找它"), "请上传参考图。"])
        self.state.tasks.plan_find_object.assert_not_called()

    def test_request_validation(self):
        for payload in ({"question": ""}, {"question": "hi", "history": {}}, {"question": ["hi"]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.state.agent_chat(payload)

    def test_parallel_request_fails_without_model_or_hardware(self):
        self.state.agent_lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                self.call(["你好"])
            self.state.agent_complete.assert_not_called()
        finally:
            self.state.agent_lock.release()

    def test_lock_released_after_model_error(self):
        with self.assertRaises(RuntimeError):
            self.call([RuntimeError("unavailable")])
        self.assertFalse(self.state.agent_lock.locked())

    def test_model_transport_uses_native_tools_and_timeouts(self):
        message = SimpleNamespace(content='<tool_call><function=get_robot_status></function></tool_call>', tool_calls=None)
        client = Mock()
        client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        module = SimpleNamespace(PLAN_MODEL='local-model', _client=lambda: client)
        with patch.dict(sys.modules, {'jaka_agent.models.settings': module, 'jaka_agent.models.runtime': module}):
            raw = self.state.agent_complete([{'role': 'user', 'content': '电量'}], [{'type': 'function'}])
        self.assertEqual(parse_action(raw)['tool'], 'get_robot_status')
        client.with_options.assert_called_once_with(timeout=45, max_retries=0)
        self.assertEqual(client.with_options.return_value.chat.completions.create.call_args.kwargs['tool_choice'], 'auto')

    def test_transport_also_accepts_parsed_tool_calls(self):
        message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(function=SimpleNamespace(name='get_robot_status', arguments='{}'))])
        client = Mock()
        client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        with patch.dict(sys.modules, {'jaka_agent.models.settings': SimpleNamespace(PLAN_MODEL='local'), 'jaka_agent.models.runtime': SimpleNamespace(_client=lambda: client)}):
            self.assertEqual(parse_action(self.state.agent_complete([], [])), {'tool': 'get_robot_status', 'arguments': {}})

    def test_decision_endpoint_is_configurable_without_reusing_other_credentials(self):
        client = Mock()
        client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="你好", tool_calls=None))])
        constructor = Mock(return_value=client)
        fallback = Mock()
        modules = {'qwen_planner': SimpleNamespace(PLAN_MODEL='vision-default', _client=fallback),
                   'openai': SimpleNamespace(OpenAI=constructor)}
        with patch.dict(os.environ, {'JAKA_AGENT_BASE_URL': 'http://127.0.0.1:9000/v1', 'JAKA_AGENT_MODEL': 'decision-model'}, clear=True), patch.dict(sys.modules, modules):
            self.state.agent_complete([], [])
        constructor.assert_called_once_with(base_url='http://127.0.0.1:9000/v1', api_key='EMPTY')
        fallback.assert_not_called()
        self.assertEqual(client.with_options.return_value.chat.completions.create.call_args.kwargs['model'], 'decision-model')

    def test_project_metrics_are_model_selected_reference_tool(self):
        result = self.call([action("query_project_info"), "项目资料记录105,000个节点。"], question='查询语义图谱规模')
        self.assertIn('105,000', result['text'])
        self.assertEqual(result["tool_trace"][0]["tool"], "query_project_info")

    def test_map_count_includes_every_object_in_selected_categories(self):
        self.state.graph_snapshot.return_value = {"name": "demo", "objects": [
            {"ann_id": 21, "category": "Lamp"}, {"ann_id": 22, "category": "Desk light"},
            {"ann_id": 23, "category": "Table"}]}
        result = self.call([action("query_map", question="照明设备", categories=['Lamp', 'Desk light']),
            "保存地图里有2件灯具。"], question="保存的数据中照明设备一共几件？")
        self.assertEqual(result["target_ids"], ["21", "22"])
        self.assertEqual([s["tool"] for s in result["tool_trace"]], ["query_map"])
        self.assertEqual(len(decision_calls(self.state.agent_complete)), 2)
        self.state.capture.assert_not_called()

    def test_session_remembers_goal_and_evidence_without_keyword_resolution(self):
        self.call([action("observe_scene", question="描述"), "照片中有蓝色椅子。"],
                  question="看看周围", conversation_id="session-one")
        self.call(["之前照片里提到的是蓝色椅子。"], question="那个呢", conversation_id="session-one")
        prompt = json.dumps(decision_calls(self.state.agent_complete)[0].args[0], ensure_ascii=False)
        self.assertIn("看看周围", prompt)
        self.assertIn("previous_evidence_not_live", prompt)
        self.assertIn("蓝色椅子", prompt)
        self.call(["你好。"], question="你好", conversation_id="session-two")
        prompt = json.dumps(decision_calls(self.state.agent_complete)[0].args[0], ensure_ascii=False)
        self.assertNotIn("蓝色椅子", prompt)

    def test_session_expiry_and_capacity(self):
        self.state.agent_sessions = {"expired": {"updated_at": 0, "history": [{"role": "user", "text": "过期秘密"}], "evidence": []}}
        self.call(["你好。"], question="你好", conversation_id="session-new")
        self.assertNotIn("expired", self.state.agent_sessions)
        self.assertEqual(len(self.state.agent_sessions), 1)

    def test_clarification_goal_survives_short_reply_then_query_resolves_it(self):
        sid = 'clarify-session'
        self.call([action('ask_user', goal='查找指定区域的灯具数量', question='你指哪个区域？', skill_id='none', missing_inputs=[])],
                  question='那里有多少灯', conversation_id=sid)
        self.assertEqual(self.state.agent_sessions[sid]['pending_request']['status'], 'needs_clarification')
        self.call([action('query_map', question='大厅灯具数量', categories=[],
                          filters=[{'field': 'position', 'operator': 'contains', 'value': '大厅'}]),
                   '地图没有符合条件的对象。'], question='大厅', conversation_id=sid)
        prompt = decision_calls(self.state.agent_complete)[0].args[0][0]['content']
        self.assertIn('查找指定区域的灯具数量', prompt)
        memory = self.state.agent_sessions[sid]
        self.assertIsNone(memory['pending_request'])
        self.assertEqual(memory['focus']['source_type'], 'saved_map')
        self.assertEqual(memory['focus']['selection']['filters'][0]['value'], '大厅')

    def test_failed_tool_retains_goal_for_retry(self):
        self.state.capture.side_effect = RuntimeError('hardware unavailable')
        self.call([action('observe_scene', question='确认门是否打开'), '相机暂时不可用。'],
                  question='门开着吗', conversation_id='retry-session')
        memory = self.state.agent_sessions['retry-session']
        self.assertEqual(memory['pending_request'], {'goal': '确认门是否打开', 'status': 'tool_failed'})
        self.assertEqual(memory['evidence'], [])
        self.call(['可以稍后再试。'], question='那稍后再说', conversation_id='retry-session')
        self.assertIn('确认门是否打开', decision_calls(self.state.agent_complete)[0].args[0][0]['content'])

    def test_model_failure_persists_unfinished_exchange(self):
        with self.assertRaises(RuntimeError):
            self.call([RuntimeError('offline')], question='查询大厅灯具', conversation_id='error-session')
        memory = self.state.agent_sessions['error-session']
        self.assertEqual(memory['pending_request']['status'], 'interrupted')
        self.assertEqual(memory['pending_request']['goal'], '查询大厅灯具')
        self.assertTrue(memory['last_exchange']['incomplete'])

    def test_photo_memory_survives_history_window_and_scope_switch(self):
        sid = 'photo-session'
        with patch.object(self.state, 'capture_path', return_value=Path(__file__)):
            self.call([action('inspect_previous_scene', question='描述旧照片'), '蓝色椅子。'],
                      history=[{'role': 'assistant', 'text': '旧照片', 'capture_id': 'b' * 32}], conversation_id=sid)
            for _ in range(9):
                self.call(['你好'], question='你好', conversation_id=sid)
            self.call([action('query_map', question='保存地图总数', categories=['*']), '地图中为0。'], conversation_id=sid)
            self.assertEqual(self.state.agent_sessions[sid]['focus']['source_type'], 'saved_map')
            self.call([action('inspect_previous_scene', question='之前照片颜色'), '旧照片里是蓝色。'], conversation_id=sid)
        memory = self.state.agent_sessions[sid]
        self.assertEqual(memory['last_scene']['capture_id'], 'b' * 32)
        self.assertEqual(memory['focus']['source_type'], 'scene_photo')
        self.assertIn('不是实时', memory['evidence'][-1]['image_source'])
        self.assertIsInstance(memory['evidence'][-1]['observed_at'], float)
        self.state.capture.assert_not_called()

    def test_plan_memory_is_not_execution_and_status_is_authoritative(self):
        sid = 'plan-session'
        self.call([action('plan_navigate', instruction='去大厅', target_ann_ids=[7], return_to_start=False)], conversation_id=sid)
        self.call(['请点击任务卡确认执行。'], question='对的', conversation_id=sid)
        self.assertEqual(self.state.agent_sessions[sid]['planned_task']['status'], 'planned')
        self.assertIn('planned_task_not_execution_proof', decision_calls(self.state.agent_complete)[0].args[0][0]['content'])
        self.state.tasks.execute.assert_not_called()
        self.state.tasks.snapshot.return_value = {'id': 'p', 'status': 'completed', 'result_text': '到达大厅'}
        self.call([action('get_robot_status'), '任务已完成。'], conversation_id=sid)
        memory = self.state.agent_sessions[sid]
        self.assertEqual(memory['planned_task']['status'], 'completed')
        self.assertEqual(memory['evidence'][-1]['task']['result_text'], '到达大厅')

    def test_task_needing_clarification_is_not_ready_to_execute(self):
        self.state.tasks.plan_skill.return_value = {'id': 'p', 'status': 'needs_clarification',
            'clarification': {'question': '哪个大厅？'}}
        self.call([action('plan_navigate', instruction='去大厅', target_ann_ids=[7], return_to_start=False)], conversation_id='plan-clarify')
        self.assertEqual(self.state.agent_sessions['plan-clarify']['pending_request']['status'], 'needs_clarification')
        self.state.tasks.execute.assert_not_called()

    def test_current_map_metric_request_has_no_hardcoded_shortcut(self):
        self.call([action("query_map", question="当前地图的拟物体总数", categories=['*']), "保存地图没有有效对象。"], question="当前地图的拟物体总数")
        self.assertEqual(self.state.agent_complete.call_args_list[0].args[0][1]["content"], "当前地图的拟物体总数")

    def test_progress_does_not_leak_schema(self):
        events = []
        self.state.agent_complete = model_fixture([action("observe_scene", question="周围"), "照片里有椅子。"])
        self.state.agent_chat({"question": "周围有什么"}, events.append)
        text = " ".join(e.get("text", "") for e in events)
        self.assertNotIn("fresh", text)

    def test_http_json_and_progress_stream(self):
        server = ja_web_server.RobotWebServer(("127.0.0.1", 0), ja_web_handler.RobotWebHandler, self.state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.state.agent_complete = model_fixture(["你好，我是小卡。"] * 2)
        try:
            for accept in ("application/json", "application/x-ndjson"):
                request = Request(f"http://127.0.0.1:{server.server_port}/api/agent/chat",
                    data=json.dumps({"question": "你好"}).encode(),
                    headers={"Content-Type": "application/json", "Accept": accept})
                with urlopen(request, timeout=3) as response:
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                    body = response.read().decode()
                if accept.endswith("ndjson"):
                    events = [json.loads(line) for line in body.splitlines()]
                    self.assertEqual(events[-1]["type"], "final")
                    self.assertEqual(events[0]["type"], "status")
                else:
                    self.assertEqual(json.loads(body)["text"], "你好，我是小卡。")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
