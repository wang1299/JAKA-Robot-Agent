"""Temporal evidence contracts, including wrong tool choices; no robot IO."""
import json
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from jaka_agent.agent.runner import AgentRunner, Tool, determine_scene_requirement, ProtocolError
import tests.test_robot_agent as test_robot_agent
from tests.test_robot_agent import action, model_fixture


class SceneEvidenceTests(unittest.TestCase):
    def run_case(self, outputs, scope='current', capture_ok=True):
        self.capture = Mock(return_value={'ok':capture_ok, 'capture_id':'new',
                            'captured_this_turn':capture_ok, 'image_url':'/new.jpg'} if capture_ok
                            else {'ok':False,'error':'camera offline'})
        self.old = Mock(return_value={'ok':True,'capture_id':'old',
                         'captured_this_turn':False,'image_url':'/old.jpg'})
        self.plan = Mock(return_value={'ok':True,'task':{'status':'planned','id':'p'}})
        tools = {
            'observe_scene':Tool('新拍摄',{'question':{'type':'string'}},self.capture,effect='observe'),
            'inspect_previous_scene':Tool('旧图',{'question':{'type':'string'}},self.old,effect='historical_observe'),
            'inspect_memory_image':Tool('历史任务图',{'question':{'type':'string'}},self.old,effect='historical_observe'),
            'plan_navigate':Tool('导航',{},self.plan,effect='plan'),
        }
        self.complete = model_fixture(outputs, scene_scope=scope)
        return AgentRunner(self.complete,tools,native=True,require_final_tool=True,
                           require_scene_contract=True).run('你现在面前有人吗',
                           [{'role':'assistant','text':'旧照片中没有人，后来导航到厕所。'}])

    def test_wrong_history_tool_rejected_then_new_photo(self):
        for tool in ('inspect_previous_scene','inspect_memory_image'):
            with self.subTest(tool=tool):
                r=self.run_case([action(tool,question='画面中是否有人'),
                                 action('observe_scene',question='有没有人'),'新画面中有人。'])
                self.old.assert_not_called()
                self.capture.assert_called_once()
                self.assertFalse(r['tool_trace'][0]['ok'])
                self.assertEqual(r['image_url'],'/new.jpg')

    def test_no_tool_answer_cannot_satisfy_live_request(self):
        r=self.run_case(['根据旧图没有人',action('observe_scene',question='重新拍摄'),'有人'])
        self.capture.assert_called_once()
        self.assertFalse(r.get('incomplete',False))

    def test_failed_capture_does_not_fall_back_to_old_photo_or_false_answer(self):
        r=self.run_case([action('observe_scene',question='现在'),
                         action('inspect_previous_scene',question='看看'),'当前没有人'],capture_ok=False)
        self.old.assert_not_called()
        self.assertTrue(r['incomplete'])
        self.assertNotIn('image_url',r)
        self.assertIn('无法确认当前',r['text'])

    def test_historical_question_does_not_start_camera(self):
        r=self.run_case([action('observe_scene',question='旧图'),
                         action('inspect_previous_scene',question='旧图'),'旧照片中无人'],scope='historical')
        self.capture.assert_not_called()
        self.old.assert_called_once()
        self.assertEqual(r['image_url'],'/old.jpg')

    def test_comparison_needs_both_sources(self):
        r=self.run_case([action('inspect_previous_scene',question='之前'), '现在也无人',
                         action('observe_scene',question='现在'),'以前无人，现在有人'],scope='both')
        self.capture.assert_called_once()
        self.old.assert_called_once()
        self.assertFalse(r.get('incomplete',False))

    def test_pending_plan_can_include_future_observation(self):
        r=self.run_case([action('plan_navigate')])
        self.plan.assert_called_once()
        self.capture.assert_not_called()
        self.assertTrue(r['requires_confirmation'])

    def test_chat_needs_no_camera(self):
        self.run_case(['你好'],scope='none')
        self.old.assert_not_called()
        self.capture.assert_not_called()

    def test_invalid_requirement_fails_closed(self):
        complete=Mock(return_value='随便看看吧')
        with self.assertRaises(ProtocolError):
            determine_scene_requirement(complete,'现在呢',[],{})
        self.assertEqual(complete.call_count,2)


class WebFreshnessTests(unittest.TestCase):
    def test_new_photo_does_not_replace_old_photo_for_same_turn_comparison(self):
        fixture=test_robot_agent.WebAgentTests(); fixture.setUp()
        web=fixture.state
        web.capture_path=Mock(return_value=Path(__file__))
        web.agent_complete=model_fixture([
            action('observe_scene',question='现在'),
            action('inspect_previous_scene',question='之前'),'对比完成'],scene_scope='both')
        memory={'evidence':[], 'last_scene':{'capture_id':'b'*32,'observed_at':1}}
        result=web._agent_chat_turn({'question':'重新拍摄并和之前比较'},lambda e:None,memory)
        self.assertFalse(result.get('incomplete',False))
        self.assertEqual(result['capture_id'],'b'*32)
        self.assertEqual(memory['last_scene']['capture_id'],'a'*32)
        web.capture.assert_called_once()

    def test_full_request_reaches_vision_and_old_image_is_never_read(self):
        fixture=test_robot_agent.WebAgentTests(); fixture.setUp()
        web=fixture.state
        web.capture_path=Mock(return_value=Path(__file__))
        web.agent_complete=model_fixture([
            action('inspect_previous_scene',question='画面中是否有人'),
            action('observe_scene',question='画面中是否有人'),'新画面中有人'],scene_scope='current')
        result=web.agent_chat({'question':'不要看之前的，请重新拍摄看看有没有人',
                              'history':[{'role':'assistant','text':'旧照片', 'capture_id':'b'*32}]})
        web.capture.assert_called_once()
        web.agent_vision.assert_called_once()
        self.assertIn('请重新拍摄',web.agent_vision.call_args.args[1])
        self.assertIn('本轮原地新拍摄',web.agent_vision.call_args.args[1])
        self.assertEqual(result['image_source'],'现场拍摄')


if __name__ == '__main__':
    unittest.main()
