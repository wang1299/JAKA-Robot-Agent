"""Visual-answer presentation and shared evidence rules; no hardware calls."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_planner as planner
import robot_web


class VisualAnswerTests(unittest.TestCase):
    def test_plain_visible_facts_and_uncertainty_preserved(self):
        answer = '左侧有一块白板。箱盖关闭，无法看到内部物品。'
        self.assertEqual(planner.format_visual_answer(answer), answer)

    def test_legacy_detail_is_prose_not_dropped(self):
        raw = {'answer': '箱盖关闭。', 'detail': '看不到内部，无法判断里面有什么。',
               'reasoning': 'INTERNAL_REASON', 'debug': 'INTERNAL_DEBUG'}
        self.assertEqual(planner.format_visual_answer(json.dumps(raw, ensure_ascii=False)),
                         '箱盖关闭。\n看不到内部，无法判断里面有什么。')

    def test_legacy_labels_and_duplicate_lines(self):
        for text in ('answer: 箱盖关闭。\ndetail: 看不到内部。',
                     '**answer:** 箱盖关闭。\n**detail:** 看不到内部。',
                     '箱盖关闭。\ndetail：看不到内部。\n看不到内部。'):
            self.assertEqual(planner.format_visual_answer(text), '箱盖关闭。\n看不到内部。')

    def test_json_markdown_and_repeated_fields(self):
        self.assertEqual(planner.format_visual_answer('```json\n{"answer":"看不到内部。","detail":"看不到内部。"}\n```'),
                         '看不到内部。')

    def test_missing_answer_can_use_meaningful_detail(self):
        self.assertEqual(planner.format_visual_answer({'answer': '', 'detail': '标牌模糊，读不清文字。'}),
                         '标牌模糊，读不清文字。')

    def test_invalid_protocol_never_leaks(self):
        for raw in ('{"answer":', '{"reasoning":"SECRET"}', '["SECRET"]', '', None,
                    {'answer': {'debug': 'SECRET'}}, '<think>unfinished reasoning'):
            self.assertEqual(planner.format_visual_answer(raw), planner.VISUAL_ANSWER_FALLBACK)

    def test_thinking_not_public(self):
        self.assertEqual(planner.format_visual_answer('<think>INTERNAL</think>画面有白板。'), '画面有白板。')
        self.assertEqual(planner.format_visual_answer('画面有白板。\n{"answer":"重复协议"}'), '画面有白板。')

    def test_execution_annotations_still_removed(self):
        self.assertEqual(planner.clean_observation_answer('detail: 看不到内部。 (使用到达点当前画面)'), '看不到内部。')

    def client(self, text):
        client = Mock()
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        client.chat.completions.create.return_value = response
        client.with_options.return_value = client
        return client

    def test_task_read_image_uses_shared_rules_and_clean_output(self):
        client = self.client('answer: 箱盖关闭。\ndetail: 看不到内部。')
        with patch.object(planner, '_client', return_value=client), patch.object(planner, '_image_data_url', return_value='image'):
            self.assertEqual(planner.read_image('unused.png', '里面有什么'), '箱盖关闭。\n看不到内部。')
        messages = client.chat.completions.create.call_args.kwargs['messages']
        self.assertIn(planner.VISUAL_EVIDENCE_RULES, messages[0]['content'][1]['text'])
        self.assertIn('不输出JSON', messages[0]['content'][1]['text'])
        self.assertNotIn('response_format', client.chat.completions.create.call_args.kwargs)

    def test_agent_and_legacy_web_use_same_rules(self):
        web = object.__new__(robot_web.RobotWebState)
        web.mock = False
        web.infer_lock = threading.Lock()
        with tempfile.TemporaryDirectory() as directory:
            # An existing directory satisfies the precondition; image encoding is mocked.
            web.capture_path = Mock(return_value=Path(directory))
            for method in (lambda: web.agent_vision(Path(directory), '里面有什么', []),
                           lambda: web.infer('unused', '里面有什么', [])):
                client = self.client('{"answer":"看不到内部。","detail":"看不到内部。"}')
                with patch.object(planner, '_client', return_value=client), patch.object(planner, '_image_data_url', return_value='image'):
                    self.assertEqual(method(), '看不到内部。')
                messages = client.chat.completions.create.call_args.kwargs['messages']
                text = messages[0]['content'][1]['text']
                self.assertIn(planner.VISUAL_EVIDENCE_RULES, text)


if __name__ == '__main__':
    unittest.main()
