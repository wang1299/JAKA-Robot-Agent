"""Regression cases for an empty pickup lobby hallucinated as a guest."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import robot_web


class WelcomeFalsePositiveTests(unittest.TestCase):
    @staticmethod
    def match_json(placeholders=False):
        raw = {'candidate_visible': True,
               'candidate_region': '候选位置或无' if placeholders else '图2中央电梯前',
               'contradictions': [], 'reason': '可见细节一致'}
        for key, label, description in [
            ('upper_clothing', '衣着', '白色短袖，上有黄绿色图案'),
            ('face_hair', '头脸发型', '黑色短发，戴眼镜'),
            ('accessories', '配饰', '眼镜和红色手绳'),
            ('body_shape', '体型', '中等身材'),
        ]:
            raw['reference_' + key] = '参考图' + label if placeholders else description
            raw['candidate_' + key] = '现场' + label if placeholders else description
            raw[key] = 'match'
        return json.dumps(raw, ensure_ascii=False)

    def test_single_scene_gate_rejects_empty_and_malformed_answers(self):
        for raw in [
            '```json\n{"person_visible":false,"person_count":0,"visible_evidence":""}\n```',
            '{"person_visible":true,"person_count":0,"visible_evidence":"有人"}',
            '{"person_visible":"true","person_count":1,"visible_evidence":"白衣人"}',
            '{"person_visible":true,"person_count":true,"visible_evidence":"白衣人"}',
            '无法判断',
        ]:
            self.assertFalse(robot_web._parse_welcome_presence_result(raw)['person_visible'])

    def test_single_scene_gate_accepts_explicit_person_evidence(self):
        result = robot_web._parse_welcome_presence_result(
            '{"person_visible":true,"person_count":1,"visible_evidence":"画面左侧一人穿白色上衣"}'
        )
        self.assertTrue(result['person_visible'])

    def test_schema_placeholders_cannot_score_a_match(self):
        result = robot_web._parse_person_reference_result(self.match_json(placeholders=True))
        self.assertFalse(result['found'])
        self.assertEqual(result['confidence'], 'low')
        self.assertFalse(result['schema_valid'])

    def test_real_descriptions_keep_existing_scoring(self):
        raw = {'candidate_visible': True, 'candidate_region': '画面中央',
               'contradictions': [], 'reason': '可见细节一致'}
        for key, description in [('upper_clothing', '白色上衣'), ('face_hair', '黑色短发'),
                                 ('accessories', '黑框眼镜'), ('body_shape', '中等身材')]:
            raw['reference_' + key] = description
            raw['candidate_' + key] = description
            raw[key] = 'match'
        self.assertTrue(robot_web._parse_person_reference_result(json.dumps(raw, ensure_ascii=False))['found'])

    def test_invalid_formatter_retries_once_with_same_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'photo.jpg'
            path.write_bytes(b'JPEG')
            first = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='两图衣着外观相似'))])
            invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.match_json(True)))])
            client = Mock()
            client.chat.completions.create.side_effect = [first, invalid]
            valid_raw = self.match_json()
            retry = Mock(return_value=(robot_web._parse_person_reference_result(valid_raw), valid_raw))
            state = SimpleNamespace(mock=False, infer_lock=threading.Lock(),
                                    reference_path=lambda *_: path,
                                    _welcome_scene_presence=Mock(return_value={
                                        'person_visible': True, 'person_count': 1,
                                        'visible_evidence': '画面中央站立一人', 'schema_valid': True}),
                                    _retry_welcome_person_comparison=retry)
            with patch('qwen_planner._client', return_value=client), \
                 patch('qwen_planner._image_data_url', return_value='data:image/jpeg;base64,AA=='):
                result = robot_web.RobotWebState.compare_person_reference(
                    state, {'reference_id': 'sample', 'suffix': '.jpg'}, path)
            self.assertTrue(result['found'])
            self.assertEqual(result['comparison_source'], 'qwen_retry')
            retry.assert_called_once_with(path, path)
            self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_failed_retry_stays_unconfirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'photo.jpg'
            path.write_bytes(b'JPEG')
            first = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='两图外观相似'))])
            invalid = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.match_json(True)))])
            client = Mock()
            client.chat.completions.create.side_effect = [first, invalid]
            state = SimpleNamespace(mock=False, infer_lock=threading.Lock(), reference_path=lambda *_: path,
                                    _welcome_scene_presence=Mock(return_value={'person_visible': True, 'schema_valid': True}),
                                    _retry_welcome_person_comparison=Mock(return_value=({'schema_valid':False,'found':False},'invalid')))
            with patch('qwen_planner._client', return_value=client), \
                 patch('qwen_planner._image_data_url', return_value='data:image/jpeg;base64,AA=='):
                result = robot_web.RobotWebState.compare_person_reference(
                    state, {'reference_id':'sample','suffix':'.jpg'}, path)
            self.assertFalse(result['found'])
            self.assertFalse(result['schema_valid'])

    def test_retry_sends_two_images_only(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)/'reference.jpg'
            scene = Path(directory)/'scene.jpg'
            reference.write_bytes(b'REF')
            scene.write_bytes(b'SCENE')
            client = Mock()
            client.with_options.return_value.chat.completions.create.return_value.choices = [
                SimpleNamespace(message=SimpleNamespace(content=self.match_json()))]
            with patch.dict(os.environ, {'JAKA_AGENT_BASE_URL':'http://127.0.0.1:8001/v1',
                                          'JAKA_AGENT_MODEL':'/data2/models/Qwen3.5-9B'}), \
                 patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=lambda **_: client)}):
                result, _ = robot_web.RobotWebState._retry_welcome_person_comparison(
                    SimpleNamespace(), reference, scene)
            self.assertTrue(result['found'])
            parts = client.with_options.return_value.chat.completions.create.call_args.kwargs['messages'][0]['content']
            images = [part['image_url']['url'] for part in parts if part['type']=='image_url']
            self.assertEqual(len(images), 2)
            self.assertNotEqual(images[0], images[1])

    def test_empty_scene_never_enters_identity_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.jpg'
            path.write_bytes(b'not-decoded-in-this-test')
            state = SimpleNamespace(mock=False, infer_lock=threading.Lock(),
                                    reference_path=lambda *_: path,
                                    _welcome_scene_presence=Mock(return_value={
                                        'person_visible': False, 'person_count': 0,
                                        'visible_evidence': '', 'schema_valid': True}))
            result = robot_web.RobotWebState.compare_person_reference(
                state, {'reference_id': 'sample', 'suffix': '.jpg'}, path)
            self.assertFalse(result['found'])
            self.assertIn('未确认有人', result['reason'])
            state._welcome_scene_presence.assert_called_once_with(path)

    def test_presence_request_sends_only_scene_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.jpg'
            path.write_bytes(b'JPEG')
            client = Mock()
            client.with_options.return_value.chat.completions.create.return_value.choices = [
                SimpleNamespace(message=SimpleNamespace(content='{"person_visible":false,"person_count":0,"visible_evidence":""}'))
            ]
            with patch.dict(os.environ, {'JAKA_AGENT_BASE_URL':'http://127.0.0.1:8001/v1',
                                          'JAKA_AGENT_MODEL':'/data2/models/Qwen3.5-9B'}), \
                 patch.dict(sys.modules, {'openai': SimpleNamespace(OpenAI=lambda **_: client)}):
                result = robot_web.RobotWebState._welcome_scene_presence(SimpleNamespace(), path)
            self.assertFalse(result['person_visible'])
            messages = client.with_options.return_value.chat.completions.create.call_args.kwargs['messages']
            self.assertEqual(len(messages), 1)
            self.assertEqual(len([part for part in messages[0]['content'] if part['type']=='image_url']), 1)


if __name__ == '__main__':
    unittest.main()
