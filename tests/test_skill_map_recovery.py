"""Regression: uploaded-object search and room-name lookup, no hardware."""
import json
import unittest
from unittest.mock import Mock
from jaka_agent.agent.map_evidence import MapEvidence
from jaka_agent.agent.skills import skill_resources
from tests.test_robot_skills import web_state, REFERENCE
from tests.test_robot_agent import model_fixture

def call(tool, **args):
    return json.dumps({'tool': tool, 'arguments': args}, ensure_ascii=False)

class NameLookupTests(unittest.TestCase):
    def setUp(self):
        self.evidence = MapEvidence({'objects': [
            {'ann_id': 91, 'category': 'door frame', 'semantic_name': '512实验准备室', 'floor_xy': [1, 2]},
            {'ann_id': 92, 'category': 'door frame', 'semantic_name': '513实验准备室', 'floor_xy': [3, 4]}]})

    def test_partial_name_never_selects_first_ambiguous_match(self):
        r = self.evidence.resolve_target('实验准备室')
        self.assertFalse(r['unique'])
        self.assertEqual(set(r['target_ids']), {'91', '92'})

    def test_exact_name_without_room_metadata(self):
        r = self.evidence.resolve_target('512实验准备室')
        self.assertTrue(r['unique'])
        self.assertEqual(r['match_type'], 'exact')
        self.assertEqual(r['target_ids'], ['91'])

    def test_failed_metadata_filter_returns_separate_name_evidence(self):
        r = self.evidence.query('去看看', ['*'], [
            {'field': 'room_name', 'operator': 'equals', 'value': '512实验准备室'}], count_unit='rooms')
        self.assertEqual(r['count'], 0)
        self.assertEqual(r['objects'], [])
        self.assertTrue(r['absence_not_established'])
        self.assertEqual(r['name_lookup_suggestions'][0]['target_ids'], ['91'])

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError): self.evidence.resolve_target('  ')

    def test_exact_match_wins_over_partial(self):
        e = MapEvidence({'objects': [{'ann_id':1,'semantic_name':'大厅'},
                                    {'ann_id':2,'semantic_name':'大厅侧门'}]})
        self.assertEqual(e.resolve_target('大厅')['target_ids'], ['1'])

class FindContractTests(unittest.TestCase):
    def test_find_does_not_require_known_location(self):
        r = skill_resources('find_object', {'objects':[]}, True, 'current_turn')
        self.assertEqual(set(r['accepted_inputs']), {'instruction','reference'})

    def test_invented_missing_location_is_rejected_then_model_can_plan(self):
        web = web_state()
        web.agent_complete = model_fixture([
            call('ask_user',goal='寻找参考物',question='请先告诉我位置',skill_id='find_object',missing_inputs=['target_location_hint']),
            call('get_skill_context',skill_id='find_object'),
            call('plan_find_object',instruction='寻找照片中的物品')])
        r = web.agent_chat({'question':'帮我找一下这个物品','reference':REFERENCE})
        self.assertFalse(r['tool_trace'][0]['ok'])
        self.assertTrue(r['requires_confirmation'])
        self.assertEqual(r['task']['kind'], 'find_object')
        self.assertEqual(r['task']['reference'], REFERENCE)
        self.assertEqual(r['task']['status'], 'planned')
        web.tasks.execute.assert_not_called()
        web.capture.assert_not_called()
        web.speak.assert_not_called()

    def test_genuine_ambiguity_can_still_be_clarified(self):
        web = web_state()
        web.agent_complete = model_fixture([call('ask_user',goal='寻找图中物品',
            question='图片里有两个目标，要找哪一个？',skill_id='find_object',missing_inputs=[])])
        r=web.agent_chat({'question':'找这个','reference':REFERENCE})
        self.assertTrue(r['requires_clarification'])
        self.assertIsNone(web.tasks.task)

if __name__ == '__main__': unittest.main()
