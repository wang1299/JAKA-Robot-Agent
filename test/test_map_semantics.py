import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent_map import MapEvidence
from robot_web import _normalize_graph
from test_robot_skills import web_state
from test_robot_agent import model_fixture


class SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.graph = _normalize_graph(json.loads(Path(__file__).resolve().parents[1].joinpath('zmq_scene_graph.json').read_text(encoding='utf-8')))
        self.map = MapEvidence(self.graph)

    def test_rooms_are_not_doors(self):
        result = self.map.query('工作室数量', ['*'], [{'field':'room_type','operator':'equals','value':'研究生工作室'}], count_unit='rooms')
        self.assertEqual(result['count'], 3)
        self.assertEqual(result['object_count'], 6)
        self.assertEqual({r['room_id'] for r in result['rooms']}, {'326','330','334'})
        self.assertTrue(all(len(r['entrance_ann_ids']) == 2 for r in result['rooms']))
        self.assertNotIn('弱电间', str(result['rooms']))

    def test_teacher_room_is_resolvable_and_has_room_metadata(self):
        for name in ('教师工作室', '332教师工作室'):
            self.assertEqual(self.map.resolve_target(name)['target_ids'], ['384'])
        r = self.map.query('教师工作室', ['*'],
            [{'field':'room_type','operator':'equals','value':'教师工作室'}], count_unit='rooms')
        self.assertEqual(r['count'], 1)
        self.assertEqual(r['rooms'][0]['room_id'], '332')

    def test_class_label_never_uses_last_instances_name(self):
        catalog = {c['category']:c for c in self.map.category_catalog()['categories']}
        self.assertNotIn('326', catalog['door frame']['label'])
        self.assertNotIn('320', catalog['Ordinary office door']['label'])
        self.assertIn('334研究生工作室前门',catalog['door frame']['example_names'])

    def test_resolve_exact_identity_and_unknown(self):
        self.assertEqual(self.map.resolve_target('334研究生工作室前门')['target_ids'], ['330'])
        self.assertEqual(self.map.resolve_target('不存在的位置')['count'], 0)
        self.assertFalse(self.map.resolve_target('334')['unique'])

    def test_duplicate_alias_does_not_prove_unique(self):
        evidence = MapEvidence({'objects':[{'ann_id':1,'aliases':['门口']},{'ann_id':2,'aliases':['门口']}]})
        self.assertFalse(evidence.resolve_target('门口')['unique'])

    def test_glass_portal_has_explicit_approach_target_not_traversal(self):
        portal = next(p for p in self.graph['doorways'] if p['portal_id']=='visual_doorway:27173835ed8bfbcf')
        self.assertEqual(portal['category_zh'], '玻璃门')
        self.assertEqual(len(portal['floor_xy']), 2)
        self.assertFalse(portal['display_only'])
        self.assertEqual(portal['navigation_mode'], 'approach')
        target = self.map.objects[str(portal['navigation_ann_id'])]
        self.assertEqual(target['source_portal_id'], portal['portal_id'])
        self.assertNotIn('navigation_pose', target)
        self.assertNotIn('ann_id', portal)
        self.assertNotIn('907',self.map.objects)
        self.assertEqual(_normalize_graph(self.graph)['doorways'], self.graph['doorways'])

    def test_invalid_portal_geometry_skipped(self):
        graph={'objects':[{'ann_id':1,'floor_xy':[1,2]}], 'doorways':{
            'bad':{'portal_id':'bad','geometry':{'center_world':[float('nan'),2]}}}}
        self.assertEqual(_normalize_graph(graph)['doorways'], [])

    def test_exact_lookup_overrides_broad_scope_without_execution(self):
        web = web_state()
        web.graph_snapshot.return_value = self.graph
        def call(tool, **args):return json.dumps({'tool':tool,'arguments':args}, ensure_ascii=False)
        web.agent_complete = model_fixture([
            call('query_map',question='门',categories=['door frame']),
            call('plan_navigate',instruction='去334研究生工作室前门',target_ann_ids=[330],return_to_start=False),
            call('resolve_map_target',name='334研究生工作室前门'),
            call('plan_navigate',instruction='去334研究生工作室前门',target_ann_ids=[330],return_to_start=False)])
        result=web.agent_chat({'question':'导航到334研究生工作室前门','conversation_id':'test-room'})
        self.assertTrue(result['requires_confirmation'])
        self.assertEqual(result['task']['steps'][0]['target_ann_id'],330)
        self.assertEqual(result['task']['status'],'planned')
        web.tasks.execute.assert_not_called()
        web.capture.assert_not_called()
        self.assertFalse(result['tool_trace'][1]['ok'])

    def test_unknown_room_membership_is_disclosed(self):
        evidence=MapEvidence({'objects':[{'ann_id':1,'category':'door'}]})
        result=evidence.query('房间',['*'],count_unit='rooms')
        self.assertEqual(result['count'],0)
        self.assertEqual(result['objects_without_room_metadata'],1)

    def test_old_choices_do_not_leak_into_unrelated_clarification(self):
        web=web_state()
        web.graph_snapshot.return_value=self.graph
        web.agent_complete=Mock(return_value=json.dumps({'tool':'ask_user','arguments':{
            'goal':'新任务','question':'请说明新的目的地','skill_id':'navigate','missing_inputs':[]}}))
        result=web._agent_chat_turn({'question':'换一个目的地'},None,{'evidence':[],
            'target_offer':{'snapshot':self.map.version,'ids':[330,382]}})
        self.assertFalse(result.get('target_choices'))


if __name__ == '__main__': unittest.main()
