import jaka_agent.mapping.graphs as ja_mapping_graphs
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from jaka_agent.agent.routing import _parse_find_object_result
from jaka_agent.agent.map_evidence import MapEvidence
from tests.test_robot_skills import web_state

def raw(confidence, found=True):
    return json.dumps(dict(found=found,candidate_visible=found,candidate_region='柜上白色物体',confidence=confidence,reason='形状一致'))

class FindResultTests(unittest.TestCase):
    def test_boolean_confidence_is_invalid_not_high(self):
        r=_parse_find_object_result(raw(True))
        self.assertTrue(r['found'])
        self.assertEqual(r['confidence'],'low')
        self.assertNotIn('format_valid',r)

    def test_valid_confidence_preserved(self):
        for value in ('high','medium','low'):
            r=_parse_find_object_result(raw(value))
            self.assertEqual(r['confidence'],value)

    def compare(self, first, second):
        web=web_state();web.infer_lock=threading.Lock()
        web.reference_path=Mock(return_value=Path(__file__))
        response=lambda body:SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=body))])
        client=Mock(); client.with_options.return_value=client
        client.chat.completions.create.side_effect=[response(first),response(second)]
        with patch('jaka_agent.models.runtime._client',return_value=client),patch('jaka_agent.models.vision._image_data_url',return_value='data:image/jpeg;base64,test'):
            result=web.compare_reference({'reference_id':'a'*32,'suffix':'.jpg'},Path(__file__))
        return result,client

    def test_original_invalid_confidence_does_not_retry(self):
        result,client=self.compare(raw(True),raw('high'))
        self.assertEqual(client.chat.completions.create.call_count,1)
        self.assertEqual(result['confidence'],'low')

    def test_explicit_found_is_authoritative(self):
        data=json.loads(raw('high'));data['candidate_visible']=False
        self.assertTrue(_parse_find_object_result(json.dumps(data))['found'])

    def test_found_does_not_require_high(self):
        web=web_state()
        for found,confidence,expected in [(True,'high','match'),(True,'medium','match'),
                                          (True,'low','match'),(True,True,'match'),(False,'high','miss')]:
            result=_parse_find_object_result(raw(confidence,found))
            obs=web.tasks._find_object_match_value(7,'/image',result,{7:{'ann_id':7,'floor_xy':[1,2]}})
            self.assertEqual(obs['verdict'],expected)

    def test_valid_low_confidence_does_not_retry(self):
        result,client=self.compare(raw('low'),raw('high'))
        self.assertEqual(result['confidence'],'low')
        self.assertEqual(client.chat.completions.create.call_count,1)

    def test_concise_chinese_speech_and_truthful_transit(self):
        web=web_state()
        for obs,result,wanted in [
            ({'name':'Black flight case'}, {'found':True,'confidence':'high'},'到达黑色航空箱，找到参考图中的物体。'),
            ({'name':'table'}, {'found':False},'到达桌子，未找到参考图中的物体。'),
            ({'name':'Unknown English'}, {'found':False},'到达当前物体，未找到参考图中的物体。'),
            ({'name':'木柜','in_transit':True}, {'found':True,'confidence':'high'},'前往木柜途中，找到参考图中的物体。')]:
            web.tasks._announce_find_observation(obs,result)
            web.speak.assert_called_with(wanted)

class PortalDataTests(unittest.TestCase):
    def setUp(self):
        self.raw=json.loads((Path(__file__).resolve().parents[1]/'src/jaka_agent/resources/maps/zmq_scene_graph.json').read_text(encoding='utf-8'))
        self.graph=ja_mapping_graphs._normalize_graph(self.raw)

    def test_hydrants_removed_and_counts_consistent(self):
        self.assertNotIn('357',self.raw['objects']);self.assertNotIn('707',self.raw['objects'])
        self.assertEqual(self.raw['stats']['num_objects'],len(self.raw['objects']))
        self.assertNotIn('Fire hydrant cabinet',json.dumps(self.raw))

    def test_named_portals_resolve_and_normalize_idempotently(self):
        evidence=MapEvidence(self.graph)
        for name,expected in [('玻璃门','1000001'),('厕所','1000002')]:
            self.assertEqual(evidence.resolve_target(name)['target_ids'],[expected])
        self.assertEqual(self.graph,ja_mapping_graphs._normalize_graph(self.graph))

    def test_unconfigured_portal_remains_display_only(self):
        self.raw['doorways']['visual_doorway:27173835ed8bfbcf'].pop('navigation_mode')
        g=ja_mapping_graphs._normalize_graph(self.raw)
        self.assertNotIn(1000001,[o['ann_id'] for o in g['objects']])

    def test_portal_id_collision_rejected(self):
        self.raw['doorways']['visual_doorway:27173835ed8bfbcf']['navigation_ann_id']=384
        with self.assertRaises(ValueError):ja_mapping_graphs._normalize_graph(self.raw)

if __name__=='__main__':unittest.main()
