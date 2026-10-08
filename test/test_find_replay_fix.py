"""Exercise deployed functions without importing/initializing robot hardware."""
import ast
import copy
from contextvars import copy_context
import json
import logging
from pathlib import Path
import re
import threading
import time
import uuid
from types import SimpleNamespace
import unittest

from robot_web_routing import _parse_find_object_result

ROOT=Path(__file__).resolve().parents[1]
class TaskCancelled(Exception): pass

def methods():
    tree=ast.parse((ROOT/'robot_web.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RobotTaskManager')
    names={'_public_find_observation','_find_object_match_value','_announce_find_observation',
           '_start_reference_snapshot_loop','_capture_reference_snapshot',
           '_finish_find_object_no_match','_canceled'}
    env=dict(re=re,copy=copy,copy_context=copy_context,threading=threading,time=time,
             uuid=uuid,guarded_call=lambda fn,*args:fn(*args),_log_exception=lambda *args,**kwargs:None,
             LOGGER=logging.getLogger('find-replay-test'),CATEGORY_ZH_MAP={},TaskCancelled=TaskCancelled,
             check_cancelled=lambda:None,FIND_SNAPSHOT_INTERVAL_SECONDS=.001,FIND_SNAPSHOT_MAX_ERRORS=3)
    for node in cls.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'robot_web.py'),'exec'),env)
    return env

class FindReplayFix(unittest.TestCase):
    def test_found_confidence_and_output_agree(self):
        env=methods()
        for found in [True,False]:
            for confidence in [True,False,'high','medium','low',None,.1]:
                with self.subTest(found=found,confidence=confidence):
                    result=_parse_find_object_result(json.dumps(dict(found=found,confidence=confidence)))
                    self.assertIs(result['found'],found)
                    speech=[]
                    stub=SimpleNamespace(_public_find_observation=env['_public_find_observation'],web=SimpleNamespace(speak=speech.append))
                    obs=env['_find_object_match_value'](stub,88,'saved.jpg',result,{88:{'floor_xy':[0,0],'category_zh':'木柜'}})
                    env['_announce_find_observation'](stub,obs,result)
                    self.assertEqual(obs['verdict'],'match' if found else 'miss')
                    self.assertEqual('未找到' in speech[0],not found)
                    self.assertEqual('未在' in obs['text'],not found)

    def test_invalid_found_is_error_not_miss(self):
        for raw in ['{}','超时','{"found":"false"}','{"found":null}','{"found":1}']:
            with self.subTest(raw=raw),self.assertRaises(ValueError):_parse_find_object_result(raw)

    def test_minor_confidence_format_repair(self):
        for found in ['true','false']:
            result=_parse_find_object_result('{"found":'+found+',"confidence":low}')
            self.assertEqual(result['found'],found=='true')

    def test_real_minicpm_field_separator_error_is_repaired(self):
        raw=('根据当前图像分析，没有目标。\n\n'
             '{"found":false,"candidate_visible":false,'
             '"candidate_region":"整个场景," "confidence":"low",'
             '"reason":"没有匹配目标"}')
        result=_parse_find_object_result(raw)
        self.assertIs(result['found'],False)
        self.assertEqual(result['candidate_region'],'整个场景')

    def test_invalid_result_retries_same_photo_then_marks_unknown(self):
        env=methods()
        captures=[];calls=[]
        def compare(reference,path):
            calls.append((reference,path))
            raise ValueError('寻物识别结果格式异常：缺少有效的 found 布尔值')
        task=SimpleNamespace(task={'id':'task-1','snapshot_count':0,'search_attempts':[]},
                             lock=threading.RLock(),web=SimpleNamespace(compare_reference=compare))
        task.snapshot=lambda:dict(task.task)
        task._update=lambda **values:task.task.update(values)
        task._capture=lambda *args,**kwargs:('capture-1',Path('same-photo.jpg'))
        task._record_capture=lambda *args,**kwargs:captures.append((args,kwargs))
        task._find_object_match_value=lambda *args:env['_find_object_match_value'](task,*args)
        task._public_find_observation=env['_public_find_observation']
        objects={739:{'floor_xy':[1,2],'category_zh':'木质储物柜'}}
        probe=env['_capture_reference_snapshot'](task,None,{'reference_id':'ref'},739,objects,
                                                  '到达点目标核验',publish_mode='always')
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0],calls[1])
        self.assertIsNone(probe['comparison']['found'])
        self.assertEqual(probe['observation']['verdict'],'uncertain')
        self.assertFalse(probe['matched'])
        self.assertEqual(len(task.task['search_attempts']),1)

    def test_retry_can_recover_valid_match(self):
        env=methods()
        calls=[]
        def compare(reference,path):
            calls.append(path)
            if len(calls)==1:
                raise ValueError('寻物识别结果格式异常：缺少有效的 found 布尔值')
            return {'found':True,'confidence':'low','reason':'外观一致'}
        task=SimpleNamespace(task={'id':'task-2','snapshot_count':0,'search_attempts':[]},
                             lock=threading.RLock(),web=SimpleNamespace(compare_reference=compare))
        task.snapshot=lambda:dict(task.task)
        task._update=lambda **values:task.task.update(values)
        task._capture=lambda *args,**kwargs:('capture-2',Path('same-photo.jpg'))
        task._record_capture=lambda *args,**kwargs:None
        task._find_object_match_value=lambda *args:env['_find_object_match_value'](task,*args)
        task._public_find_observation=env['_public_find_observation']
        probe=env['_capture_reference_snapshot'](task,None,{},739,
              {739:{'floor_xy':[1,2],'category_zh':'木质储物柜'}},'到达点目标核验')
        self.assertEqual(calls,[Path('same-photo.jpg')]*2)
        self.assertTrue(probe['matched'])
        self.assertEqual(probe['observation']['verdict'],'match')

    def test_unknown_point_does_not_become_not_found_or_success(self):
        env=methods()
        task=SimpleNamespace(lock=threading.RLock(),task={
            'checked_ann_ids':[739], 'candidate_status':{'739':'uncertain'},
        })
        env['_finish_find_object_no_match'](task)
        self.assertEqual(task.task['status'],'inconclusive')
        self.assertIn('不能确认',task.task['result_text'])

    def test_real_worker_lifecycle(self):
        env=methods()
        for event in ['normal','arrival','cancel','failed','replacement','inference_error']:
            with self.subTest(event=event):
                stop,entered,released=threading.Event(),threading.Event(),threading.Event()
                calls=[];result={}
                task=SimpleNamespace(task={'id':'first','status':'running','reference':{},'cancel_requested':False},lock=threading.RLock(),_capture_workers=[])
                task.snapshot=lambda:dict(task.task)
                task._canceled=lambda:env['_canceled'](task)
                def capture(*args,**kwargs):
                    entered.set()
                    if not released.wait(3):raise RuntimeError('test release timeout')
                    if event=='inference_error':raise ValueError('invalid response')
                    return dict(matched=True,attempt={},observation={'verdict':'match'},comparison={'found':True},capture_path='saved.jpg')
                task._capture_reference_snapshot=capture
                driver=SimpleNamespace(cancel_move=lambda:calls.append('fake-stop'))
                thread=env['_start_reference_snapshot_loop'](task,driver,88,{},stop,result)
                try:
                    self.assertTrue(entered.wait(3))
                    if event=='arrival':stop.set()
                    elif event=='cancel':task.task['cancel_requested']=True;stop.set()
                    elif event=='failed':task.task['status']='failed';stop.set()
                    elif event=='replacement':task.task['id']='second';stop.set()
                finally:
                    released.set();thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertEqual('match' in result,event in ['normal','arrival'])
                self.assertEqual(calls,['fake-stop'] if event in ['normal','inference_error'] else [])
                self.assertEqual('error' in result,event=='inference_error')

if __name__=='__main__': unittest.main()
