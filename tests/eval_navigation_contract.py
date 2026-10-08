"""Opt-in real decision-model tests with hardware and task execution replaced.

No camera, navigation, microphone, map writes or real task cards are created.
"""
import jaka_agent.mapping.graphs as ja_mapping_graphs
import jaka_agent.web.state as ja_web_state
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.agent.skills import prepare_skill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--repeat', type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    graph = ja_mapping_graphs._normalize_graph(json.loads((root/'src/jaka_agent/resources/maps/zmq_scene_graph.json').read_text(encoding='utf-8')), 'zmq_scene_graph.json')

    def complete(messages, tools):
        request = Request(args.base_url+'/chat/completions', data=json.dumps({
            'model':'Qwen3.5-9B', 'messages':messages, 'tools':tools, 'tool_choice':'auto',
            'parallel_tool_calls':False, 'temperature':0, 'max_tokens':700}).encode(),
            headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=45) as response:message=json.load(response)['choices'][0]['message']
        if message.get('tool_calls'):
            calls=message['tool_calls']
            if len(calls)!=1:return '<tool_call>invalid parallel calls'
            call=calls[0]['function']
            return json.dumps({'tool':call['name'],'arguments':json.loads(call['arguments'])},ensure_ascii=False)
        return message.get('content') or ''

    def state():
        web=object.__new__(ja_web_state.RobotWebState)
        web.mock=False
        web.agent_lock=threading.Lock()
        web.agent_sessions={}
        web.graph_snapshot=lambda:graph
        web.agent_complete=complete
        pending={'task':None}
        def plan(skill_id,instruction,reference=None,**params):
            skill,_,steps=prepare_skill(skill_id,{'instruction':instruction,**params},graph,reference)
            pending['task']={'id':'fixture-only','status':'planned','skill':{'id':skill.id},'plan':steps}
            return pending['task']
        web.tasks=SimpleNamespace(snapshot=lambda:pending['task'], plan_skill=plan,
            plan=Mock(side_effect=AssertionError('Use typed skill in this test')),
            robot_status=lambda:{'online':True,'estop_state':True},
            execute=Mock(side_effect=AssertionError('NO PHYSICAL EXECUTION')))
        web.mapping=SimpleNamespace(is_busy=lambda:False)
        web.capture=Mock(side_effect=AssertionError('NO CAMERA'))
        return web

    results=[]
    for repeat in range(args.repeat):
        cases=[('explicit', ['到地图中 ann_id 713 的沙发处拍照，告诉我那里有什么。']),
               ('screenshot', ['你在整个场景中能找到几个椅子', '导航到沙发看看', '它旁边有个黑色航空箱']),
               ('status', ['你已经开始导航了吗？'])]
        for name,questions in cases:
            web=state()
            start=time.monotonic()
            turns=[]
            for question in questions:
                result=web.agent_chat({'question':question,'conversation_id':'fixture-'+name})
                turns.append(result)
                print(json.dumps({'case':name,'repeat':repeat+1,'question':question,'result':result},ensure_ascii=False),flush=True)
            if name=='explicit':
                passed=bool(turns[-1].get('task')) and turns[-1].get('requires_confirmation') is True
            elif name=='screenshot':
                # Exact distance tools or automatic pairwise query evidence can
                # ground this; the essential requirement is no arbitrary plan.
                passed=(not any(t.get('task') for t in turns)
                        and turns[-1].get('requires_clarification') is True)
            else:
                # Either direct status intent or a status tool is acceptable.
                passed=(turns[-1]['execution']['state']=='none' and not turns[-1].get('incomplete')
                        and (turns[-1].get('response_kind')=='task_status'
                             or any(t['tool']=='get_robot_status' for t in turns[-1].get('tool_trace',[]))))
            web.tasks.execute.assert_not_called()
            web.capture.assert_not_called()
            results.append({'case':name,'repeat':repeat+1,'passed':passed,'seconds':round(time.monotonic()-start,2)})
    print(json.dumps({'summary':results,'passed':sum(r['passed'] for r in results),'total':len(results)},ensure_ascii=False),flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__=='__main__':
    raise SystemExit(main())
