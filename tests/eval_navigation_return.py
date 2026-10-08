"""Real Qwen navigation semantics, mocked task manager/hardware only."""
import jaka_agent.mapping.graphs as ja_mapping_graphs
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tests.test_robot_skills import web_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--case',default='')
    parser.add_argument('--debug',action='store_true')
    args=parser.parse_args()
    data=json.loads((Path(__file__).resolve().parents[1]/'src/jaka_agent/resources/maps/zmq_scene_graph.json').read_text(encoding='utf-8'))
    graph=ja_mapping_graphs._normalize_graph(data,'zmq_scene_graph.json')
    ids={o.get('skill_defaults',{}).get('welcome'):o['ann_id'] for o in graph['objects'] if o.get('skill_defaults',{}).get('welcome')}
    pickup,dropoff=ids['pickup_ann_id'],ids['return_ann_id']
    command='python3 -c '+shlex.quote("import sys,urllib.request; data=sys.stdin.buffer.read(); r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions',data=data,headers={'Content-Type':'application/json'}),timeout=45); sys.stdout.buffer.write(r.read())")
    def complete(messages,tools):
        body={'model':'Qwen3.5-9B','messages':messages,'tools':tools,'tool_choice':'required',
              'parallel_tool_calls':False,'temperature':0,'max_tokens':700}
        proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','myserver',command],
                            input=json.dumps(body,ensure_ascii=False).encode(),capture_output=True,timeout=55)
        if proc.returncode:raise RuntimeError(proc.stderr.decode(errors='replace'))
        message=json.loads(proc.stdout)['choices'][0]['message']
        if args.debug:print(json.dumps(message,ensure_ascii=False),flush=True)
        calls=message.get('tool_calls') or []
        if len(calls)==1:
            f=calls[0]['function']
            return json.dumps({'tool':f['name'],'arguments':json.loads(f['arguments'])},ensure_ascii=False)
        return message.get('content') or ''
    cases=[('named_return','你能不能返回送客点',[dropoff],False),
           ('back_to_place','回到接客点',[pickup],False),
           ('negated_return','去送客点，不用回来',[dropoff],False),
           ('round_trip','去送客点，然后回本次出发点',[dropoff],True),
           ('ordered','先去接客点，再去送客点，到那里等着',[pickup,dropoff],False),
           ('paraphrase','去送客点，办完后原路折返',[dropoff],True),
           ('unknown_origin','回去',None,None),
           ('ambiguous_target','去沙发那里',None,None)]
    results=[]
    for repeat in range(args.repeat):
        for name,question,targets,returning in cases:
            if args.case and args.case!=name:continue
            web=web_state();web.graph_snapshot.return_value=graph;web.agent_complete=complete
            start=time.monotonic()
            with patch('jaka_agent.tasks.planning.plan_task',side_effect=AssertionError('Legacy planner called')) as legacy:
                result=web.agent_chat({'question':question,'conversation_id':'eval-navigation'})
            task=result.get('task') or {};steps=task.get('steps') or []
            if targets is None:
                passed=bool(result.get('requires_clarification') and not task)
            else:
                passed=bool(result.get('requires_confirmation') and task.get('skill',{}).get('id')=='navigate'
                    and [s.get('target_ann_id') for s in steps if s['type']=='navigate']==targets
                    and sum(s['type']=='return' for s in steps)==int(returning)
                    and task['plan']['need_return']==returning and all(s['type'] in ('navigate','return') for s in steps))
            legacy.assert_not_called();web.tasks.execute.assert_not_called();web.capture.assert_not_called();web.speak.assert_not_called()
            report={'case':name,'repeat':repeat+1,'passed':passed,'seconds':round(time.monotonic()-start,2),
                    'question':question,'tools':result['tool_trace'],'steps':steps,'reply':result['text']}
            results.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
    failures=sum(not r['passed'] for r in results)
    print(json.dumps({'cases':len(results),'failures':failures,'hardware':'mocked'}),flush=True)
    raise SystemExit(bool(failures))


if __name__=='__main__':main()
