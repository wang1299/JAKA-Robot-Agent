"""Real Qwen decision evaluation with persistent history; NO physical actions."""
import json
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_web
from robot_memory import ConversationStore
from test_robot_skills import web_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='')
    options=parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    graph = robot_web._normalize_graph(json.loads((root/'zmq_scene_graph.json').read_text(encoding='utf-8')), 'zmq_scene_graph.json')
    command = 'python3 -c ' + shlex.quote("import sys,urllib.request; r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions',data=sys.stdin.buffer.read(),headers={'Content-Type':'application/json'}),timeout=45); sys.stdout.buffer.write(r.read())")
    def complete(messages, tools):
        body = {'model':'Qwen3.5-9B','messages':messages,'tools':tools,'tool_choice':'required',
                'parallel_tool_calls':False,'temperature':0,'max_tokens':700}
        proc = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','myserver',command],
                              input=json.dumps(body,ensure_ascii=False).encode(),capture_output=True,timeout=55)
        if proc.returncode: raise RuntimeError(proc.stderr.decode(errors='replace'))
        message = json.loads(proc.stdout)['choices'][0]['message']
        calls = message.get('tool_calls') or []
        if len(calls)==1:
            function=calls[0]['function']
            return json.dumps({'tool':function['name'],'arguments':json.loads(function['arguments'])},ensure_ascii=False)
        return message.get('content') or ''

    cases = [('inside','你觉得它里面有什么','answer'),
             ('left_board','刚才它左边那块白板写着什么','answer'),
             ('old_photo','再分析刚才那张照片，箱盖是开着的吗','answer'),
             ('current_position','你现在还在航空箱旁边吗','status'),
             ('new_navigation','现在去送客点，到那里停留，不用回来','navigate'),
             ('independent','它里面有什么','independent'),
             ('earlier_memory','还记得我之前给箱子取的名字吗','recall')]
    if options.case: cases=[c for c in cases if c[0]==options.case]
    failures=0
    with tempfile.TemporaryDirectory() as directory:
        for name,question,expected in cases:
            store=ConversationStore(Path(directory)/(name+'.sqlite3'))
            for cid in ('memory-eval-A','memory-eval-B'): store.create(cid)
            web=web_state();web.conversation_store=store;web.agent_sessions={}
            web.graph_snapshot.return_value=graph;web.agent_complete=complete
            web.capture_path=Mock(return_value=Path(__file__))
            web.agent_vision=Mock(return_value='照片中是关闭的黑色航空箱，左侧白板上有文字和示意图但无法清晰辨认；箱内不可见。')
            web.tasks.robot_status=Mock(return_value={'online':True,'pose':{'x':1,'y':2},'move_status':'idle'})
            def msg(role,text,index):
                message={'id':f'history-{index:03d}','role':role,'text':text,'createdAt':(time.time()-300+index)*1000}
                store.message('memory-eval-A',message);store.event('memory-eval-A','dialogue',message)
            msg('user','我把这个航空箱叫做小黑盒，请记住这个名字。',0)
            for i in range(1,40): msg('user' if i%2 else 'assistant','讨论其他普通问题。',i)
            task=web.plan_for_conversation('memory-eval-A',lambda:web.tasks.plan_skill('inspect_location',
                '去黑色航空箱旁边看看',target_ann_id=71,question='旁边有什么'))
            msg('user','你可以去黑色航空箱那边看看吗',40)
            store.message('memory-eval-A',{'id':'assistant-plan','role':'assistant','text':'计划已生成，等待确认。',
                'taskId':task['id'],'createdAt':(time.time()-200)*1000})
            store.save_memory('memory-eval-A',{'evidence':[],'planned_task':{'id':task['id'],'status':'planned'},
                'pending_request':{'task_id':task['id'],'goal':'去航空箱查看','status':'awaiting_confirmation'}})
            web.tasks._update(status='succeeded',finished_at=time.time()-120,observations=[{
                'ann_id':71,'image_url':'/captures/'+'a'*32+'.jpg','observed_at':time.time()-130,
                'text':'前方是关闭的黑色航空箱，左侧有白板，后方有金属门。'}])
            # Reload from disk; not a RAM-only happy-path test.
            web.conversation_store=ConversationStore(store.path)
            start=time.monotonic()
            result=web.agent_chat({'conversation_id':'memory-eval-B' if expected=='independent' else 'memory-eval-A','question':question})
            names=[t['tool'] for t in result['tool_trace']]
            passed=not result.get('incomplete')
            if expected=='navigate':
                passed &= (result.get('task') or {}).get('skill',{}).get('id')=='navigate'
            else:
                passed &= not result.get('task') and not any(n.startswith('plan_') for n in names)
                if expected=='status': passed &= 'get_robot_status' in names
                if expected=='independent': passed &= '航空箱' not in result['text'] and not web.agent_vision.called
                if expected=='recall': passed &= 'recall_conversation' in names and '小黑盒' in result['text']
            passed &= not web.capture.called and not web.tasks.execute.called and not web.speak.called
            report={'case':name,'passed':bool(passed),'seconds':round(time.monotonic()-start,2),'tools':names,'reply':result['text']}
            print(json.dumps(report,ensure_ascii=False),flush=True)
            failures += not passed
    print(json.dumps({'cases':len(cases),'failures':failures,'hardware_and_vision':'mocked; decision_model=real_Qwen'}),flush=True)
    raise SystemExit(bool(failures))


if __name__=='__main__': main()
