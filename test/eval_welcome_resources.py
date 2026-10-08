"""Opt-in real Qwen decisions over mocked hardware/tasks; no deployment or motion."""
import argparse
import copy
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_web
from test_robot_skills import web_state, REFERENCE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--case', default='')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    raw = json.loads((Path(__file__).resolve().parents[1] / 'zmq_scene_graph.json').read_text(encoding='utf-8'))
    graph = robot_web._normalize_graph(raw, 'zmq_scene_graph.json')
    # Test IDs are discovered from the bindings, never part of production prompts.
    defaults = {o.get('skill_defaults', {}).get('welcome'): o['ann_id'] for o in graph['objects'] if o.get('skill_defaults', {}).get('welcome')}
    pickup, dropoff = defaults['pickup_ann_id'], defaults['return_ann_id']
    request_code = "import sys,urllib.request; p=sys.stdin.buffer.read(); r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions',data=p,headers={'Content-Type':'application/json'}),timeout=45); sys.stdout.buffer.write(r.read())"
    remote_command = 'python3 -c ' + shlex.quote(request_code)

    def complete(messages, tools):
        payload = {'model': 'Qwen3.5-9B', 'messages': messages, 'tools': tools,
                   'tool_choice': 'required', 'parallel_tool_calls': False, 'temperature': 0, 'max_tokens': 700}
        proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', 'myserver', remote_command],
                              input=json.dumps(payload, ensure_ascii=False).encode(), capture_output=True, timeout=55)
        if proc.returncode:
            raise RuntimeError(proc.stderr.decode(errors='replace'))
        message = json.loads(proc.stdout)['choices'][0]['message']
        if args.debug:print(json.dumps({'model_reply': message}, ensure_ascii=False), flush=True)
        calls = message.get('tool_calls') or []
        if len(calls) == 1:
            f = calls[0]['function']
            return json.dumps({'tool': f['name'], 'arguments': json.loads(f['arguments'])}, ensure_ascii=False)
        return message.get('content') or ''

    results = []
    for repeat in range(args.repeat):
        for case in ('two_turn_upload', 'photo_first', 'changed_wording', 'missing_destination', 'ambiguous_pickup', 'explicit_override', 'chat', 'map_count'):
            if args.case and case != args.case:continue
            web = web_state()
            web.graph_snapshot.return_value = copy.deepcopy(graph)
            web.tasks.task = {'id': 'previous-find', 'kind': 'find_object', 'status': 'succeeded'}
            web.agent_complete = complete
            if case == 'missing_destination':
                web.graph_snapshot.return_value['objects'] = [o for o in web.graph_snapshot.return_value['objects'] if o['ann_id'] != dropoff]
            if case == 'ambiguous_pickup':
                obj = copy.deepcopy(next(o for o in graph['objects'] if o['ann_id'] == pickup))
                obj['ann_id'] = 900001
                web.graph_snapshot.return_value['objects'].append(obj)
            start = time.monotonic()
            preliminary = None
            if case == 'two_turn_upload':
                preliminary = web.agent_chat({'question': '你能去帮我接个人吗', 'conversation_id': 'eval-welcome'})
                if args.debug:print(json.dumps({'preliminary': preliminary}, ensure_ascii=False), flush=True)
                assert preliminary.get('requires_clarification') and preliminary['execution']['task_id'] is None
            question = {'chat': '你好，你叫什么名字？', 'map_count': '只查询保存地图，接客点和送客点一共有几个？不要移动。',
                        'changed_wording': '照片里这位要来，麻烦到平时接人的地方等他，再带去安排好的地方。',
                        'explicit_override': f'去地图#{dropoff}接照片中的客人，再带到#{pickup}，不要用默认路线。'}.get(case, '你能去帮我接个人吗')
            result = web.agent_chat({'question': question, 'reference': None if case in ('chat','map_count') else REFERENCE, 'conversation_id': 'eval-welcome'})
            expected_plan = case not in ('missing_destination', 'ambiguous_pickup')
            if case in ('chat','map_count'):
                called = [t['tool'] for t in result['tool_trace']]
                success = (not result.get('task') and result.get('response_kind') == 'answer'
                           and result['execution']['task_id'] is None
                           and (not called if case == 'chat' else 'query_map' in called))
            elif expected_plan:
                task = result.get('task') or {}
                success = (result.get('requires_confirmation') and task.get('kind') == 'welcome'
                           and task.get('reference') == REFERENCE and task.get('status') == 'planned'
                           and task.get('pickup_ann_id') == (dropoff if case == 'explicit_override' else pickup)
                           and task.get('return_ann_id') == (pickup if case == 'explicit_override' else dropoff))
            else:
                success = bool(result.get('requires_clarification') and result['execution']['task_id'] is None)
            web.capture.assert_not_called(); web.tasks.execute.assert_not_called(); web.speak.assert_not_called()
            report = {'case': case, 'repeat': repeat + 1, 'passed': bool(success), 'seconds': round(time.monotonic()-start, 2),
                      'tools': result['tool_trace'], 'text': result['text'], 'first_reply': preliminary['text'] if preliminary else None}
            print(json.dumps(report, ensure_ascii=False), flush=True)
            results.append(report)
    failures = sum(not r['passed'] for r in results)
    print(json.dumps({'cases': len(results), 'failures': failures, 'hardware': 'mocked'}, ensure_ascii=False), flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
