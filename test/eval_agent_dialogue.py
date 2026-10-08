"""Opt-in real MiniCPM dialogue eval through the actual web Agent entry point.

All hardware/task handlers are fixtures. Nothing is deployed, photographed or
moved. Use --base-url with an existing model tunnel; default is loopback only.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot_web
from robot_skills import prepare_skill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--model', default='/home/admin1/MiniCPM/Model/MiniCPM-V-4.6')
    parser.add_argument('--case', default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--map-file', type=Path)
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        parser.error('--repeat must be between 1 and 10')
    model_calls = 0

    def complete(messages, tools):
        nonlocal model_calls
        model_calls += 1
        request = Request(args.base_url + '/chat/completions', data=json.dumps({
            'model': args.model, 'messages': messages, 'tools': tools,
            'tool_choice': 'auto', 'max_tokens': 700, 'temperature': 0,
        }).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=45) as response:
            message = json.load(response)['choices'][0]['message']
        if message.get('tool_calls'):
            calls = message['tool_calls']
            if len(calls) != 1:
                return '{invalid parallel tool calls'
            raw = json.dumps({'tool': calls[0]['function']['name'],
                              'arguments': json.loads(calls[0]['function']['arguments'])}, ensure_ascii=False)
        else:
            raw = message.get('content') or ''
        if args.debug:
            print(json.dumps({'raw': raw}, ensure_ascii=False), flush=True)
        return raw

    categories = ['Blue chair', 'Blue chair', 'Leather armchair', 'Leather armchair', 'Leather armchair', 'Sofa', 'Table']
    graph = {'name': 'fixture-map', 'objects': [
        {'ann_id': i, 'category': label, 'position': '测试区域', 'floor_xy': [i, 0]}
        for i, label in enumerate(categories)]}
    if args.map_file:
        graph = robot_web._normalize_graph(json.loads(args.map_file.read_text(encoding='utf-8')), args.map_file.name)
    # Semantic target labels are evaluation ground truth, never production logic.
    chair_ids = {str(obj['ann_id']) for obj in graph['objects'] if obj['category'].lower() in ('blue chair', 'leather armchair')}

    def state():
        result = object.__new__(robot_web.RobotWebState)
        result.mock = False
        result.agent_lock = threading.Lock()
        result.agent_sessions = {}
        result.graph_snapshot = lambda: graph
        result.agent_complete = complete
        def plan_skill(skill_id, instruction, reference=None, **parameters):
            skill, _, plan = prepare_skill(skill_id, {'instruction': instruction, **parameters}, graph, reference)
            return {'id': 'fixture-only', 'status': 'planned', 'skill': {'id': skill.id, 'name': skill.name}, 'plan': plan}
        result.tasks = SimpleNamespace(snapshot=lambda: None,
            robot_status=lambda: {'online': True, 'battery': 62},
            plan=lambda instruction: {'id': 'fixture-only', 'status': 'planned'},
            plan_skill=plan_skill,
            execute=Mock(side_effect=AssertionError('No execution in evaluation')))
        result.mapping = SimpleNamespace(is_busy=lambda: False)
        result.capture = Mock(return_value=('a' * 32, Path(__file__)))
        result.capture_path = lambda capture_id: Path(__file__)
        result.reference_path = lambda rid, suffix: Path(__file__)
        result.agent_vision = Mock(return_value='这张照片可确认看到1把蓝色椅子，旁边有一张桌子。右侧被墙遮挡，无法判断遮挡区域。不能代表全房间总数。')
        return result

    point_ids = [obj['ann_id'] for obj in graph['objects'] if obj.get('lifecycle', 'active') == 'active']
    if len(point_ids) < 2:
        parser.error('Skill evaluation needs at least two map points')
    first, second = point_ids[:2]
    cases = [
        ('chat', ['你好'], None),
        ('capabilities', ['你平时能帮我做什么呀'], None),
        ('map', ['帮我查询地图里有几个椅子'], 'query_map'),
        ('paraphrase', ['替我数数保存的环境记录里那些供人坐的椅子，扶手椅也算，沙发不算'], 'query_map'),
        ('no_camera', ['不要拍照，只查地图里有几把椅子'], 'query_map'),
        ('scene', ['你知道当前场景有几个椅子吗'], 'observe_scene'),
        ('history', ['你看看面前有什么', '刚才照片里的椅子是什么颜色'], 'observe_scene'),
        ('switch_scope', ['你看看面前有什么', '帮我查询地图里有几个椅子'], 'query_map'),
        ('confirmation', ['对的'], 'query_map'),
        ('count_followup', ['地图里有椅子吗', '那一共几个'], 'query_map'),
        ('plan', [f'去地图编号#{first}处拍照看看周围有没有障碍物，然后回来'], 'plan_inspect_location'),
        ('patrol', [f'在地图编号#{first}和#{second}这两处检查环境变化，比较两轮'], 'plan_patrol'),
        ('welcome', [f'到地图编号#{first}接照片中的人，确认后带到#{second}'], 'plan_welcome'),
        ('find', ['帮我寻找参考照片里的物品'], 'plan_find_object'),
        ('status', ['你的电量还剩多少'], 'get_robot_status'),
    ]
    failures = []
    runs = [(repeat, case) for repeat in range(1, args.repeat + 1) for case in cases]
    evaluated = 0
    for repeat, (name, questions, expected) in runs:
        if args.case and name not in args.case.split(','):
            continue
        web = state()
        evaluated += 1
        calls_before = model_calls
        started = time.monotonic()
        try:
            history = []
            if name == 'confirmation':
                history = [{'role': 'user', 'text': '帮我查询地图里有几个椅子'},
                           {'role': 'assistant', 'text': '你想让我查询地图里的椅子总数吗？'}]
            turns = []
            for question in questions:
                reference = {'reference_id': 'b' * 32, 'suffix': '.jpg'} if name in ('welcome', 'find') else None
                result = web.agent_chat({'question': question, 'conversation_id': 'eval-' + name, 'history': history, 'reference': reference})
                turns.append(result)
            called = [step['tool'] for result in turns for step in result['tool_trace']]
            passed = not any(r.get('incomplete') for r in turns)
            passed = passed and (expected in called if expected else not called)
            passed = passed and not any(re.search(r'query_map|observe_scene|inspect_previous_scene|无需调用工具|summarize_map_selection', r['text']) for r in turns)
            passed = passed and all(re.search(r'[\u4e00-\u9fff]', r['text']) and not re.search(r'Note:|The user|无需调用工具', r['text']) for r in turns)
            if expected == 'query_map':
                # A follow-up can cite the already verified count without doing
                # another identical query. Validate the last actual selection.
                selections = [turn for turn in turns if any(step['tool'] == 'query_map' and step['ok'] for step in turn['tool_trace'])]
                passed = passed and bool(selections) and set(selections[-1]['target_ids']) == chair_ids
                # The current fixture and supplied project map both contain five.
                if len(chair_ids) == 5:
                    passed = passed and bool(re.search(r'5|五', result['text']))
            if name in ('map', 'paraphrase', 'no_camera', 'confirmation', 'count_followup'):
                passed = passed and not web.capture.called
            if name == 'history':
                passed = passed and web.capture.call_count == 1
            if name == 'switch_scope':
                passed = passed and web.capture.call_count == 1 and any(s['tool'] == 'observe_scene' for s in turns[0]['tool_trace'])
            if name in ('plan', 'patrol', 'welcome', 'find'):
                passed = passed and result.get('requires_confirmation') is True and not web.capture.called
                passed = passed and (result.get('task', {}).get('skill') or {}).get('id') == expected.removeprefix('plan_')
            if name == 'capabilities':
                passed = passed and '迎宾' in result['text'] and '巡逻' in result['text']
            if name == 'status':
                passed = passed and '62' in result['text']
            web.tasks.execute.assert_not_called()
            print(json.dumps({'case': name, 'repeat': repeat, 'passed': bool(passed), 'model_calls': model_calls-calls_before,
                'seconds': round(time.monotonic()-started, 1),
                'tools': called, 'answers': [r['text'] for r in turns]}, ensure_ascii=False), flush=True)
        except Exception as exc:
            passed = False
            print(json.dumps({'case': name, 'passed': False, 'error_type': type(exc).__name__, 'error': str(exc)}, ensure_ascii=False), flush=True)
        if not passed:
            failures.append(f'{repeat}:{name}')
    if not evaluated:
        parser.error('No matching evaluation cases')
    print(json.dumps({'evaluated': evaluated, 'failed': failures, 'model_calls': model_calls}, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
