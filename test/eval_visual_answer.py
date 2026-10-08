"""Opt-in MiniCPM evaluation of an EXISTING photo; never opens the camera."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen_planner
import robot_web


class SSHVisionClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, **options):
        return self

    def create(self, **payload):
        payload.setdefault('max_tokens', 512)
        command = 'python3 -c ' + shlex.quote("import sys,urllib.request; data=sys.stdin.buffer.read(); r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=data,headers={'Content-Type':'application/json'}),timeout=45); sys.stdout.buffer.write(r.read())")
        result = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','myserver',command],
                                input=json.dumps(payload,ensure_ascii=False).encode(),capture_output=True,timeout=55)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors='replace'))
        text = json.loads(result.stdout)['choices'][0]['message']['content']
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error('An existing photo is required')
    web = object.__new__(robot_web.RobotWebState)
    web.mock = False
    web.infer_lock = threading.Lock()
    questions = ('黑色航空箱里面有什么？', '描述一下画面里能看到什么。', '这张照片能确定箱子里有工具吗？')
    with patch.object(qwen_planner, '_client', return_value=SSHVisionClient()):
        for mode in ('task', 'chat'):
            for question in questions:
                answer = (qwen_planner.read_image(str(args.image), question) if mode == 'task'
                          else web.agent_vision(args.image, question, []))
                print(json.dumps({'mode':mode,'question':question,'answer':answer},ensure_ascii=False),flush=True)
                assert 'detail:' not in answer.lower() and 'detail：' not in answer.lower()
    print('6 responses: formatting checked; visual grounding requires review against input photo.',flush=True)


if __name__ == '__main__':
    main()
