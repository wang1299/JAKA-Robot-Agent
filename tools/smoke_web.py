"""Check a real Mock HTTP process, including installed assets and writable data."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def smoke(python=sys.executable, installed=False, legacy=False, replay=False):
    with tempfile.TemporaryDirectory(prefix='jaka-smoke-') as directory:
        work = Path(directory)
        env = os.environ.copy()
        env['JAKA_DATA_DIR'] = str(work / 'runtime')
        env['PYTHONUNBUFFERED'] = '1'
        if installed:
            env.pop('PYTHONPATH', None)
        else:
            env['PYTHONPATH'] = str(ROOT / 'src')
        if installed:
            check = "import pathlib,sys,jaka_agent; assert pathlib.Path(jaka_agent.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()), jaka_agent.__file__"
            subprocess.run([python, '-c', check], cwd=work, env=env, check=True, capture_output=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        entry = [str(ROOT / 'robot_web.py')] if legacy else ['-m', 'jaka_agent']
        command = [python, *entry, '--replay' if replay else '--mock', '--host', '127.0.0.1', '--port', str(port)]
        url = f'http://127.0.0.1:{port}'
        with (work / 'server.log').open('w', encoding='utf8') as log:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        with urlopen(url + '/api/health', timeout=1) as response:
                            assert json.load(response) == {'ok': True, 'mock': True, 'replay': replay}
                        break
                    except (URLError, TimeoutError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError((work / 'server.log').read_text(encoding='utf8'))
                        time.sleep(.1)
                assets = {
                    '/': ('text/html', b'/static/js/app.js'),
                    '/scene-viewer': ('text/html', b'/static/js/scene.js'),
                    '/static/css/app.css': ('text/css', b'{'),
                    '/static/js/app.js': ('text/javascript', b'function'),
                    '/static/css/scene.css': ('text/css', b'{'),
                    '/static/js/scene.js': ('text/javascript', b'import'),
                    '/service-worker.js': ('text/javascript', b'/static/js/app.js'),
                    '/manifest.webmanifest': ('application/manifest+json', b'icons'),
                    '/map-icons/door.svg': ('image/svg+xml', b'<svg'),
                    '/slam-map/image': ('image/png', b'\x89PNG'),
                }
                if replay:
                    assets.pop('/slam-map/image')
                    assets.update({'/replay': ('text/html', b'/static/js/replay.js'),
                                   '/static/js/replay.js': ('text/javascript', b'confirmation_id'),
                                   '/static/css/replay.css': ('text/css', b'{')})
                for path, (mime, marker) in assets.items():
                    with urlopen(url + path, timeout=3) as response:
                        assert response.headers.get_content_type() == mime, path
                        assert marker in response.read(), path
                for path in ['/static/../resources/model_config.json', '/static/js/missing.js']:
                    try:
                        urlopen(url + path, timeout=3)
                    except HTTPError as exc:
                        assert exc.code == 404, path
                    else:
                        raise AssertionError('Unexpected static resource: ' + path)
                with urlopen(url + '/api/map', timeout=3) as response:
                    assert json.load(response)['objects']
                if replay:
                    def post(path, payload):
                        request = Request(url + path, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                        with urlopen(request, timeout=5) as response:
                            return json.load(response)
                    answer = post('/api/replay/plan', {'case_id': 'find_object', 'conversation_id': 'wheel-smoke'})
                    task = answer['result']['task']
                    assert task['status'] == 'planned'
                    with urlopen(url + '/api/replay?conversation_id=wheel-smoke', timeout=3) as response:
                        assert json.load(response)['robot']['pose']['x'] == 0
                    post('/api/task/cancel', {'task_id': task['id'], 'conversation_id': 'wheel-smoke'})
                assert (work / 'runtime/conversation_data/conversations.sqlite3').is_file()
                assert (work / 'runtime/logs/robot_web.log').is_file()
            finally:
                if os.name == 'nt' and process.poll() is None:
                    # A venv launcher can own a child Python on Windows; stop both.
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                elif process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    return len(assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--installed', action='store_true', help='Use installed package without source paths')
    parser.add_argument('--legacy', action='store_true', help='Check the original robot_web.py entry point')
    parser.add_argument('--replay', action='store_true', help='Check synthetic task replay and its packaged assets')
    args = parser.parse_args()
    count = smoke(args.python, args.installed, args.legacy, args.replay)
    print(f'{"Replay" if args.replay else "Mock"} HTTP smoke passed: health, map, {count} assets, resource boundaries and data directory.')


if __name__ == '__main__':
    main()
