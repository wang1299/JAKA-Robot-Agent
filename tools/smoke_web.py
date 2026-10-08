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
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def smoke(python=sys.executable, installed=False, legacy=False):
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
        command = [python, *entry, '--mock', '--host', '127.0.0.1', '--port', str(port)]
        url = f'http://127.0.0.1:{port}'
        with (work / 'server.log').open('w', encoding='utf8') as log:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        with urlopen(url + '/api/health', timeout=1) as response:
                            assert json.load(response) == {'ok': True, 'mock': True}
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
                assert (work / 'runtime/conversation_data/conversations.sqlite3').is_file()
                assert (work / 'runtime/logs/robot_web.log').is_file()
            finally:
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
    args = parser.parse_args()
    count = smoke(args.python, args.installed, args.legacy)
    print(f'Mock HTTP smoke passed: health, map, {count} assets, resource boundaries and data directory.')


if __name__ == '__main__':
    main()
