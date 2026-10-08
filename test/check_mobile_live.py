"""Explicit live smoke test; only reads pages and transcribes audio, never submits robot tasks."""
import argparse
import io
import json
import wave
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_url')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')

    def request(path, data=None, content_type=None):
        headers = {'Content-Type': content_type} if content_type else {}
        req = Request(base + path, data=data, headers=headers)
        try:
            with urlopen(req, timeout=45) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    status, body = request('/')
    assert status == 200
    assert b'MediaRecorder' in body and b'/api/audio/transcribe' in body
    assert b'/api/listen' not in body
    for path in ('/manifest.webmanifest', '/service-worker.js', '/pwa-icon.svg', '/api/health'):
        assert request(path)[0] == 200, path
    assert json.loads(request('/manifest.webmanifest')[1])['display'] == 'standalone'
    print('Phone recording page and PWA assets: OK', flush=True)
    for content_type, payload, expected in [('audio/webm', b'', 400), ('text/plain', b'audio', 415)]:
        status, _ = request('/api/audio/transcribe', payload, content_type)
        assert status == expected, (status, expected)
    silent = io.BytesIO()
    with wave.open(silent, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\0\0' * 16000)
    assert request('/api/audio/transcribe', silent.getvalue(), 'audio/wav')[0] == 422
    print('Empty, invalid-format, and silent audio handling: OK', flush=True)


if __name__ == '__main__':
    main()
