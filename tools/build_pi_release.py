"""Build a Raspberry Pi runtime archive from an explicit allowlist; never deploy tests or secrets."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'deploy/raspberrypi/runtime-files.json').read_text(encoding='utf-8'))
    names = config['files']
    if len(names) != len(set(names)):
        raise ValueError('Duplicate release entries')
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    manifest = {}
    with tarfile.open(output / 'jaka-pi-runtime.tar.gz', 'w:gz') as archive:
        for name in names:
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError(f'Invalid release file: {name}')
            data = path.read_bytes()
            if path.suffix in {'.py', '.html', '.js', '.sh', '.json', '.svg', '.webmanifest', '.txt', '.css', '.toml', '.md', '.example'}:
                data = data.replace(b'\r\n', b'\n')
            info = tarfile.TarInfo(name)
            info.mode = 0o755 if path.suffix == '.sh' else 0o644
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            manifest[name] = hashlib.sha256(data).hexdigest()
    (output / 'runtime-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'{len(manifest)} runtime files: {output / "jaka-pi-runtime.tar.gz"}')


if __name__ == '__main__':
    main()
