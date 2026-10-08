"""Check publishable deployment files; report locations, never secret values."""
from __future__ import annotations

import ipaddress
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
SECRET = re.compile(
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b|"
    r"^\s*(?:export\s+)?(?:\w*(?:PASSWORD|PASSWD)|HF_TOKEN|DASHSCOPE_API_KEY|JAKA_AGENT_API_KEY)"
    r"\s*=\s*(?!not-needed(?:\s|$)|[\"']?\$|[\"']?<)[^\s#]+", re.I | re.M)


def inspect_file(name: str, data: str) -> list[str]:
    issues = []
    for number, line in enumerate(data.splitlines(), 1):
        if SECRET.search(line):
            issues.append(f"{name}:{number}: possible credential")
        for match in IPV4.finditer(line):
            try:
                address = ipaddress.ip_address(match.group())
            except ValueError:
                continue
            if not address.is_loopback:
                issues.append(f"{name}:{number}: non-loopback address")
    return issues


def main():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=ROOT, check=True, capture_output=True)
    names = set(result.stdout.decode("utf-8").split("\0")) - {""}
    issues = []
    for name in sorted(names):
        path = ROOT / name
        if name.endswith((".local.env", ".local.json", ".safetensors", ".onnx", ".pth", ".pt")):
            issues.append(f"{name}: private config or weights included")
        # Robot addresses and synthetic network fixtures elsewhere have separate purposes.
        selected = (name.startswith(("deploy/server/", "deploy/raspberrypi/", "configs/"))
                    or name in {"tunnel.sh", "docs/models.md"})
        if selected and path.is_file() and path.stat().st_size < 1024 * 1024:
            issues.extend(inspect_file(name, path.read_text(encoding="utf-8-sig")))
    for issue in issues:
        print(issue)
    print(f"Deployment privacy: {'FAIL' if issues else 'PASS'}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
