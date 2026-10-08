#!/usr/bin/env python3
"""Download official sherpa-onnx voice assets, validate, then publish atomically."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
ASSETS = {
    "asr": ("asr-paraformer", RELEASE + "asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2"),
    "tts": ("vits-piper-zh_CN-huayan-medium", RELEASE + "tts-models/vits-piper-zh_CN-huayan-medium.tar.bz2"),
}


def validate(directory: Path, role: str) -> None:
    required = ["tokens.txt", "encoder.int8.onnx", "decoder.int8.onnx"] if role == "asr" else ["tokens.txt"]
    for name in required:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/empty voice asset: {name}")
    if role == "tts":
        if not any(p.stat().st_size for p in directory.glob("*.onnx")):
            raise ValueError("Missing TTS ONNX weights")
        if not (directory / "espeak-ng-data").is_dir() or not any((directory / "espeak-ng-data").iterdir()):
            raise ValueError("Missing TTS espeak-ng-data")


def safe_extract(archive: Path, destination: Path) -> None:
    """Allow regular files/directories only; reject traversal, links and devices."""
    destination = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        if sum(m.size for m in members) > 4 * 1024**3:
            raise ValueError("Voice archive exceeds the extraction limit")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name or ":" in member.name:
                raise ValueError("Unsafe archive path")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Archive contains a link or special file")
            target = destination.joinpath(*path.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive path escapes destination")
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def download(role: str, base: Path, expected_sha256: str | None = None) -> Path:
    name, url = ASSETS[role]
    base.mkdir(parents=True, exist_ok=True)
    target = base / name
    if target.exists():
        validate(target, role)
        print(f"{role}: existing assets validated; no download")
        return target
    with tempfile.TemporaryDirectory(prefix=".voice-download-", dir=base) as temporary:
        staging = Path(temporary)
        archive = staging / "model.tar.bz2"
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256.lower():
            raise ValueError("Downloaded archive SHA256 mismatch")
        extracted = staging / "extracted"
        extracted.mkdir()
        safe_extract(archive, extracted)
        candidates = [p for p in extracted.iterdir() if p.is_dir()]
        if len(candidates) != 1:
            raise ValueError("Expected one top-level model directory")
        model = candidates[0]
        validate(model, role)
        (model / "download-manifest.json").write_text(json.dumps(
            {"source": url, "archive_sha256": actual}, indent=2) + "\n", encoding="utf-8")
        model.rename(target)
    print(f"{role}: downloaded and validated")
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["all", "asr", "tts"], default="all")
    parser.add_argument("--base", type=Path,
                        default=Path(os.environ.get("JAKA_VOICE_BASE", "/home/pi/voice")).expanduser())
    parser.add_argument("--sha256", help="Optional independently obtained archive digest (one role only)")
    parser.add_argument("--check", action="store_true", help="Validate existing assets without downloading")
    args = parser.parse_args(argv)
    if args.sha256 and (args.role == "all" or len(args.sha256) != 64 or
                        any(c not in "0123456789abcdefABCDEF" for c in args.sha256)):
        parser.error("--sha256 requires one role and a 64-digit hex digest")
    try:
        for role in ASSETS if args.role == "all" else [args.role]:
            if args.check:
                validate(args.base / ASSETS[role][0], role)
                print(f"{role}: assets OK")
            else:
                download(role, args.base, args.sha256)
    except (OSError, ValueError, tarfile.TarError):
        print("Voice model operation failed; check assets, permissions or network")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
