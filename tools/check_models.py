"""Manual smoke check: model HTTP endpoints and synthetic text/image requests only."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jaka_agent.models.config import load_model_config


def request(base: str, route: str, key: str, timeout: float, payload=None):
    url = urlsplit(base)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError("Invalid model endpoint")
    root = base.rstrip("/").removesuffix("/v1")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(root + route, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read(2 * 1024 * 1024)
    return json.loads(raw)


def parse_answer(response):
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Expected text content")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    return json.loads(content)


def image(color: str) -> dict:
    from PIL import Image
    output = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(output, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")}}


def completion(base, key, timeout, model, content):
    return parse_answer(request(base, "/v1/chat/completions", key, timeout,
        {"model": model, "messages": [{"role": "user", "content": content}],
         "max_tokens": 512, "temperature": 0.0, "stream": False}))


def check(role: str, base: str, model: str, key: str, timeout: float, health_only=False):
    request(base, "/health", key, timeout)
    catalog = request(base, "/v1/models", key, timeout)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise ValueError("Invalid /v1/models response")
    # Local weight directories may not appear in cache-based model listings.
    stages = ["health", "model-list"]
    if health_only:
        return stages
    if role == "qwen":
        answer = completion(base, key, timeout, model,
                            'Return exactly this JSON object, without explanation: {"status":"ok"}')
        if answer != {"status": "ok"}:
            raise ValueError("Text JSON smoke check failed")
        stages.append("text-json")
    else:
        answer = completion(base, key, timeout, model, [image("red"),
            {"type": "text", "text": 'What solid color is this image? Return JSON {"color":"red"} using the actual English color name.'}])
        if not isinstance(answer, dict) or str(answer.get("color", "")).lower() != "red":
            raise ValueError("Single-image smoke check failed")
        stages.append("single-image")
        answer = completion(base, key, timeout, model, [image("red"), image("blue"),
            {"type": "text", "text": 'Identify the solid color of each image in order. Return only JSON {"colors":["first English color","second English color"]}.'}])
        if not isinstance(answer, dict) or answer.get("colors") != ["red", "blue"]:
            raise ValueError("Two-image smoke check failed")
        stages.append("two-image")
    return stages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Same private model JSON used by the application")
    parser.add_argument("--role", choices=["all", "qwen", "minicpm"], default="all")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", type=Path, help="Optional sanitized result JSON")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        load_model_config(args.config)
    except (ValueError, OSError):
        print("Model configuration is invalid or missing; no endpoint contacted")
        return 1
    results = []
    for role in ["qwen", "minicpm"] if args.role == "all" else [args.role]:
        agent = role == "qwen"
        base = os.environ.get("JAKA_AGENT_BASE_URL" if agent else "DASHSCOPE_BASE_URL", "")
        model = os.environ.get("JAKA_AGENT_MODEL" if agent else "QWEN_VISION_MODEL", "")
        key = os.environ.get("JAKA_AGENT_API_KEY" if agent else "DASHSCOPE_API_KEY", "not-needed")
        try:
            if not base or not model:
                raise ValueError("Missing endpoint/model")
            stages = check(role, base, model, key, args.timeout, args.health_only)
            results.append({"role": role, "successful": True, "checks": stages})
            print(f"{role}: PASS ({', '.join(stages)})")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Do not print URLs, response bodies, model paths or credentials.
            results.append({"role": role, "successful": False, "error_type": type(exc).__name__})
            print(f"{role}: FAIL ({type(exc).__name__}); see docs/models.md troubleshooting")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(item["successful"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
