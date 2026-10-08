"""Non-secret, persistent model routing for the web entry point.

Loaded before the lazy planner import; explicit environment overrides take priority.
An absent file preserves legacy single-model behavior.
"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from jaka_agent.paths import model_config_path


ALLOWED = {
    "JAKA_AGENT_BASE_URL", "JAKA_AGENT_MODEL", "DASHSCOPE_BASE_URL",
    "QWEN_PLAN_MODEL", "QWEN_VISION_MODEL", "JAKA_AGENT_PROTOCOL",
}


def load_model_config(path=None):
    explicit = os.environ.get("JAKA_MODEL_CONFIG")
    config_path = Path(path or explicit or model_config_path())
    if not config_path.exists() and path is None and not explicit:
        return {}
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) - ALLOWED:
        raise ValueError("模型配置必须是对象，且只能包含模型名称和接口地址")
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError("模型配置值不能为空: " + key)
    effective = {key: os.environ.get(key, value).strip() for key, value in values.items()}
    for key, value in effective.items():
        if not value:
            raise ValueError("模型配置值不能为空: " + key)
        if key == "JAKA_AGENT_PROTOCOL" and value not in ("native", "json"):
            raise ValueError("Agent 协议必须为 native 或 json")
        if key.endswith("BASE_URL"):
            url = urlsplit(value)
            if (url.scheme not in ("http", "https") or not url.hostname
                    or url.username or url.password or url.query or url.fragment):
                raise ValueError("模型接口地址不合法: " + key)
    # Validate everything before modifying the process environment.
    for key, value in effective.items():
        os.environ[key] = value
    return effective
