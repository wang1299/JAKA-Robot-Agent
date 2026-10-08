"""Create cancellation-aware model clients."""
from __future__ import annotations
import jaka_agent.models.settings as models_settings


def _client():
    """DashScope OpenAI 兼容客户端 (懒导入 openai)。"""
    from openai import OpenAI
    if not models_settings.API_KEY:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY")
    from jaka_agent.tasks.runtime import CancellationClient
    return CancellationClient(OpenAI(api_key=models_settings.API_KEY, base_url=models_settings.BASE_URL))
