"""Execution event hooks and shared speech output."""
from __future__ import annotations


_VOICE = None


def announce(msg, level="info"):
    """关键节点播报；已取消任务不得新增播报。"""
    from jaka_agent.tasks.runtime import check_cancelled
    check_cancelled()
    print(msg)
    if _VOICE is not None:
        try:
            _VOICE.say(msg, wait=False)
        except Exception as e:
            print(f"[TTS 失败] {e}")
