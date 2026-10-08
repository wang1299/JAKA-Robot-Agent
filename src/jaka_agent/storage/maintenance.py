"""Explicit capture retention helpers."""
from __future__ import annotations
import jaka_agent.web.settings as web_settings
import time

def _clean_old_captures(max_age_days=14, pinned=()):
    """启动时清理过旧图片(best-effort), 避免树莓派磁盘无限增长。"""
    if not web_settings.CAPTURE_DIR.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    patterns = {"*.jpg", *(f"*{suffix}" for suffix in web_settings.REFERENCE_IMAGE_TYPES)}
    for pattern in patterns:
        for path in web_settings.CAPTURE_DIR.glob(pattern):
            try:
                if not {f"/captures/{path.name}", f"/references/{path.name}"}.intersection(pinned) and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass
