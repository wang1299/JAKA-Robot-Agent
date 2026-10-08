"""Shared backend logging and exception diagnostics."""
from __future__ import annotations
import jaka_agent.web.settings as web_settings
import logging
import sys
import traceback

LOG_PATH = web_settings.HERE / "logs/robot_web.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger("jaka_vision")


if not LOGGER.handlers:
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def _log_exception(context: str, exc: BaseException, include_traceback: bool = True, **fields):
    """记录可定位的后端异常，不把图片内容或 API 密钥写入日志。"""
    response = getattr(exc, "response", None)
    if response is not None:
        fields.setdefault("http_status", getattr(response, "status_code", None))
        headers = getattr(response, "headers", None)
        if headers:
            fields.setdefault("request_id", headers.get("x-request-id") or headers.get("request-id"))
        body = getattr(response, "text", None)
        if body:
            # 服务端有时只返回一行 Internal Server Error；保留有限长度便于定位，避免日志被图片/HTML撑爆。
            fields.setdefault("response_body", str(body)[:2000])
    if response is not None and body:
        fields.setdefault("response_body", str(body)[:2000])
    fields.setdefault("exception_type", type(exc).__name__)
    details = " ".join(f"{key}={value!r}" for key, value in fields.items() if value is not None)
    message = f"{context}: {exc}"
    if details:
        message += f" | {details}"
    LOGGER.error(message)
    if include_traceback:
        LOGGER.error("Traceback follows:\n%s", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
