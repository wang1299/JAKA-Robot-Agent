"""Validate and transcode browser audio for speech recognition."""
from __future__ import annotations
import jaka_agent.web.settings as web_settings
import shutil
import subprocess
from http import HTTPStatus

class AudioRequestError(ValueError):
    """可直接映射为 HTTP 状态码的手机音频请求错误。"""

    def __init__(self, message: str, status: HTTPStatus):
        super().__init__(message)
        self.status = status


def _decode_mobile_audio(audio_bytes: bytes) -> tuple[bytes, int]:
    """通过 ffmpeg 把浏览器录音转换为 16 kHz 单声道 PCM16。"""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("服务器未安装 ffmpeg，无法解码手机录音")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-t", str(web_settings.MAX_AUDIO_SECONDS + 1),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        "-f", "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=audio_bytes,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("手机录音解码超时") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        hint = detail[-1][:200] if detail else "未知格式错误"
        raise RuntimeError(f"手机录音解码失败：{hint}")
    pcm = result.stdout
    if not pcm:
        raise AudioRequestError("录音中没有可识别的音频", HTTPStatus.UNPROCESSABLE_ENTITY)
    duration_ms = round(len(pcm) / (2 * 16000) * 1000)
    if duration_ms > web_settings.MAX_AUDIO_SECONDS * 1000:
        raise AudioRequestError(
            f"录音时长不能超过 {web_settings.MAX_AUDIO_SECONDS} 秒",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
    return pcm, duration_ms
