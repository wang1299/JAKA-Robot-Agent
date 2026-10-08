"""Share capture, microphone and speech resources across web requests."""
from __future__ import annotations
import jaka_agent.agent.runner as agent_runner
import jaka_agent.diagnostics as diagnostics
import jaka_agent.hardware.audio as hardware_audio
import jaka_agent.tasks.events as tasks_events
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import base64
import os
import shutil
import time
import uuid
from http import HTTPStatus
from pathlib import Path

class HardwareServiceMixin:
    def capture_path(self, capture_id: str) -> Path:
        if not web_settings.CAPTURE_ID_RE.fullmatch(capture_id or ""):
            raise ValueError("非法 capture_id")
        archive = getattr(self, 'media_archive', None)
        return archive.resolve(f'/captures/{capture_id}.jpg') if archive else web_settings.CAPTURE_DIR / f"{capture_id}.jpg"

    def reference_path(self, reference_id: str, suffix: str) -> Path:
        if not web_settings.CAPTURE_ID_RE.fullmatch(reference_id or ""):
            raise ValueError("非法 reference_id")
        if suffix not in web_settings.REFERENCE_IMAGE_TYPES:
            raise ValueError("图片格式不支持")
        archive = getattr(self, 'media_archive', None)
        return archive.resolve(f'/references/{reference_id}{suffix}') if archive else web_settings.CAPTURE_DIR / f"{reference_id}{suffix}"

    def new_capture_path(self, capture_id, media_scope=None, kind='observations', metadata=None):
        archive = getattr(self, 'media_archive', None)
        if archive and media_scope:
            return archive.allocate(f'/captures/{capture_id}.jpg', media_scope['conversation_id'],
                                    media_scope['turn_id'], kind, media_scope.get('task_id'), metadata)
        return self.capture_path(capture_id)

    def save_reference_image(self, name: str, encoded: str, media_scope=None) -> dict:
        """兼容旧 JSON 上传格式；Web 页面发送原始图片字节。"""
        try:
            payload = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("参考图片不是合法 Base64") from exc
        return self.save_reference_upload(name, payload, media_scope=media_scope)

    def save_reference_upload(self, name: str, payload: bytes, media_scope=None) -> dict:
        """校验并保存原始参考图片，避免 Base64 编解码。"""
        suffix = Path(name).suffix.lower()
        info = web_settings.REFERENCE_IMAGE_TYPES.get(suffix)
        if info is None:
            raise ValueError("参考图片仅支持 PNG、JPEG 或 WebP")
        if not payload or len(payload) > web_settings.MAX_REFERENCE_BYTES:
            raise ValueError("参考图片为空或超过 6 MB")
        _, magic = info
        if not payload.startswith(magic) or (suffix == ".webp" and payload[8:12] != b"WEBP"):
            raise ValueError("参考图片内容与文件格式不匹配")
        reference_id = uuid.uuid4().hex
        path = self.reference_path(reference_id, suffix)
        archive = getattr(self, 'media_archive', None)
        url = f'/references/{reference_id}{suffix}'
        if archive and media_scope:
            path = archive.allocate(url, media_scope['conversation_id'], media_scope['turn_id'], 'uploads',
                                    metadata={'original_name':str(name)[:255]})
        path.write_bytes(payload)
        if archive: archive.record(url)
        return {
            "reference_id": reference_id,
            "suffix": suffix,
            "content_type": info[0],
            "image_url": f"/references/{reference_id}{suffix}",
        }

    def capture(self, media_scope=None, kind='observations', metadata=None) -> tuple[str, Path]:
        """拍一张彩色图; 同一时刻只允许一个请求占用 Orbbec。"""
        capture_id = uuid.uuid4().hex
        path = self.new_capture_path(capture_id, media_scope, kind, metadata)
        with self.camera_lock:
            if self.mock:
                from jaka_agent.paths import runtime_path
                source = runtime_path("shot.jpg")
                if not source.exists():
                    raise RuntimeError("mock 模式需要脚本目录下存在 shot.jpg")
                shutil.copyfile(source, path)
                if getattr(self, 'media_archive', None): self.media_archive.record(f'/captures/{capture_id}.jpg')
                return capture_id, path

            try:
                self.tasks._hardware_camera_driver().capture(str(path))
            except Exception as exc:
                diagnostics.LOGGER.warning("[camera] scene capture failed type=%s", type(exc).__name__)
                raise agent_runner.ToolInputError("本次相机采集失败，请检查相机连接或占用后重试；不是机器人不具备视觉能力。") from exc
        if getattr(self, 'media_archive', None): self.media_archive.record(f'/captures/{capture_id}.jpg', captured_at=time.time())
        return capture_id, path

    def speak(self, text: str):
        """复用 qwen_planner 的 announce + VoiceAssistant 队列，播报不阻塞任务线程。"""
        tasks_runtime.check_cancelled()
        text = str(text or "").strip()
        if not text:
            return
        if self.mock:
            print(f"[tts mock] {text}")
            return
        try:
            with self.voice_lock:
                if self.voice is None:
                    from jaka_agent.hardware.voice import VoiceAssistant

                    self.voice = VoiceAssistant()
                voice = self.voice


                tasks_events._VOICE = voice
            tasks_events.announce(text)
        except Exception as exc:
            print(f"[TTS 失败] {exc}")

    def ask_and_listen(self, prompt: str) -> str:
        """同步播完迎宾询问后收听一句，避免把喇叭余音误识别为对方回答。"""
        tasks_runtime.check_cancelled()
        prompt = str(prompt or "").strip()
        if self.mock:
            print(f"[tts mock] {prompt}")
            time.sleep(0.3)
            return os.getenv("JAKA_WEB_MOCK_WELCOME_ANSWER", "是的").strip()
        with self.voice_lock:
            tasks_runtime.check_cancelled()
            if self.voice is None:
                from jaka_agent.hardware.voice import VoiceAssistant

                self.voice = VoiceAssistant()
            print(prompt)
            self.voice.say(prompt, wait=True)
            time.sleep(0.25)
            return (tasks_runtime.guarded_call(self.voice.listen_utterance) or "").strip()

    def listen(self) -> str:
        """用树莓派麦克风听一句话并返回文字; 不在网页请求线程间并发录音。"""
        if self.mock:
            # 仅供没有麦克风/语音模型的电脑测试网页按钮。可用环境变量换成任意测试句，
            # 不参与树莓派真实模式; 真实模式始终返回用户当次说出的自然语言。
            time.sleep(0.7)
            return os.getenv("JAKA_WEB_MOCK_VOICE_TEXT", "这是一条模拟语音输入")
        with self.voice_lock:
            if self.voice is None:
                from jaka_agent.hardware.voice import VoiceAssistant

                self.voice = VoiceAssistant()
            return (self.voice.listen_utterance() or "").strip()

    def transcribe_mobile_audio(self, audio_bytes: bytes) -> dict:
        """识别手机浏览器上传的录音，不保存原始音频。"""
        request_id = uuid.uuid4().hex
        if self.mock:
            time.sleep(0.1)
            text = os.getenv("JAKA_WEB_MOCK_VOICE_TEXT", "这是一条模拟语音输入").strip()
            if not text:
                raise hardware_audio.AudioRequestError("没有识别到有效语音", HTTPStatus.UNPROCESSABLE_ENTITY)
            return {"text": text, "duration_ms": 0, "request_id": request_id}

        pcm, duration_ms = hardware_audio._decode_mobile_audio(audio_bytes)
        with self.voice_lock:
            if self.voice is None:
                from jaka_agent.hardware.voice import VoiceAssistant

                self.voice = VoiceAssistant()
            text = (self.voice.transcribe_pcm16(pcm, sr=16000) or "").strip()
        if not text:
            raise hardware_audio.AudioRequestError("没有识别到有效语音", HTTPStatus.UNPROCESSABLE_ENTITY)
        return {"text": text, "duration_ms": duration_ms, "request_id": request_id}
