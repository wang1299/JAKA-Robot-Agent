# -*- coding: utf-8 -*-
"""语音助手模块: 唤醒词 + STT 下指令 + TTS 播报。

把树莓派上跑通的 voice_test.py 抽成可复用的 VoiceAssistant,供 qwen_planner 的
`voice` 子命令调用。硬件/模型均按树莓派实测钉死:
  - 麦阵: 启动时按名 AIUI 自动探测 PortAudio 索引 (8ch → 降混单声道 16k)
  - 喇叭: 启动时按名自动探测, aplay -D plughw:CARD=<id> (plughw 自动重采样; 用卡名寻址抗重编号)
  - STT : 流式 Paraformer 中英 int8  (/home/pi/voice/asr-paraformer)
  - TTS : Piper zh_CN-huayan-medium  (/home/pi/voice/vits-piper-zh_CN-huayan-medium)

并发模型: 执行主路径(qwen_planner 的 plan→导航→observe→返回)仍是单线程同步。
本模块只加两个【不碰运动、不在运动期开麦】的守护线程:
  - TTS 播报线程: 队列串行外放, 执行期 say(wait=False) 入队即返回, 不阻塞机器人。
  - stdin 读取线程: 把终端输入行入队, 实现"终端指令优先于语音"。
"""

import os
import re
import sys
import time
import glob
import wave
import queue
import tempfile
import subprocess
import threading

import numpy as np
import sounddevice as sd
import sherpa_onnx

# ---------- 硬编码(树莓派实测) ----------
BASE = "/home/pi/voice"
ASR_DIR = os.path.join(BASE, "asr-paraformer")
TTS_DIR = os.path.join(BASE, "vits-piper-zh_CN-huayan-medium")
MIC_DEV = None            # 启动时自动探测 AIUI 麦阵(见 _detect_mic_pa); 卡号/索引随热插拔会变
SPEAKER_ALSA = None       # 启动时自动探测喇叭(见 _detect_speaker_alsa); 用 CARD=<名字> 抗重编号
SPEAKER_VOLUME_PCT = 80   # 启动时把喇叭【系统音量】设到此 %(0~100)
SAMPLE_RATE = 16000

# ---------- 默认词表(可在 qwen_planner 里覆盖传参) ----------
DEFAULT_WAKE_WORDS = ("小卡", "小咔", "小卡卡")
DEFAULT_YES_WORDS = ("好", "好的", "好吧", "行", "可以", "没问题", "出发",
                     "出发吧", "走吧", "执行", "开始", "确认", "是的", "对",
                      "开始执行吧","开始执行","开始吧","go")
DEFAULT_NO_WORDS = ("取消", "算了", "不要", "不行", "停", "别", "改主意", "no")


def _find(*patterns):
    """根级优先, 退而递归找; 都没有就抛 FileNotFoundError。"""
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"找不到: {patterns}")


def _normalize(text):
    """去标点/空格/英文小写, 便于子串匹配。"""
    if not text:
        return ""
    for ch in " ,.，。！!？?、；;:：\"'""''()（）[]【】\n\r\t":
        text = text.replace(ch, "")
    return text.strip().lower()


def _match_any(text, words):
    """text 是否包含 words 中任一词(子串匹配)。"""
    norm = _normalize(text)
    if not norm:
        return False
    return any(_normalize(w) and _normalize(w) in norm for w in words)


def _detect_speaker_alsa():
    """从 aplay -l 找一个非 HDMI 的播放卡, 返回 'plughw:CARD=<id>'。
    ALSA 卡号随热插拔/枚举顺序会变(Generalplus 不一定总是 card 3), 用 CARD=<id 名字> 寻址才稳。"""
    try:
        out = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return "default"
    fallback = None
    for line in out.splitlines():
        m = re.match(r"card\s+(\d+):\s+(\S+)\s+\[", line)
        if not m:
            continue
        cid = m.group(2)                  # 卡 id, 如 "Device" / "vc4hdmi0"
        if cid.startswith("vc4hdmi"):     # HDMI 不算
            continue
        if "generalplus" in line.lower() or cid == "Device":   # 喇叭优先
            return f"plughw:CARD={cid}"
        fallback = fallback or cid
    return f"plughw:CARD={fallback}" if fallback else "default"


def _detect_mic_pa():
    """从 sounddevice 找讯飞麦阵(AIUI)的 PortAudio 索引; 找不到退回首个输入设备/None。"""
    try:
        devs = sd.query_devices()
    except Exception:
        return None
    fallback = None
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            name = (d.get("name") or "").lower()
            if "aiui" in name:            # 讯飞麦阵
                return i
            if fallback is None:
                fallback = i
    return fallback


class VoiceAssistant:
    """唤醒词 + 一句指令捕获 + TTS 播报。

    生命周期: 构造(加载模型+起线程) → 调用 say/listen_keyword/listen_utterance/confirm
    → close() 释放。
    """

    def __init__(self, asr_dir=ASR_DIR, tts_dir=TTS_DIR, mic_dev=MIC_DEV,
                 speaker_alsa=SPEAKER_ALSA, num_threads=2):
        self.mic_dev = mic_dev if mic_dev is not None else _detect_mic_pa()
        self.speaker_alsa = speaker_alsa if speaker_alsa is not None else _detect_speaker_alsa()
        print(f"[voice] 麦阵=PA设备{self.mic_dev} | 喇叭={self.speaker_alsa}")
        self._set_speaker_volume(SPEAKER_VOLUME_PCT)   # 启动即把系统音量调到设定值

        # ---- STT: 流式 Paraformer ----
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens  = _find(f"{asr_dir}/tokens.txt",        f"{asr_dir}/**/tokens.txt"),
            encoder = _find(f"{asr_dir}/encoder.int8.onnx", f"{asr_dir}/**/encoder.int8.onnx"),
            decoder = _find(f"{asr_dir}/decoder.int8.onnx", f"{asr_dir}/**/decoder.int8.onnx"),
            num_threads=num_threads, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search",
        )
        # sherpa-onnx 的在线识别器实例不是为并发 stream 设计的。网页上传语音与
        # 机器人本机麦克风可能同时触发识别，因此在模型入口统一串行化。
        self._asr_lock = threading.Lock()

        # ---- TTS: Piper (config 写法, 这版 wheel 没有 from_piper) ----
        tts_cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model    = _find(f"{tts_dir}/*.onnx",         f"{tts_dir}/**/*.onnx"),
                    tokens   = _find(f"{tts_dir}/tokens.txt",     f"{tts_dir}/**/tokens.txt"),
                    data_dir = _find(f"{tts_dir}/espeak-ng-data", f"{tts_dir}/**/espeak-ng-data"),
                ),
                num_threads=num_threads,
                debug=False,
            ),
        )
        self.tts = sherpa_onnx.OfflineTts(tts_cfg)

        # ---- TTS 播报线程(队列串行外放) ----
        self._say_queue = queue.Queue()
        self._say_thread = threading.Thread(target=self._say_worker, daemon=True)
        self._say_thread.start()

        # ---- stdin 读取线程(终端指令优先) ----
        self._kb_queue = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._stdin_thread = threading.Thread(target=self._stdin_worker, daemon=True)
        self._stdin_thread.start()

    def _set_speaker_volume(self, pct):
        """启动时把喇叭【系统音量】设到 pct%(best-effort: 失败就用当前系统音量, 不报错)。
        卡号会变 → 先按卡名查当前卡号; 控制名(PCM/Speaker/Master)因卡而异 → 自动挑一个。"""
        m = re.search(r"CARD=([^,\s]+)", self.speaker_alsa or "")
        card = m.group(1) if m else None
        if not card:
            return
        try:
            out = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=3).stdout
            num = None
            for line in out.splitlines():
                mm = re.match(r"card\s+(\d+):\s+(\S+)\s+\[", line)
                if mm and mm.group(2) == card:
                    num = mm.group(1)
                    break
            if num is None:
                return
            ctrls = subprocess.run(["amixer", "-c", num, "scontrols"],
                                   capture_output=True, text=True, timeout=3).stdout
            names = re.findall(r"control '([^']+)'", ctrls)
            pick = next((n for n in names if n in ("PCM", "Speaker", "Master", "Headphone")),
                        names[0] if names else None)
            if not pick:
                return
            subprocess.run(["amixer", "-c", num, "sset", pick, f"{int(pct)}%", "unmute"],
                           capture_output=True, timeout=3)
            print(f"[voice] 喇叭系统音量 → {pct}% (card {num}, 控制={pick})")
        except Exception as e:
            print(f"[voice] 设系统音量失败({e}), 用当前音量")

    # ---------------- 录音 / 识别 ----------------
    def _record(self, seconds, sr=SAMPLE_RATE):
        """录一段: 用 with InputStream 保证流干净开关(避免反复 sd.rec 让 ALSA/PortAudio 卡死),
        8 通道麦阵降混成单声道 float32。单声道开失败则退回 8 通道求平均。"""
        last_err = None
        for ch in (1, 8):
            chunks = []

            def _cb(indata, frames, time_info, status):
                chunks.append(indata.copy())

            try:
                with sd.InputStream(samplerate=sr, channels=ch, device=self.mic_dev,
                                    dtype="float32", callback=_cb):
                    sd.sleep(int(seconds * 1000))
            except Exception as e:
                last_err = e
                print(f"  [mic] channels={ch} 失败: {e}")
                try:
                    sd.stop()
                except Exception:
                    pass
                continue
            if not chunks:
                continue
            data = np.concatenate(chunks, axis=0)
            return data.mean(axis=1) if ch > 1 else data.flatten()
        raise RuntimeError(f"录音失败(检查麦阵): {last_err}")

    @staticmethod
    def _is_silent(samples, thr=0.012):
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) < thr

    def transcribe_samples(self, samples, sr=SAMPLE_RATE):
        """识别任意来源的单声道 float 音频，供本机麦克风和网页上传共用。"""
        samples = np.asarray(samples, dtype="float32").reshape(-1)
        if samples.size == 0 or self._is_silent(samples):
            return ""
        with self._asr_lock:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sr, samples)
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            return self.recognizer.get_result(stream).strip()

    def transcribe_pcm16(self, pcm_bytes, sr=SAMPLE_RATE):
        """识别 little-endian signed 16-bit PCM 字节。"""
        if not pcm_bytes or len(pcm_bytes) % 2:
            return ""
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype("float32") / 32768.0
        return self.transcribe_samples(samples, sr)

    def _transcribe(self, samples, sr=SAMPLE_RATE):
        """兼容原有内部调用。"""
        return self.transcribe_samples(samples, sr)

    def _transcribe_window(self, seconds=2.0, sr=SAMPLE_RATE):
        return self._transcribe(self._record(seconds, sr), sr)

    # ---------------- TTS ----------------
    def _play_sync(self, text):
        """合成并外放一句(阻塞直到播完)——复用 voice_test.py 跑通的路径。"""
        if not text:
            return
        out = self.tts.generate(text, sid=0, speed=1.0)
        pcm = (np.asarray(out.samples, "float32") * 32767).astype("<i2").tobytes()
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(out.sample_rate)
                w.writeframes(pcm)
            subprocess.run(["aplay", "-D", self.speaker_alsa, "-q", path], check=True)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _say_worker(self):
        """串行消费播报队列。"""
        while True:
            text, done = self._say_queue.get()
            if text is None:           # 关停哨兵
                if done:
                    done.set()
                break
            if text:                   # 空串 = drain 标记, 跳过合成
                try:
                    self._play_sync(text)
                except Exception as e:
                    print(f"[TTS 失败] {e}")
            if done:
                done.set()

    def say(self, text, wait=True):
        """播报一句。wait=True 阻塞到播完(IDLE/CONFIRM 前); wait=False 入队即返回(执行期)。"""
        done = threading.Event() if wait else None
        self._say_queue.put((text, done))
        if wait and done:
            done.wait()

    def drain_speaker(self, timeout=15):
        """等队列里已有的句子全部播完(回 IDLE 前用)。"""
        done = threading.Event()
        self._say_queue.put(("", done))   # 空串: worker 跳过合成, 只置位
        done.wait(timeout=timeout)

    # ---------------- 终端输入 ----------------
    def _stdin_worker(self):
        while self._running.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":              # EOF
                break
            self._kb_queue.put(line.rstrip("\n"))   # 裸回车也入队(confirm 里当 yes)

    def pop_terminal(self):
        """取一条终端输入(无则 None)。"""
        try:
            return self._kb_queue.get_nowait()
        except queue.Empty:
            return None

    def has_terminal(self):
        return not self._kb_queue.empty()

    # ---------------- 监听 ----------------
    def listen_keyword(self, keywords=DEFAULT_WAKE_WORDS, window=2.0,
                       max_norm_len=6, idle_timeout=None):
        """循环短窗监听唤醒词。命中返回 True; 检测到终端输入返回 False(让主循环优先处理)。

        唤醒判定: 非静音 → 转录 → 去标点后【片段够短】(≤max_norm_len)且含关键词才算唤醒。
        "够短"是为了排除长句闲聊里碰巧含"小卡"(如"那个小卡车的轮子")。唤醒就是唤醒,
        指令由后续 listen_utterance 单独听一句 —— 不再把唤醒词后面的尾巴当指令(那会把
        犹豫/环境音当指令, 极易出错)。
        """
        print('[待机] 听唤醒词「小卡」…(或直接在终端敲指令)')
        start = time.time()
        while self._running.is_set():
            if self.has_terminal():           # 终端优先 → 让位
                return False
            if idle_timeout is not None and time.time() - start > idle_timeout:
                return False
            try:
                samples = self._record(window)
            except RuntimeError as e:
                print(f"[待机] {e} — 1 秒后重试")
                time.sleep(1.0)               # 录音失败不卡死: 退一步再试
                continue
            if self._is_silent(samples):      # 静音跳过, 省 CPU + 不误识环境音
                continue
            text = self._transcribe(samples)
            norm = _normalize(text)
            if not norm or len(norm) > max_norm_len:
                continue                      # 太长 = 闲聊/噪声, 不是单纯唤醒
            print(f"  [听到] {text!r}")
            for kw in keywords:
                k = _normalize(kw)
                if k and k in norm:
                    return True
        return False

    def listen_utterance(self, max_seconds=6.0, chunk=1.5):
        """唤醒后捕获一句指令: 分块录, 【说到停就收】(检测到说完后的一个静音块即截断),
        最多 max_seconds。比固定窗口响应更快、也不易把整段环境音吞进去。全程可被终端插队。"""
        if self.has_terminal():               # 终端优先
            return ""
        print("[听指令] 请说…")
        frames = []
        total, saw_speech, sil_after = 0.0, False, 0
        while total < max_seconds and self._running.is_set():
            if self.has_terminal():           # 说一半也能被终端打断
                return ""
            try:
                seg = self._record(chunk)
            except RuntimeError as e:
                print(f"  [听指令] {e}")
                break
            frames.append(seg)
            total += chunk
            if self._is_silent(seg):
                if saw_speech:
                    sil_after += 1
                    if sil_after >= 1:        # 说完一句后的停顿 → 收尾
                        break
            else:
                saw_speech = True
                sil_after = 0
        if not frames or not saw_speech:      # 全程静音 = 没说
            return ""
        text = self._transcribe(np.concatenate(frames))
        print(f"  [指令] {text!r}")
        return text

    def confirm(self, prompt, yes_words=DEFAULT_YES_WORDS, no_words=DEFAULT_NO_WORDS,
                timeout=8.0, window=1.5):
        """语音/终端二选一确认, 终端优先; 超时按"否"处理(不动车)。"""
        self.say(prompt, wait=True)
        time.sleep(0.2)                       # 让喇叭余音散掉, 免得灌进麦
        start = time.time()
        while time.time() - start < timeout:
            kb = self.pop_terminal()
            if kb is not None:                # 终端优先
                if kb.strip() == "":          # 裸回车 = 确认
                    return True
                if _match_any(kb, no_words):
                    return False
                return True                   # 其它非"否"的输入 → 确认
            try:
                text = self._transcribe_window(seconds=window)
            except RuntimeError:
                continue
            if _match_any(text, yes_words):
                return True
            if _match_any(text, no_words):
                return False
        return False

    # ---------------- 关闭 ----------------
    def close(self):
        self._running.clear()
        try:
            self._say_queue.put((None, None))     # 停 TTS 线程
            self._say_thread.join(timeout=2)
        except Exception:
            pass
        # stdin 线程 readline 阻塞中, daemon 随进程退出而结束


# ---------------- 自测 ----------------
if __name__ == "__main__":
    va = VoiceAssistant()
    try:
        va.say("语音模块就绪, 现在测试。说小卡唤醒。", wait=True)
        if va.listen_keyword(DEFAULT_WAKE_WORDS):
            va.say("我在, 请说一句话", wait=True)
            text = va.listen_utterance()
            va.say("你说的是, " + (text or "没有听到"), wait=True)
        else:
            va.say("没有听到唤醒词", wait=True)
    finally:
        va.close()
