#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAKA 视觉助手 Web 界面 —— 服务端入口 + 对话历史 + 文字/语音输入
==================================================================
设计目标:
  ① 轻量部署: Python 后端与 robot_web_page.html 前端分离, 不引入 Flask。
  ② 智能路由: 现场视觉问题才拍照; 能力/用法/一般对话直接返回文字。
     视觉问题采用两阶段回答: 先拍照并立刻返回图片 → 再调用 Qwen-VL 返回文字。
  ③ 对话历史: 会话和消息保存在浏览器 localStorage, 树莓派不额外维护数据库。
  ④ 语音输入: 手机录音上传或树莓派本机麦克风 → Sherpa-ONNX 离线识别 → 回填输入框。
  ⑤ 硬件懒加载: 启动网页不占用相机/麦克风; 真正请求时才加载并在结束后释放。

树莓派运行:
  cd ~/jaka-robot
  source ~/jaka/bin/activate
  python robot_web.py --host 0.0.0.0 --port 8080
  浏览器访问 http://<树莓派IP>:8080

本地界面测试(不需要 Orbbec / DashScope / sherpa_onnx):
  python robot_web.py --mock --port 8081

安全说明:
  本服务默认无登录, 仅应部署在可信局域网或 Tailscale 网络内, 不要直接暴露到公网。
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextvars import copy_context
from types import SimpleNamespace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from robot_agent import AgentRunner, Tool, ToolInputError, clean_history
from robot_agent_map import MapEvidence, FILTER_SCHEMA
from robot_skills import prepare_skill, skill_catalog, skill_summary, skill_tools, skill_resources
from robot_runtime import (TaskCancelled, TaskDriver, SharedMappingCamera,
                           cancellation_scope, check_cancelled, guarded_call)

from robot_web_routing import (
    FIND_OBJECT_WORDS,
    FIND_TARGET_COT_MAX_TOKENS,
    FIND_TARGET_COT_SYSTEM,
    FIND_TARGET_COT_USER,
    ROUTER_PROMPT,
    _fallback_route_mode,
    _find_object_candidates,
    _find_target_label_from_instruction,
    _has_find_object_intent,
    _history_lines,
    _is_reference_find_request,
    _json_text,
    _map_query,
    _mock_route_mode,
    _parse_find_object_result,
    _parse_model_json_object,
    _parse_patrol_result,
    _project_metric_query,
    _parse_route_decision,
)


# ======================================================================
#  配置
# ======================================================================
HERE = Path(__file__).resolve().parent
CAPTURE_DIR = HERE / "web_captures"
VIDEO_DIR = HERE / "web_videos"
MAP_ICON_DIR = HERE / "map_icons"
SLAM_CONFIG_PATH = HERE / "slam_calibration.json"
TRACK_HISTORY_PATH = HERE / "robot_tracks.json"
MAPPING_RUNS_DIR = HERE / "mapping_runs"
MAPPING_SERVER = os.getenv("JAKA_MAPPING_SERVER", "tcp://127.0.0.1:5560").strip()
MAPPING_RESULT_TIMEOUT = float(os.getenv("JAKA_MAPPING_RESULT_TIMEOUT", "1800"))
# 当前观察/录像统一使用头部相机。DG 是头部相机；VF 备注为手部相机。
HEAD_CAMERA_SN = os.getenv("JAKA_HEAD_CAM_SN", "AY8V74300DG").strip()
HAND_CAMERA_SN = os.getenv("JAKA_HAND_CAM_SN", "AY8V74300VF").strip()
MAX_BODY_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_BYTES = 6 * 1024 * 1024
MAX_AUDIO_BYTES = 8 * 1024 * 1024
MAX_AUDIO_SECONDS = 30
TASK_VIDEO_ENABLED = os.getenv("JAKA_RECORD_TASK_VIDEO", "1").strip().lower() not in {"0", "false", "no", "off"}
TASK_VIDEO_FPS = max(0.5, min(15.0, float(os.getenv("JAKA_TASK_VIDEO_FPS", "5"))))
TASK_VIDEO_MAX_WIDTH = max(320, int(os.getenv("JAKA_TASK_VIDEO_MAX_WIDTH", "960")))
FIND_SNAPSHOT_INTERVAL_SECONDS = max(0.5, float(os.getenv("JAKA_FIND_SNAPSHOT_INTERVAL_SECONDS", "1")))
FIND_SNAPSHOT_ATTEMPTS_PER_POINT = max(1, int(os.getenv("JAKA_FIND_SNAPSHOT_ATTEMPTS_PER_POINT", "2")))
FIND_SNAPSHOT_MAX_ERRORS = max(1, int(os.getenv("JAKA_FIND_SNAPSHOT_MAX_ERRORS", "3")))
# 行进中寻物采用单帧高置信触发；保留任务字段值 1 供前端/API 展示。
FIND_SNAPSHOT_CONFIRM_FRAMES = 1
WELCOME_SNAPSHOT_INTERVAL_SECONDS = max(
    0.5, float(os.getenv("JAKA_WELCOME_SNAPSHOT_INTERVAL_SECONDS", "1"))
)
WELCOME_QUESTION = os.getenv("JAKA_WELCOME_QUESTION", "你是小卡的朋友吗").strip() or "你是小卡的朋友吗"
WELCOME_PERSON_SYSTEM = (
    "你是迎宾机器人的严格人物外观比对器。图片1是目标人物参考图，图片2是现场广角图。"
    "采用瘦身CoT：先定位图片2中最可能的一个候选人物，再分别核对衣着、头脸与发型、配饰、体型。"
    "不要根据地点、性别、年龄或常见衣服颜色猜测身份。人物太小、模糊或遮挡时，必须明确写无法核对；"
    "普通黑色上衣、短发、戴眼镜等常见单项相似不能单独证明是同一人。"
    "本阶段只输出简短的观察与比对摘要，不输出JSON、found或confidence。"
)
WELCOME_PERSON_USER = (
    "请按“候选定位、衣着、头脸发型、配饰、体型、明确冲突”六项做简短比对。"
    "每个外观项目都要分别说明参考图和现场候选实际看到了什么；看不清就写无法核对。"
)
WELCOME_PERSON_JSON_SYSTEM = (
    "你是人物图像比对结果格式化器。把上一阶段的观察摘要转换成唯一一个合法JSON对象。"
    "只能使用摘要中明确写出的可见证据，不得补充、猜测或改变结论。"
    "match表示两边可见细节一致，mismatch表示有明确冲突，unknown表示任一边无法核对。"
    "只输出JSON，不要分析、Markdown、代码块、前后缀或多个JSON。"
)
WELCOME_PERSON_JSON_SCHEMA = (
    '{"candidate_visible":true,"candidate_region":"候选位置或无",'
    '"reference_upper_clothing":"参考图衣着",'
    '"candidate_upper_clothing":"现场衣着","upper_clothing":"match|mismatch|unknown",'
    '"reference_face_hair":"参考图头脸发型",'
    '"candidate_face_hair":"现场头脸发型","face_hair":"match|mismatch|unknown",'
    '"reference_accessories":"参考图配饰",'
    '"candidate_accessories":"现场配饰","accessories":"match|mismatch|unknown",'
    '"reference_body_shape":"参考图体型",'
    '"candidate_body_shape":"现场体型","body_shape":"match|mismatch|unknown",'
    '"contradictions":["明确冲突，没有则为空数组"],"reason":"简短依据"}'
)
WELCOME_AFFIRMATIVE_WORDS = (
    "是的", "是", "对的", "对", "没错", "我是", "我就是", "就是", "嗯",
    "好的", "好", "可以", "没问题",
)
WELCOME_NEGATIVE_WORDS = (
    "不是", "不对", "没有", "认错", "找错", "错了", "不认识", "不行", "不好",
    "否", "不是我", "我不是",
)
CAPTURE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SLAM_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
REFERENCE_IMAGE_TYPES = {
    ".png": ("image/png", b"\x89PNG\r\n\x1a\n"),
    ".jpg": ("image/jpeg", b"\xff\xd8"),
    ".jpeg": ("image/jpeg", b"\xff\xd8"),
    ".webp": ("image/webp", b"RIFF"),
}
CATEGORY_ZH_MAP = {
    "black flight case": "黑色航空箱",
    "blue chair": "蓝色椅子",
    "ceiling spotlights": "天花板射灯",
    "glass coffee table": "玻璃茶几",
    "glass_table": "玻璃桌",
    "guest drop-off point": "送客点",
    "guest pickup point": "接客点",
    "leather armchair": "皮质扶手椅",
    "ordinary office door": "普通办公室门",
    "door_panel": "门板",
    "paper wall poster": "纸质墙面海报",
    "potted plant": "盆栽",
    "shrub or bush": "灌木",
    "silver elevator door": "银色电梯门",
    "white utility table": "白色工作台",
    "window": "窗户",
    "wooden storage cabinet": "木质储物柜",
    "wooden study desk": "木质书桌",
    "chair": "椅子",
    "coffee table": "茶几",
    "desk": "办公桌",
    "dining table": "餐桌",
    "elevator display": "电梯显示屏",
    "plant": "植物",
    "sofa": "沙发",
}
MAP_ICON_NAMES = {
    "chair", "plant", "shrub", "case", "table", "coffeetable", "desk", "cabinet",
    "door", "window", "poster", "elevator", "light", "object", "pickup", "dropoff",
}
AUDIO_CONTENT_TYPES = {
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


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
        "-t", str(MAX_AUDIO_SECONDS + 1),
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
    if duration_ms > MAX_AUDIO_SECONDS * 1000:
        raise AudioRequestError(
            f"录音时长不能超过 {MAX_AUDIO_SECONDS} 秒",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )
    return pcm, duration_ms
DEFAULT_SLAM_IMAGE_URL = "http://192.168.10.10:8809/maps/demo/3/map.png"
SLAM_IMAGE_URL = os.getenv("JAKA_SLAM_IMAGE_URL", DEFAULT_SLAM_IMAGE_URL).strip()
SLAM_YAML_URL = os.getenv("JAKA_SLAM_YAML_URL", urljoin(SLAM_IMAGE_URL, "map.yaml")).strip()
SLAM_FETCH_TIMEOUT = float(os.getenv("JAKA_SLAM_FETCH_TIMEOUT", "5"))

# 后端日志同时输出到终端和文件；网页端只显示面向用户的简短错误。
LOG_PATH = HERE / "robot_web.log"
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


def _vision_completion(client, **kwargs):
    """调用视觉模型；结构化输出参数被服务端拒绝时自动无该参数重试一次。"""
    response_format = kwargs.pop("response_format", {"type": "json_object"})
    try:
        return client.chat.completions.create(response_format=response_format, **kwargs)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
        if status != 500:
            raise
        LOGGER.warning("[vision] structured request returned HTTP 500; retrying without response_format")
        return client.chat.completions.create(**kwargs)

def _clean_old_captures(max_age_days=14):
    """启动时清理过旧图片(best-effort), 避免树莓派磁盘无限增长。"""
    if not CAPTURE_DIR.exists():
        return
    cutoff = time.time() - max_age_days * 86400
    patterns = {"*.jpg", *(f"*{suffix}" for suffix in REFERENCE_IMAGE_TYPES)}
    for pattern in patterns:
        for path in CAPTURE_DIR.glob(pattern):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass


def _image_size(path: Path) -> tuple[int, int]:
    """只用标准库读取 PNG/JPEG 尺寸，避免树莓派额外安装 Pillow。"""
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if suffix in (".jpg", ".jpeg") and data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            if marker in (0xD8, 0xD9):
                continue
            if index + 2 > len(data):
                break
            length = struct.unpack(">H", data[index:index + 2])[0]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if index + 7 > len(data):
                    break
                height, width = struct.unpack(">HH", data[index + 3:index + 7])
                return int(width), int(height)
            index += max(2, length)
    raise ValueError("SLAM 图片必须是有效的 PNG 或 JPEG")


def _parse_slam_yaml(text: str) -> dict:
    """解析底盘 map.yaml 的简单键值；仅提取地图配准必需字段。"""
    values = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    try:
        resolution = float(values["resolution"])
        origin = json.loads(values["origin"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("map.yaml 缺少有效的 resolution/origin") from exc
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("map.yaml resolution 必须是正数")
    if not isinstance(origin, list) or len(origin) < 3:
        raise ValueError("map.yaml origin 必须是 [x,y,yaw]")
    try:
        origin = [float(origin[0]), float(origin[1]), float(origin[2])]
    except (TypeError, ValueError) as exc:
        raise ValueError("map.yaml origin 含非数值") from exc
    if not all(math.isfinite(value) for value in origin):
        raise ValueError("map.yaml origin 必须是有限数值")
    return {
        "resolution": resolution,
        "origin": origin,
        "image": values.get("image", "map.png").strip("'\""),
        "map_type": values.get("map_type", ""),
        "create_time": values.get("create_time"),
        "modify_time": values.get("modify_time"),
    }


# ======================================================================
#  拟物体地图: 兼容 planner 格式和 ZMQ geometry 格式
# ======================================================================
def _normalized_ann_id(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _normalize_graph(data: dict, name="scene_graph.json") -> dict:
    """任意场景图 → planner 可用结构; 无二维世界坐标的对象不进入导航地图。"""
    raw = data.get("objects", [])
    objects = list(raw.values()) if isinstance(raw, dict) else list(raw or [])
    normalized = []
    for source in objects:
        if not isinstance(source, dict):
            continue
        obj = copy.deepcopy(source)
        geometry = obj.get("geometry") or {}
        center = obj.get("floor_xy") or geometry.get("center_world")
        if not isinstance(center, list) or len(center) < 2:
            continue
        try:
            x, y = float(center[0]), float(center[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        obj["ann_id"] = _normalized_ann_id(obj.get("ann_id"))
        obj["category"] = obj.get("category") or "Unknown"
        category_key = re.sub(r"\s+", " ", str(obj["category"]).strip().lower())
        obj["category_zh"] = obj.get("category_zh") or CATEGORY_ZH_MAP.get(category_key, obj["category"])
        obj["func_desc"] = obj.get("func_desc") or ""
        obj["position"] = obj.get("position") or f"世界系({x:.1f},{y:.1f})"
        obj["floor_xy"] = [x, y]
        obj.setdefault("viewpoint", None)
        obj.setdefault("marker", None)
        if not obj.get("box3d"):
            center3 = geometry.get("center_world") or [x, y, 0.0]
            half = geometry.get("aabb_half_sizes_world") or [0.25, 0.25, 0.25]
            if len(center3) >= 3 and len(half) >= 3:
                obj["box3d"] = {
                    "center": [float(center3[0]), float(center3[1]), float(center3[2])],
                    "size": [float(half[0]) * 2, float(half[1]) * 2, float(half[2]) * 2],
                    "yaw": 0.0,
                }
        normalized.append(obj)
    if not normalized:
        raise ValueError(f"{name} 中没有带 floor_xy/center_world 的有效物体")
    return {
        "name": name,
        "frame": data.get("frame") or data.get("video_id") or "robot_map",
        "objects": normalized,
        "func_relationships": data.get("func_relationships") or (data.get("relationships") or {}).get("functional", []),
        "pos_relationships": data.get("pos_relationships") or (data.get("relationships") or {}).get("positional", []),
    }


def _step_target_ids(step: dict) -> list:
    if step.get("type") in ("cruise", "patrol"):
        return list(step.get("target_ann_ids") or [])
    target = step.get("target_ann_id")
    return [] if target is None else [target]


def _fallback_clarification(instruction: str, plan: dict, objects: list):
    """模型漏问时兜底: 同名的单目标导航必须先让用户选定 ann_id。"""
    if plan.get("clarification"):
        return plan["clarification"]
    object_by_id = {obj["ann_id"]: obj for obj in objects}
    explicit_ids = {
        int(value)
        for value in re.findall(r"(?:ann_id\s*[=:：]\s*|#|编号\s*)(\d+)", instruction, re.I)
    }
    for step in plan.get("steps", []):
        if step.get("type") not in ("navigate", "observe"):
            continue
        chosen = object_by_id.get(step.get("target_ann_id"))
        if not chosen:
            continue
        category_zh = str(chosen.get("category_zh") or "").strip().lower()
        category = str(chosen.get("category") or "").strip().lower()
        siblings = [
            obj for obj in objects
            if (category_zh and str(obj.get("category_zh") or "").strip().lower() == category_zh)
            or (category and str(obj.get("category") or "").strip().lower() == category)
        ]
        if len(siblings) < 2 or explicit_ids.intersection(obj["ann_id"] for obj in siblings):
            continue
        aliases = [value for value in (category_zh, category) if value]
        if not any(alias in instruction.lower() for alias in aliases):
            continue
        described = []
        for obj in siblings:
            fields = [str(obj.get("position") or ""), str(obj.get("func_desc") or "")]
            if obj.get("marker") is not None:
                fields.append(f"marker={obj['marker']}")
            if any(len(field) >= 2 and field in instruction for field in fields):
                described.append(obj["ann_id"])
        if len(set(described)) == 1:
            continue
        name = chosen.get("category_zh") or chosen.get("category") or "目标"
        return {
            "question": f"地图中有 {len(siblings)} 个{name}，请选择要前往哪一个。",
            "candidate_ann_ids": [obj["ann_id"] for obj in siblings],
        }
    return None


OBSERVATION_INTENT_WORDS = (
    "观察", "查看", "看一下", "看一看", "检查", "拍照", "现场情况", "现场环境", "汇报",
)


def _ensure_explicit_observation_steps(plan: dict, instruction: str) -> bool:
    """把模型遗漏的到点观察补成显式步骤，保证任务卡片与真实执行一致。"""
    if not any(word in str(instruction) for word in OBSERVATION_INTENT_WORDS):
        return False
    steps = list(plan.get("steps") or [])
    navigate_indexes = [index for index, step in enumerate(steps) if step.get("type") == "navigate"]
    if not navigate_indexes or any(step.get("type") == "observe" for step in steps):
        return False

    observe_all = any(word in str(instruction) for word in ("分别观察", "逐个观察", "依次观察", "每个"))
    target_indexes = set(navigate_indexes if observe_all else navigate_indexes[-1:])
    repaired = []
    for index, step in enumerate(steps):
        repaired.append(step)
        if index not in target_indexes:
            continue
        ann_id = step.get("target_ann_id")
        repaired.append({
            "type": "observe",
            "target_ann_id": ann_id,
            "focus": "目标当前状态及周围环境",
            "question": str(instruction),
            "fine_adjust": False,
            "capture_only": True,
        })
    plan["steps"] = repaired
    return True


PERSON_MATCH_WEIGHTS = {
    "upper_clothing": 0.40,
    "face_hair": 0.30,
    "accessories": 0.10,
    "body_shape": 0.20,
}


def _parse_person_reference_result(raw: str) -> dict:
    """从逐项外观比对计算人物置信度；不信任模型自报的 found/confidence。"""
    text = _json_text(raw).strip()
    value = None
    candidates = [text]
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])
    for candidate in candidates:
        repaired = re.sub(r'"\s*:\s*=\s*', '":', candidate)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        try:
            parsed = json.loads(repaired)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    if value is None:
        value = _parse_model_json_object(text)
    fallback = {
        "found": False,
        "candidate_visible": False,
        "candidate_region": "",
        "confidence": "low",
        "score": 0.0,
        "reason": "模型未返回完整的人物逐项比对，不能确认目标人物。",
        "comparisons": {},
        "contradictions": [],
        "schema_valid": False,
    }
    if not isinstance(value, dict):
        return fallback

    candidate_visible = value.get("candidate_visible")
    if isinstance(candidate_visible, str):
        normalized_visible = candidate_visible.strip().lower()
        if normalized_visible in {"true", "1", "yes", "是", "可见", "存在"}:
            candidate_visible = True
        elif normalized_visible in {"false", "0", "no", "否", "不可见", "不存在"}:
            candidate_visible = False
    region = str(value.get("candidate_region") or "").strip()[:300]
    reference_features = value.get("reference_features")
    candidate_features = value.get("candidate_features")
    comparisons = value.get("comparisons")
    contradictions_raw = value.get("contradictions")
    if not isinstance(reference_features, dict):
        reference_features = {
            key: value.get(f"reference_{key}")
            for key in PERSON_MATCH_WEIGHTS
        }
    if not isinstance(candidate_features, dict):
        candidate_features = {
            key: value.get(f"candidate_{key}")
            for key in PERSON_MATCH_WEIGHTS
        }
    if not isinstance(comparisons, dict):
        comparisons = {
            key: value.get(key, value.get(f"{key}_comparison"))
            for key in PERSON_MATCH_WEIGHTS
        }
    if isinstance(contradictions_raw, str):
        contradiction_text = contradictions_raw.strip()
        contradictions_raw = (
            []
            if not contradiction_text
            or contradiction_text.lower() in {"无", "没有", "none", "no"}
            else [contradiction_text]
        )
    if (
        not isinstance(candidate_visible, bool)
        or not isinstance(reference_features, dict)
        or not isinstance(candidate_features, dict)
        or not isinstance(comparisons, dict)
        or not isinstance(contradictions_raw, list)
    ):
        fallback["candidate_visible"] = candidate_visible is True
        fallback["candidate_region"] = region
        return fallback

    state_aliases = {
        "match": "match",
        "匹配": "match",
        "一致": "match",
        "same": "match",
        "mismatch": "mismatch",
        "不匹配": "mismatch",
        "冲突": "mismatch",
        "different": "mismatch",
        "unknown": "unknown",
        "未知": "unknown",
        "不可见": "unknown",
        "无法判断": "unknown",
        "unclear": "unknown",
    }
    normalized = {}
    descriptions_complete = True
    unknown_markers = (
        "无法核对", "无法看到", "无法判断", "看不清", "不可见", "不清楚", "未说明",
        "unknown", "unclear", "not visible",
    )
    mismatch_markers = ("不一致", "明显不同", "存在冲突", "mismatch")
    for key in PERSON_MATCH_WEIGHTS:
        raw_state = str(comparisons.get(key) or "").strip().lower()
        normalized[key] = state_aliases.get(raw_state, "")
        reference_description = str(reference_features.get(key) or "").strip()
        candidate_description = str(candidate_features.get(key) or "").strip()
        combined_description = (
            f"{reference_description} {candidate_description}"
        ).lower()
        if any(marker in combined_description for marker in unknown_markers):
            normalized[key] = "unknown"
        elif any(marker in combined_description for marker in mismatch_markers):
            normalized[key] = "mismatch"
        if not normalized[key]:
            descriptions_complete = False
        if not reference_description:
            descriptions_complete = False
        if not candidate_description:
            descriptions_complete = False

    if not descriptions_complete:
        fallback["candidate_visible"] = candidate_visible
        fallback["candidate_region"] = region
        fallback["comparisons"] = normalized
        return fallback

    score = sum(
        PERSON_MATCH_WEIGHTS[key]
        for key, state in normalized.items()
        if state == "match"
    )
    observable_weight = sum(
        PERSON_MATCH_WEIGHTS[key]
        for key, state in normalized.items()
        if state != "unknown"
    )
    mismatch_weight = sum(
        PERSON_MATCH_WEIGHTS[key]
        for key, state in normalized.items()
        if state == "mismatch"
    )
    contradictions = [
        str(item).strip()[:160]
        for item in contradictions_raw[:8]
        if str(item).strip()
    ]

    found = bool(
        candidate_visible
        and score >= 0.80
        and observable_weight >= 0.80
        and mismatch_weight == 0.0
        and not contradictions
    )
    if found:
        confidence = "high"
    elif (
        candidate_visible
        and score >= 0.55
        and mismatch_weight <= 0.20
        and not contradictions
    ):
        confidence = "medium"
    else:
        confidence = "low"

    reason = str(value.get("reason") or "").strip()[:500]
    if not reason:
        reason = (
            f"程序计分 {score:.2f}，可核对权重 {observable_weight:.2f}，"
            f"冲突权重 {mismatch_weight:.2f}。"
        )
    return {
        "found": found,
        "candidate_visible": candidate_visible,
        "candidate_region": region,
        "confidence": confidence,
        "score": round(score, 2),
        "observable_weight": round(observable_weight, 2),
        "mismatch_weight": round(mismatch_weight, 2),
        "reason": reason,
        "reference_features": reference_features,
        "candidate_features": candidate_features,
        "comparisons": normalized,
        "contradictions": contradictions,
        "schema_valid": True,
    }


# ======================================================================
#  机器人任务状态机: Qwen 规划 → 人工确认 → 后台确定性执行
# ======================================================================
class TaskVideoRecorder:
    """任务旁路录像器：失败只记日志，不影响导航、观察和寻物主流程。"""

    def __init__(self, task_id: str, driver, fps: float = TASK_VIDEO_FPS, max_width: int = TASK_VIDEO_MAX_WIDTH):
        self.task_id = task_id
        self.driver = driver
        self.fps = float(fps)
        self.max_width = int(max_width)
        self.video_id = uuid.uuid4().hex
        self.path = VIDEO_DIR / f"{self.video_id}.mp4"
        self.url = f"/videos/{self.video_id}.mp4"
        self.started_at = None
        self.finished_at = None
        self.frame_count = 0
        self.error = None
        self._stop = threading.Event()
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"TaskVideo-{self.video_id[:8]}")
        self._thread.start()

    def request_stop(self):
        self._stop.set()
        self._pause_requested.clear()

    def stop(self, timeout=None):
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def pause(self, timeout=5.0):
        """等待录像线程退出相机抓帧区，供到点转向和正式拍照独占相机。"""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        if not self._pause_requested.is_set():
            self._paused.clear()
        self._pause_requested.set()
        return self._paused.wait(timeout=max(0.1, float(timeout)))

    def resume(self, timeout=2.0):
        self._pause_requested.clear()
        deadline = time.monotonic() + max(0.1, float(timeout))
        while self._paused.is_set() and self._thread is not None and self._thread.is_alive():
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)

    def snapshot(self):
        return {
            "id": self.video_id,
            "url": self.url,
            "path": str(self.path),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "frame_count": self.frame_count,
            "error": self.error,
        }

    def _run(self):
        writer = None
        try:
            import cv2

            interval = 1.0 / max(0.5, self.fps)
            while not self._stop.is_set():
                if self._pause_requested.is_set():
                    self._paused.set()
                    while self._pause_requested.is_set() and not self._stop.is_set():
                        self._stop.wait(0.05)
                    self._paused.clear()
                    continue
                started = time.time()
                if not hasattr(self.driver, "grab_color_frame"):
                    self.error = "driver has no grab_color_frame"
                    return
                frame = self.driver.grab_color_frame()
                if self._stop.is_set():
                    break
                if frame is None:
                    time.sleep(interval)
                    continue
                height, width = frame.shape[:2]
                if width > self.max_width:
                    scale = self.max_width / float(width)
                    width = self.max_width
                    height = max(1, round(height * scale))
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
                    if not writer.isOpened():
                        self.error = "cv2.VideoWriter open failed"
                        return
                    LOGGER.info("[video] recording started task_id=%r path=%s fps=%.1f size=%dx%d",
                                self.task_id, self.path, self.fps, width, height)
                writer.write(frame)
                self.frame_count += 1
                delay = interval - (time.time() - started)
                if delay > 0:
                    self._stop.wait(delay)
        except Exception as exc:
            if not isinstance(exc, TaskCancelled):
                self.error = str(exc)
                _log_exception("[video] recording failed", exc, include_traceback=True, task_id=self.task_id)
        finally:
            self.finished_at = time.time()
            self._paused.set()
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            LOGGER.info("[video] recording stopped task_id=%r path=%s frames=%d error=%r",
                        self.task_id, self.path, self.frame_count, self.error)


class TaskSafetyError(RuntimeError):
    """An expected execution blocker, not an internal HTTP server failure."""


def _estop_flag(value):
    """Normalize documented boolean encodings; missing/invalid is not safe."""
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "0", "1"):
        return value.strip().lower() in ("true", "1")
    return None


class RobotTaskManager:
    """维护唯一活动任务; 所有可变状态通过 lock 快照给网页轮询。"""

    def __init__(self, web_state):
        self.web = web_state
        self.lock = threading.RLock()
        self.task = None
        self.thread = None
        self.driver = None
        self.hardware_driver = None
        self._capture_workers = []
        self.video_recorder = None
        self.mock_pose = [0.0, 0.0, 0.0]
        self.track = []
        self.track_started_at = time.time()
        self.saved_tracks = self._load_saved_tracks()

    def _load_saved_tracks(self):
        try:
            value = json.loads(TRACK_HISTORY_PATH.read_text(encoding="utf-8"))
            tracks = value.get("tracks", []) if isinstance(value, dict) else value
            if not isinstance(tracks, list):
                return []
            return [track for track in tracks if isinstance(track, dict) and isinstance(track.get("points"), list)][:20]
        except FileNotFoundError:
            return []
        except Exception as exc:
            print(f"[track] 忽略无效轨迹文件: {exc}")
            return []

    def _save_tracks_file(self):
        temp = TRACK_HISTORY_PATH.with_suffix(".tmp")
        temp.write_text(
            json.dumps({"tracks": self.saved_tracks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(TRACK_HISTORY_PATH)

    @staticmethod
    def _clean_track(points):
        cleaned = []
        for point in points:
            try:
                current = [float(point[0]), float(point[1]), float(point[2] if len(point) > 2 else 0.0)]
            except (TypeError, ValueError, IndexError):
                continue
            if not all(math.isfinite(value) for value in current):
                continue
            if cleaned:
                distance = math.hypot(current[0] - cleaned[-1][0], current[1] - cleaned[-1][1])
                angle = abs((current[2] - cleaned[-1][2] + math.pi) % (2 * math.pi) - math.pi)
                if distance < 0.02 and angle < math.radians(3):
                    continue
            cleaned.append(current)
        return cleaned

    def tracks_snapshot(self):
        with self.lock:
            return {
                "tracks": copy.deepcopy(self.saved_tracks),
                "current_points": len(self._clean_track(self.track)),
            }

    def save_current_track(self, name=""):
        with self.lock:
            points = self._clean_track(self.track)
            if len(points) < 2:
                raise ValueError("当前轨迹有效点不足，机器人移动后再保存")
            distance = sum(
                math.hypot(right[0] - left[0], right[1] - left[1])
                for left, right in zip(points, points[1:])
            )
            created_at = time.time()
            item = {
                "id": uuid.uuid4().hex,
                "name": str(name or time.strftime("轨迹 %m-%d %H:%M", time.localtime(created_at)))[:40],
                "created_at": created_at,
                "map_name": self.web.graph_snapshot().get("name"),
                "point_count": len(points),
                "distance_m": round(distance, 3),
                "points": points,
            }
            self.saved_tracks.insert(0, item)
            self.saved_tracks = self.saved_tracks[:20]
            self._save_tracks_file()
            self.track = []
            self.track_started_at = time.time()
            return copy.deepcopy(item)

    def delete_saved_track(self, track_id):
        with self.lock:
            before = len(self.saved_tracks)
            self.saved_tracks = [track for track in self.saved_tracks if str(track.get("id")) != str(track_id)]
            if len(self.saved_tracks) == before:
                raise ValueError("轨迹不存在")
            self._save_tracks_file()
            return self.tracks_snapshot()

    def clear_current_track(self):
        with self.lock:
            self.track = []
            self.track_started_at = time.time()
            return self.tracks_snapshot()

    def snapshot(self):
        with self.lock:
            task = copy.deepcopy(self.task)
            recorder = self.video_recorder
        if task is not None and recorder is not None:
            task["video"] = recorder.snapshot()
        return task

    def _update(self, **values):
        with self.lock:
            if self.task is not None:
                if self.task.get("cancel_requested"):
                    # Only worker cleanup may acknowledge completed cancellation.
                    values.pop("finished_at", None)
                    values["status"] = "canceling"
                self.task.update(values)

    def _update_step(self, index, **values):
        with self.lock:
            if self.task is not None and 0 <= index < len(self.task["steps"]):
                self.task["steps"][index].update(values)

    def _objects(self):
        return self.web.graph_snapshot()["objects"]

    def _mock_plan(self, instruction: str, objects: list) -> dict:
        """仅 --mock 使用的界面测试计划; 真机模式始终由 Qwen plan_task 生成。"""
        valid_ids = {obj["ann_id"] for obj in objects}
        explicit_ids = [
            int(value)
            for value in re.findall(r"(?:ann_id\s*[=:：]\s*|#|编号\s*)(\d+)", instruction, re.I)
            if int(value) in valid_ids
        ]
        mentioned = list(dict.fromkeys(explicit_ids))
        if not mentioned:
            for obj in objects:
                names = [obj.get("category_zh", ""), obj.get("category", ""), str(obj.get("ann_id"))]
                if any(name and name in instruction for name in names):
                    mentioned.append(obj["ann_id"])
        if not mentioned:
            mentioned = [objects[0]["ann_id"]]
        observe = any(word in instruction for word in ("看", "观察", "状态", "检查", "寻找", "找"))
        patrol = any(word in instruction.lower() for word in ("巡视", "巡逻", "patrol")) and len(mentioned) >= 2
        need_return = False  # Mock text is not an authorization to append travel.
        steps = []
        if patrol:
            steps.append({
                "type": "patrol", "target_ann_ids": mentioned, "count": -1,
                "reason": "循环拍照并对比各点位是否出现环境变化",
            })
        else:
            for ann_id in mentioned:
                steps.append({"type": "navigate", "target_ann_id": ann_id, "use_viewpoint": False, "reason": "前往目标"})
                if observe:
                    steps.append({"type": "observe", "target_ann_id": ann_id, "question": instruction, "fine_adjust": False})
        if need_return:
            steps.append({"type": "return"})
        return {"understanding": instruction, "steps": steps, "need_return": need_return}

    def plan(self, instruction: str) -> dict:
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")
        graph = self.web.graph_snapshot()
        objects = graph["objects"]
        if self.web.mock:
            plan = self._mock_plan(instruction, objects)
        else:
            from qwen_planner import plan_task

            plan = plan_task(instruction, objects)
        LOGGER.info(
            "[plan] generated instruction=%r map=%r plan=%s",
            instruction,
            graph.get("name"),
            json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
        )
        if _ensure_explicit_observation_steps(plan, instruction):
            LOGGER.warning(
                "[plan] model omitted observe; repaired explicit observation step instruction=%r plan=%s",
                instruction,
                json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
            )
        clarification = _fallback_clarification(instruction, plan, objects)
        if clarification:
            candidate_ids = list(dict.fromkeys(clarification.get("candidate_ann_ids") or []))
            candidates = []
            for ann_id in candidate_ids:
                obj = next((value for value in objects if value["ann_id"] == ann_id), None)
                if not obj:
                    continue
                xy = obj.get("nav_xy") or obj["floor_xy"]
                candidates.append({
                    "ann_id": ann_id,
                    "name": obj.get("category_zh") or obj.get("category") or f"目标{ann_id}",
                    "position": obj.get("position") or "",
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                })
            if len(candidates) < 2:
                raise RuntimeError("目标澄清候选不足，请重新描述目标")
            task = {
                "id": uuid.uuid4().hex,
                "instruction": instruction,
                "understanding": plan.get("understanding") or clarification.get("question"),
                "status": "needs_clarification",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "current_step": None,
                "steps": [],
                "plan": plan,
                "route_points": [],
                "observations": [],
                "logs": [],
                "error": None,
                "cancel_requested": False,
                "map_name": graph["name"],
                "clarification": {
                    "question": str(clarification.get("question") or "请选择具体目标"),
                    "candidate_ann_ids": [value["ann_id"] for value in candidates],
                },
                "candidates": candidates,
            }
            with self.lock:
                self.task = task
            return copy.deepcopy(task)
        return self._publish_plan(instruction, graph, plan)

    def _publish_plan(self, instruction, graph, plan):
        """Shared pending-task builder; never starts the robot."""
        objects = graph["objects"]
        if plan.get("need_return") and not any(step.get("type") == "return" for step in plan.get("steps", [])):
            plan["steps"].append({"type": "return"})
        if not plan.get("steps"):
            raise RuntimeError("小卡 没有生成可执行步骤")
        names = {obj["ann_id"]: obj.get("category_zh") or obj.get("category") for obj in objects}
        steps = []
        route_points = []
        patrol_step = next((step for step in plan.get("steps", []) if step.get("type") == "patrol"), None)
        for index, step in enumerate(plan.get("steps", [])):
            item = copy.deepcopy(step)
            item.update({
                "index": index,
                "status": "pending",
                "target_names": [names.get(ann_id, str(ann_id)) for ann_id in _step_target_ids(step)],
            })
            steps.append(item)
            if step.get("type") not in ("navigate", "cruise", "patrol"):
                continue
            for visit_order, ann_id in enumerate(_step_target_ids(step)):
                obj = next((value for value in objects if value["ann_id"] == ann_id), None)
                if obj:
                    display_xy = obj.get("nav_xy") or obj["floor_xy"]
                    route_points.append({
                        "ann_id": ann_id,
                        "step_index": index,
                        "visit_order": visit_order,
                        "x": display_xy[0],
                        "y": display_xy[1],
                    })
        task = {
            "id": uuid.uuid4().hex,
            "kind": "patrol" if patrol_step else "robot_task",
            "instruction": instruction,
            "understanding": plan.get("understanding") or instruction,
            "status": "planned",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "current_target_ann_id": None,
            "steps": steps,
            "plan": plan,
            "route_points": route_points,
            "observations": [],
            "logs": [],
            "error": None,
            "cancel_requested": False,
            "map_name": graph["name"],
            "video": None,
        }
        if patrol_step:
            task.update({
                "patrol_round": 0,
                "patrol_count": int(patrol_step.get("count", -1)),
                "patrol_baselines": {},
                "patrol_comparisons": [],
                "patrol_status_by_ann": {
                    str(ann_id): "pending" for ann_id in _step_target_ids(patrol_step)
                },
                "anomaly": None,
                "result_text": "",
            })
        with self.lock:
            self.task = task
        return copy.deepcopy(task)

    def plan_skill(self, skill_id, instruction, reference=None, **parameters):
        """One adapter for robot skill proposals; confirmation still uses execute()."""
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")
            graph = self.web.graph_snapshot()
            map_snapshot = MapEvidence(graph).version
            skill, args, plan = prepare_skill(skill_id, {"instruction": instruction, **parameters}, graph, reference)
            if skill.requires_reference:
                if not isinstance(reference, dict):
                    raise ValueError("参考图片无效")
                path = self.web.reference_path(reference.get("reference_id"), reference.get("suffix"))
                if not path.is_file():
                    raise ValueError("参考图片不存在或已过期，请重新上传")
            if skill_id == "welcome":
                self.plan_welcome(args["instruction"], reference, args["pickup_ann_id"], args["return_ann_id"], announce=False)
            elif skill_id == "find_object":
                self.plan_find_object(args["instruction"], reference, announce=False)
            else:
                self._publish_plan(args["instruction"], graph, plan)
            self.task["skill"] = {"id": skill.id, "name": skill.name, "version": 1,
                                  "completion": skill.completion, "failure_policy": skill.failure_policy,
                                  "map_snapshot": map_snapshot}
            return copy.deepcopy(self.task)

    def plan_find_object(self, instruction: str, reference: dict, *, announce=True) -> dict:
        """为上传参考图生成确定性的逐点寻物任务，不依赖通用任务规划模型。"""
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")
        graph = self.web.graph_snapshot()
        candidates = _find_object_candidates(graph["objects"])
        if not candidates:
            raise RuntimeError("当前地图没有适合放置物品的可巡检点位")
        target_label = self.web.find_target_label(instruction, reference)
        start_xy = None
        try:
            status = self.robot_status()
            if status.get("online") and status.get("pose"):
                start_xy = (float(status["pose"]["x"]), float(status["pose"]["y"]))
        except Exception:
            pass
        if start_xy is not None:
            ordered = []
            remaining = list(candidates)
            cursor = start_xy
            while remaining:
                nearest = min(
                    remaining,
                    key=lambda obj: math.hypot(
                        float((obj.get("nav_xy") or obj["floor_xy"])[0]) - cursor[0],
                        float((obj.get("nav_xy") or obj["floor_xy"])[1]) - cursor[1],
                    ),
                )
                remaining.remove(nearest)
                ordered.append(nearest)
                point = nearest.get("nav_xy") or nearest["floor_xy"]
                cursor = (float(point[0]), float(point[1]))
            candidates = ordered
        steps = []
        route_points = []
        route_distance = 0.0
        route_cursor = start_xy
        for index, obj in enumerate(candidates):
            ann_id = obj["ann_id"]
            name = obj.get("category_zh") or obj.get("category") or f"目标{ann_id}"
            xy = obj.get("nav_xy") or obj["floor_xy"]
            steps.append({
                "type": "find_object",
                "target_ann_id": ann_id,
                "index": index,
                "status": "pending",
                "target_names": [name],
                "reason": "在可放置物品的点位寻找参考物",
            })
            route_points.append({
                "ann_id": ann_id, "step_index": index, "visit_order": index,
                "x": xy[0], "y": xy[1], "verdict": "candidate",
            })
            if route_cursor is not None:
                route_distance += math.hypot(float(xy[0]) - route_cursor[0], float(xy[1]) - route_cursor[1])
            route_cursor = (float(xy[0]), float(xy[1]))
        task = {
            "id": uuid.uuid4().hex,
            "kind": "find_object",
            "instruction": instruction,
            "understanding": (
                f"先用拟物体地图筛选 {len(candidates)} 个可放置物品的候选点，机器人沿路线前往时"
                f"每 {int(FIND_SNAPSHOT_INTERVAL_SECONDS)} 秒抓拍一次，"
                "单帧高置信匹配后立即停止。"
            ),
            "status": "planned",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "current_target_ann_id": None,
            "steps": steps,
            "plan": {"steps": copy.deepcopy(steps), "need_return": False},
            "route_points": route_points,
            "observations": [],
            "logs": [],
            "error": None,
            "cancel_requested": False,
            "map_name": graph["name"],
            "reference": copy.deepcopy(reference),
            "target_label": target_label,
            "candidate_ann_ids": [obj["ann_id"] for obj in candidates],
            "checked_ann_ids": [],
            "candidate_status": {str(obj["ann_id"]): "candidate" for obj in candidates},
            "planned_route_m": round(route_distance, 2) if start_xy is not None else None,
            "search_attempts": [],
            "snapshot_count": 0,
            "snapshot_interval_seconds": FIND_SNAPSHOT_INTERVAL_SECONDS,
            "snapshot_confirmation_frames": FIND_SNAPSHOT_CONFIRM_FRAMES,
            "current_search_stage": None,
            "match": None,
            "result_text": "",
            "video": None,
        }
        with self.lock:
            self.task = task
        if announce:
            self.web.speak(f"确认要开始寻找这个{target_label}吗？")
        return copy.deepcopy(task)

    def plan_welcome(
        self,
        instruction: str,
        reference: dict,
        pickup_ann_id: int,
        return_ann_id: int,
        *, announce=True,
    ) -> dict:
        """生成“前往接人点等待目标人物，确认后带回指定点”的迎宾任务。"""
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有任务正在执行，请先停止当前任务")

        graph = self.web.graph_snapshot()
        objects = {int(obj["ann_id"]): obj for obj in graph["objects"]}
        if pickup_ann_id not in objects:
            raise ValueError(f"接人点 #{pickup_ann_id} 不在当前地图中")
        if return_ann_id not in objects:
            raise ValueError(f"返回点 #{return_ann_id} 不在当前地图中")
        if pickup_ann_id == return_ann_id:
            raise ValueError("接人点和返回点不能是同一个地图目标")

        def point_value(ann_id):
            obj = objects[ann_id]
            xy = obj.get("nav_xy") or obj.get("floor_xy")
            if not isinstance(xy, (list, tuple)) or len(xy) < 2:
                raise ValueError(f"地图目标 #{ann_id} 没有可用导航坐标")
            return obj, (float(xy[0]), float(xy[1]))

        pickup, pickup_xy = point_value(pickup_ann_id)
        return_point, return_xy = point_value(return_ann_id)
        pickup_name = pickup.get("category_zh") or pickup.get("category") or f"目标{pickup_ann_id}"
        return_name = (
            return_point.get("category_zh")
            or return_point.get("category")
            or f"目标{return_ann_id}"
        )
        steps = [
            {
                "type": "navigate",
                "target_ann_id": pickup_ann_id,
                "index": 0,
                "status": "pending",
                "target_names": [pickup_name],
                "reason": "前往接人点并正对所选地图目标",
            },
            {
                "type": "wait_guest",
                "target_ann_id": pickup_ann_id,
                "index": 1,
                "status": "pending",
                "target_names": [pickup_name],
                "reason": (
                    f"每 {WELCOME_SNAPSHOT_INTERVAL_SECONDS:g} 秒抓拍识别目标人物，"
                    "单帧高置信识别后立即进行语音确认"
                ),
            },
            {
                "type": "navigate",
                "target_ann_id": return_ann_id,
                "index": 2,
                "status": "pending",
                "target_names": [return_name],
                "reason": "对方肯定身份后带领其返回所选地点",
            },
        ]
        task = {
            "id": uuid.uuid4().hex,
            "kind": "welcome",
            "instruction": instruction,
            "understanding": (
                f"前往{pickup_name}等待参考照片中的人物；看到目标后询问“{WELCOME_QUESTION}”，"
                f"得到肯定回答后带领对方返回{return_name}。否定回答后继续等待，未听清时重复询问。"
            ),
            "status": "planned",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "current_target_ann_id": None,
            "current_stage": "等待执行",
            "steps": steps,
            "plan": {"steps": copy.deepcopy(steps), "need_return": False},
            "route_points": [
                {
                    "ann_id": pickup_ann_id,
                    "step_index": 0,
                    "visit_order": 0,
                    "x": pickup_xy[0],
                    "y": pickup_xy[1],
                    "role": "pickup",
                },
                {
                    "ann_id": return_ann_id,
                    "step_index": 2,
                    "visit_order": 1,
                    "x": return_xy[0],
                    "y": return_xy[1],
                    "role": "return",
                },
            ],
            "observations": [],
            "logs": [],
            "error": None,
            "cancel_requested": False,
            "map_name": graph["name"],
            "reference": copy.deepcopy(reference),
            "pickup_ann_id": pickup_ann_id,
            "pickup_name": pickup_name,
            "return_ann_id": return_ann_id,
            "return_name": return_name,
            "snapshot_interval_seconds": WELCOME_SNAPSHOT_INTERVAL_SECONDS,
            "snapshot_count": 0,
            "rejected_count": 0,
            "unclear_count": 0,
            "last_heard_text": "",
            "result_text": "",
            "video": None,
        }
        with self.lock:
            self.task = task
        if announce:
            self.web.speak(f"迎宾路线已设置，接人点是{pickup_name}，返回点是{return_name}。")
        return copy.deepcopy(task)

    def execute(self, task_id: str):
        if getattr(self.web, "mapping", None) and self.web.mapping.is_busy():
            raise RuntimeError("机器人正在执行自动建图，请先停止建图任务")
        with self.lock:
            if not self.task or self.task["id"] != task_id:
                raise ValueError("任务不存在或已被替换")
            if self.task["status"] != "planned":
                raise RuntimeError(f"任务当前状态不能执行: {self.task['status']}")
            if self.is_busy():
                raise RuntimeError("上一任务仍在收尾，请稍后再执行")
            if self.task.get("map_name") != self.web.graph_snapshot().get("name"):
                raise RuntimeError("规划后地图已切换，请重新生成任务计划")
            skill_snapshot = (self.task.get("skill") or {}).get("map_snapshot")
            if skill_snapshot and skill_snapshot != MapEvidence(self.web.graph_snapshot()).version:
                raise RuntimeError("规划后地图内容已变化，请重新生成技能计划")
            # Fail closed before spawning any execution/video/camera thread.
            status = self.robot_status()
            if status.get("online") is not True:
                raise TaskSafetyError("底盘离线或无法读取状态，未启动任务")
            if status.get("estop_state") is True:
                raise TaskSafetyError("底盘处于急停状态，未启动任务。请现场确认安全并检查急停装置；系统不会自动解除急停")
            if status.get("estop_state") is not False:
                raise TaskSafetyError("无法确认底盘急停状态，未启动任务，请先检查底盘连接")
            self.task.update({"status": "running", "started_at": time.time()})
            self.thread = threading.Thread(target=self._run, daemon=True, name="RobotWebTask")
            self.thread.start()
        return self.snapshot()

    def cancel(self, task_id: str):
        with self.lock:
            if not self.task or self.task["id"] != task_id:
                raise ValueError("任务不存在")
            if self.task["status"] == "planned":
                self.task.update({"status": "canceled", "finished_at": time.time()})
                return self.snapshot()
            if self.task["status"] not in ("running", "canceling"):
                return self.snapshot()
            self.task.update({"status": "canceling", "cancel_requested": True})
            driver = self.driver
            for stop_event, _ in getattr(self, "_capture_workers", []):
                stop_event.set()
            if self.video_recorder is not None:
                self.video_recorder.request_stop()
        if driver is not None:
            try:
                driver.cancel_move()
            except Exception as exc:
                self._update(error=f"取消指令失败: {exc}")
        return self.snapshot()

    def _canceled(self):
        with self.lock:
            return bool(self.task and self.task.get("cancel_requested"))

    def _task_cancel_check(self):
        task_id = (self.snapshot() or {}).get("id")
        def canceled():
            with self.lock:
                return (not self.task or self.task.get("id") != task_id
                        or bool(self.task.get("cancel_requested")))
        return canceled

    def _task_camera_driver(self):
        return TaskDriver(self._hardware_camera_driver(), self._task_cancel_check(), self.lock)

    def is_busy(self):
        with self.lock:
            thread = getattr(self, "thread", None)
            return bool((self.task and self.task.get("status") in ("running", "canceling"))
                        or (thread is not None and thread.is_alive()))

    def _start_video_recording(self, driver):
        if self.web.mock or not TASK_VIDEO_ENABLED or driver is None:
            return None
        if not hasattr(driver, "grab_color_frame"):
            return None
        task = self.snapshot() or {}
        recorder = TaskVideoRecorder(task.get("id") or uuid.uuid4().hex, driver)
        with self.lock:
            check_cancelled()
            self.video_recorder = recorder
            if self.task is not None:
                self.task["video"] = recorder.snapshot()
        recorder.start()
        with self.lock:
            if self.task is not None:
                self.task["video"] = recorder.snapshot()
        return recorder

    def _stop_video_recording(self):
        with self.lock:
            recorder = self.video_recorder
            self.video_recorder = None
        if recorder is None:
            return
        recorder.stop()
        with self.lock:
            if self.task is not None:
                self.task["video"] = recorder.snapshot()

    def _pause_video_recording(self):
        with self.lock:
            recorder = self.video_recorder
        if recorder is None:
            return True
        paused = recorder.pause(timeout=5.0)
        if not paused:
            LOGGER.error("[video] recorder did not pause before arrival orientation")
        return paused

    def _resume_video_recording(self):
        with self.lock:
            recorder = self.video_recorder
        if recorder is not None:
            recorder.resume()

    def _run(self):
        with cancellation_scope(self._task_cancel_check()):
            self._run_scoped()

    def _run_scoped(self):
        task_at_start = self.snapshot() or {}
        LOGGER.info(
            "[task] started task_id=%r kind=%r steps=%d mode=%s",
            task_at_start.get("id"),
            task_at_start.get("kind"),
            len(task_at_start.get("steps") or []),
            "mock" if self.web.mock else "hardware",
        )
        try:
            if self.web.mock:
                self._run_mock()
            else:
                self._run_hardware()
        except TaskCancelled:
            LOGGER.info("[task] canceled; discarded outstanding result task_id=%s", task_at_start.get("id"))
        except Exception as exc:
            task = self.snapshot() or {}
            stage = task.get("current_search_stage") or task.get("current_stage")
            _log_exception(
                "[task] execution failed",
                exc,
                task_id=task.get("id"),
                kind=task.get("kind"),
                step=task.get("current_step"),
                target_ann_id=task.get("current_target_ann_id"),
                stage=stage,
            )
            public_error = str(exc)
            if stage and stage not in public_error:
                public_error = f"{stage}: {public_error}"
            self._update(
                status="error",
                error=public_error,
                current_stage=stage,
                finished_at=time.time(),
            )
        else:
            finished = self.snapshot() or {}
            LOGGER.info(
                "[task] finished task_id=%r status=%r error=%r",
                finished.get("id"), finished.get("status"), finished.get("error"),
            )
        finally:
            # Remain canceling until every producer has stopped using the camera.
            for stop_event, _ in self._capture_workers:
                stop_event.set()
            for _, thread in self._capture_workers:
                thread.join()
            self._capture_workers.clear()
            self._stop_video_recording()
            with self.lock:
                self.driver = None
                if self.task and self.task.get("cancel_requested"):
                    self.task.update(status="canceled", finished_at=time.time(),
                                     current_stage="任务已取消", current_search_stage=None)

    def _hardware_camera_driver(self):
        """Web 服务内复用相机管线，避免任务结束时释放 Orbbec SDK 导致进程崩溃。"""
        from qwen_planner import CameraBackedDriver

        with self.lock:
            if self.hardware_driver is None:
                self.hardware_driver = CameraBackedDriver(
                    camera_factory=self._camera_factory,
                    keep_camera_open=True,
                    warmup_frames=max(0, int(os.getenv("JAKA_WEB_WARMUP_FRAMES", "30"))),
                )
            return self.hardware_driver

    def close(self):
        """只在整个 Web 服务退出时释放常驻相机。"""
        thread = self.thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self.lock:
            driver = self.hardware_driver
            self.hardware_driver = None
            self.driver = None
        if driver is not None and hasattr(driver, "close_camera"):
            try:
                driver.close_camera(force=True)
            except Exception as exc:
                print(f"[task] 关闭相机失败: {exc}")

    def _mock_move(self, target_xy):
        start = self.mock_pose[:2]
        for index in range(1, 11):
            if self._canceled():
                return "canceled"
            ratio = index / 10
            self.mock_pose[0] = start[0] + (target_xy[0] - start[0]) * ratio
            self.mock_pose[1] = start[1] + (target_xy[1] - start[1]) * ratio
            self.track.append(self.mock_pose[:])
            self.track = self.track[-500:]
            time.sleep(0.12)
        return "succeeded"

    def _mock_observation(self, step):
        source = HERE / "shot.jpg"
        capture_id = uuid.uuid4().hex
        if source.exists():
            shutil.copyfile(source, self.web.capture_path(capture_id))
            image_url = f"/captures/{capture_id}.jpg"
        else:
            image_url = None
        observation = {
            "ann_id": step.get("target_ann_id"),
            "text": "模拟观察完成：已到达目标附近并拍摄现场画面。",
            "image_url": image_url,
        }
        with self.lock:
            self.task["observations"].append(observation)
        return observation

    @staticmethod
    def _public_find_observation(comparison):
        """将模型内部比对说明转换成用户能直接理解的结果，不暴露拼图布局。"""
        found = bool(comparison.get("found"))
        confidence = str(comparison.get("confidence") or "").lower()
        if found and confidence == "high":
            return "已在当前现场确认找到与参考图外观一致的目标物品。"
        if found:
            return "现场疑似发现目标物品，但目前无法完全确认。"
        return "未在当前现场发现与参考图外观匹配的目标物品。"

    def _find_object_match_value(self, ann_id, image_url, comparison, objects):
        obj = objects[ann_id]
        xy = obj.get("nav_xy") or obj["floor_xy"]
        name = obj.get("category_zh") or obj.get("category") or f"目标{ann_id}"
        return {
            "ann_id": ann_id,
            "name": name,
            "position": obj.get("position") or "",
            "x": float(xy[0]),
            "y": float(xy[1]),
            "image_url": image_url,
            "text": self._public_find_observation(comparison),
            "model_reason": str(comparison.get("reason") or "").strip()[:500],
            "confidence": str(comparison.get("confidence") or ""),
            "found": bool(comparison.get("found")),
            "verdict": "match" if comparison.get("found") and comparison.get("confidence") == "high" else "miss",
        }

    def _set_find_verdict(self, ann_id, verdict):
        """把寻物核验结论同步到任务步骤、候选状态和地图路线。"""
        with self.lock:
            if not self.task:
                return
            self.task.setdefault("candidate_status", {})[str(ann_id)] = verdict
            for point in self.task.get("route_points") or []:
                if str(point.get("ann_id")) == str(ann_id):
                    point["verdict"] = verdict

    def _announce_find_observation(self, observation, comparison):
        check_cancelled()
        target_label = str(self.task.get("target_label") or "物品")
        location = str(observation.get("name") or "当前点位")
        position = str(observation.get("position") or "").strip()
        if observation.get("in_transit"):
            pose = observation.get("capture_pose") or {}
            where = (
                f"前往{location}途中（坐标 {float(pose.get('x', 0.0)):.2f},"
                f" {float(pose.get('y', 0.0)):.2f}）"
            )
        else:
            where = f"{location}附近" if not position else f"{location}的{position}"
        if comparison.get("found") and comparison.get("confidence") == "high":
            self.web.speak(f"在{where}找到了{target_label}。")
        else:
            self.web.speak(f"在{where}没有找到{target_label}。")

    def _finish_find_object_match(self, index, match):
        check_cancelled()
        if match.get("in_transit"):
            result_text = (
                f"已在前往{match['name']}途中找到目标（抓拍坐标 "
                f"{match['x']:.2f}, {match['y']:.2f}）。"
            )
        else:
            result_text = (
                f"已找到目标，位于{match['name']}（#{match['ann_id']}，"
                f"坐标 {match['x']:.2f}, {match['y']:.2f}；{match['position']}）。"
            )
        with self.lock:
            for later in self.task["steps"][index + 1:]:
                if later.get("status") == "pending":
                    later.update({"status": "skipped", "finished_at": time.time()})
                    ann_id = later.get("target_ann_id")
                    self.task.setdefault("candidate_status", {})[str(ann_id)] = "skipped"
                    for point in self.task.get("route_points") or []:
                        if str(point.get("ann_id")) == str(ann_id):
                            point["verdict"] = "skipped"
            self.task.update({
                "match": match,
                "result_text": result_text,
                "status": "succeeded",
                "current_step": None,
                "current_target_ann_id": None,
                "finished_at": time.time(),
            })

    def _finish_find_object_no_match(self):
        with self.lock:
            check_cancelled()
            checked = len(self.task.get("checked_ann_ids") or [])
            result_text = f"已完成 {checked} 个候选点的行进中抓拍与到点核验，仍未发现可信匹配。"
            self.task.update({
                "result_text": result_text,
                "status": "not_found",
                "current_step": None,
                "current_target_ann_id": None,
                "current_search_stage": None,
                "finished_at": time.time(),
            })

    def _run_find_object_mock(self):
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        for index, step in enumerate(self.task["steps"]):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            ann_id = step["target_ann_id"]
            LOGGER.info(
                "[find] candidate start task_id=%r index=%d/%d ann_id=%r name=%r",
                (self.snapshot() or {}).get("id"), index + 1, len(self.task["steps"]),
                ann_id, step.get("target_names"),
            )
            self._update(current_step=index, current_target_ann_id=ann_id)
            self._set_find_verdict(ann_id, "active")
            self._update_step(index, status="running", started_at=time.time())
            stop_event = threading.Event()
            probe_result = {}
            probe_thread = self._start_reference_snapshot_loop(
                None, ann_id, objects, stop_event, probe_result, stage="行进中抓拍检测",
            )
            try:
                status = self._mock_move(objects[ann_id]["floor_xy"])
            finally:
                stop_event.set()
                probe_thread.join(timeout=30.0)
                if probe_thread.is_alive():
                    probe_result.setdefault("error", "行进中抓拍线程未能在 30 秒内停止")
            if status != "succeeded":
                LOGGER.error(
                    "[find] navigation failed task_id=%r ann_id=%r status=%r",
                    (self.snapshot() or {}).get("id"), ann_id, status,
                )
                self._update_step(index, status=status, finished_at=time.time())
                self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
                return
            if probe_result.get("error"):
                raise RuntimeError(str(probe_result["error"]))
            match = probe_result.get("match")
            if match is not None:
                with self.lock:
                    self.task["checked_ann_ids"].append(ann_id)
                self._set_find_verdict(ann_id, "match")
                self._update_step(index, status="succeeded", finished_at=time.time())
                self._announce_find_observation(match, probe_result.get("comparison") or match)
                self._finish_find_object_match(index, match)
                return
            if self._canceled():
                self._update_step(index, status="canceled", finished_at=time.time())
                self._update(status="canceled", current_step=None, current_target_ann_id=None, finished_at=time.time())
                return
            match, observation, canceled = self._search_reference_at_point(None, ann_id, objects)
            if canceled:
                self._update_step(index, status="canceled", finished_at=time.time())
                self._update(status="canceled", current_step=None, current_target_ann_id=None, finished_at=time.time())
                return
            if observation is None:
                raise RuntimeError(f"在候选点 #{ann_id} 没有生成有效寻物画面")
            with self.lock:
                self.task["checked_ann_ids"].append(ann_id)
            self._set_find_verdict(ann_id, observation["verdict"])
            self._update_step(index, status="succeeded", finished_at=time.time())
            self._announce_find_observation(observation, observation)
            if match is not None:
                self._finish_find_object_match(index, match)
                return
        self._finish_find_object_no_match()

    def _patrol_spec(self):
        """返回巡逻步骤序号、目标列表和轮数；网页巡逻只接受一个完整多点步骤。"""
        for index, step in enumerate(self.task.get("steps") or []):
            if step.get("type") == "patrol":
                target_ids = list(dict.fromkeys(step.get("target_ann_ids") or []))
                if len(target_ids) < 2:
                    raise RuntimeError("巡逻至少需要两个不同点位")
                count = int(step.get("count", -1))
                return index, target_ids, count
        raise RuntimeError("巡逻任务缺少 patrol 步骤")

    def _patrol_location(self, ann_id, objects):
        obj = objects[ann_id]
        return obj.get("category_zh") or obj.get("category") or f"目标{ann_id}"

    def _patrol_capture_value(self, ann_id, capture_id, round_no, objects):
        return {
            "ann_id": ann_id,
            "name": self._patrol_location(ann_id, objects),
            "capture_id": capture_id,
            "image_url": f"/captures/{capture_id}.jpg",
            "round": round_no,
            "captured_at": time.time(),
        }

    def _set_patrol_baseline(self, ann_id, value):
        with self.lock:
            self.task.setdefault("patrol_baselines", {})[str(ann_id)] = value
            self.task.setdefault("patrol_status_by_ann", {})[str(ann_id)] = "baseline"

    def _record_patrol_comparison(self, baseline, current, comparison):
        check_cancelled()
        abnormal = bool(comparison.get("changed")) and comparison.get("confidence") in ("high", "medium")
        value = {
            "ann_id": current["ann_id"],
            "name": current["name"],
            "round": current["round"],
            "before_image_url": baseline["image_url"],
            "after_image_url": current["image_url"],
            "changed": bool(comparison.get("changed")),
            "abnormal": abnormal,
            "confidence": str(comparison.get("confidence") or "low"),
            "summary": str(comparison.get("summary") or "未发现明确变化。"),
            "changes": copy.deepcopy(comparison.get("changes") or []),
            "compared_at": time.time(),
        }
        with self.lock:
            self.task.setdefault("patrol_comparisons", []).append(value)
            self.task["patrol_comparisons"] = self.task["patrol_comparisons"][-30:]
            self.task.setdefault("patrol_status_by_ann", {})[str(current["ann_id"])] = (
                "anomaly" if abnormal else "normal"
            )
            if not abnormal:
                self.task.setdefault("patrol_baselines", {})[str(current["ann_id"])] = current
        return value

    def _finish_patrol_anomaly(self, step_index, value):
        check_cancelled()
        changes = "；".join(
            f"{item.get('item') or '物体'}：{item.get('detail') or item.get('type') or '发生变化'}"
            for item in value.get("changes") or []
        )
        result = f"巡逻已在{value['name']}（#{value['ann_id']}）停止：{value['summary']}"
        if changes:
            result += f" 变化明细：{changes}。"
        self._update_step(step_index, status="anomaly", finished_at=time.time())
        self._update(
            status="anomaly", anomaly=value, result_text=result,
            current_step=None, current_target_ann_id=value["ann_id"], finished_at=time.time(),
        )
        self.web.speak(result)

    def _finish_patrol_normal(self, step_index, rounds):
        check_cancelled()
        result = f"已完成 {rounds} 轮多点巡逻，没有发现可信环境变化。"
        self._update_step(step_index, status="succeeded", finished_at=time.time())
        self._update(
            status="succeeded", result_text=result,
            current_step=None, current_target_ann_id=None, finished_at=time.time(),
        )

    def _run_patrol_mock(self):
        """模拟两轮多点巡逻；第二轮第二个点产生一次异常，便于离线验收完整界面。"""
        step_index, target_ids, count = self._patrol_spec()
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        max_rounds = count + 1 if count > 0 else 2
        self._update_step(step_index, status="running", started_at=time.time())
        for round_no in range(1, max_rounds + 1):
            self._update(patrol_round=round_no, current_step=step_index)
            for visit_index, ann_id in enumerate(target_ids):
                if self._canceled():
                    self._update_step(step_index, status="canceled", finished_at=time.time())
                    self._update(status="canceled", current_target_ann_id=None, finished_at=time.time())
                    return
                self._update(current_target_ann_id=ann_id)
                self.task["patrol_status_by_ann"][str(ann_id)] = "active"
                status = self._mock_move(objects[ann_id].get("nav_xy") or objects[ann_id]["floor_xy"])
                if status != "succeeded":
                    self._update_step(step_index, status=status, finished_at=time.time())
                    self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
                    return
                capture_id, _ = self.web.capture()
                current = self._patrol_capture_value(ann_id, capture_id, round_no, objects)
                baseline = self.task.get("patrol_baselines", {}).get(str(ann_id))
                if baseline is None:
                    self._set_patrol_baseline(ann_id, current)
                    continue
                comparison = guarded_call(self.web.compare_patrol_images,
                    self.web.capture_path(baseline["capture_id"]),
                    self.web.capture_path(current["capture_id"]),
                    current["name"],
                )
                if count < 0 and round_no == 2 and visit_index == 1:
                    comparison = {
                        "changed": True, "confidence": "high",
                        "summary": "模拟发现点位内新增了一个遗留物品。",
                        "changes": [{"type": "added", "item": "纸箱", "detail": "基线图中没有，本次画面中出现"}],
                    }
                value = self._record_patrol_comparison(baseline, current, comparison)
                if value["abnormal"]:
                    self._finish_patrol_anomaly(step_index, value)
                    return
        self._finish_patrol_normal(step_index, max_rounds)

    @staticmethod
    def _welcome_voice_intent(text: str) -> str:
        """将一次迎宾回答分为肯定、否定或未听清；否定词必须优先于“是”。"""
        normalized = re.sub(r"[\s，。！？、,.!?；;：:]+", "", str(text or "").lower())
        if not normalized:
            return "unclear"
        if any(word in normalized for word in WELCOME_NEGATIVE_WORDS):
            return "negative"
        if any(word in normalized for word in WELCOME_AFFIRMATIVE_WORDS):
            return "affirmative"
        return "unclear"

    def _wait_cancellable(self, seconds: float) -> bool:
        """等待指定时间；任务被取消时提前返回 False。"""
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            if self._canceled():
                return False
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        return not self._canceled()

    def _capture_welcome_snapshot(self, driver, objects):
        """迎宾点抓拍一帧并计分；单帧高置信命中后立即发布并进入语音确认。"""
        task = self.snapshot() or {}
        ann_id = int(task["pickup_ann_id"])
        capture_id = uuid.uuid4().hex
        capture_path = self.web.capture_path(capture_id)
        task_id = task.get("id")
        try:
            if driver is None:
                capture_id, capture_path = self.web.capture()
            else:
                driver.capture(str(capture_path))
        except TaskCancelled:
            raise
        except Exception as exc:
            _log_exception(
                "[welcome] camera capture failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                capture_path=str(capture_path),
            )
            raise RuntimeError(f"迎宾抓拍失败: {exc}") from exc

        try:
            comparison = guarded_call(self.web.compare_person_reference, task.get("reference") or {}, capture_path)
        except TaskCancelled:
            raise
        except Exception as exc:
            _log_exception(
                "[welcome] person comparison failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                capture_path=str(capture_path),
            )
            raise RuntimeError(f"迎宾人物识别失败: {exc}") from exc

        obj = objects[ann_id]
        xy = obj.get("nav_xy") or obj["floor_xy"]
        matched = (
            bool(comparison.get("found"))
            and str(comparison.get("confidence") or "").lower() == "high"
        )
        observation = {
            "ann_id": ann_id,
            "name": obj.get("category_zh") or obj.get("category") or f"目标{ann_id}",
            "x": float(xy[0]),
            "y": float(xy[1]),
            "image_url": f"/captures/{capture_id}.jpg",
            "text": (
                "现场发现与迎宾参考图外观一致的人物，正在进行语音确认。"
                if matched
                else "本次抓拍未确认发现目标人物。"
            ),
            "found": bool(comparison.get("found")),
            "confidence": str(comparison.get("confidence") or "low"),
            "match_score": float(comparison.get("score") or 0.0),
            "match_breakdown": copy.deepcopy(comparison.get("comparisons") or {}),
            "contradictions": copy.deepcopy(comparison.get("contradictions") or []),
            "candidate_region": str(comparison.get("candidate_region") or ""),
            "model_reason": str(comparison.get("reason") or "")[:500],
            "captured_at": time.time(),
            "stage": "迎宾点人物检测",
        }
        with self.lock:
            if self.task:
                check_cancelled()
                self.task["snapshot_count"] = int(self.task.get("snapshot_count") or 0) + 1
                if matched:
                    self.task.setdefault("observations", []).append(observation)
                    self.task["observations"] = self.task["observations"][-12:]
        LOGGER.info(
            "[welcome] person probe task_id=%r ann_id=%r found=%s confidence=%r "
            "score=%.2f schema_valid=%s publish=%s image=%s",
            task_id,
            ann_id,
            bool(comparison.get("found")),
            comparison.get("confidence"),
            float(comparison.get("score") or 0.0),
            bool(comparison.get("schema_valid")),
            matched,
            capture_path,
        )
        return matched, observation, comparison

    def _run_welcome_mock(self):
        objects = {int(obj["ann_id"]): obj for obj in self._objects()}
        pickup_ann_id = int(self.task["pickup_ann_id"])
        return_ann_id = int(self.task["return_ann_id"])
        self._update(current_step=0, current_target_ann_id=pickup_ann_id, current_stage="前往接人点")
        self._update_step(0, status="running", started_at=time.time())
        pickup_xy = objects[pickup_ann_id].get("nav_xy") or objects[pickup_ann_id]["floor_xy"]
        status = self._mock_move(pickup_xy)
        self._update_step(0, status=status, finished_at=time.time())
        if status != "succeeded":
            self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
            return

        self._update(current_step=1, current_stage="已到接人点，等待目标人物")
        self._update_step(1, status="running", started_at=time.time())
        accepted = False
        while not self._canceled() and not accepted:
            if not self._wait_cancellable(WELCOME_SNAPSHOT_INTERVAL_SECONDS):
                break
            source = HERE / "shot.jpg"
            if not source.exists():
                reference = self.task.get("reference") or {}
                source = self.web.reference_path(reference["reference_id"], reference["suffix"])
            capture_id = uuid.uuid4().hex
            capture_path = self.web.capture_path(capture_id)
            shutil.copyfile(source, capture_path)
            observation = {
                "ann_id": pickup_ann_id,
                "name": self.task.get("pickup_name"),
                "image_url": f"/captures/{capture_id}.jpg",
                "text": "模拟发现目标人物，正在进行语音确认。",
                "found": True,
                "confidence": "high",
                "stage": "迎宾点人物检测",
                "captured_at": time.time(),
            }
            with self.lock:
                self.task["snapshot_count"] = int(self.task.get("snapshot_count") or 0) + 1
                self.task["observations"].append(observation)
            while not self._canceled():
                self._update(current_stage="发现疑似目标人物，等待语音确认")
                heard = self.web.ask_and_listen(WELCOME_QUESTION)
                intent = self._welcome_voice_intent(heard)
                self._update(last_heard_text=heard)
                if intent == "affirmative":
                    accepted = True
                    self._update_step(1, status="succeeded", finished_at=time.time())
                    break
                if intent == "negative":
                    self._update(
                        rejected_count=int(self.task.get("rejected_count") or 0) + 1,
                        current_stage="对方否认，继续等待目标人物",
                    )
                    break
                self._update(
                    unclear_count=int(self.task.get("unclear_count") or 0) + 1,
                    current_stage="未听清回答，正在重新询问",
                )
        if self._canceled():
            self._update_step(1, status="canceled", finished_at=time.time())
            self._update(status="canceled", finished_at=time.time())
            return

        self.web.speak("好的，请跟我来。")
        self._update(current_step=2, current_target_ann_id=return_ann_id, current_stage="带领客人返回")
        self._update_step(2, status="running", started_at=time.time())
        return_xy = objects[return_ann_id].get("nav_xy") or objects[return_ann_id]["floor_xy"]
        status = self._mock_move(return_xy)
        self._update_step(2, status=status, finished_at=time.time())
        if status == "succeeded":
            result = (
                f"已在{self.task.get('pickup_name')}接到目标人物，"
                f"并带领其返回{self.task.get('return_name')}。"
            )
            self._update(
                status="succeeded",
                current_step=None,
                current_target_ann_id=None,
                current_stage="迎宾任务完成",
                result_text=result,
                finished_at=time.time(),
            )
        else:
            self._update(
                status="canceled" if status == "canceled" else "aborted",
                error=status,
                finished_at=time.time(),
            )

    def _run_mock(self):
        if self.task.get("kind") == "welcome":
            self._run_welcome_mock()
            return
        if self.task.get("kind") == "find_object":
            self._run_find_object_mock()
            return
        if self.task.get("kind") == "patrol":
            self._run_patrol_mock()
            return
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        start_pose = self.mock_pose[:]
        for index, step in enumerate(self.task["steps"]):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            self._update(current_step=index)
            self._update_step(index, status="running", started_at=time.time())
            status = "succeeded"
            if step["type"] in ("navigate", "cruise"):
                for ann_id in _step_target_ids(step):
                    status = self._mock_move(objects[ann_id]["floor_xy"])
                    if status != "succeeded":
                        break
            elif step["type"] == "observe":
                self._mock_observation(step)
            elif step["type"] == "return":
                status = self._mock_move(start_pose[:2])
            elif step["type"] == "cancel":
                status = "canceled"
            self._update_step(index, status=status, finished_at=time.time())
            if status != "succeeded":
                self._update(status="canceled" if status == "canceled" else "aborted", finished_at=time.time())
                return
        self._update(status="succeeded", current_step=None, finished_at=time.time())

    def _camera_factory(self):
        try:
            from example_capture_infer import ColorDepthCamera

            cam = ColorDepthCamera(serial=HEAD_CAMERA_SN, strict_serial=True)
        except Exception as depth_error:
            print(f"[task] 头部相机深度不可用({depth_error}), 回退同一头部相机的纯彩色流")
            from example_capture_infer import ColorCamera

            cam = ColorCamera(serial=HEAD_CAMERA_SN, strict_serial=True)
        return cam

    def _save_observation(self, step, log):
        ann_id = step.get("target_ann_id")
        source = HERE / f"obs_ann{ann_id}.png"
        image_url = None
        if source.exists():
            capture_id = uuid.uuid4().hex
            import cv2

            image = cv2.imread(str(source))
            if image is not None:
                cv2.imwrite(str(self.web.capture_path(capture_id)), image)
                image_url = f"/captures/{capture_id}.jpg"
        value = {"ann_id": ann_id, "text": log.observation or "", "image_url": image_url}
        with self.lock:
            self.task["observations"].append(value)
        return value

    def _append_executor_logs(self, executor, start_index):
        with self.lock:
            self.task["logs"].extend([
                {
                    "step_type": item.step_type,
                    "target_ann_id": item.target_ann_id,
                    "detail": item.detail,
                    "observation": item.observation,
                    "status": item.status,
                }
                for item in executor.log[start_index:]
            ])

    def _capture_reference_snapshot(self, driver, reference, ann_id, objects, stage, publish_mode="none", **motion):
        """抓拍一帧并与参考图比对；publish_mode 控制是否进入前端任务流。

        - match: 单帧命中时立即入前端
        - always: 不论命中与否都入前端（用于到点后的正式核验）
        - none: 暂不写前端（用于行进中累计连续命中）
        """
        self._update(current_search_stage=stage)
        capture_id = uuid.uuid4().hex
        capture_path = self.web.capture_path(capture_id)
        task_id = (self.snapshot() or {}).get("id")
        captured_pose = None
        try:
            if driver is None:
                capture_id, capture_path = self.web.capture()
            else:
                driver.capture(str(capture_path))
                if str(stage).startswith("行进中"):
                    try:
                        captured_pose = tuple(map(float, driver.get_pose()))
                    except Exception as exc:
                        LOGGER.warning(
                            "[snapshot] failed to read capture pose task_id=%r ann_id=%r error=%s",
                            task_id, ann_id, exc,
                        )
        except TaskCancelled:
            raise
        except Exception as exc:
            _log_exception(
                "[snapshot] camera capture failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                stage=stage,
                capture_path=str(capture_path),
            )
            raise RuntimeError(f"reference snapshot camera capture failed at ann_id={ann_id}: {exc}") from exc
        # 视觉比对只接收参考图本身，不传物品名称或地图位置，避免模型被
        # “水杯”“白板附近”等文字先验诱导成类别匹配或定向猜测。
        model_reference = dict(reference or {})
        try:
            comparison = guarded_call(self.web.compare_reference, model_reference, capture_path)
        except TaskCancelled:
            raise
        except Exception as exc:
            _log_exception(
                "[snapshot] reference comparison failed",
                exc,
                include_traceback=False,
                task_id=task_id,
                ann_id=ann_id,
                stage=stage,
                capture_path=str(capture_path),
                reference_id=(reference or {}).get("reference_id"),
            )
            raise RuntimeError(f"reference snapshot vision comparison failed at ann_id={ann_id}: {exc}") from exc
        observation = self._find_object_match_value(
            ann_id, f"/captures/{capture_id}.jpg", comparison, objects,
        )
        if captured_pose is not None:
            pose_x, pose_y, pose_theta = captured_pose
            observation.update({
                "in_transit": True,
                "capture_pose": {"x": pose_x, "y": pose_y, "theta": pose_theta},
                "x": pose_x,
                "y": pose_y,
                "position": f"行进中抓拍坐标({pose_x:.2f},{pose_y:.2f})",
            })
        matched = observation["verdict"] == "match"
        attempt = {
            **copy.deepcopy(observation),
            "stage": stage,
            "captured_at": time.time(),
            **motion,
        }
        with self.lock:
            if self.task:
                self.task["snapshot_count"] = int(self.task.get("snapshot_count") or 0) + 1
                should_publish = publish_mode == "always" or (publish_mode == "match" and matched)
                check_cancelled()
                if should_publish:
                    self.task.setdefault("search_attempts", []).append(attempt)
                    self.task["search_attempts"] = self.task["search_attempts"][-12:]
        LOGGER.info(
            "[snapshot] reference probe task_id=%r ann_id=%r stage=%r found=%s confidence=%r publish=%s image=%s",
            task_id, ann_id, stage, bool(comparison.get("found")),
            comparison.get("confidence"), bool(publish_mode == "always" or (publish_mode == "match" and matched)), capture_path,
        )
        return {
            "capture_id": capture_id,
            "capture_path": capture_path,
            "observation": observation,
            "comparison": comparison,
            "matched": matched,
            "attempt": attempt,
        }

    def _start_reference_snapshot_loop(self, driver, ann_id, objects, stop_event, result, stage="行进中抓拍检测"):
        """后台周期抓拍；单帧高置信命中后立即发布并请求停止移动。"""
        reference = copy.deepcopy(self.task.get("reference") or {})
        task_id = (self.snapshot() or {}).get("id")

        def worker():
            consecutive_errors = 0
            # 线程由移动命令发送后的回调启动；再等满一个采样周期，让首帧确实来自途中。
            if stop_event.wait(FIND_SNAPSHOT_INTERVAL_SECONDS):
                return
            while not stop_event.is_set() and not self._canceled():
                cycle_started = time.monotonic()
                try:
                    probe = self._capture_reference_snapshot(
                        driver, reference, ann_id, objects, stage, publish_mode="none",
                    )
                except TaskCancelled:
                    break
                except Exception as exc:
                    if stop_event.is_set() or self._canceled():
                        break
                    consecutive_errors += 1
                    LOGGER.warning(
                        "[snapshot] reference probe failed task_id=%r ann_id=%r errors=%d/%d error=%s",
                        task_id, ann_id, consecutive_errors, FIND_SNAPSHOT_MAX_ERRORS, exc,
                    )
                    if consecutive_errors >= FIND_SNAPSHOT_MAX_ERRORS:
                        result["error"] = (
                            f"连续 {consecutive_errors} 次抓拍识别失败，已停止本次寻物：{exc}"
                        )
                        stop_event.set()
                        try:
                            if driver is not None:
                                driver.cancel_move()
                        except Exception:
                            pass
                        break
                else:
                    consecutive_errors = 0
                    # 到达点后主线程会先置 stop_event，再等待本线程退出，然后才开始
                    # 朝向修正。丢弃恰好跨越到达时刻完成的行进检测，正式结果由到点
                    # 后唯一的一张照片给出。
                    if stop_event.is_set() or self._canceled():
                        break
                    if probe.get("matched"):
                        with self.lock:
                            check_cancelled()
                            if self.task:
                                self.task.setdefault("search_attempts", []).append(probe["attempt"])
                                self.task["search_attempts"] = self.task["search_attempts"][-12:]
                        LOGGER.info(
                            "[snapshot] single-frame match confirmed task_id=%r ann_id=%r "
                            "publish=True image=%s",
                            task_id, ann_id, probe["capture_path"],
                        )
                        result["match"] = probe["observation"]
                        result["comparison"] = probe["comparison"]
                        stop_event.set()
                        try:
                            if driver is not None:
                                driver.cancel_move()
                        except Exception:
                            pass
                        break
                remaining = FIND_SNAPSHOT_INTERVAL_SECONDS - (time.monotonic() - cycle_started)
                if remaining > 0:
                    stop_event.wait(remaining)

        context = copy_context()
        def run_worker():
            try:
                context.run(worker)
            except TaskCancelled:
                pass
        thread = threading.Thread(target=run_worker, daemon=True, name=f"FindSnapshot-{ann_id}")
        with self.lock:
            check_cancelled()
            self._capture_workers.append((stop_event, thread))
            thread.start()
        return thread

    def _search_reference_at_point(self, driver, ann_id, objects):
        """到达目标点后做一次正式核验；由 observations 在前端只展示一次。"""
        if self._canceled():
            return None, None, True
        probe = self._capture_reference_snapshot(
            driver, self.task.get("reference") or {}, ann_id, objects,
            "到达点目标核验", publish_mode="always",
        )
        check_cancelled()
        observation = probe["observation"]
        comparison = probe["comparison"]
        matched = bool(probe["matched"])
        return observation if matched else None, observation, False

    def _run_patrol_hardware(self):
        """逐点导航并维护每个点的视觉基线；发现可信变化后立即停止巡逻。"""
        from qwen_planner import PlanExecutor

        step_index, target_ids, count = self._patrol_spec()
        objects = {obj["ann_id"]: obj for obj in self._objects()}
        driver = self._task_camera_driver()
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)
        self._update_step(step_index, status="running", started_at=time.time())
        round_no = 0
        while count < 0 or round_no < count + 1:
            round_no += 1
            self._update(patrol_round=round_no, current_step=step_index)
            for ann_id in target_ids:
                if self._canceled():
                    self._update_step(step_index, status="canceled", finished_at=time.time())
                    self._update(status="canceled", current_target_ann_id=None, finished_at=time.time())
                    return
                self._update(current_target_ann_id=ann_id)
                with self.lock:
                    self.task["patrol_status_by_ann"][str(ann_id)] = "active"
                before_log = len(executor.log)
                status = executor._do_navigate({
                    "type": "navigate", "target_ann_id": ann_id,
                    "use_viewpoint": bool(objects[ann_id].get("viewpoint")),
                })
                self._append_executor_logs(executor, before_log)
                if status != "succeeded":
                    self._update_step(step_index, status=status, finished_at=time.time())
                    if status in ("failed", "timeout"):
                        try:
                            driver.cancel_move()
                        except Exception:
                            pass
                    self._update(
                        status="canceled" if status == "canceled" else "aborted",
                        error=f"巡逻点 #{ann_id} 导航状态: {status}", finished_at=time.time(),
                    )
                    return

                before_observe = len(executor.log)
                observe_status = executor._do_observe({
                    "type": "observe",
                    "target_ann_id": ann_id,
                    "question": "确认巡逻目标在画面中，并拍摄适合与历史基线比较的现场环境。",
                    "fine_adjust": True,
                })
                self._append_executor_logs(executor, before_observe)
                if observe_status != "succeeded":
                    self._update_step(step_index, status=observe_status, finished_at=time.time())
                    self._update(
                        status="error", current_target_ann_id=ann_id,
                        error=f"巡逻点 #{ann_id} 到点后未找到目标，已停止巡逻。",
                        finished_at=time.time(),
                    )
                    return
                source = HERE / f"obs_ann{ann_id}.png"
                if not source.exists():
                    raise RuntimeError(f"巡逻点 #{ann_id} 未生成有效观察画面")
                import cv2

                image = cv2.imread(str(source))
                if image is None:
                    raise RuntimeError(f"巡逻点 #{ann_id} 的观察画面无法读取")
                capture_id = uuid.uuid4().hex
                capture_path = self.web.capture_path(capture_id)
                cv2.imwrite(str(capture_path), image)
                current = self._patrol_capture_value(ann_id, capture_id, round_no, objects)
                baseline = self.task.get("patrol_baselines", {}).get(str(ann_id))
                if baseline is None:
                    self._set_patrol_baseline(ann_id, current)
                    continue
                comparison = guarded_call(self.web.compare_patrol_images,
                    self.web.capture_path(baseline["capture_id"]), capture_path, current["name"],
                )
                value = self._record_patrol_comparison(baseline, current, comparison)
                if value["abnormal"]:
                    self._finish_patrol_anomaly(step_index, value)
                    return
        self._finish_patrol_normal(step_index, round_no)

    def _run_welcome_navigation(self, executor, step_index, ann_id, stage):
        """执行迎宾任务的一段导航，并同步任务步骤和执行日志。"""
        self._update(
            current_step=step_index,
            current_target_ann_id=ann_id,
            current_stage=stage,
        )
        self._update_step(step_index, status="running", started_at=time.time())
        before = len(executor.log)
        status = executor._do_navigate({
            "type": "navigate",
            "target_ann_id": ann_id,
            "use_viewpoint": False,
        })
        self._append_executor_logs(executor, before)
        self._update_step(step_index, status=status, finished_at=time.time())
        return status

    def _run_welcome_hardware(self):
        """前往接人点，按配置间隔识别人像；语音确认后带领客人返回指定点。"""
        from qwen_planner import PlanExecutor

        objects = {int(obj["ann_id"]): obj for obj in self._objects()}
        pickup_ann_id = int(self.task["pickup_ann_id"])
        return_ann_id = int(self.task["return_ann_id"])
        driver = self._task_camera_driver()
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)

        status = self._run_welcome_navigation(
            executor, 0, pickup_ann_id, "前往接人点"
        )
        if status != "succeeded":
            if status in ("failed", "timeout"):
                try:
                    driver.cancel_move()
                except Exception:
                    pass
            self._update(
                status="canceled" if status == "canceled" else "aborted",
                error=status,
                finished_at=time.time(),
            )
            return
        if self._canceled():
            self._update(status="canceled", finished_at=time.time())
            return

        self._update(
            current_step=1,
            current_target_ann_id=pickup_ann_id,
            current_stage="已到接人点，等待目标人物",
        )
        self._update_step(1, status="running", started_at=time.time())
        consecutive_errors = 0
        accepted = False
        if not self._wait_cancellable(WELCOME_SNAPSHOT_INTERVAL_SECONDS):
            self._update_step(1, status="canceled", finished_at=time.time())
            self._update(status="canceled", finished_at=time.time())
            return
        while not self._canceled() and not accepted:
            cycle_started = time.monotonic()
            try:
                matched, _observation, _comparison = self._capture_welcome_snapshot(
                    driver, objects
                )
                consecutive_errors = 0
                self._update(last_detection_error=None)
            except Exception as exc:
                consecutive_errors += 1
                self._update(
                    current_stage="人物识别暂时失败，正在继续重试",
                    last_detection_error=str(exc),
                )
                LOGGER.warning(
                    "[welcome] person probe retry task_id=%r errors=%d error=%s",
                    (self.snapshot() or {}).get("id"),
                    consecutive_errors,
                    exc,
                )
                matched = False

            if self._canceled():
                break
            if matched:
                self._update(current_stage="发现疑似目标人物，等待语音确认")
                while not self._canceled():
                    heard = self.web.ask_and_listen(WELCOME_QUESTION)
                    intent = self._welcome_voice_intent(heard)
                    self._update(last_heard_text=heard)
                    LOGGER.info(
                        "[welcome] voice confirmation task_id=%r intent=%r heard=%r",
                        (self.snapshot() or {}).get("id"),
                        intent,
                        heard[:100],
                    )
                    if intent == "affirmative":
                        accepted = True
                        self._update_step(1, status="succeeded", finished_at=time.time())
                        break
                    if intent == "negative":
                        self._update(
                            rejected_count=int(self.task.get("rejected_count") or 0) + 1,
                            current_stage="对方否认，继续等待目标人物",
                        )
                        break
                    self._update(
                        unclear_count=int(self.task.get("unclear_count") or 0) + 1,
                        current_stage="未听清回答，正在重新询问",
                    )
                if accepted:
                    break

            remaining = WELCOME_SNAPSHOT_INTERVAL_SECONDS - (
                time.monotonic() - cycle_started
            )
            if remaining > 0 and not self._wait_cancellable(remaining):
                break

        if self._canceled():
            self._update_step(1, status="canceled", finished_at=time.time())
            self._update(
                status="canceled",
                current_stage="迎宾任务已取消",
                finished_at=time.time(),
            )
            return

        self.web.speak("好的，请跟我来。")
        status = self._run_welcome_navigation(
            executor, 2, return_ann_id, "带领客人返回"
        )
        if status != "succeeded":
            if status in ("failed", "timeout"):
                try:
                    driver.cancel_move()
                except Exception:
                    pass
            self._update(
                status="canceled" if status == "canceled" else "aborted",
                error=status,
                finished_at=time.time(),
            )
            return

        result = (
            f"已在{self.task.get('pickup_name')}接到目标人物，"
            f"并带领其返回{self.task.get('return_name')}。"
        )
        self._update(
            status="succeeded",
            current_step=None,
            current_target_ann_id=None,
            current_stage="迎宾任务完成",
            result_text=result,
            finished_at=time.time(),
        )
        self.web.speak("已到达，请慢走。")

    def _run_hardware(self):
        if self.task.get("kind") == "welcome":
            self._run_welcome_hardware()
            return
        if self.task.get("kind") == "find_object":
            self._run_find_object_hardware()
            return
        if self.task.get("kind") == "patrol":
            self._run_patrol_hardware()
            return
        from qwen_planner import JakaTCPDriver, PlanExecutor

        plan = self.task["plan"]
        need_camera = any(step.get("type") == "observe" for step in plan.get("steps", []))
        driver = (self._task_camera_driver() if (need_camera or TASK_VIDEO_ENABLED)
                  else TaskDriver(JakaTCPDriver(), self._task_cancel_check(), self.lock))
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)
        plan_steps = plan.get("steps", [])
        for index, step in enumerate(plan_steps):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            self._update(current_step=index)
            self._update_step(index, status="running", started_at=time.time())
            before = len(executor.log)
            step_type = step["type"]
            if step_type == "navigate":
                status = executor._do_navigate(step)
            elif step_type == "cruise":
                status = executor._do_cruise(step)
            elif step_type == "observe":
                status = executor._do_observe(step)
            elif step_type == "return":
                status = executor._do_return()
            elif step_type == "cancel":
                executor._do_cancel()
                status = "canceled"
            else:
                status = "failed"
            check_cancelled()
            new_logs = executor.log[before:]
            with self.lock:
                self.task["logs"].extend([
                    {
                        "step_type": item.step_type,
                        "target_ann_id": item.target_ann_id,
                        "detail": item.detail,
                        "observation": item.observation,
                        "status": item.status,
                    }
                    for item in new_logs
                ])
            observe_logs = [item for item in new_logs if item.step_type == "observe"]
            if observe_logs:
                fallback_ann_id = step.get("target_ann_id")
                if fallback_ann_id is None and step_type == "cruise":
                    ids = list(step.get("target_ann_ids") or [])
                    fallback_ann_id = ids[-1] if ids else None
                observation_step = step if step_type == "observe" else {
                    "target_ann_id": fallback_ann_id,
                }
                self._save_observation(observation_step, observe_logs[-1])
            self._update_step(index, status=status, finished_at=time.time())
            if status != "succeeded":
                if status in ("failed", "timeout"):
                    try:
                        driver.cancel_move()
                    except Exception:
                        pass
                final = "canceled" if status == "canceled" else "aborted"
                error = status
                if status == "not_found" and observe_logs:
                    error = observe_logs[-1].observation or "到点后未找到目标"
                self._update(status=final, error=error, finished_at=time.time())
                return
        self._update(status="succeeded", current_step=None, finished_at=time.time())

    def _run_find_object_hardware(self):
        """真机按拟物体地图候选点逐点导航，每个候选点只拍一张核验参考物。"""
        from qwen_planner import PlanExecutor

        objects = {obj["ann_id"]: obj for obj in self._objects()}
        driver = self._task_camera_driver()
        with self.lock:
            self.driver = driver
        self._start_video_recording(driver)
        executor = PlanExecutor(self._objects(), driver)
        for index, step in enumerate(self.task["steps"]):
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            ann_id = step["target_ann_id"]
            self._update(current_step=index, current_target_ann_id=ann_id)
            self._set_find_verdict(ann_id, "active")
            self._update_step(index, status="running", started_at=time.time())
            before = len(executor.log)
            stop_event = threading.Event()
            probe_result = {}
            probe_thread = None

            def start_transit_snapshots_after_move_command():
                nonlocal probe_thread
                self._resume_video_recording()
                self._update(current_search_stage="行进中抓拍检测")
                probe_thread = self._start_reference_snapshot_loop(
                    driver, ann_id, objects, stop_event, probe_result,
                    stage="行进中抓拍检测",
                )

            def stop_transit_snapshots_before_facing():
                stop_event.set()
                if probe_thread is not None:
                    probe_thread.join(timeout=30.0)
                if probe_thread is not None and probe_thread.is_alive():
                    raise RuntimeError("行进中抓拍线程未能在 30 秒内停止，已取消到点朝向修正")
                if not self._pause_video_recording():
                    raise RuntimeError("任务录像线程未能停止相机抓帧，已取消到点朝向修正")
                self._update(current_search_stage="到达目标点，正在正对目标")

            try:
                status = executor._do_navigate({
                    "type": "navigate", "target_ann_id": ann_id, "use_viewpoint": False,
                    "_on_motion_started": start_transit_snapshots_after_move_command,
                    "_on_translation_arrived": stop_transit_snapshots_before_facing,
                })
            finally:
                stop_event.set()
                if probe_thread is not None:
                    probe_thread.join(timeout=30.0)
                if probe_thread is not None and probe_thread.is_alive():
                    probe_result.setdefault("error", "行进中抓拍线程未能在 30 秒内停止")
            with self.lock:
                self.task["logs"].extend([
                    {
                        "step_type": item.step_type,
                        "target_ann_id": item.target_ann_id,
                        "detail": item.detail,
                        "observation": item.observation,
                        "status": item.status,
                    }
                    for item in executor.log[before:]
                ])
            check_cancelled()
            if probe_result.get("error"):
                self._update_step(index, status="failed", finished_at=time.time())
                self._update(
                    status="aborted", error=str(probe_result["error"]),
                    current_step=None, current_target_ann_id=None, current_search_stage=None,
                    finished_at=time.time(),
                )
                return
            match = probe_result.get("match")
            if match is not None:
                with self.lock:
                    self.task["checked_ann_ids"].append(ann_id)
                self._set_find_verdict(ann_id, "match")
                self._update_step(index, status="succeeded", finished_at=time.time())
                self._announce_find_observation(match, probe_result.get("comparison") or match)
                self._finish_find_object_match(index, match)
                return
            if status != "succeeded":
                self._update_step(index, status=status, finished_at=time.time())
                if status in ("failed", "timeout"):
                    try:
                        driver.cancel_move()
                    except Exception:
                        pass
                self._update(status="canceled" if status == "canceled" else "aborted", error=status, finished_at=time.time())
                return
            LOGGER.info("[find] navigation succeeded task_id=%r ann_id=%r", (self.snapshot() or {}).get("id"), ann_id)
            if self._canceled():
                self._update(status="canceled", finished_at=time.time())
                return
            match, observation, canceled = self._search_reference_at_point(driver, ann_id, objects)
            if canceled:
                self._update(status="canceled", finished_at=time.time())
                return
            if observation is None:
                raise RuntimeError(f"在候选点 #{ann_id} 没有生成有效寻物画面")
            with self.lock:
                self.task["checked_ann_ids"].append(ann_id)
            self._set_find_verdict(ann_id, observation["verdict"])
            self._update_step(index, status="succeeded", finished_at=time.time())
            self._announce_find_observation(observation, observation)
            if match is not None:
                self._finish_find_object_match(index, match)
                return
        self._update(current_search_stage=None)
        self._finish_find_object_no_match()

    def robot_status(self) -> dict:
        if self.web.mock:
            return {
                "online": True,
                "pose": {"x": self.mock_pose[0], "y": self.mock_pose[1], "theta": self.mock_pose[2]},
                "move_status": self.task.get("status") if self.task else "idle",
                "estop_state": False,
                "track": copy.deepcopy(self.track),
            }
        try:
            from qwen_planner import JakaTCPDriver

            status = JakaTCPDriver().robot_status()
            pose = status.get("current_pose") or {}
            point = [pose.get("x", 0.0), pose.get("y", 0.0), pose.get("theta", 0.0)]
            with self.lock:
                cleaned = self._clean_track([self.track[-1], point] if self.track else [point])
                if cleaned and (not self.track or len(cleaned) > 1):
                    self.track.append(cleaned[-1])
                self.track = self.track[-500:]
                track = copy.deepcopy(self.track)
            return {
                "online": True,
                "pose": {"x": point[0], "y": point[1], "theta": point[2]},
                "move_status": status.get("move_status"),
                "estop_state": _estop_flag(status.get("estop_state")),
                "battery": status.get("battery") or status.get("battery_percent"),
                "track": track,
            }
        except Exception as exc:
            return {"online": False, "error": str(exc), "track": copy.deepcopy(self.track)}


# ======================================================================
#  自动探索建图: 路线预览 → RGB-D 采集 → BoxFusion → 新场景图
# ======================================================================
class MappingManager:
    """维护唯一活动建图任务，并将每次任务完整保存在 mapping_runs。"""

    ACTIVE_STATUSES = {"running", "canceling", "uploading", "server_processing"}

    def __init__(self, web_state):
        self.web = web_state
        self.lock = threading.RLock()
        self.sessions = {}
        self.active_id = None
        self.thread = None
        self.stop_event = threading.Event()
        self.bridge = None
        MAPPING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_sessions()

    @staticmethod
    def _session_path(session_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(session_id or "")):
            raise ValueError("建图任务 ID 非法")
        return MAPPING_RUNS_DIR / str(session_id)

    def _load_sessions(self):
        for path in sorted(MAPPING_RUNS_DIR.glob("*/session.json"), reverse=True):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                session_id = str(value.get("id") or "")
                if not re.fullmatch(r"[0-9a-f]{32}", session_id):
                    continue
                if value.get("status") in self.ACTIVE_STATUSES:
                    value["status"] = "interrupted"
                    value["error"] = "网页服务重启，任务可从未完成探索点继续"
                    self._write_session(value)
                self.sessions[session_id] = value
            except Exception as exc:
                print(f"[mapping] 忽略无效历史 {path}: {exc}")

    def _write_session(self, session):
        run_dir = self._session_path(session["id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        temp = run_dir / "session.json.tmp"
        temp.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(run_dir / "session.json")

    def _update(self, session_id, **values):
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            session.update(values)
            session["updated_at"] = time.time()
            self._write_session(session)
            return copy.deepcopy(session)

    @staticmethod
    def _summary(session):
        plan = session.get("plan") or {}
        progress = session.get("progress") or {}
        bridge = session.get("bridge") or {}
        result = session.get("result") or {}
        return {
            "id": session.get("id"), "name": session.get("name"),
            "status": session.get("status"), "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"), "error": session.get("error"),
            "point_count": len(plan.get("route") or []),
            "estimated_frames": int(plan.get("estimated_frames") or 0),
            "frames": int(progress.get("frames") or 0),
            "uploaded": int(bridge.get("acked") or 0),
            "processed": int(bridge.get("processed") or 0),
            "objects": int(result.get("objects") or bridge.get("objects") or 0),
            "latest_preview": session.get("latest_preview"),
            "map_name": result.get("map_name"),
        }

    def is_busy(self):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return True
            if not self.active_id:
                return False
            session = self.sessions.get(self.active_id) or {}
            return session.get("status") in self.ACTIVE_STATUSES

    def sessions_snapshot(self):
        with self.lock:
            values = sorted(self.sessions.values(), key=lambda item: item.get("created_at", 0), reverse=True)
            return {"active_id": self.active_id, "sessions": [self._summary(value) for value in values]}

    def snapshot(self, session_id=None):
        with self.lock:
            selected = session_id or self.active_id
            if not selected and self.sessions:
                selected = max(self.sessions.values(), key=lambda item: item.get("created_at", 0))["id"]
            session = copy.deepcopy(self.sessions.get(selected)) if selected else None
        if session and self.bridge and session["id"] == self.active_id:
            bridge_state = self.bridge.snapshot()
            session["bridge"] = bridge_state
            self._update(session["id"], bridge=bridge_state)
        if session:
            preview_dir = self._session_path(session["id"]) / "previews"
            session["previews"] = [
                path.name for path in sorted(
                    (path for path in preview_dir.glob("*.*") if path.suffix.lower() in (".jpg", ".jpeg", ".png")),
                    key=lambda path: path.stat().st_mtime_ns,
                )[-60:]
            ]
        return {"active_id": self.active_id, "session": session}

    def _args(self, options):
        from auto_explore import load_agv_defaults

        host, http_port, tcp_port = load_agv_defaults(None)
        return SimpleNamespace(
            execute=True, confirm=True, config=None,
            agv_host=host, agv_http_port=http_port, agv_tcp_port=tcp_port, proto="auto",
            map_yaml_url=SLAM_YAML_URL, map_image_url=SLAM_IMAGE_URL,
            fetch_timeout=max(5.0, SLAM_FETCH_TIMEOUT), out="",
            camera_index=0, camera_width=1280, camera_height=720, camera_fps=30,
            robot_radius=float(options["robot_radius"]), safety_margin=float(options["safety_margin"]),
            point_spacing=float(options["point_spacing"]), max_points=int(options["max_points"]),
            start_order=0, heading_step_deg=float(options["heading_step_deg"]),
            settle_time=float(options["settle_time"]), min_depth_ratio=0.05, zmq_server=None,
            distance_tolerance=0.08, theta_tolerance=0.08,
            move_timeout=120.0, status_interval=0.5,
        )

    def _plan_from_cached_slam(self, args, run_dir, pose):
        from auto_explore import SlamMap, build_route, heading_count

        image_bytes, _ = self.web.slam_image()
        slam = self.web.slam_snapshot()
        calibration = slam.get("calibration") or {}
        metadata = slam.get("metadata") or {}
        resolution = float(metadata.get("resolution") or calibration.get("resolution") or 0)
        origin = metadata.get("origin") or [
            float(calibration.get("origin_x") or 0),
            float(calibration.get("origin_y") or 0),
            math.radians(float(calibration.get("rotation") or 0)),
        ]
        if resolution <= 0:
            raise RuntimeError("SLAM 地图缺少有效分辨率")
        yaml_text = (
            "image: slam_map.png\n"
            f"resolution: {resolution}\n"
            f"origin: [{float(origin[0])}, {float(origin[1])}, {float(origin[2])}]\n"
            f"negate: {int(metadata.get('negate') or 0)}\n"
            f"occupied_thresh: {float(metadata.get('occupied_thresh') or 0.65)}\n"
            f"free_thresh: {float(metadata.get('free_thresh') or 0.196)}\n"
        ).encode("utf-8")
        map_obj = SlamMap.from_data(yaml_text, image_bytes, slam.get("image_url") or "local://slam_map.png")
        points = map_obj.make_coverage_points(
            pose, args.robot_radius, args.safety_margin, args.point_spacing, args.max_points,
        )
        if not points:
            raise RuntimeError("SLAM 地图中没有生成可达探索点")
        route = build_route(points, pose)
        count = heading_count(args.heading_step_deg)
        plan = {
            "created_at": time.time(), "map_yaml_url": SLAM_YAML_URL,
            "map_image_url": slam.get("image_url"), "map": map_obj.metadata,
            "initial_pose": [float(value) for value in pose[:3]],
            "points": [[float(x), float(y)] for x, y in points], "route": route,
            "heading_count": count, "estimated_frames": len(route) * count,
            "options": vars(args),
        }
        map_obj.save_assets(run_dir)
        temp = run_dir / "explore_plan.json.tmp"
        temp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(run_dir / "explore_plan.json")
        return plan

    def plan(self, payload):
        if self.is_busy():
            raise RuntimeError("已有建图任务正在执行")
        task = self.web.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            raise RuntimeError("机器人正在执行功能任务，请先停止后再建图")
        try:
            options = {
                "max_points": int(payload.get("max_points", 30)),
                "point_spacing": float(payload.get("point_spacing", 1.0)),
                "heading_step_deg": float(payload.get("heading_step_deg", 120.0)),
                "robot_radius": float(payload.get("robot_radius", 0.30)),
                "safety_margin": float(payload.get("safety_margin", 0.15)),
                "settle_time": float(payload.get("settle_time", 0.8)),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("建图参数必须是数字") from exc
        if not 1 <= options["max_points"] <= 250:
            raise ValueError("探索点数量必须在 1 到 250 之间")
        if not 0.25 <= options["point_spacing"] <= 10:
            raise ValueError("探索点间距必须在 0.25 到 10 米之间")
        if not 1 <= options["heading_step_deg"] <= 360:
            raise ValueError("拍摄角度间隔必须在 1 到 360 度之间")
        if options["robot_radius"] < 0 or options["safety_margin"] < 0 or options["settle_time"] < 0:
            raise ValueError("安全距离和稳定时间不能为负数")

        session_id = uuid.uuid4().hex
        run_dir = self._session_path(session_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        args = self._args(options)
        status = self.web.tasks.robot_status()
        pose_data = status.get("pose") if status.get("online") else None
        if pose_data:
            pose = [float(pose_data["x"]), float(pose_data["y"]), float(pose_data.get("theta") or 0)]
        elif self.web.mock:
            points = [obj["floor_xy"] for obj in self.web.graph_snapshot()["objects"]]
            pose = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points), 0.0]
        else:
            raise RuntimeError(f"无法读取机器人当前位置: {status.get('error') or '底盘离线'}")
        plan = self._plan_from_cached_slam(args, run_dir, pose)
        created = time.time()
        session = {
            "id": session_id,
            "name": str(payload.get("name") or time.strftime("建图 %m-%d %H:%M", time.localtime(created)))[:50],
            "status": "planned", "created_at": created, "updated_at": created,
            "options": options, "plan": plan,
            "progress": {"status": "planned", "frames": 0, "completed": [], "failed": [], "next_order": 0},
            "bridge": {}, "latest_preview": None, "result": {}, "error": None,
        }
        with self.lock:
            self.sessions[session_id] = session
            self._write_session(session)
        return copy.deepcopy(session)

    def start(self, session_id, resume=False):
        with self.lock:
            if self.is_busy():
                raise RuntimeError("已有建图任务正在执行")
            session = self.sessions.get(str(session_id))
            if session is None:
                raise ValueError("建图任务不存在")
            allowed = {"planned"} if not resume else {"canceled", "interrupted", "failed"}
            if session.get("status") not in allowed:
                raise RuntimeError("当前任务状态不能执行此操作")
            task = self.web.tasks.snapshot()
            if task and task.get("status") in ("running", "canceling"):
                raise RuntimeError("机器人正在执行功能任务")
            self.active_id = session["id"]
            self.stop_event = threading.Event()
            start_order = int((session.get("progress") or {}).get("next_order") or 0) if resume else 0
            session.update(status="running", error=None, started_at=time.time())
            self._write_session(session)
            self.thread = threading.Thread(
                target=self._run_session, args=(session["id"], start_order),
                daemon=True, name=f"MappingTask-{session['id'][:8]}",
            )
            self.thread.start()
            return copy.deepcopy(session)

    def cancel(self, session_id):
        with self.lock:
            session = self.sessions.get(str(session_id))
            if session is None:
                raise ValueError("建图任务不存在")
            if session.get("status") not in self.ACTIVE_STATUSES:
                raise RuntimeError("建图任务当前未执行")
            session["status"] = "canceling"
            self._write_session(session)
        self.stop_event.set()
        try:
            from auto_explore import cancel_move
            cancel_move(self._args(session["options"]))
        except Exception as exc:
            print(f"[mapping] 停止底盘移动失败: {exc}")
        return self.snapshot(session_id)["session"]

    def _on_progress(self, session_id, progress):
        status = "running" if progress.get("status") == "running" else progress.get("status")
        self._update(session_id, status=status, progress=progress)

    def _on_bridge(self, session_id, header, _payload):
        bridge_state = self.bridge.snapshot() if self.bridge else {}
        values = {"bridge": bridge_state}
        if header.get("type") == "preview":
            values["latest_preview"] = header.get("local_name")
        if header.get("type") == "progress" and header.get("status"):
            values["status"] = header["status"]
        if header.get("type") == "error":
            values.update(status="failed", error=header.get("message"))
        self._update(session_id, **values)

    def _run_mock(self, session_id, start_order):
        session = self.snapshot(session_id)["session"]
        run_dir = self._session_path(session_id)
        plan = session["plan"]
        progress = session.get("progress") or {}
        completed = list(progress.get("completed") or []) if start_order else []
        frames = int(progress.get("frames") or 0) if start_order else 0
        source = next((path for path in (HERE / "shot.jpg", HERE / "obs_ann18.png", HERE / "slam.png") if path.exists()), None)
        preview_dir = run_dir / "previews"
        preview_dir.mkdir(exist_ok=True)
        for order in range(start_order, len(plan["route"])):
            if self.stop_event.is_set():
                self.bridge.cancel()
                time.sleep(0.2)
                self._update(session_id, status="canceled", error=None)
                return
            point = plan["route"][order]
            for _ in range(plan["heading_count"]):
                frames += 1
                time.sleep(0.02)
            completed.append({"point": point, "order": order, "frames": plan["heading_count"]})
            latest = None
            if source:
                latest = f"preview_{frames - 1:08d}{source.suffix.lower()}"
                shutil.copyfile(source, preview_dir / latest)
            self._update(session_id, status="running", latest_preview=latest, progress={
                "status": "running", "frames": frames, "completed": completed,
                "failed": [], "current_order": order, "current_point": point, "next_order": order + 1,
            }, bridge={"status": "processing", "acked": frames, "processed": frames, "objects": 0})
        graph = self.web.graph_snapshot()
        raw = {
            "video_id": session_id, "objects": graph["objects"],
            "relationships": {"functional": graph.get("func_relationships", []), "positional": graph.get("pos_relationships", [])},
        }
        result_path = run_dir / "server_result.json"
        result_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        self._finalize_result(session_id, raw)

    def _run_session(self, session_id, start_order):
        if self.web.mock:
            try:
                self._run_mock(session_id, start_order)
            except Exception as exc:
                self._update(session_id, status="failed", error=str(exc))
            finally:
                with self.lock:
                    self.active_id = None
            return
        run_dir = self._session_path(session_id)
        session = self.snapshot(session_id)["session"]
        args = self._args(session["options"])
        args.out = str(run_dir)
        try:
            from auto_explore import execute_plan
            from mapping_bridge import MappingBridgeClient

            self.bridge = MappingBridgeClient(
                session_id, run_dir, MAPPING_SERVER,
                on_message=lambda header, payload: self._on_bridge(session_id, header, payload),
            )
            self.bridge.start({"estimated_frames": session["plan"]["estimated_frames"], **session["options"]})
            if start_order > 0:
                restored = self.bridge.requeue_saved_frames()
                print(f"[mapping] 续传已恢复 {restored} 帧")
            if start_order < len(session["plan"]["route"]):
                execute_plan(
                    args, session["plan"], run_dir, start_order=start_order, stop_event=self.stop_event,
                    progress_callback=lambda progress: self._on_progress(session_id, progress),
                    frame_callback=self.bridge.enqueue_frame,
                    camera_factory=lambda *_: SharedMappingCamera(
                        self.web.tasks._hardware_camera_driver(), self.stop_event.is_set),
                )
            if self.stop_event.is_set():
                self.bridge.cancel()
                time.sleep(0.2)
                self._update(session_id, status="canceled", error=None)
                return
            self._update(session_id, status="uploading")
            self.bridge.finish()
            deadline = time.time() + max(30.0, MAPPING_RESULT_TIMEOUT)
            while time.time() < deadline and not self.stop_event.is_set():
                bridge_state = self.bridge.snapshot()
                self._update(session_id, bridge=bridge_state, status=bridge_state.get("status") or "server_processing")
                if self.bridge.wait_complete(0.5):
                    break
                if bridge_state.get("status") == "error":
                    raise RuntimeError(bridge_state.get("error") or "建图服务器处理失败")
            if self.stop_event.is_set():
                self._update(session_id, status="canceled", error=None)
                return
            if not self.bridge.result_path.exists():
                raise TimeoutError("等待建图服务器返回最终 JSON 超时")
            raw = json.loads(self.bridge.result_path.read_text(encoding="utf-8"))
            self._finalize_result(session_id, raw)
        except TaskCancelled:
            self.bridge.cancel()
            self._update(session_id, status="canceled", error=None)
        except Exception as exc:
            self._update(session_id, status="failed", error=str(exc))
            print(f"[mapping] 任务失败 {session_id}: {exc}")
        finally:
            if self.bridge:
                self.bridge.close()
            self.bridge = None
            with self.lock:
                self.active_id = None

    def _finalize_result(self, session_id, raw):
        run_dir = self._session_path(session_id)
        name = f"scene_graph_mapping_{session_id}.json"
        normalized = _normalize_graph(raw, name)
        raw_objects = raw.get("objects") or []
        raw_count = len(raw_objects)
        functional = raw.get("func_relationships") or (raw.get("relationships") or {}).get("functional") or []
        positional = raw.get("pos_relationships") or (raw.get("relationships") or {}).get("positional") or []
        categories = {str(obj.get("category") or "Unknown") for obj in normalized["objects"]}
        temp = HERE / f".{name}.tmp"
        temp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(HERE / name)
        result = {
            "map_name": name, "objects": len(normalized["objects"]), "categories": len(categories),
            "functional_relationships": len(functional), "positional_relationships": len(positional),
            "invalid_objects": max(0, raw_count - len(normalized["objects"])),
            "result_path": str(run_dir / "server_result.json"),
        }
        self._update(session_id, status="complete", completed_at=time.time(), result=result, error=None)

    def apply(self, session_id):
        session = self.snapshot(session_id)["session"]
        if not session or session.get("status") != "complete":
            raise RuntimeError("只有已完成且校验通过的地图才能启用")
        name = (session.get("result") or {}).get("map_name")
        graph = self.web.select_map(name)
        self._update(session_id, applied_at=time.time())
        return graph

    def delete(self, session_id):
        with self.lock:
            if str(session_id) == self.active_id:
                raise RuntimeError("不能删除正在执行的建图任务")
            if str(session_id) not in self.sessions:
                raise ValueError("建图任务不存在")
            self.sessions.pop(str(session_id))
        shutil.rmtree(self._session_path(session_id))
        return self.sessions_snapshot()

    def preview_path(self, session_id, name):
        safe_name = Path(str(name or "")).name
        if safe_name != name or Path(safe_name).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ValueError("预览图文件名非法")
        path = self._session_path(session_id) / "previews" / safe_name
        if not path.is_file():
            raise FileNotFoundError("预览图不存在")
        return path


# ======================================================================
#  硬件 / 模型能力: 全部懒加载, 请求结束立即释放相机
# ======================================================================
class RobotWebState:
    """跨 HTTP 请求共享的硬件锁与语音实例。"""

    def __init__(self, mock=False):
        self.mock = mock
        self.camera_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        self.voice_lock = threading.Lock()
        self.agent_lock = threading.Lock()
        self.agent_sessions = {}
        self.graph_lock = threading.RLock()
        self.slam_lock = threading.RLock()
        self.voice = None
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        _clean_old_captures()
        self.graph = self._load_initial_graph()
        self.slam_image_path = None
        self.slam_calibration = None
        self.slam_metadata = None
        self.slam_fetched_at = None
        self.slam_fetch_error = None
        self._load_slam_image()
        self.tasks = RobotTaskManager(self)
        self.mapping = MappingManager(self)
        # 默认固定使用脚本目录中的去噪 slam.png；确需底盘原图时显式设置 JAKA_SLAM_AUTO_FETCH=1。
        if not self.mock and SLAM_IMAGE_URL and os.getenv("JAKA_SLAM_AUTO_FETCH", "0") == "1":
            threading.Thread(target=self._auto_refresh_slam, daemon=True, name="SlamMapFetch").start()

    def _default_slam_calibration(self, width: int, height: int) -> dict:
        """没有元数据时先按拟物体范围铺开，用户再在网页中精确标定。"""
        if (width, height) == (2502, 2797):
            # 当前 demo/3 地图的底盘 map.yaml 权威参数；断网使用本地 slam.png 时仍能对齐。
            return {
                "origin_x": -31.062019,
                "origin_y": -39.285198,
                "resolution": 0.03,
                "rotation": 0.0,
                "opacity": 0.58,
            }
        points = [obj["floor_xy"] for obj in self.graph["objects"]]
        min_x = min(float(point[0]) for point in points)
        max_x = max(float(point[0]) for point in points)
        min_y = min(float(point[1]) for point in points)
        max_y = max(float(point[1]) for point in points)
        range_x = max(4.0, max_x - min_x)
        range_y = max(4.0, max_y - min_y)
        resolution = max(range_x * 1.16 / width, range_y * 1.16 / height)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return {
            "origin_x": center_x - width * resolution / 2,
            "origin_y": center_y - height * resolution / 2,
            "resolution": resolution,
            "rotation": 0.0,
            "opacity": 0.58,
        }

    def _load_slam_image(self):
        """优先读取保存配置，其次自动发现脚本同目录下的 slam.png。"""
        config = {}
        try:
            if SLAM_CONFIG_PATH.exists():
                config = json.loads(SLAM_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[slam] 忽略无效标定配置: {exc}")
        candidates = []
        env_path = os.getenv("JAKA_WEB_SLAM_IMAGE", "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())
        # 默认优先采用人工确认过、去噪后的固定底图；slam_live 仅作为后备或手动刷新结果。
        candidates.append(HERE / "slam.png")
        configured = str(config.get("image") or "").strip()
        if configured and Path(configured).name == configured:
            candidates.append(HERE / configured)
        candidates.extend(HERE / name for name in (
            "slam_live.png", "slam_live.jpg", "slam.jpg", "slam.jpeg",
        ))
        path = next((value.resolve() for value in candidates if value.is_file()), None)
        if path is None:
            return
        try:
            width, height = _image_size(path)
        except Exception as exc:
            print(f"[slam] 跳过 {path.name}: {exc}")
            return
        defaults = self._default_slam_calibration(width, height)
        use_saved = (width, height) != (2502, 2797) or config.get("calibration_version") == 3
        calibration = {}
        for key, default in defaults.items():
            try:
                value = float(config.get(key, default) if use_saved else default)
                calibration[key] = value if math.isfinite(value) else default
            except (TypeError, ValueError):
                calibration[key] = default
        if calibration["resolution"] <= 0:
            calibration["resolution"] = defaults["resolution"]
        calibration["opacity"] = min(1.0, max(0.05, calibration["opacity"]))
        self.slam_image_path = path
        self.slam_calibration = calibration
        print(f"[slam] 已加载 {path.name} ({width}x{height})，等待/沿用网页标定")

    def slam_snapshot(self) -> dict:
        with self.slam_lock:
            path = self.slam_image_path
            if path is None or not path.exists() or self.slam_calibration is None:
                return {
                    "available": False,
                    "source_url": SLAM_IMAGE_URL,
                    "metadata_url": SLAM_YAML_URL,
                    "fetch_error": self.slam_fetch_error,
                }
            width, height = _image_size(path)
            return {
                "available": True,
                "name": path.name,
                "width": width,
                "height": height,
                "image_url": f"/slam-map/image?v={path.stat().st_mtime_ns}",
                "calibration": copy.deepcopy(self.slam_calibration),
                "source_url": SLAM_IMAGE_URL,
                "metadata_url": SLAM_YAML_URL,
                "metadata": copy.deepcopy(self.slam_metadata),
                "fetched_at": self.slam_fetched_at,
                "fetch_error": self.slam_fetch_error,
            }

    def reload_local_slam(self) -> dict:
        """重新读取脚本目录中的固定底图，避免网页刷新按钮切回未去噪的底盘图片。"""
        with self.slam_lock:
            self.slam_metadata = None
            self.slam_fetched_at = None
            self.slam_fetch_error = None
            self._load_slam_image()
        return self.slam_snapshot()

    def slam_image(self) -> tuple[bytes, str]:
        with self.slam_lock:
            if self.slam_image_path is None or not self.slam_image_path.exists():
                raise FileNotFoundError("尚未加载 SLAM 地图图片")
            content_type = SLAM_IMAGE_TYPES.get(self.slam_image_path.suffix.lower())
            if not content_type:
                raise ValueError("不支持的 SLAM 图片格式")
            return self.slam_image_path.read_bytes(), content_type

    def _auto_refresh_slam(self):
        try:
            result = self.refresh_slam_from_chassis()
            print(f"[slam] 已从底盘刷新 {result['name']} ({result['width']}x{result['height']})")
        except Exception as exc:
            self.slam_fetch_error = str(exc)
            print(f"[slam] 底盘地图抓取失败，继续使用本地图片: {exc}")

    def refresh_slam_from_chassis(self) -> dict:
        """成对抓取 map.yaml + 图片，以底盘 origin/resolution 做权威配准。"""
        parsed = urlparse(SLAM_IMAGE_URL)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("JAKA_SLAM_IMAGE_URL 必须是有效的 HTTP/HTTPS 地址")
        yaml_parsed = urlparse(SLAM_YAML_URL)
        if yaml_parsed.scheme not in ("http", "https") or not yaml_parsed.hostname:
            raise ValueError("JAKA_SLAM_YAML_URL 必须是有效的 HTTP/HTTPS 地址")
        yaml_request = Request(
            SLAM_YAML_URL,
            headers={"User-Agent": "JAKA-Vision/1.0", "Accept": "text/yaml,text/plain,*/*;q=0.5"},
        )
        with urlopen(yaml_request, timeout=max(1.0, SLAM_FETCH_TIMEOUT)) as response:
            yaml_payload = response.read(64 * 1024 + 1)
        if not yaml_payload or len(yaml_payload) > 64 * 1024:
            raise ValueError("底盘返回的 map.yaml 为空或过大")
        try:
            metadata = _parse_slam_yaml(yaml_payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("底盘 map.yaml 不是 UTF-8 文本") from exc
        image_url = urljoin(SLAM_YAML_URL, metadata["image"])
        request = Request(
            image_url,
            headers={"User-Agent": "JAKA-Vision/1.0", "Accept": "image/png,image/jpeg;q=0.9"},
        )
        with urlopen(request, timeout=max(1.0, SLAM_FETCH_TIMEOUT)) as response:
            payload = response.read(14 * 1024 * 1024 + 1)
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
        if not payload or len(payload) > 14 * 1024 * 1024:
            raise ValueError("底盘返回的 SLAM 图片为空或超过 14 MB")
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            suffix = ".png"
        elif payload.startswith(b"\xff\xd8"):
            suffix = ".jpg"
        else:
            raise ValueError(f"底盘返回的不是 PNG/JPEG（Content-Type={content_type or '未知'}）")
        temp = HERE / f".slam_fetch_{uuid.uuid4().hex}{suffix}"
        final = HERE / f"slam_live{suffix}"
        try:
            temp.write_bytes(payload)
            width, height = _image_size(temp)
            with self.slam_lock:
                temp.replace(final)
                self.slam_image_path = final.resolve()
                opacity = float((self.slam_calibration or {}).get("opacity", 0.58))
                origin = metadata["origin"]
                self.slam_calibration = {
                    "origin_x": origin[0],
                    "origin_y": origin[1],
                    "resolution": metadata["resolution"],
                    "rotation": math.degrees(origin[2]),
                    "opacity": min(1.0, max(0.05, opacity)),
                }
                self.slam_metadata = metadata
                self.slam_fetched_at = time.time()
                self.slam_fetch_error = None
                self._save_slam_config()
        finally:
            if temp.exists():
                temp.unlink()
        return self.slam_snapshot()

    def upload_slam_image(self, name: str, encoded: str) -> dict:
        suffix = Path(name).suffix.lower()
        if suffix not in SLAM_IMAGE_TYPES:
            raise ValueError("SLAM 图片仅支持 PNG 或 JPEG")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("SLAM 图片不是合法 Base64") from exc
        if not payload or len(payload) > 14 * 1024 * 1024:
            raise ValueError("SLAM 图片为空或超过 14 MB")
        temp = HERE / f".slam_upload_{uuid.uuid4().hex}{suffix}"
        final = HERE / f"slam_uploaded{suffix}"
        try:
            temp.write_bytes(payload)
            width, height = _image_size(temp)
            temp.replace(final)
        finally:
            if temp.exists():
                temp.unlink()
        self.slam_image_path = final.resolve()
        self.slam_calibration = self._default_slam_calibration(width, height)
        self.slam_metadata = None
        self._save_slam_config()
        return self.slam_snapshot()

    def _save_slam_config(self):
        if self.slam_image_path is None or self.slam_calibration is None:
            return
        value = {"calibration_version": 3, "image": self.slam_image_path.name, **self.slam_calibration}
        SLAM_CONFIG_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_slam_calibration(self, value: dict) -> dict:
        if self.slam_image_path is None or self.slam_calibration is None:
            raise RuntimeError("请先加载 SLAM 地图图片")
        calibration = {}
        for key in ("origin_x", "origin_y", "resolution", "rotation", "opacity"):
            try:
                number = float(value.get(key))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"SLAM 标定参数 {key} 无效") from exc
            if not math.isfinite(number):
                raise ValueError(f"SLAM 标定参数 {key} 必须是有限数值")
            calibration[key] = number
        if not 0.0001 <= calibration["resolution"] <= 10:
            raise ValueError("米/像素必须在 0.0001 到 10 之间")
        if not 0.05 <= calibration["opacity"] <= 1:
            raise ValueError("透明度必须在 0.05 到 1 之间")
        calibration["rotation"] = ((calibration["rotation"] + 180) % 360) - 180
        self.slam_calibration = calibration
        self._save_slam_config()
        return self.slam_snapshot()

    def reset_slam_calibration(self) -> dict:
        if self.slam_image_path is None:
            raise RuntimeError("请先加载 SLAM 地图图片")
        if self.slam_metadata:
            origin = self.slam_metadata["origin"]
            opacity = float((self.slam_calibration or {}).get("opacity", 0.58))
            self.slam_calibration = {
                "origin_x": origin[0],
                "origin_y": origin[1],
                "resolution": self.slam_metadata["resolution"],
                "rotation": math.degrees(origin[2]),
                "opacity": min(1.0, max(0.05, opacity)),
            }
        else:
            width, height = _image_size(self.slam_image_path)
            self.slam_calibration = self._default_slam_calibration(width, height)
        self._save_slam_config()
        return self.slam_snapshot()

    def _load_initial_graph(self):
        preferred = os.getenv("JAKA_WEB_GRAPH", "zmq_scene_graph.json")
        candidates = []
        for name in (preferred, "zmq_scene_graph.json", "scene_graph_real.json", "scene_graph_sample.json"):
            path = HERE / name
            if path not in candidates:
                candidates.append(path)
        for path in candidates:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return _normalize_graph(json.load(handle), path.name)
            except Exception as exc:
                print(f"[map] 跳过 {path.name}: {exc}")
        raise RuntimeError("找不到可用场景图; 请放置 zmq_scene_graph.json 或设置 JAKA_WEB_GRAPH")

    def graph_snapshot(self):
        with self.graph_lock:
            return copy.deepcopy(self.graph)

    def map_files(self):
        """列出脚本目录下可转换为导航地图的 JSON, 大文件只在选择时完整返回。"""
        names = []
        for path in sorted(HERE.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    _normalize_graph(json.load(handle), path.name)
                names.append(path.name)
            except Exception:
                continue
        return names

    def select_map(self, name: str):
        task = self.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            raise RuntimeError("任务执行期间不能切换地图")
        safe_name = Path(name).name
        if safe_name != name or not safe_name.lower().endswith(".json"):
            raise ValueError("地图文件名非法")
        path = HERE / safe_name
        with path.open("r", encoding="utf-8") as handle:
            graph = _normalize_graph(json.load(handle), safe_name)
        with self.graph_lock:
            self.graph = graph
        return self.graph_snapshot()

    def upload_map(self, name: str, data: dict):
        task = self.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            raise RuntimeError("任务执行期间不能切换地图")
        graph = _normalize_graph(data, Path(name or "浏览器地图.json").name)
        with self.graph_lock:
            self.graph = graph
        return self.graph_snapshot()

    @staticmethod
    def capture_path(capture_id: str) -> Path:
        if not CAPTURE_ID_RE.fullmatch(capture_id or ""):
            raise ValueError("非法 capture_id")
        return CAPTURE_DIR / f"{capture_id}.jpg"

    @staticmethod
    def reference_path(reference_id: str, suffix: str) -> Path:
        if not CAPTURE_ID_RE.fullmatch(reference_id or ""):
            raise ValueError("非法 reference_id")
        if suffix not in REFERENCE_IMAGE_TYPES:
            raise ValueError("图片格式不支持")
        return CAPTURE_DIR / f"{reference_id}{suffix}"

    def save_reference_image(self, name: str, encoded: str) -> dict:
        """兼容旧 JSON 上传格式；Web 页面发送原始图片字节。"""
        try:
            payload = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("参考图片不是合法 Base64") from exc
        return self.save_reference_upload(name, payload)

    def save_reference_upload(self, name: str, payload: bytes) -> dict:
        """校验并保存原始参考图片，避免 Base64 编解码。"""
        suffix = Path(name).suffix.lower()
        info = REFERENCE_IMAGE_TYPES.get(suffix)
        if info is None:
            raise ValueError("参考图片仅支持 PNG、JPEG 或 WebP")
        if not payload or len(payload) > MAX_REFERENCE_BYTES:
            raise ValueError("参考图片为空或超过 6 MB")
        _, magic = info
        if not payload.startswith(magic) or (suffix == ".webp" and payload[8:12] != b"WEBP"):
            raise ValueError("参考图片内容与文件格式不匹配")
        reference_id = uuid.uuid4().hex
        path = self.reference_path(reference_id, suffix)
        path.write_bytes(payload)
        return {
            "reference_id": reference_id,
            "suffix": suffix,
            "content_type": info[0],
            "image_url": f"/references/{reference_id}{suffix}",
        }

    def capture(self) -> tuple[str, Path]:
        """拍一张彩色图; 同一时刻只允许一个请求占用 Orbbec。"""
        capture_id = uuid.uuid4().hex
        path = self.capture_path(capture_id)
        with self.camera_lock:
            if self.mock:
                source = HERE / "shot.jpg"
                if not source.exists():
                    raise RuntimeError("mock 模式需要脚本目录下存在 shot.jpg")
                shutil.copyfile(source, path)
                return capture_id, path

            try:
                self.tasks._hardware_camera_driver().capture(str(path))
            except Exception as exc:
                LOGGER.warning("[camera] scene capture failed type=%s", type(exc).__name__)
                raise ToolInputError("本次相机采集失败，请检查相机连接或占用后重试；不是机器人不具备视觉能力。") from exc
        return capture_id, path

    def agent_complete(self, messages, tools):
        """Use the native tool template; validate emitted Qwen-style calls ourselves."""
        from qwen_planner import PLAN_MODEL, _client

        # Optional decision-model endpoint. Vision continues using VISION_MODEL
        # and the existing endpoint; no implicit cloud fallback or credential reuse.
        agent_base = os.getenv("JAKA_AGENT_BASE_URL", "").strip()
        if agent_base:
            from openai import OpenAI
            client = OpenAI(base_url=agent_base, api_key=os.getenv("JAKA_AGENT_API_KEY", "EMPTY"))
        else:
            client = _client()
        response = client.with_options(timeout=45, max_retries=0).chat.completions.create(
            model=os.getenv("JAKA_AGENT_MODEL", "").strip() or PLAN_MODEL,
            messages=messages, tools=tools,
            tool_choice="required" if agent_base and any(t.get("function", {}).get("name") == "finish_response" for t in tools) else "auto",
            temperature=0, max_tokens=700,
            **({"parallel_tool_calls": False} if agent_base else {}),
        )
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            if len(message.tool_calls) != 1:
                return "<tool_call>invalid parallel calls"
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments)
            except (TypeError, ValueError):
                return "{invalid native tool arguments"
            return json.dumps({"tool": call.function.name, "arguments": arguments}, ensure_ascii=False)
        return _json_text(message.content)

    def agent_vision(self, path, question, history):
        if self.mock:
            return "模拟画面中有桌椅。此结果仅用于测试，不代表真实现场。"
        from qwen_planner import VISION_MODEL, VISUAL_EVIDENCE_RULES, format_visual_answer, _client, _image_data_url

        with self.infer_lock:
            response = _client().with_options(timeout=45, max_retries=0).chat.completions.create(
                model=VISION_MODEL, temperature=0, max_tokens=512,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(str(path), VISION_MODEL)}},
                    {"type": "text", "text": VISUAL_EVIDENCE_RULES + "\n只输出给用户看的自然语言，不输出JSON。\n"
                     + "最近对话（只用于理解指代）：\n" + "\n".join(_history_lines(history))
                     + "\n问题：" + question},
                ]}],
            )
        return format_visual_answer(_json_text(response.choices[0].message.content))

    def agent_chat(self, payload, emit=None):
        """Bounded, per-conversation evidence memory; no inferred action dispatch."""
        clean_history(payload.get("history", []))
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("问题必须是 1～4000 字符的文字")
        session_id = payload.get("conversation_id")
        if session_id is not None and (not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id)):
            raise ValueError("会话标识无效")
        if not self.agent_lock.acquire(blocking=False):
            raise RuntimeError("小卡正在处理另一条对话，请稍后再试")
        try:
            sessions = getattr(self, "agent_sessions", {})
            self.agent_sessions = sessions
            now = time.time()
            for key in list(sessions):
                if now - sessions[key]["updated_at"] > 1800:
                    del sessions[key]
            initial_history = []
            for item in payload.get("history", [])[-16:]:
                if (isinstance(item, dict) and item.get("role") in ("user", "assistant")
                        and isinstance(item.get("text"), str) and item["text"].strip()):
                    initial_history.append({"role": item["role"], "text": item["text"][:2000],
                        "capture_id": item.get("capture_id") if isinstance(item.get("capture_id"), str)
                        and CAPTURE_ID_RE.fullmatch(item["capture_id"]) else None})
            memory = copy.deepcopy(sessions.get(session_id, {"history": initial_history, "evidence": []}))
            memory.setdefault("pending_request", None)
            memory.setdefault("focus", None)
            memory.setdefault("last_scene", None)
            memory.setdefault("planned_task", None)
            memory.setdefault("reference", None)
            selection = payload.get("target_selection")
            if selection is not None:
                offer = memory.get("target_offer") or {}
                if (not isinstance(selection, dict) or type(selection.get("ann_id")) is not int
                        or selection.get("snapshot") != offer.get("snapshot")
                        or selection["ann_id"] not in offer.get("ids", [])
                        or MapEvidence(self.graph_snapshot()).version != offer.get("snapshot")):
                    raise ValueError("目标选项已失效，请重新查询地图并选择")
            memory["confirmed_target"] = selection
            memory["turn_goal"] = question
            turn = {**payload, "history": memory["history"]}
            def save_exchange(answer):
                memory["history"] = (memory["history"] + [
                    {"role": "user", "text": question[:2000]},
                    {"role": "assistant", "text": answer["text"][:2000], "capture_id": answer.get("capture_id")}
                ])[-16:]
                memory["last_exchange"] = {"request": question[:2000], "reply": answer["text"][:2000],
                                           "incomplete": bool(answer.get("incomplete"))}
                memory["updated_at"] = time.time()
                memory.pop("turn_goal", None)
                if session_id:
                    if session_id not in sessions and len(sessions) >= 64:
                        del sessions[min(sessions, key=lambda key: sessions[key]["updated_at"])]
                    sessions[session_id] = memory
            try:
                result = self._agent_chat_turn(turn, emit, memory)
            except Exception:
                memory["pending_request"] = {"goal": memory["turn_goal"][:1200], "status": "interrupted"}
                save_exchange({"text": "本轮处理未完成。", "incomplete": True})
                raise
            if result.get("requires_clarification"):
                memory["pending_request"] = result["pending_request"]
            elif result.get("requires_confirmation"):
                task = result["task"]
                # Do not interpret a generated plan as a completed physical task.
                memory["planned_task"] = {"id": task.get("id"), "status": task.get("status"),
                    "goal": memory["turn_goal"][:1200], "recorded_at": time.time(), "skill": task.get("skill")}
                memory["pending_request"] = {"goal": memory["turn_goal"][:1200],
                    "status": "needs_clarification" if task.get("status") == "needs_clarification" else "awaiting_confirmation",
                    "task_id": task.get("id"), "clarification": task.get("clarification")}
            elif result.get("incomplete"):
                memory["pending_request"] = {"goal": memory["turn_goal"][:1200], "status": "interrupted"}
            elif result.get("tool_trace"):
                # Tool failures remain unresolved even when a polite explanation
                # was generated. Successful new queries replace the old focus.
                memory["pending_request"] = (None if result["tool_trace"][-1]["ok"] else
                    {"goal": memory["turn_goal"][:1200], "status": "tool_failed"})
            save_exchange(result)
            return result
        finally:
            self.agent_lock.release()

    def _agent_chat_turn(self, payload, emit, memory):
        question = payload.get("question")
        history = payload.get("history", [])
        clean_history(history)  # Validate before touching hardware or contacting the model.
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("问题必须是 1～4000 字符的文字")
        reference = payload.get("reference")
        reference_source = "current_turn" if reference is not None else None
        if reference is None and memory.get("reference"):
            saved_reference = memory["reference"]
            if self.reference_path(saved_reference["reference_id"], saved_reference["suffix"]).is_file():
                reference = saved_reference
                reference_source = "previous_turn"
            else:
                memory["reference"] = None
        if reference is not None:
            if not isinstance(reference, dict):
                raise ValueError("参考图片元信息无效")
            rid, suffix = reference.get("reference_id"), reference.get("suffix")
            if not isinstance(rid, str) or not isinstance(suffix, str):
                raise ValueError("参考图片元信息无效")
            reference_path = self.reference_path(rid, suffix)
            if not reference_path.exists():
                raise ValueError("参考图片不存在或已过期，请重新上传")
            reference = {"reference_id": rid, "suffix": suffix,
                         "image_url": f"/references/{rid}{suffix}",
                         "content_type": REFERENCE_IMAGE_TYPES[suffix][0]}
            memory["reference"] = copy.deepcopy(reference)
        last_scene = memory.get("last_scene") or {}
        previous_capture = last_scene.get("capture_id")
        if not (isinstance(previous_capture, str) and CAPTURE_ID_RE.fullmatch(previous_capture)
                and self.capture_path(previous_capture).exists()):
            previous_capture = None
        for turn in reversed(history[-16:]):
            if previous_capture:
                break
            if not isinstance(turn, dict) or turn.get("role") != "assistant":
                continue
            capture_id = turn.get("capture_id")
            if isinstance(capture_id, str) and CAPTURE_ID_RE.fullmatch(capture_id):
                if self.capture_path(capture_id).exists():
                    previous_capture = capture_id
                    break
        turn_graph = self.graph_snapshot()
        map_evidence = MapEvidence(turn_graph)

        def get_skill_context(skill_id):
            result = skill_resources(skill_id, turn_graph, bool(reference), reference_source)
            result["snapshot"] = map_evidence.version
            return result

        def scene_result(capture_id, path, question, fresh):
            try:
                observation = self.agent_vision(path, question, history)
            except Exception as exc:
                LOGGER.warning("[vision] scene analysis failed type=%s", type(exc).__name__)
                raise ToolInputError("照片已采集，但本次视觉分析失败，请检查模型连接后重试。") from exc
            return {"ok": True, "capture_id": capture_id, "image_url": f"/captures/{capture_id}.jpg",
                    "image_source": "现场拍摄" if fresh else "历史照片（不是实时画面）",
                    "source_type": "scene_photo", "captured_this_turn": fresh,
                    "observed_at": path.stat().st_mtime,
                    "observation": observation}

        def inspect_previous_scene(question):
            if not previous_capture or not self.capture_path(previous_capture).exists():
                return {"ok": False, "error": "本会话的历史照片不存在或已过期，没有重新拍照。"}
            path = self.capture_path(previous_capture)
            memory["last_scene"] = {"capture_id": previous_capture, "observed_at": path.stat().st_mtime}
            return scene_result(previous_capture, path, question, False)

        def observe_scene(question):
            nonlocal previous_capture
            if self.tasks.is_busy() or self.mapping.is_busy():
                return {"ok": False, "error": "机器人正在执行任务，请查询任务状态或停止任务后再拍照。"}
            capture_id, path = self.capture()
            previous_capture = capture_id
            memory["last_scene"] = {"capture_id": capture_id, "observed_at": path.stat().st_mtime}
            tools["inspect_previous_scene"] = history_tool
            return scene_result(capture_id, path, question, True)

        def inspect_reference(question):
            if not reference:
                return {"ok": False, "error": "本轮没有上传图片，请用户上传。"}
            return {"ok": True, "image_url": reference["image_url"], "image_source": "用户上传的图片", "source_type": "uploaded_image",
                    "observation": self.agent_vision(reference_path, question, history)}

        def query_project_info():
            # Reference material, not a keyword shortcut or current map inventory.
            return {"ok": True, "source": "项目历史指标资料，非当前部署的实时测量或承诺",
                    "records": [_project_metric_query(q) for q in
                                ("空间拟物体数量", "语义图谱规模", "空间对象识别指标")]}

        def get_robot_status():
            task = self.tasks.snapshot()
            fields = ("id", "instruction", "status", "current_step", "error", "result_text", "observations", "skill")
            robot = self.tasks.robot_status()
            task_data = {k: task[k] for k in fields if k in task} if task else None
            if task_data and isinstance(task_data.get("observations"), list):
                task_data["observations"] = task_data["observations"][-4:]
            if task and (memory.get("planned_task") or {}).get("id") == task.get("id"):
                memory["planned_task"].update(status=task.get("status"), recorded_at=time.time())
            return {"ok": True, "source_type": "robot_status", "observed_at": time.time(),
                    "robot": {k: v for k, v in robot.items() if k != "track"},
                    "task": task_data,
                    "mapping_busy": self.mapping.is_busy()}

        def plan_robot_skill(skill_id, **arguments):
            if MapEvidence(self.graph_snapshot()).version != map_evidence.version:
                raise ToolInputError("本轮查询后地图已更新，请重新查询并规划，不能沿用旧目标。")
            targets = (arguments.get("target_ann_ids", []) if skill_id == "navigate" else
                       [arguments.get("target_ann_id")] if skill_id == "inspect_location" else [])
            for target in targets:
                selected = memory.get("confirmed_target")
                if selected and target != selected["ann_id"]:
                    return {"ok": False, "error": "请使用用户在候选按钮中选定的目标，不要替换为其他对象"}
                obj = map_evidence.objects.get(str(target)) or {}
                scope = (memory.get("target_queries") or {}).get(obj.get("category"), {})
                candidates = [map_evidence.objects[key] for key in scope.get("ids", [])
                              if key in map_evidence.objects
                              and map_evidence.objects[key].get("category") == obj.get("category")]
                if len(candidates) > 1 and not selected and scope.get("snapshot") == map_evidence.version:
                    choices = [{k: obj.get(k) for k in ("ann_id", "category", "category_zh", "position", "floor_xy")}
                               for obj in candidates[:20]]
                    memory["target_offer"] = {"ids": [c["ann_id"] for c in choices], "snapshot": map_evidence.version}
                    return {"ok": True, "clarification": "地图查询仍有多个同类目标，尚未创建移动任务。请从下方选择，或补充可区分的位置线索；不会擅自选择其中一个。",
                            "target_choices": choices, "target_choice_snapshot": map_evidence.version,
                            "pending_request": {"goal": arguments.get("instruction"), "status": "needs_clarification"}}
            task = self.tasks.plan_skill(skill_id, reference=reference, **arguments)
            return {"ok": True, "task": task, "requires_confirmation": True}

        def ask_user(goal, question, skill_id, missing_inputs):
            # Validate declared missing resources; do not classify user wording.
            if skill_id != "none":
                resources = get_skill_context(skill_id)
                available = set(resources["default_arguments"])
                if reference_source == "current_turn" and reference:
                    available.add("reference")
                if available.intersection(missing_inputs):
                    return {"ok": False, "error": "先依据已核验资源重新判断。不要索要本轮已上传的照片或唯一默认点；用户另指定目标、历史照片指代不明或多候选仍可澄清。",
                            "resources": resources}
            elif "reference" in missing_inputs and reference_source == "current_turn" and reference:
                return {"ok": False, "error": "本轮参考图已上传成功，不需要重新上传；可查看参考图或继续规划。"}
            return {"ok": True, "clarification": question, "pending_request": {
                "goal": goal, "clarification": question, "status": "needs_clarification"}}

        text_arg = {"type": "string"}
        catalog = map_evidence.category_catalog()
        history_tool = Tool("分析本会话最近一张历史现场照片，不启动相机、不代表实时画面。", {"question": text_arg},
                            inspect_previous_scene, "正在查看之前的照片", "历史照片分析")
        category_arg = {"type": "array", "items": {"type": "string", "maxLength": 256}, "maxItems": 200,
            "description": "选择所有语义相关类别的原始 category 值组成字符串数组；全图传 [\"*\"]，*不能与具体类别混用；没有相关类别传 []。当前目录：" +
                json.dumps(catalog["categories"], ensure_ascii=False)}
        if catalog["next_offset"] is None:
            category_arg["items"]["enum"] = ["*", *map_evidence.by_category]
        else:
            category_arg["description"] += "目录未完整，可用 list_map_categories 继续读取。"
        tools = {
            "get_skill_context": Tool("准备技能或询问缺少资料前，读取技能所需的参考图状态及地图配置的默认点位。默认点来自当前地图而非固定编号；用户指定地点优先，多候选必须消歧。仅查询不执行。", {
                "skill_id": {"type": "string", "enum": [s["id"] for s in skill_summary()]}},
                get_skill_context, "正在核对任务所需资料和默认点位", "技能资源查询", "prepare"),
            "observe_scene": Tool("拍摄并分析原地当前的新照片，不移动。仅用于用户需要现在的现场信息。", {"question": text_arg}, observe_scene, "正在查看当前画面", "现场观察", "observe"),
            "inspect_reference": Tool("查看用户上传的图片", {"question": text_arg}, inspect_reference, "正在查看上传的图片", "图片分析"),
            "query_map": Tool("查询已保存的环境记录/拟物体地图。按语义选择全部相关类别和筛选条件，工具一次返回完整计数、分类数量和位置；不需要提供对象 ID，不需再调用统计工具。", {
                "question": {"type": "string", "description": "结合上下文还原的完整查询需求"},
                "categories": category_arg, "filters": FILTER_SCHEMA,
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "default": 0}}, map_evidence.query, "正在查询地图", "地图查询"),
            "query_project_info": Tool("查阅项目历史验收指标，不是当前地图或实时状态", {}, query_project_info, "正在查阅项目资料", "项目资料"),
            "compare_map_positions": Tool("核对地图中参考物与候选物的距离。先查询双方真实 ID；用于旁边、附近等空间线索消歧，不能仅凭位置描述相似选目标。不自动导航。", {
                "reference_ann_id": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "candidate_ann_ids": {"type": "array", "minItems": 1, "maxItems": 40,
                    "items": {"type": "integer", "minimum": 0, "maximum": 2147483647}}},
                map_evidence.compare_positions, "正在核对地图空间关系", "地图空间关系"),
            "get_robot_status": Tool("查询机器人状态与任务结果", {}, get_robot_status, "正在读取机器人状态", "机器人状态查询"),
            "ask_user": Tool("仅当缺少必要信息时澄清；保留未完成需求等待回答。明确的只读请求直接调用对应工具，不再确认。", {
                "goal": {"type": "string", "description": "尚未完成的完整需求"}, "question": text_arg,
                "skill_id": {"type": "string", "enum": ["none", *[s["id"] for s in skill_summary()]], "description": "当前准备的技能；不是技能需求才填none。先核对技能资源。"},
                "missing_inputs": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 80},
                    "description": "确实缺少的资源参数名，如reference、pickup_ann_id、return_ann_id。已有本轮参考图和唯一默认点不能称为缺失；用户偏好、目标歧义或明确另指定地点时用空数组，并在question说明真正需要确认的内容。"}}, ask_user, "有一处信息需要确认", "需求澄清"),
        }
        tools.update(skill_tools(plan_robot_skill))
        if previous_capture:
            tools["inspect_previous_scene"] = history_tool
        if catalog["next_offset"] is not None:
            tools["list_map_categories"] = Tool("分页查看地图其余类别；仅目录很大时需要。", {
                "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "default": 0}},
                map_evidence.category_catalog, "正在读取地图类别", "地图类别目录")
        if not reference:
            # Offer only currently available resources; never imply that an
            # absent upload can answer questions about a saved environment.
            del tools["inspect_reference"]
        def remember(name, arguments, result):
            LOGGER.info("[agent-tool] tool=%s ok=%s task_id=%s task_status=%s",
                        name, result.get("ok", True), (result.get("task") or {}).get("id"),
                        (result.get("task") or {}).get("status"))
            memory["turn_goal"] = str(arguments.get("instruction") or arguments.get("goal") or arguments.get("question") or question)
            if not result.get("ok", True):
                return
            if name == "query_map":
                scopes = memory.setdefault("target_queries", {})
                for category in result.get("selection", {}).get("categories", []):
                    scopes[category] = {"ids": [key for key in result.get("target_ids", [])
                        if map_evidence.objects[key].get("category") == category], "snapshot": result.get("snapshot")}
            # Keep small, server-produced evidence, not copied client claims or a
            # second full map. Memory is historical, never a live sensor cache.
            keys = ("source", "source_type", "snapshot", "map_name", "capture_id", "image_source", "observed_at",
                    "observation", "count", "groups", "selection", "count_scope", "missing_filter_fields")
            evidence = {key: result[key] for key in keys if key in result}
            if name == "get_robot_status":
                task = result.get("task") or {}
                evidence["task"] = {key: task[key] for key in ("id", "status", "result_text", "error") if key in task}
            if evidence and len(json.dumps(evidence, ensure_ascii=False)) <= 6000:
                memory["evidence"] = (memory["evidence"] + [{"tool": name, "recorded_at": time.time(), **evidence}])[-4:]
                memory["focus"] = {"source_type": result.get("source_type"), "question": memory["turn_goal"][:1200],
                    "snapshot": result.get("snapshot"), "selection": result.get("selection"),
                    "capture_id": result.get("capture_id"), "observed_at": result.get("observed_at")}

        if self.mock:
            return {"text": "当前为模拟模式，没有连接模型或操作真实机器人。", "tool_trace": [], "target_ids": []}
        return AgentRunner(self.agent_complete, tools, native=True, require_final_tool=True,
                           task_snapshot=self.tasks.snapshot).run(question, history, {
            "has_uploaded_image": reference_source == "current_turn", "has_reference_image": bool(reference),
            "reference_source": reference_source, "has_previous_scene_image": bool(previous_capture),
            "skills": skill_summary(),
            "available_map": {"name": map_evidence.name, "snapshot": map_evidence.version}, "now": time.time(),
            "pending_request": memory.get("pending_request"), "focus": memory.get("focus"),
            "user_selected_target": memory.get("confirmed_target"),
            "planned_task_not_execution_proof": memory.get("planned_task"),
            "previous_evidence_not_live": memory["evidence"],
        }, emit, record=remember)

    def route(self, question: str, history=None, has_reference=False) -> dict:
        """判断本轮是否需要新照片；纯文字问题在同一次模型调用中直接回答。"""
        metric_answer = _project_metric_query(question)
        if metric_answer:
            LOGGER.info("[route] deterministic mode=project_metric question=%r", question)
            return {"mode": "text", "answer": metric_answer}
        if _has_find_object_intent(question) and not has_reference:
            LOGGER.info("[route] find request is missing reference image question=%r", question)
            return {"mode": "text", "answer": "请先上传要寻找物品的参考照片，我才能开始地图巡检。"}
        if _is_reference_find_request(question, has_reference):
            LOGGER.info(
                "[route] forcing find_object for reference request question=%r has_reference=%s",
                question,
                has_reference,
            )
            return {"mode": "find_object", "answer": ""}
        fallback_mode = _fallback_route_mode(question, has_reference)
        if fallback_mode in ("robot_task", "vision"):
            LOGGER.info(
                "[route] deterministic mode=%s question=%r has_reference=%s",
                fallback_mode,
                question,
                has_reference,
            )
            return {"mode": fallback_mode, "answer": ""}
        if fallback_mode == "map_query":
            LOGGER.info("[route] deterministic mode=map_query question=%r", question)
            answer, target_ids = _map_query(self.graph_snapshot()["objects"], question)
            return {"mode": "map_query", "answer": answer, "target_ids": target_ids}
        if self.mock:
            if not has_reference and any(word in question for word in FIND_OBJECT_WORDS):
                return {"mode": "text", "answer": "请先上传要寻找物品的参考照片，我才能开始地图巡检。"}
            mode = _mock_route_mode(question, has_reference=has_reference)
            if mode == "map_query":
                answer, target_ids = _map_query(self.graph_snapshot()["objects"], question)
                return {"mode": mode, "answer": answer, "target_ids": target_ids}
            if mode != "text":
                return {"mode": mode, "answer": ""}
            if "导航" in question or "指令" in question:
                answer = (
                    "底层 `qwen_planner.py` 支持按场景图导航、巡游、观察和返回。"
                    "移动任务会先由 小卡 生成计划并显示在地图上，确认后才驱动底盘。"
                )
            else:
                answer = "这是一个不依赖当前相机画面的文字问题，因此本轮不会启动相机。"
            return {"mode": "text", "answer": answer}

        from qwen_planner import PLAN_MODEL, _client

        context = "\n".join(_history_lines(history)) or "(无历史)"
        user_message = (
            f"最近对话:\n{context}\n\n本轮是否有参考图: {'是' if has_reference else '否'}"
            f"\n当前问题: {question}"
        )
        response = _client().chat.completions.create(
            model=PLAN_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        LOGGER.info("[route-model] raw question=%r response=%r", question, raw)
        try:
            decision = _parse_route_decision(raw)
        except ValueError as exc:
            raise RuntimeError(f"对话路由结果不是合法 JSON: {raw}") from exc
        if not isinstance(decision, dict):
            raise RuntimeError(f"对话路由结果不是对象: {raw}")
        mode = str(decision.get("mode") or "text")
        fallback_mode = _fallback_route_mode(question, has_reference)
        if mode == "text" and fallback_mode:
            LOGGER.warning(
                "[route] corrected model mode text -> %s question=%r raw=%r",
                fallback_mode,
                question,
                raw,
            )
            mode = fallback_mode
        if mode not in ("text", "vision", "map_query", "robot_task", "find_object"):
            raise RuntimeError(f"未知对话模式: {mode}")
        if mode == "find_object" and not has_reference:
            return {"mode": "text", "answer": "请先上传要寻找物品的参考照片，我才能开始地图巡检。"}
        if mode == "map_query":
            answer, target_ids = _map_query(self.graph_snapshot()["objects"], question)
            return {"mode": mode, "answer": answer, "target_ids": target_ids}
        answer = str(decision.get("answer") or "").strip()
        if mode == "text" and not answer:
            answer = "我暂时没有理解这句话。你可以直接说要去哪里、观察什么，或上传参考图片后说“帮我找这个物品”。"
        return {"mode": mode, "answer": answer}

    def infer(self, capture_id: str, question: str, history=None) -> str:
        """对已拍图片做视觉问答; 带最近对话以理解"它/那里/再看"等自然指代。"""
        path = self.capture_path(capture_id)
        if not path.exists():
            raise FileNotFoundError("拍摄图片不存在或已过期")
        if self.mock:
            return (
                "画面是一个室内办公区域，可以看到门、墙面窗户、桌椅和右侧的设备。"
                "当前环境光线较暗，画面带有轻微紫红色偏色，细小物体不容易准确辨认。"
            )
        with self.infer_lock:
            from qwen_planner import VISION_MODEL, VISUAL_EVIDENCE_RULES, format_visual_answer, _client, _image_data_url

            turns = _history_lines(history)
            if turns:
                model_question = (
                    "以下是同一会话最近的对话，仅用于理解当前问题里的自然指代；"
                    "事实判断仍以这次新拍摄的图片为准。\n"
                    + "\n".join(turns)
                    + f"\n当前用户问题: {question}"
                )
            else:
                model_question = question
            image_url = _image_data_url(str(path), VISION_MODEL)
            response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": VISUAL_EVIDENCE_RULES + "\n只输出给用户看的自然语言，不输出JSON。\n" + model_question},
                    ],
                }],
                max_tokens=512,
                temperature=0.0,
            )
            return format_visual_answer(_json_text(response.choices[0].message.content))

    def compare_reference(self, reference: dict, scene_path: Path) -> dict:
        """将参考图和当前现场图一起交给视觉模型，返回受限的命中结论。"""
        if self.mock:
            return {"found": False, "confidence": "low", "reason": "模拟模式不进行真实物品比对。"}
        reference_id = str(reference.get("reference_id") or "")
        suffix = str(reference.get("suffix") or "").lower()
        reference_path = self.reference_path(reference_id, suffix)
        if not reference_path.exists():
            raise FileNotFoundError("参考图片不存在或已过期")
        if not scene_path.exists():
            raise FileNotFoundError("现场拍摄图片不存在")

        with self.infer_lock:
            from qwen_planner import (
                BASE_URL,
                VISION_MODEL,
                _client,
                _image_data_url,
            )

            # 所有模型输入统一预处理；两张图仍作为两个独立 image_url 发送。
            reference_url = _image_data_url(str(reference_path), VISION_MODEL)
            scene_url = _image_data_url(str(scene_path), VISION_MODEL)
            LOGGER.info(
                "[vision] compare_reference request model=%r base_url=%r input_mode=multi-image "
                "reasoning=slim-cot guidance=reference-image-only verification=disabled "
                "reference_bytes=%d scene_bytes=%d",
                VISION_MODEL,
                BASE_URL,
                reference_path.stat().st_size,
                scene_path.stat().st_size,
            )

            # 与 robot_client.py 已验证的瘦身 CoT 保持一致：
            # system 要求先做一句候选 grounding，再输出最终 JSON；
            # user 中两张图片仍独立输入，并全部放在文本之前。
            response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": FIND_TARGET_COT_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": reference_url}},
                            {"type": "image_url", "image_url": {"url": scene_url}},
                            {"type": "text", "text": FIND_TARGET_COT_USER},
                        ],
                    },
                ],
                max_tokens=FIND_TARGET_COT_MAX_TOKENS,
                temperature=0.0,
            )
        raw = _json_text(response.choices[0].message.content).strip()
        comparison = _parse_find_object_result(raw)
        LOGGER.info(
            "[vision] compare_reference response scene=%s found=%s confidence=%r raw=%r",
            scene_path,
            bool(comparison.get("found")),
            comparison.get("confidence"),
            raw[:2000],
        )
        return comparison

    def compare_person_reference(self, reference: dict, scene_path: Path) -> dict:
        """以两张独立图片比对迎宾人物外观，不向模型提供地点或身份文字先验。"""
        if self.mock:
            return {
                "found": True,
                "confidence": "high",
                "candidate_region": "模拟画面中央",
                "reason": "模拟模式固定命中目标人物。",
            }
        reference_id = str(reference.get("reference_id") or "")
        suffix = str(reference.get("suffix") or "").lower()
        reference_path = self.reference_path(reference_id, suffix)
        if not reference_path.exists():
            raise FileNotFoundError("迎宾人物参考图片不存在或已过期")
        if not scene_path.exists():
            raise FileNotFoundError("迎宾现场抓拍图片不存在")

        with self.infer_lock:
            from qwen_planner import BASE_URL, VISION_MODEL, _client, _image_data_url

            reference_url = _image_data_url(str(reference_path), VISION_MODEL)
            scene_url = _image_data_url(str(scene_path), VISION_MODEL)
            LOGGER.info(
                "[vision] compare_person_reference request model=%r base_url=%r "
                "input_mode=multi-image reasoning=two-stage-slim-cot interval=%gs "
                "reference_bytes=%d scene_bytes=%d",
                VISION_MODEL,
                BASE_URL,
                WELCOME_SNAPSHOT_INTERVAL_SECONDS,
                reference_path.stat().st_size,
                scene_path.stat().st_size,
            )
            response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": WELCOME_PERSON_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": reference_url}},
                            {"type": "image_url", "image_url": {"url": scene_url}},
                            {"type": "text", "text": WELCOME_PERSON_USER},
                        ],
                    },
                ],
                max_tokens=640,
                temperature=0.0,
            )
            reasoning_raw = _json_text(response.choices[0].message.content).strip()
            format_prompt = (
                "下面是上一阶段根据两张图片得到的观察摘要：\n"
                "<<<观察摘要开始>>>\n"
                f"{reasoning_raw[:4000]}\n"
                "<<<观察摘要结束>>>\n"
                "请把它转换为以下扁平JSON结构。没有明确证据的项目必须填“无法核对”并设为unknown：\n"
                f"{WELCOME_PERSON_JSON_SCHEMA}"
            )
            formatted_response = _client().chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": WELCOME_PERSON_JSON_SYSTEM},
                    {"role": "user", "content": format_prompt},
                ],
                max_tokens=640,
                temperature=0.0,
            )
        raw = _json_text(formatted_response.choices[0].message.content).strip()
        comparison = _parse_person_reference_result(raw)
        comparison["reasoning_summary"] = reasoning_raw[:2000]
        LOGGER.info(
            "[vision] compare_person_reference response scene=%s found=%s confidence=%r "
            "score=%.2f schema_valid=%s comparisons=%r reasoning=%r raw=%r",
            scene_path,
            bool(comparison.get("found")),
            comparison.get("confidence"),
            float(comparison.get("score") or 0.0),
            bool(comparison.get("schema_valid")),
            comparison.get("comparisons"),
            reasoning_raw[:2000],
            raw[:2000],
        )
        return comparison

    def compare_patrol_images(self, before_path: Path, after_path: Path, location_desc: str) -> dict:
        """比较同一巡逻点的前后画面，忽略轻微视角和光照变化，只报告物体级异常。"""
        if self.mock:
            return {"changed": False, "confidence": "high", "summary": "模拟比较未发现异常。", "changes": []}
        if not before_path.exists() or not after_path.exists():
            raise FileNotFoundError("巡逻基线图片或当前图片不存在")

        with self.infer_lock:
            from qwen_planner import VISION_MODEL, _client, _image_data_url, _is_minicpm_model

            is_minicpm = _is_minicpm_model(VISION_MODEL)
            before_url = _image_data_url(str(before_path), VISION_MODEL)
            after_url = _image_data_url(str(after_path), VISION_MODEL)
            patrol_prompt = (
                f"你是机器人巡逻的严格变化检测器。位置：{location_desc}。"
                "图片1是上次基线，图片2是本次画面。忽略轻微相机位置、透视、曝光、阴影和人员姿态差异；"
                "只把明确的物品新增、缺失、移动，门窗或设备状态改变、明显环境异常判为 changed=true。"
                "不确定时必须 changed=false 或 confidence=low。只输出 JSON："
                '{"changed":true,"confidence":"high|medium|low","summary":"中文结论",'
                '"changes":[{"type":"added|removed|moved|state|changed","item":"物品","detail":"变化说明"}]}'
            )
            if is_minicpm:
                response = _client().chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": before_url}},
                            {"type": "image_url", "image_url": {"url": after_url}},
                            {"type": "text", "text": patrol_prompt},
                        ],
                    }],
                    max_tokens=384,
                    temperature=0.0,
                )
            else:
                response = _vision_completion(
                    _client(),
                    model=VISION_MODEL,
                    messages=[
                        {"role": "system", "content": patrol_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": before_url}},
                                {"type": "image_url", "image_url": {"url": after_url}},
                                {"type": "text", "text": "请比较图片1和图片2。"},
                            ],
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
        return _parse_patrol_result(response.choices[0].message.content)

    def find_target_label(self, instruction: str, reference: dict) -> str:
        """优先使用用户文字；指令没有物品名时让视觉模型识别参考图。"""
        from_text = _find_target_label_from_instruction(instruction)
        if from_text:
            return from_text
        if self.mock:
            return "物品"
        try:
            reference_id = str(reference.get("reference_id") or "")
            suffix = str(reference.get("suffix") or "").lower()
            reference_path = self.reference_path(reference_id, suffix)
            if not reference_path.exists():
                return "物品"
            with self.infer_lock:
                from qwen_planner import VISION_MODEL, _client, _image_data_url, _is_minicpm_model

                image_url = _image_data_url(str(reference_path), VISION_MODEL)
                label_prompt = (
                    "识别这张参考图中的主要待寻找物，只输出一个简短中文物品名，"
                    f"不要标点、解释或 Markdown。用户寻物指令：{instruction or '未说明'}"
                )
                if _is_minicpm_model(VISION_MODEL):
                    request_kwargs = {
                        "model": VISION_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": label_prompt},
                            ],
                        }],
                        "max_tokens": 32,
                        "temperature": 0.0,
                    }
                else:
                    request_kwargs = {
                        "model": VISION_MODEL,
                        "messages": [
                            {"role": "system", "content": label_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_url}},
                                    {"type": "text", "text": "请识别参考物品。"},
                                ],
                            },
                        ],
                        "temperature": 0.0,
                    }
                response = None
                for attempt in range(2):
                    try:
                        response = _client().chat.completions.create(**request_kwargs)
                        break
                    except Exception as exc:
                        http_response = getattr(exc, "response", None)
                        status = (
                            getattr(exc, "status_code", None)
                            or getattr(http_response, "status_code", None)
                        )
                        if status != 500 or attempt > 0:
                            raise
                        LOGGER.warning(
                            "[find] reference label first vision request returned HTTP 500; retrying once"
                        )
                        time.sleep(0.5)
            label = re.sub(r"\s+", "", _json_text(response.choices[0].message.content)).strip("，。！？,.!?；; ")
            return label[:16] or "物品"
        except Exception as exc:
            _log_exception(
                "[find] reference label inference failed",
                exc,
                reference_id=reference.get("reference_id"),
            )
            return "物品"

    def speak(self, text: str):
        """复用 qwen_planner 的 announce + VoiceAssistant 队列，播报不阻塞任务线程。"""
        check_cancelled()
        text = str(text or "").strip()
        if not text:
            return
        if self.mock:
            print(f"[tts mock] {text}")
            return
        try:
            with self.voice_lock:
                if self.voice is None:
                    from voice import VoiceAssistant

                    self.voice = VoiceAssistant()
                voice = self.voice
                import qwen_planner

                qwen_planner._VOICE = voice
            qwen_planner.announce(text)
        except Exception as exc:
            print(f"[TTS 失败] {exc}")

    def ask_and_listen(self, prompt: str) -> str:
        """同步播完迎宾询问后收听一句，避免把喇叭余音误识别为对方回答。"""
        check_cancelled()
        prompt = str(prompt or "").strip()
        if self.mock:
            print(f"[tts mock] {prompt}")
            time.sleep(0.3)
            return os.getenv("JAKA_WEB_MOCK_WELCOME_ANSWER", "是的").strip()
        with self.voice_lock:
            check_cancelled()
            if self.voice is None:
                from voice import VoiceAssistant

                self.voice = VoiceAssistant()
            print(prompt)
            self.voice.say(prompt, wait=True)
            time.sleep(0.25)
            return (guarded_call(self.voice.listen_utterance) or "").strip()

    def listen(self) -> str:
        """用树莓派麦克风听一句话并返回文字; 不在网页请求线程间并发录音。"""
        if self.mock:
            # 仅供没有麦克风/语音模型的电脑测试网页按钮。可用环境变量换成任意测试句，
            # 不参与树莓派真实模式; 真实模式始终返回用户当次说出的自然语言。
            time.sleep(0.7)
            return os.getenv("JAKA_WEB_MOCK_VOICE_TEXT", "这是一条模拟语音输入")
        with self.voice_lock:
            if self.voice is None:
                from voice import VoiceAssistant

                self.voice = VoiceAssistant()
            return (self.voice.listen_utterance() or "").strip()

    def transcribe_mobile_audio(self, audio_bytes: bytes) -> dict:
        """识别手机浏览器上传的录音，不保存原始音频。"""
        request_id = uuid.uuid4().hex
        if self.mock:
            time.sleep(0.1)
            text = os.getenv("JAKA_WEB_MOCK_VOICE_TEXT", "这是一条模拟语音输入").strip()
            if not text:
                raise AudioRequestError("没有识别到有效语音", HTTPStatus.UNPROCESSABLE_ENTITY)
            return {"text": text, "duration_ms": 0, "request_id": request_id}

        pcm, duration_ms = _decode_mobile_audio(audio_bytes)
        with self.voice_lock:
            if self.voice is None:
                from voice import VoiceAssistant

                self.voice = VoiceAssistant()
            text = (self.voice.transcribe_pcm16(pcm, sr=16000) or "").strip()
        if not text:
            raise AudioRequestError("没有识别到有效语音", HTTPStatus.UNPROCESSABLE_ENTITY)
        return {"text": text, "duration_ms": duration_ms, "request_id": request_id}

    def close(self):
        task = self.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            try:
                self.tasks.cancel(task["id"])
            except Exception:
                pass
        self.tasks.close()
        if self.voice is not None:
            voice = self.voice
            try:
                import qwen_planner

                if qwen_planner._VOICE is voice:
                    qwen_planner._VOICE = None
            except Exception:
                pass
            try:
                voice.close()
            except Exception:
                pass
            self.voice = None


# ======================================================================
#  单页界面
# ======================================================================
PAGE_PATH = HERE / "robot_web_page.html"
SCENE_VIEWER_PATH = HERE / "scene_viewer.html"
PWA_MANIFEST_PATH = HERE / "manifest.webmanifest"
SERVICE_WORKER_PATH = HERE / "service-worker.js"
PWA_ICON_PATH = HERE / "pwa-icon.svg"


def _document() -> bytes:
    """读取前端片段并组装完整 HTML 文档。"""
    try:
        page_html = PAGE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"前端页面文件不可用: {PAGE_PATH}") from exc
    head = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#f6f7f8">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <meta name="apple-mobile-web-app-title" content="JAKA Vision">
  <link rel="manifest" href="/manifest.webmanifest">
  <link rel="icon" href="/pwa-icon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/pwa-icon.svg">
  <title>JAKA Vision</title>
</head>
<body>
"""
    return (head + page_html + "\n</body>\n</html>\n").encode("utf-8")


# ======================================================================
#  HTTP API
# ======================================================================
class RobotWebHandler(BaseHTTPRequestHandler):
    server_version = "JakaVision/1.0"

    @property
    def state(self) -> RobotWebState:
        return self.server.state

    def log_message(self, fmt, *args):
        LOGGER.info("[http] %s %s - %s", self.command, self.path, fmt % args)

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Permissions-Policy", "microphone=(self), camera=(self)")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "media-src 'self' blob:; connect-src 'self'; worker-src 'self'; manifest-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com",
        )
        self.end_headers()

    def _send_bytes(self, body, content_type, status=HTTPStatus.OK):
        self._headers(status, content_type, len(body))
        self.wfile.write(body)

    def _send_json(self, value, status=HTTPStatus.OK):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _error(self, message, status=HTTPStatus.BAD_REQUEST):
        self._send_json({"error": str(message)}, status)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("无效 Content-Length") from exc
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("请求体为空或过大")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是合法 JSON") from exc

    def _read_reference_image(self) -> bytes:
        """读取与 /captures 相同的原始图片字节 HTTP body。"""
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type not in {info[0] for info in REFERENCE_IMAGE_TYPES.values()}:
            raise ValueError("参考图片 Content-Type 不支持")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("无效 Content-Length") from exc
        if length <= 0 or length > MAX_REFERENCE_BYTES:
            raise ValueError("参考图片请求体为空或过大")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("参考图片上传不完整")
        return payload

    def _read_mobile_audio(self) -> bytes:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in AUDIO_CONTENT_TYPES:
            raise AudioRequestError("录音格式不支持", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise AudioRequestError("无效 Content-Length", HTTPStatus.BAD_REQUEST) from exc
        if length <= 0:
            raise AudioRequestError("录音内容为空", HTTPStatus.BAD_REQUEST)
        if length > MAX_AUDIO_BYTES:
            raise AudioRequestError(
                f"录音不能超过 {MAX_AUDIO_BYTES // (1024 * 1024)} MB",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise AudioRequestError("录音上传不完整", HTTPStatus.BAD_REQUEST)
        return payload

    def do_GET(self):
        request_url = urlparse(self.path)
        path = request_url.path
        if path == "/":
            self._send_bytes(_document(), "text/html; charset=utf-8")
            return
        static_assets = {
            "/manifest.webmanifest": (PWA_MANIFEST_PATH, "application/manifest+json; charset=utf-8"),
            "/service-worker.js": (SERVICE_WORKER_PATH, "text/javascript; charset=utf-8"),
            "/pwa-icon.svg": (PWA_ICON_PATH, "image/svg+xml; charset=utf-8"),
        }
        if path in static_assets:
            asset_path, content_type = static_assets[path]
            try:
                self._send_bytes(asset_path.read_bytes(), content_type)
            except OSError as exc:
                self._error(f"PWA 资源不可用: {exc}", HTTPStatus.NOT_FOUND)
            return
        if path in {"/scene-viewer", "/scene-viewer/", "/scene-viewer.html"}:
            try:
                self._send_bytes(SCENE_VIEWER_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self._error(f"三维拟物体查看页不可用: {exc}", HTTPStatus.NOT_FOUND)
            return
        if path == "/api/health":
            self._send_json({"ok": True, "mock": self.state.mock})
            return
        if path == "/api/skills":
            self._send_json({"skills": skill_catalog()})
            return
        if path == "/api/maps":
            graph = self.state.graph_snapshot()
            self._send_json({"current": graph["name"], "files": self.state.map_files()})
            return
        if path == "/api/map":
            self._send_json(self.state.graph_snapshot())
            return
        if path == "/api/slam":
            self._send_json(self.state.slam_snapshot())
            return
        if path == "/slam-map/image":
            try:
                body, content_type = self.state.slam_image()
                self._send_bytes(body, content_type)
            except FileNotFoundError as exc:
                self._error(exc, HTTPStatus.NOT_FOUND)
            return
        if path == "/api/robot/status":
            self._send_json({
                "robot": self.state.tasks.robot_status(),
                "task": self.state.tasks.snapshot(),
            })
            return
        if path == "/api/tracks":
            self._send_json(self.state.tasks.tracks_snapshot())
            return
        if path == "/api/task/status":
            self._send_json({"task": self.state.tasks.snapshot()})
            return
        if path == "/api/mapping/sessions":
            self._send_json(self.state.mapping.sessions_snapshot())
            return
        if path == "/api/mapping/status":
            session_id = (parse_qs(request_url.query).get("session_id") or [None])[0]
            self._send_json(self.state.mapping.snapshot(session_id))
            return
        if path == "/api/mapping/preview":
            query = parse_qs(request_url.query)
            session_id = (query.get("session_id") or [""])[0]
            name = (query.get("name") or [""])[0]
            try:
                preview_path = self.state.mapping.preview_path(session_id, name)
                content_type = mimetypes.guess_type(preview_path.name)[0] or "image/jpeg"
                self._send_bytes(preview_path.read_bytes(), content_type)
            except FileNotFoundError as exc:
                self._error(exc, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._error(exc, HTTPStatus.BAD_REQUEST)
            return
        icon_match = re.fullmatch(r"/map-icons/([a-z]+)\.svg", path)
        if icon_match:
            name = icon_match.group(1)
            if name not in MAP_ICON_NAMES:
                self._error("图标不存在", HTTPStatus.NOT_FOUND)
                return
            icon_path = MAP_ICON_DIR / f"{name}.svg"
            if not icon_path.exists():
                self._error("图标文件不存在", HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(icon_path.read_bytes(), "image/svg+xml; charset=utf-8")
            return
        match = re.fullmatch(r"/captures/([0-9a-f]{32})\.jpg", path)
        if match:
            image_path = self.state.capture_path(match.group(1))
            if not image_path.exists():
                self._error("图片不存在", HTTPStatus.NOT_FOUND)
                return
            body = image_path.read_bytes()
            content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            self._send_bytes(body, content_type)
            return
        video_match = re.fullmatch(r"/videos/([0-9a-f]{32})\.(mp4|avi)", path)
        if video_match:
            video_id, suffix = video_match.groups()
            video_path = VIDEO_DIR / f"{video_id}.{suffix}"
            if not video_path.exists():
                self._error("视频不存在", HTTPStatus.NOT_FOUND)
                return
            content_type = "video/mp4" if suffix == "mp4" else "video/x-msvideo"
            self._send_bytes(video_path.read_bytes(), content_type)
            return
        reference_match = re.fullmatch(r"/references/([0-9a-f]{32})(\.(?:png|jpg|jpeg|webp))", path)
        if reference_match:
            reference_id, suffix = reference_match.groups()
            try:
                image_path = self.state.reference_path(reference_id, suffix)
            except ValueError as exc:
                self._error(exc, HTTPStatus.NOT_FOUND)
                return
            if not image_path.exists():
                self._error("参考图片不存在", HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(image_path.read_bytes(), REFERENCE_IMAGE_TYPES[suffix][0])
            return
        self._error("接口不存在", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        request_url = urlparse(self.path)
        path = request_url.path
        try:
            if path == "/api/audio/transcribe":
                audio_bytes = self._read_mobile_audio()
                self._send_json(self.state.transcribe_mobile_audio(audio_bytes))
                return
            if path == "/api/reference/upload" and self.headers.get("Content-Type", "").lower().startswith("image/"):
                name = (parse_qs(request_url.query).get("name") or [""])[0]
                image_bytes = self._read_reference_image()
                self._send_json(self.state.save_reference_upload(name, image_bytes))
                return
            payload = self._read_json()
            if path == "/api/agent/chat":
                if not isinstance(payload, dict):
                    raise ValueError("请求体必须是对象")
                if self.headers.get("Accept") != "application/x-ndjson":
                    self._send_json(self.state.agent_chat(payload))
                    return
                # Close-delimited NDJSON: flush progress immediately, never cache robot data.
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def emit(event):
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()

                try:
                    result = self.state.agent_chat(payload, emit)
                    emit({"type": "final", **result})
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as exc:
                    LOGGER.warning("[agent] request failed type=%s", type(exc).__name__)
                    try:
                        emit({"type": "error", "text": str(exc) if isinstance(exc, ValueError) else "对话服务暂时不可用，请稍后重试；机器人没有通过此对话自动执行动作。"})
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                return
            if path == "/api/capture":
                capture_id, _ = self.state.capture()
                self._send_json({
                    "capture_id": capture_id,
                    "image_url": f"/captures/{capture_id}.jpg",
                })
                return
            if path == "/api/route":
                question = str(payload.get("question") or "").strip()
                if not question:
                    raise ValueError("问题不能为空")
                history = payload.get("history") or []
                if not isinstance(history, list):
                    raise ValueError("history 必须是列表")
                has_reference = payload.get("has_reference", False)
                if not isinstance(has_reference, bool):
                    raise ValueError("has_reference 必须是布尔值")
                LOGGER.info(
                    "[route] request question=%r has_reference=%s history_items=%d",
                    question,
                    has_reference,
                    len(history),
                )
                decision = self.state.route(question, history, has_reference=has_reference)
                self._send_json({
                    "mode": decision["mode"],
                    "needs_camera": decision["mode"] == "vision",
                    "text": decision["answer"],
                    "target_ids": decision.get("target_ids") or [],
                })
                return
            if path == "/api/reference/upload":
                name = str(payload.get("name") or "")
                encoded = str(payload.get("image_base64") or "")
                if not name or not encoded:
                    raise ValueError("请提供参考图片文件名和内容")
                self._send_json(self.state.save_reference_image(name, encoded))
                return
            if path == "/api/map/select":
                graph = self.state.select_map(str(payload.get("name") or ""))
                self._send_json(graph)
                return
            if path == "/api/map/upload":
                graph_data = payload.get("graph")
                if not isinstance(graph_data, dict):
                    raise ValueError("graph 必须是 JSON 对象")
                graph = self.state.upload_map(str(payload.get("name") or "浏览器地图.json"), graph_data)
                self._send_json(graph)
                return
            if path == "/api/slam/upload":
                name = str(payload.get("name") or "slam.png")
                encoded = str(payload.get("image_base64") or "")
                self._send_json(self.state.upload_slam_image(name, encoded))
                return
            if path == "/api/slam/calibration":
                calibration = payload.get("calibration")
                if not isinstance(calibration, dict):
                    raise ValueError("calibration 必须是对象")
                self._send_json(self.state.update_slam_calibration(calibration))
                return
            if path == "/api/slam/refresh":
                self._send_json(self.state.refresh_slam_from_chassis())
                return
            if path == "/api/slam/reload-local":
                self._send_json(self.state.reload_local_slam())
                return
            if path == "/api/slam/auto-calibrate":
                self._send_json(self.state.reset_slam_calibration())
                return
            if path == "/api/task/plan":
                instruction = str(payload.get("instruction") or "").strip()
                if not instruction:
                    raise ValueError("任务指令不能为空")
                self._send_json({"task": self.state.tasks.plan(instruction)})
                return
            if path == "/api/task/find-object/plan":
                instruction = str(payload.get("instruction") or "").strip()
                reference = payload.get("reference")
                if not instruction:
                    raise ValueError("寻物指令不能为空")
                if not isinstance(reference, dict):
                    raise ValueError("请先上传参考图片")
                reference_id = str(reference.get("reference_id") or "")
                suffix = str(reference.get("suffix") or "").lower()
                reference_path = self.state.reference_path(reference_id, suffix)
                if not reference_path.exists():
                    raise FileNotFoundError("参考图片不存在或已过期")
                content_type = REFERENCE_IMAGE_TYPES[suffix][0]
                expected_url = f"/references/{reference_id}{suffix}"
                if reference.get("image_url") != expected_url or reference.get("content_type") != content_type:
                    raise ValueError("参考图片元信息无效")
                safe_reference = {
                    "reference_id": reference_id,
                    "suffix": suffix,
                    "content_type": content_type,
                    "image_url": expected_url,
                }
                self._send_json({"task": self.state.tasks.plan_skill("find_object", instruction, reference=safe_reference)})
                return
            if path == "/api/task/welcome/plan":
                instruction = str(payload.get("instruction") or "").strip()
                reference = payload.get("reference")
                if not instruction:
                    raise ValueError("迎宾指令不能为空")
                if not isinstance(reference, dict):
                    raise ValueError("请先上传目标人物的参考照片")
                reference_id = str(reference.get("reference_id") or "")
                suffix = str(reference.get("suffix") or "").lower()
                reference_path = self.state.reference_path(reference_id, suffix)
                if not reference_path.exists():
                    raise FileNotFoundError("人物参考图片不存在或已过期")
                content_type = REFERENCE_IMAGE_TYPES[suffix][0]
                expected_url = f"/references/{reference_id}{suffix}"
                if (
                    reference.get("image_url") != expected_url
                    or reference.get("content_type") != content_type
                ):
                    raise ValueError("人物参考图片元信息无效")
                try:
                    pickup_ann_id = int(payload.get("pickup_ann_id"))
                    return_ann_id = int(payload.get("return_ann_id"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("请在地图中分别选择接人点和返回点") from exc
                safe_reference = {
                    "reference_id": reference_id,
                    "suffix": suffix,
                    "content_type": content_type,
                    "image_url": expected_url,
                }
                self._send_json({
                    "task": self.state.tasks.plan_skill(
                        "welcome", instruction,
                        reference=safe_reference,
                        pickup_ann_id=pickup_ann_id,
                        return_ann_id=return_ann_id,
                    )
                })
                return
            if path == "/api/task/execute":
                task_id = str(payload.get("task_id") or "")
                self._send_json({"task": self.state.tasks.execute(task_id)})
                return
            if path == "/api/task/cancel":
                task_id = str(payload.get("task_id") or "")
                self._send_json({"task": self.state.tasks.cancel(task_id)})
                return
            if path == "/api/mapping/plan":
                self._send_json({"session": self.state.mapping.plan(payload)})
                return
            if path == "/api/mapping/start":
                session_id = str(payload.get("session_id") or "")
                self._send_json({"session": self.state.mapping.start(session_id, resume=False)})
                return
            if path == "/api/mapping/resume":
                session_id = str(payload.get("session_id") or "")
                self._send_json({"session": self.state.mapping.start(session_id, resume=True)})
                return
            if path == "/api/mapping/cancel":
                session_id = str(payload.get("session_id") or "")
                self._send_json({"session": self.state.mapping.cancel(session_id)})
                return
            if path == "/api/mapping/apply":
                session_id = str(payload.get("session_id") or "")
                self._send_json({"map": self.state.mapping.apply(session_id)})
                return
            if path == "/api/tracks/save":
                track = self.state.tasks.save_current_track(str(payload.get("name") or ""))
                result = self.state.tasks.tracks_snapshot()
                result["track"] = track
                self._send_json(result)
                return
            if path == "/api/tracks/delete":
                track_id = str(payload.get("track_id") or "")
                if not track_id:
                    raise ValueError("缺少轨迹 ID")
                self._send_json(self.state.tasks.delete_saved_track(track_id))
                return
            if path == "/api/tracks/clear-current":
                self._send_json(self.state.tasks.clear_current_track())
                return
            if path == "/api/infer":
                capture_id = str(payload.get("capture_id") or "")
                question = str(payload.get("question") or "").strip()
                if not question:
                    raise ValueError("问题不能为空")
                history = payload.get("history") or []
                if not isinstance(history, list):
                    raise ValueError("history 必须是列表")
                answer = self.state.infer(capture_id, question, history)
                self._send_json({"text": answer})
                return
            if path == "/api/listen":
                self._send_json({"text": self.state.listen()})
                return
            self._error("接口不存在", HTTPStatus.NOT_FOUND)
        except TaskSafetyError as exc:
            LOGGER.warning("[task] execution blocked: %s", exc)
            self._error(exc, HTTPStatus.CONFLICT)
        except AudioRequestError as exc:
            self._error(exc, exc.status)
        except FileNotFoundError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._error(exc, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            _log_exception(
                "[http] POST request failed",
                exc,
                method="POST",
                path=path,
                client=self.address_string(),
            )
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self):
        request_url = urlparse(self.path)
        if request_url.path != "/api/mapping/session":
            self._error("接口不存在", HTTPStatus.NOT_FOUND)
            return
        session_id = (parse_qs(request_url.query).get("session_id") or [""])[0]
        try:
            self._send_json(self.state.mapping.delete(session_id))
        except ValueError as exc:
            self._error(exc, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self._error(exc, HTTPStatus.CONFLICT)
        except Exception as exc:
            _log_exception(
                "[http] DELETE request failed",
                exc,
                method="DELETE",
                path=request_url.path,
                client=self.address_string(),
            )
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)


class RobotWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, state):
        super().__init__(address, handler)
        self.state = state

    def handle_error(self, request, client_address):
        """记录未被请求处理器捕获的异常，避免只在终端看到连接断开。"""
        exc_type, exc, _ = sys.exc_info()
        if exc is not None:
            _log_exception(
                "[http] unhandled request exception",
                exc,
                method=getattr(request, "command", None),
                client=client_address,
                exception_type=getattr(exc_type, "__name__", None),
            )
        else:
            LOGGER.error("[http] unhandled request exception client=%r", client_address)


def main():
    parser = argparse.ArgumentParser(description="JAKA 视觉助手单文件 Web 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mock", action="store_true", help="不连接相机/模型/麦克风")
    args = parser.parse_args()

    from robot_model_config import load_model_config
    model_config = load_model_config()
    for key, value in sorted(model_config.items()):
        LOGGER.info("[models] %s=%s", key, value)

    state = RobotWebState(mock=args.mock)
    server = RobotWebServer((args.host, args.port), RobotWebHandler, state)
    print(f"[web] JAKA Vision 正在监听: http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"[web] 远程浏览器请访问: http://<树莓派IP>:{args.port}")
        print(f"[web] SSH 转发后请访问: http://127.0.0.1:{args.port}")
    print(f"[web] mode={'mock' if args.mock else 'hardware'} | Ctrl+C 退出")
    LOGGER.info("[web] backend log file: %s", LOG_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 正在关闭")
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
