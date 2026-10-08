"""Web limits, task thresholds and runtime resource paths."""
from __future__ import annotations
import math
import os
import re
from urllib.parse import parse_qs, urljoin, urlparse

from jaka_agent import paths

HERE = paths.DATA_DIR


CAPTURE_DIR = paths.runtime_path("web_captures")


VIDEO_DIR = paths.runtime_path("web_videos")


MAP_ICON_DIR = paths.STATIC_DIR / "map_icons"


SLAM_CONFIG_PATH = paths.runtime_path("slam_calibration.json")


TRACK_HISTORY_PATH = paths.runtime_path("robot_tracks.json")


MAPPING_RUNS_DIR = paths.runtime_path("mapping_runs")


MAPPING_SERVER = os.getenv("JAKA_MAPPING_SERVER", "tcp://127.0.0.1:5560").strip()


MAPPING_RESULT_TIMEOUT = float(os.getenv("JAKA_MAPPING_RESULT_TIMEOUT", "1800"))


HEAD_CAMERA_SN = os.getenv("JAKA_HEAD_CAM_SN", "AY8V74300DG").strip()


HAND_CAMERA_SN = os.getenv("JAKA_HAND_CAM_SN", "AY8V74300VF").strip()


MAX_BODY_BYTES = 20 * 1024 * 1024


MAX_REFERENCE_BYTES = 6 * 1024 * 1024


MAX_AUDIO_BYTES = 8 * 1024 * 1024


MAX_AUDIO_SECONDS = 30


TASK_VIDEO_ENABLED = os.getenv("JAKA_RECORD_TASK_VIDEO", "1").strip().lower() not in {"0", "false", "no", "off"}


TASK_VIDEO_FPS = max(0.5, min(15.0, float(os.getenv("JAKA_TASK_VIDEO_FPS", "5"))))


TASK_VIDEO_MAX_WIDTH = max(320, int(os.getenv("JAKA_TASK_VIDEO_MAX_WIDTH", "960")))


PATROL_POSE_DISTANCE_TOLERANCE_M = 0.08


PATROL_POSE_THETA_TOLERANCE_RAD = math.radians(3)


FIND_SNAPSHOT_INTERVAL_SECONDS = max(0.5, float(os.getenv("JAKA_FIND_SNAPSHOT_INTERVAL_SECONDS", "1")))


FIND_SNAPSHOT_ATTEMPTS_PER_POINT = max(1, int(os.getenv("JAKA_FIND_SNAPSHOT_ATTEMPTS_PER_POINT", "2")))


FIND_SNAPSHOT_MAX_ERRORS = max(1, int(os.getenv("JAKA_FIND_SNAPSHOT_MAX_ERRORS", "3")))


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
    "JSON 示例中的‘参考图衣着’‘现场衣着’‘候选位置或无’等只是占位符，绝不能原样输出；"
    "无法从摘要确定的字段写‘无法核对’，相应比较写 unknown。"
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


WELCOME_PRESENCE_PROMPT = (
    "只查看这一张现场照片，不要考虑参考图或之前的对话。画面里是否存在真实的人？"
    "海报、标识、反光和人形图案都不是人。看不清时按无人处理。"
    '仅返回 JSON：{"person_visible":false,"person_count":0,'
    '"visible_evidence":"实际可见的人体部位和衣着；无人则为空"}'
)


WELCOME_RETRY_PROMPT = (
    "图1是目标人物参考照，图2是现场照片。只根据这两张图里的真实可见细节比对人物。"
    "先分别观察两图，不根据地点猜测身份。返回一个JSON对象，包含以下字段："
    "candidate_visible（布尔）、candidate_region（图2人物的具体位置）、"
    "reference_upper_clothing、candidate_upper_clothing、upper_clothing、"
    "reference_face_hair、candidate_face_hair、face_hair、"
    "reference_accessories、candidate_accessories、accessories、"
    "reference_body_shape、candidate_body_shape、body_shape、"
    "contradictions（字符串数组）、reason。四个对比字段只能填match、mismatch或unknown。"
    "描述字段必须写各图实际看到的颜色、款式或部位，绝不能写字段名、模板占位词或泛称；"
    "看不清写‘无法核对’并将比较设为unknown。不确定时不要猜测。仅输出JSON。"
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


AUDIO_CONTENT_TYPES = {
    "audio/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


DEFAULT_SLAM_IMAGE_URL = "http://192.168.10.10:8809/maps/demo/3/map.png"


SLAM_IMAGE_URL = os.getenv("JAKA_SLAM_IMAGE_URL", DEFAULT_SLAM_IMAGE_URL).strip()


SLAM_YAML_URL = os.getenv("JAKA_SLAM_YAML_URL", urljoin(SLAM_IMAGE_URL, "map.yaml")).strip()


SLAM_FETCH_TIMEOUT = float(os.getenv("JAKA_SLAM_FETCH_TIMEOUT", "5"))


PAGE_PATH = paths.TEMPLATES_DIR / "index.html"


SCENE_VIEWER_PATH = paths.TEMPLATES_DIR / "scene_viewer.html"


PWA_MANIFEST_PATH = paths.STATIC_DIR / "manifest.webmanifest"


SERVICE_WORKER_PATH = paths.STATIC_DIR / "service-worker.js"


PWA_ICON_PATH = paths.STATIC_DIR / "pwa-icon.svg"
