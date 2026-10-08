"""Read model, robot and observation settings from the process environment."""
from __future__ import annotations
import os

API_KEY      = os.getenv("DASHSCOPE_API_KEY", "not-needed")


BASE_URL     = os.getenv("DASHSCOPE_BASE_URL", "http://localhost:8000/v1")


PLAN_MODEL   = os.getenv("QWEN_PLAN_MODEL",   "/root/Model/MiniCPM-V-4.6")


VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "/root/Model/MiniCPM-V-4.6")


JAKA_HOST = os.getenv("JAKA_HOST", "192.168.10.10")


JAKA_PORT = int(os.getenv("JAKA_PORT", "31001"))


JAKA_HTTP_PORT = int(os.getenv("JAKA_HTTP_PORT", "9001"))   # agv.py: 底盘 HTTP 通道


CAM_FX = float(os.getenv("JAKA_CAM_FX", "636.9331510849437"))


CAM_W = int(os.getenv("JAKA_CAM_W", "1280"))


NAV_MIN_CENTER_DISTANCE_M = float(os.getenv("JAKA_NAV_MIN_CENTER_DISTANCE_M", "0.35"))


NAV_FACE_MAX_DELTA_DEG = min(
    35.0, max(0.0, float(os.getenv("JAKA_NAV_FACE_MAX_DELTA_DEG", "35"))),
)


NAV_GOAL_THETA_TOLERANCE_DEG = min(
    10.0, max(1.0, float(os.getenv("JAKA_NAV_GOAL_THETA_TOLERANCE_DEG", "10"))),
)


NAV_APPROACH_MAX_HEADING_DELTA_DEG = min(
    90.0, max(5.0, float(os.getenv("JAKA_NAV_APPROACH_MAX_HEADING_DELTA_DEG", "45"))),
)


YAW_SIGN = int(os.getenv("JAKA_YAW_SIGN", "-1"))


REFRAME_MAX_DYAW = float(os.getenv("JAKA_REFRAME_MAX_DYAW", "0.5"))


CENTERED_THRESH = float(os.getenv("JAKA_CENTERED_THRESH", "0.10"))


APPROACH_STEP    = float(os.getenv("JAKA_APPROACH_STEP", "0.5"))


APPROACH_ABS_MIN = float(os.getenv("JAKA_APPROACH_ABS_MIN", "0.35"))


APPROACH_MARGIN  = float(os.getenv("JAKA_APPROACH_MARGIN", "0.05"))


APPROACH_MAX_TOTAL = float(os.getenv("JAKA_APPROACH_MAX_TOTAL", "1.5"))


APPROACH_MAX_TRIES = min(4, max(1, int(os.getenv("JAKA_APPROACH_MAX_TRIES", "5"))))


OBSERVE_MOVE_TOL = float(os.getenv("JAKA_OBSERVE_MOVE_TOL", "0.05"))


OBSERVE_NEAR_CROP_DISTANCE = min(
    1.50, max(0.0, float(os.getenv("JAKA_OBSERVE_NEAR_CROP_DISTANCE", "1.50"))),
)


OBSERVE_CROP_RETREAT_STEP = min(
    0.30, max(0.0, float(os.getenv("JAKA_OBSERVE_CROP_RETREAT_STEP", "0.30"))),
)


OBSERVE_MAX_RETREAT = min(
    1.50, max(0.0, float(os.getenv("JAKA_OBSERVE_MAX_RETREAT", "1.50"))),
)
