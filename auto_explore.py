#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动探索采集 RGB-D + 位姿，并为 BoxFusion 生成拟物体地图数据。

运行逻辑：
  1. 从底盘网页下载 map.yaml 和 map.png；
  2. 根据占据栅格、机器人安全半径生成可达覆盖点；
  3. 依次导航到覆盖点，并在每个点旋转采集 RGB-D；
  4. 按 received_frames 格式保存图片、深度和 pose_4x4；
  5. 可选地通过 ZMQ 实时发送给运行 BoxFusion 的电脑。

默认只生成计划，不会移动机器人。确认计划后加 --execute 才执行。
"""

import argparse
import copy
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

import cv2
import numpy as np

try:
    import zmq
except Exception:
    zmq = None

try:
    from pyorbbecsdk import Config, Context, OBAlignMode, OBSensorType, Pipeline
    ORBBEC_IMPORT_ERROR = None
except Exception as exc:  # 本地 --mock/路线预览不要求安装树莓派相机 SDK。
    Config = Context = OBAlignMode = OBSensorType = Pipeline = None
    ORBBEC_IMPORT_ERROR = exc


# ======================================================================
#  默认配置：与 jaka_step.py / convert_local_to_ca1m_v4.py 保持一致
# ======================================================================
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 800
DEFAULT_CAMERA_FPS = 30
# 当前 RGB-D 采集统一使用头部相机。DG 是头部相机；VF 备注为手部相机。
HEAD_CAMERA_SN = os.getenv("JAKA_HEAD_CAM_SN", "AY8V74300DG").strip()
HAND_CAMERA_SN = os.getenv("JAKA_HAND_CAM_SN", "AY8V74300VF").strip()
EXPECTED_HW = (800, 1280)

R_CAM2ROBOT = np.array([
    [-0.0032137179781592806, -0.12040115187173750, 0.99272011898858770],
    [-0.99999357528247920, 0.00196328844753335, -0.00299914858522554],
    [-0.0015878949969485023, -0.99272337945997350, -0.12040668778372776],
], dtype=np.float64)
CAM_HEIGHT_M = 1.464131
CAMERA_OFFSET_XY = (0.0, 0.0)

DEFAULT_AGV_HOST = "192.168.10.10"
DEFAULT_AGV_HTTP_PORT = 9001
DEFAULT_AGV_TCP_PORT = 31001
DEFAULT_MAP_YAML_URL = "http://192.168.10.10:8809/maps/demo/3/map.yaml"
DEFAULT_OUT = "received_frames"
JPEG_MAGIC = b"\xff\xd8\xff"


# ======================================================================
#  底盘通信：HTTP 优先，失败后自动回退 TCP
# ======================================================================
def load_agv_defaults(config_path: str | None):
    host, http_port, tcp_port = DEFAULT_AGV_HOST, DEFAULT_AGV_HTTP_PORT, DEFAULT_AGV_TCP_PORT
    path = Path(config_path) if config_path else Path("conf") / "userCmdControl.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        system = config.get("systemConfig", {}) or {}
        host = system.get("agv_ip", host)
        http_port = int(system.get("agv_http_port", http_port))
        tcp_port = int(system.get("agv_port", tcp_port))
    except Exception:
        pass
    return str(host), int(http_port), int(tcp_port)


def _http_json(host: str, port: int, path: str, timeout: float) -> dict:
    request = Request(f"http://{host}:{port}{path}", headers={"User-Agent": "JAKA-AutoExplore/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _tcp_json(host: str, port: int, path: str, timeout: float) -> dict:
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((path.rstrip("\r\n") + "\r\n").encode("utf-8"))
        payload = bytearray()
        depth = 0
        started = False
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            payload.extend(chunk)
            for byte in chunk:
                if byte == ord("{"):
                    depth += 1
                    started = True
                elif byte == ord("}") and started:
                    depth -= 1
            if started and depth <= 0:
                break
        text = bytes(payload).decode("utf-8", errors="ignore")
        left, right = text.find("{"), text.rfind("}")
        if left < 0 or right <= left:
            raise RuntimeError("TCP 未返回完整 JSON")
        return json.loads(text[left:right + 1])


def call_api(host: str, http_port: int, tcp_port: int, path: str, proto="auto", timeout=4.0):
    methods = ["http", "tcp"] if proto == "auto" else [proto]
    last_error = None
    for method in methods:
        try:
            if method == "http":
                return method, _http_json(host, http_port, path, timeout)
            return method, _tcp_json(host, tcp_port, path, timeout)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"底盘 API 请求失败: {last_error}")


def api_ok(data: dict) -> bool:
    return str(data.get("status", "")).upper() in {"OK", "SUCCEEDED"}


def status_pose(data: dict):
    results = data.get("results", {}) or {}
    pose = results.get("current_pose", {}) or {}
    try:
        return float(pose["x"]), float(pose["y"]), float(pose["theta"]), results
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"底盘状态没有有效 current_pose: {data}") from exc


def read_pose(args):
    _, data = call_api(args.agv_host, args.agv_http_port, args.agv_tcp_port, "/api/robot_status", args.proto)
    if not api_ok(data):
        raise RuntimeError(f"读取底盘状态失败: {data}")
    return status_pose(data)


def cancel_move(args):
    try:
        call_api(args.agv_host, args.agv_http_port, args.agv_tcp_port, "/api/move/cancel", args.proto)
    except Exception as exc:
        print(f"[WARN] 取消移动失败: {exc}")


def move_to(args, x: float, y: float, theta: float) -> bool:
    query = urlencode({
        "location": f"{x:.6f},{y:.6f},{theta:.6f}",
        "distance_tolerance": args.distance_tolerance,
        "theta_tolerance": args.theta_tolerance,
    }, safe=",")
    _, response = call_api(args.agv_host, args.agv_http_port, args.agv_tcp_port, f"/api/move?{query}", args.proto)
    if not api_ok(response):
        print(f"[MOVE] 请求失败: {response}")
        return False

    print(f"[MOVE] target=({x:.2f}, {y:.2f}, {theta:.2f})")
    start = time.time()
    saw_running = False
    while time.time() - start < args.move_timeout:
        _, status = call_api(args.agv_host, args.agv_http_port, args.agv_tcp_port, "/api/robot_status", args.proto, timeout=3.0)
        results = status.get("results", {}) or {}
        move_status = str(results.get("move_status") or "").lower()
        running_status = str(results.get("running_status") or "").lower()
        if move_status in {"running", "moving", "executing"} or running_status in {"running", "moving", "executing"}:
            saw_running = True
        if move_status in {"failed", "aborted", "canceled", "error"}:
            print(f"[MOVE] 底盘返回失败状态: {move_status}")
            return False
        if (saw_running or time.time() - start > 1.0) and move_status in {"idle", "stopped", "succeeded"} and running_status in {"idle", "stopped"}:
            return True
        time.sleep(args.status_interval)
    print("[MOVE] 等待到达超时")
    cancel_move(args)
    return False


# ======================================================================
#  SLAM 地图：解析 map.yaml，并生成带安全距离的可达区域
# ======================================================================
def fetch_bytes(url: str, timeout: float = 10.0) -> bytes:
    request = Request(url, headers={"User-Agent": "JAKA-AutoExplore/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_map_yaml(text: str) -> dict:
    values = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    try:
        origin = json.loads(values["origin"])
        result = {
            "resolution": float(values["resolution"]),
            "origin": [float(origin[0]), float(origin[1]), float(origin[2])],
            "image": values.get("image", "map.png").strip("'\""),
            "free_thresh": float(values.get("free_thresh", 0.196)),
            "occupied_thresh": float(values.get("occupied_thresh", 0.65)),
            "negate": int(values.get("negate", 0)),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"map.yaml 缺少有效 resolution/origin: {exc}") from exc
    if result["resolution"] <= 0 or len(result["origin"]) != 3:
        raise ValueError("map.yaml 的 resolution/origin 无效")
    return result


class SlamMap:
    def __init__(self, yaml_url: str, image_url: str | None, timeout: float):
        self.yaml_url = yaml_url
        self.timeout = timeout
        yaml_bytes = fetch_bytes(yaml_url, timeout)
        metadata = parse_map_yaml(yaml_bytes.decode("utf-8"))
        derived_url = urljoin(yaml_url, metadata["image"])
        self.image_url = image_url or derived_url
        self._load(yaml_bytes, fetch_bytes(self.image_url, timeout), metadata)

    @classmethod
    def from_data(cls, yaml_bytes: bytes, image_bytes: bytes, image_url="local://slam_map.png"):
        """从网页已缓存的 map.yaml/map.png 建图，避免重复访问底盘。"""
        instance = cls.__new__(cls)
        instance.yaml_url = "local://slam_map.yaml"
        instance.image_url = image_url
        instance.timeout = 0.0
        metadata = parse_map_yaml(yaml_bytes.decode("utf-8"))
        instance._load(yaml_bytes, image_bytes, metadata)
        return instance

    def _load(self, yaml_bytes: bytes, image_bytes: bytes, metadata: dict):
        self.yaml_bytes = bytes(yaml_bytes)
        self.image_bytes = bytes(image_bytes)
        self.metadata = metadata
        array = np.frombuffer(self.image_bytes, dtype=np.uint8)
        gray = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("无法解码 SLAM 地图图片")
        self.gray = gray
        self.height, self.width = gray.shape[:2]
        self.resolution = self.metadata["resolution"]
        self.origin_x, self.origin_y, self.origin_yaw = self.metadata["origin"]
        print(f"[MAP] {self.width}x{self.height}, resolution={self.resolution:.4f}m/pixel")
        print(f"[MAP] origin=({self.origin_x:.3f}, {self.origin_y:.3f}, {self.origin_yaw:.3f})")

    def world_to_pixel(self, x: float, y: float):
        cos_yaw, sin_yaw = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        dx, dy = x - self.origin_x, y - self.origin_y
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        col = int(local_x / self.resolution)
        row = int(self.height - local_y / self.resolution)
        return row, col

    def pixel_to_world(self, row: int, col: int):
        local_x = (col + 0.5) * self.resolution
        local_y = (self.height - row - 0.5) * self.resolution
        cos_yaw, sin_yaw = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        x = self.origin_x + cos_yaw * local_x - sin_yaw * local_y
        y = self.origin_y + sin_yaw * local_x + cos_yaw * local_y
        return float(x), float(y)

    def safe_mask(self, robot_radius: float, safety_margin: float):
        # 与底盘网页的 reachability 判断保持相同思路：白色区域才视为可通行。
        normalized = self.gray.astype(np.float32) / 255.0
        if self.metadata["negate"]:
            occupancy = normalized
        else:
            occupancy = 1.0 - normalized
        free = (occupancy <= self.metadata["free_thresh"]).astype(np.uint8)
        clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * self.resolution
        required = float(robot_radius) + float(safety_margin)
        safe = ((free > 0) & (clearance >= required)).astype(np.uint8)
        return safe, clearance

    def reachable_safe_mask(self, pose, robot_radius: float, safety_margin: float):
        safe, _ = self.safe_mask(robot_radius, safety_margin)
        row, col = self.world_to_pixel(pose[0], pose[1])
        row = max(0, min(self.height - 1, row))
        col = max(0, min(self.width - 1, col))
        if not safe[row, col]:
            ys, xs = np.where(safe > 0)
            if len(xs) == 0:
                raise RuntimeError("SLAM 图中没有满足安全距离的可通行区域")
            nearest = int(np.argmin((xs - col) ** 2 + (ys - row) ** 2))
            row, col = int(ys[nearest]), int(xs[nearest])
            print(f"[MAP] 当前位姿不在安全像素，计划从最近安全点 ({row}, {col}) 开始")
        labels_count, labels = cv2.connectedComponents(safe, connectivity=8)
        component = labels == labels[row, col]
        if labels[row, col] == 0:
            raise RuntimeError("当前位姿不在可达安全区域内")
        return component.astype(np.uint8)

    def make_coverage_points(self, pose, robot_radius, safety_margin, spacing, max_points):
        component = self.reachable_safe_mask(pose, robot_radius, safety_margin)
        stride = max(1, int(round(float(spacing) / self.resolution)))
        ys, xs = np.where(component > 0)
        if len(xs) == 0:
            return []
        min_row, max_row, min_col, max_col = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        candidates = []
        # 使用两个半步网格交错采样，减少规则网格漏掉窄通道的情况。
        for offset in (0, stride // 2):
            for row in range(min_row + offset, max_row + 1, stride):
                for col in range(min_col + offset, max_col + 1, stride):
                    if component[row, col]:
                        candidates.append((row, col))
        if not candidates:
            return [self.pixel_to_world(int(ys[0]), int(xs[0]))]

        start_row, start_col = self.world_to_pixel(pose[0], pose[1])
        candidates.sort(key=lambda rc: (rc[1] - start_col) ** 2 + (rc[0] - start_row) ** 2)
        selected = []
        min_sq = max(1, int((stride * 0.62) ** 2))
        for row, col in candidates:
            if all((row - old_row) ** 2 + (col - old_col) ** 2 >= min_sq for old_row, old_col in selected):
                selected.append((row, col))
                if max_points > 0 and len(selected) >= max_points:
                    break
        points = [self.pixel_to_world(row, col) for row, col in selected]
        print(f"[MAP] 生成 {len(points)} 个覆盖点，像素间距约 {stride} ({spacing:.2f}m)")
        return points

    def save_assets(self, output_dir: Path):
        (output_dir / "slam_map.yaml").write_bytes(self.yaml_bytes)
        (output_dir / "slam_map.png").write_bytes(self.image_bytes)


# ======================================================================
#  Orbbec RGB-D 相机与 BoxFusion 数据输出
# ======================================================================
def decode_color_frame(frame):
    """兼容 Orbbec SDK 返回 bytes 或非连续 ndarray 的彩色帧。"""
    raw = frame.get_data()
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(raw, dtype=np.uint8)
    else:
        buffer = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)).ravel()
    if buffer.size >= 3 and buffer[:3].tobytes() == JPEG_MAGIC:
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    width, height = int(frame.get_width()), int(frame.get_height())
    if buffer.size == width * height * 3:
        # 非压缩帧通常是 RGB；转换成 OpenCV 使用的 BGR。
        return cv2.cvtColor(buffer.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
    return None


def decode_depth_frame(frame):
    """按 uint16 深度值解析帧，深度单位保持为毫米。"""
    width, height = int(frame.get_width()), int(frame.get_height())
    raw = frame.get_data()
    if isinstance(raw, (bytes, bytearray, memoryview)):
        payload = bytes(raw)
    else:
        payload = np.ascontiguousarray(raw).tobytes()
    if len(payload) != width * height * 2:
        return None
    try:
        scale = float(frame.get_depth_scale())
    except Exception:
        scale = 1.0
    depth = np.frombuffer(payload, dtype=np.uint16).reshape((height, width)).astype(np.float32) * scale
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).clip(0, 65535).astype(np.uint16)


class OrbbecCamera:
    def __init__(self, device_index, width, height, fps):
        if ORBBEC_IMPORT_ERROR is not None:
            raise RuntimeError(f"当前 Python 环境无法加载 Orbbec SDK: {ORBBEC_IMPORT_ERROR}")
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.ctx = Context()
        devices = self.ctx.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("没有检测到 Orbbec 相机")
        self.device = None
        for index in range(devices.get_count()):
            candidate = devices.get_device_by_index(index)
            if candidate.get_device_info().get_serial_number() == HEAD_CAMERA_SN:
                self.device = candidate
                break
        if self.device is None:
            raise RuntimeError(f"未找到头部相机序列号 {HEAD_CAMERA_SN}，拒绝切换到其他相机")
        info = self.device.get_device_info()
        print(f"[CAM] 固定使用头部相机 {info.get_name()} | SN={info.get_serial_number()}")
        self.pipeline = Pipeline(self.device)
        self.config = Config()
        self.config.enable_stream(self._profile(OBSensorType.COLOR_SENSOR))
        self.config.enable_stream(self._profile(OBSensorType.DEPTH_SENSOR))
        self.config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.start(self.config)

    def _profile(self, sensor_type):
        profiles = self.pipeline.get_stream_profile_list(sensor_type)
        same_size = None
        for index in range(profiles.get_count()):
            profile = profiles.get_stream_profile_by_index(index)
            try:
                video = profile.as_video_stream_profile()
            except Exception:
                video = profile
            try:
                width, height, fps = int(video.get_width()), int(video.get_height()), int(video.get_fps())
            except Exception:
                continue
            if (width, height) == (self.width, self.height):
                if fps == self.fps:
                    return video
                same_size = same_size or video
        if same_size is not None:
            return same_size
        return profiles.get_default_video_stream_profile()

    def read(self, timeout_ms=200, tries=10):
        for _ in range(tries):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms)
                color_frame = frames.get_color_frame() if frames else None
                depth_frame = frames.get_depth_frame() if frames else None
                if color_frame is None or depth_frame is None:
                    continue
                color = decode_color_frame(color_frame)
                depth = decode_depth_frame(depth_frame)
                if color is None or depth is None:
                    continue
                height, width = depth.shape[:2]
                if color.shape[:2] != depth.shape[:2]:
                    color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)
                return color, depth
            except Exception:
                continue
        return None, None

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def make_t_cam2world(pose_xyz):
    x, y, theta = map(float, pose_xyz[:3])
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    body_to_world = np.array([
        [cos_theta, -sin_theta, 0.0, x],
        [sin_theta, cos_theta, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    camera_to_body = np.eye(4, dtype=np.float64)
    camera_to_body[:3, :3] = R_CAM2ROBOT
    camera_to_body[:3, 3] = [CAMERA_OFFSET_XY[0], CAMERA_OFFSET_XY[1], CAM_HEIGHT_M]
    return (body_to_world @ camera_to_body).astype(np.float32)


class DatasetWriter:
    def __init__(self, output_dir: str, zmq_server: str | None, frame_callback=None):
        self.data_dir = Path(output_dir).resolve()
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.data_dir / "metadata.jsonl"
        self.frame_id = self._next_frame_id()
        self.publisher = None
        self.zmq_context = None
        self.frame_callback = frame_callback
        if zmq_server:
            if zmq is None:
                raise RuntimeError("指定了 --zmq-server，但当前环境没有 pyzmq")
            self.zmq_context = zmq.Context()
            self.publisher = self.zmq_context.socket(zmq.PUSH)
            self.publisher.setsockopt(zmq.SNDHWM, 2)
            self.publisher.connect(zmq_server)
            print(f"[ZMQ] PUSH -> {zmq_server}")

    def _next_frame_id(self):
        if not self.meta_path.exists():
            return 0
        last = -1
        for line in self.meta_path.read_text(encoding="utf-8").splitlines():
            try:
                last = max(last, int(json.loads(line).get("frame_id", -1)))
            except Exception:
                pass
        return last + 1

    def write(self, color_bgr, depth_u16, pose_xyz, point_index, heading_index):
        if color_bgr.shape[:2] != EXPECTED_HW or depth_u16.shape[:2] != EXPECTED_HW:
            raise RuntimeError(f"相机尺寸必须为 1280x800，实际 RGB={color_bgr.shape[:2]} depth={depth_u16.shape[:2]}")
        ok_rgb, rgb_buffer = cv2.imencode(".jpg", color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        ok_depth, depth_buffer = cv2.imencode(".png", depth_u16, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if not ok_rgb or not ok_depth:
            raise RuntimeError("RGB-D 编码失败")
        timestamp = time.time()
        stem = f"{timestamp:.6f}"
        rgb_path = self.rgb_dir / f"{stem}.jpg"
        depth_path = self.depth_dir / f"{stem}.png"
        rgb_bytes, depth_bytes = rgb_buffer.tobytes(), depth_buffer.tobytes()
        rgb_path.write_bytes(rgb_bytes)
        depth_path.write_bytes(depth_bytes)
        pose = make_t_cam2world(pose_xyz)
        prefix = self.data_dir.name
        record = {
            "frame_id": self.frame_id,
            "timestamp": timestamp,
            "rgb_path": f"{prefix}/rgb/{stem}.jpg",
            "depth_path": f"{prefix}/depth/{stem}.png",
            "pose_4x4": pose.tolist(),
            "chassis_xyz": [float(value) for value in pose_xyz],
            "explore_point": int(point_index),
            "heading_index": int(heading_index),
        }
        with self.meta_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.publisher is not None:
            ts_bytes = np.array([timestamp], dtype=np.float64).tobytes()
            self.publisher.send_multipart([rgb_bytes, depth_bytes, pose.astype(np.float32).tobytes(), ts_bytes])
        if self.frame_callback is not None:
            self.frame_callback(copy.deepcopy(record), rgb_bytes, depth_bytes, pose, timestamp)
        print(f"[SAVE] frame={self.frame_id} point={point_index} heading={heading_index}")
        self.frame_id += 1
        return record

    def close(self):
        if self.publisher is not None:
            self.publisher.close(linger=0)
        if self.zmq_context is not None:
            self.zmq_context.term()


# ======================================================================
#  探索控制器
# ======================================================================
def write_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def build_route(points, start_pose):
    remaining = list(range(len(points)))
    route = []
    current = (float(start_pose[0]), float(start_pose[1]))
    while remaining:
        index = min(remaining, key=lambda i: (points[i][0] - current[0]) ** 2 + (points[i][1] - current[1]) ** 2)
        remaining.remove(index)
        route.append(index)
        current = points[index]
    return route


def _wait_or_stop(stop_event, seconds):
    return bool(stop_event and stop_event.wait(max(0.0, float(seconds))))


def capture_after_move(args, camera, writer, point_index, heading_index, stop_event=None):
    if _wait_or_stop(stop_event, args.settle_time):
        return False
    color, depth = camera.read(timeout_ms=200, tries=12)
    if color is None or depth is None:
        print("[CAM] 未能获取 RGB-D，跳过当前视角")
        return False
    valid_ratio = float(np.count_nonzero(depth > 0)) / max(1, depth.size)
    if valid_ratio < args.min_depth_ratio:
        print(f"[CAM] 深度有效率过低 ({valid_ratio:.2%})，跳过当前视角")
        return False
    pose = read_pose(args)[:3]
    if stop_event.is_set():
        return False
    writer.write(color, depth, pose, point_index, heading_index)
    return True


def heading_count(step_deg: float) -> int:
    """覆盖一周所需视角数；非整除角度也不能漏掉最后一个方向。"""
    return max(1, int(math.ceil(360.0 / float(step_deg) - 1e-9)))


def create_plan(args, output_dir=None, initial_pose=None):
    """计算并保存探索路线，不产生任何底盘移动。"""
    map_obj = SlamMap(args.map_yaml_url, args.map_image_url, args.fetch_timeout)
    initial_pose = tuple(initial_pose or read_pose(args))
    points = map_obj.make_coverage_points(
        initial_pose,
        args.robot_radius,
        args.safety_margin,
        args.point_spacing,
        args.max_points,
    )
    if not points:
        raise RuntimeError("没有生成任何探索点")
    route = build_route(points, initial_pose)
    out_dir = Path(output_dir or args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    map_obj.save_assets(out_dir)
    plan = {
        "created_at": time.time(),
        "map_yaml_url": args.map_yaml_url,
        "map_image_url": map_obj.image_url,
        "map": map_obj.metadata,
        "initial_pose": [float(value) for value in initial_pose[:3]],
        "points": [[float(x), float(y)] for x, y in points],
        "route": route,
        "heading_count": heading_count(args.heading_step_deg),
        "estimated_frames": len(route) * heading_count(args.heading_step_deg),
        "options": {key: value for key, value in vars(args).items() if not key.startswith("_")},
    }
    write_json(out_dir / "explore_plan.json", plan)
    print(f"[PLAN] 已写入 {out_dir / 'explore_plan.json'}")
    print(f"[PLAN] 起点=({initial_pose[0]:.2f}, {initial_pose[1]:.2f}), 覆盖点={len(points)}")
    print(
        f"[PLAN] 每点采集={plan['heading_count']} 张，方向间隔={args.heading_step_deg:.0f}°，"
        f"预计最多={plan['estimated_frames']} 张"
    )
    return plan


def execute_plan(args, plan, output_dir=None, start_order=0, stop_event=None,
                 progress_callback=None, frame_callback=None, camera_factory=None):
    """执行既有路线；状态写盘后再通知网页，便于中断后继续。"""
    stop_event = stop_event or threading.Event()
    out_dir = Path(output_dir or args.out).resolve()
    route = list(plan.get("route") or [])
    points = list(plan.get("points") or [])
    initial_pose = list(plan.get("initial_pose") or [0.0, 0.0, 0.0])
    if start_order < 0 or start_order >= len(route):
        raise ValueError("继续执行序号超出路线范围")
    route_slice = route[start_order:]
    camera_factory = camera_factory or OrbbecCamera
    camera = camera_factory(args.camera_index, args.camera_width, args.camera_height, args.camera_fps)
    writer = DatasetWriter(str(out_dir), args.zmq_server, frame_callback=frame_callback)
    progress = {
        "status": "running", "route": route, "completed": [], "failed": [],
        "frames": 0, "start_order": int(start_order), "current_order": int(start_order),
        "started_at": time.time(),
    }
    old_progress_path = out_dir / "explore_progress.json"
    if old_progress_path.exists() and start_order > 0:
        try:
            old = json.loads(old_progress_path.read_text(encoding="utf-8"))
            progress["completed"] = list(old.get("completed") or [])
            progress["failed"] = list(old.get("failed") or [])
            progress["frames"] = int(old.get("frames") or 0)
        except Exception:
            pass

    def publish():
        write_json(old_progress_path, progress)
        if progress_callback is not None:
            progress_callback(copy.deepcopy(progress))

    publish()
    try:
        for offset, point_index in enumerate(route_slice):
            if stop_event.is_set():
                progress["status"] = "canceled"
                break
            order = start_order + offset
            progress["current_order"] = order
            progress["current_point"] = point_index
            publish()
            x, y = points[point_index]
            next_index = route[order + 1] if order + 1 < len(route) else point_index
            next_x, next_y = points[next_index]
            travel_heading = math.atan2(next_y - y, next_x - x) if next_index != point_index else initial_pose[2]
            print(f"\n[EXPLORE] {order + 1}/{len(route)} point={point_index}")
            if not move_to(args, x, y, travel_heading):
                progress["failed"].append({"point": point_index, "order": order, "reason": "navigation"})
                publish()
                continue
            completed_frames = 0
            for heading_index in range(int(plan.get("heading_count") or heading_count(args.heading_step_deg))):
                if stop_event.is_set():
                    progress["status"] = "canceled"
                    cancel_move(args)
                    break
                theta = (travel_heading + math.radians(args.heading_step_deg * heading_index)) % (2 * math.pi)
                if heading_index > 0 and not move_to(args, x, y, theta):
                    print(f"[EXPLORE] 原地转向失败 heading={heading_index}")
                    continue
                if capture_after_move(args, camera, writer, point_index, heading_index, stop_event):
                    completed_frames += 1
                    progress["frames"] += 1
                publish()
            if progress["status"] == "canceled":
                break
            progress["completed"].append({"point": point_index, "order": order, "frames": completed_frames})
            progress["next_order"] = order + 1
            publish()
    except Exception as exc:
        progress["status"] = "failed"
        progress["error"] = str(exc)
        raise
    finally:
        if progress.get("status") == "running":
            progress["status"] = "captured"
        progress["finished_at"] = time.time()
        publish()
        writer.close()
        camera.close()
    return progress


def run(args):
    out_dir = Path(args.out).resolve()
    plan = create_plan(args, out_dir)
    route = plan["route"]
    if not args.execute:
        print("[PLAN] 当前为预览模式。确认安全后使用 --execute 执行移动。")
        for order, index in enumerate(route[: min(12, len(route))]):
            print(f"  {order + 1:03d}: ({plan['points'][index][0]:.2f}, {plan['points'][index][1]:.2f})")
        if len(route) > 12:
            print(f"  ... 其余 {len(route) - 12} 个点见 explore_plan.json")
        return

    if not args.confirm:
        answer = input("即将自动移动机器人并采集数据，输入 EXPLORE 确认：").strip()
        if answer != "EXPLORE":
            print("[PLAN] 未确认，退出。")
            return

    try:
        execute_plan(args, plan, out_dir, start_order=args.start_order)
    except KeyboardInterrupt:
        print("\n[STOP] 收到 Ctrl+C，正在取消当前移动")
        cancel_move(args)
        raise
    print(f"[DONE] 探索完成，数据目录: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="基于已知 SLAM 地图自动探索并采集拟物体地图数据")
    parser.add_argument("--execute", action="store_true", help="实际执行导航和采集；不加时只生成计划")
    parser.add_argument("--confirm", action="store_true", help="跳过 EXPLORE 交互确认，适合远程自动启动")
    parser.add_argument("--config", default=None, help="底盘配置文件，默认读取 conf/userCmdControl.json")
    parser.add_argument("--agv-host", default=None, help="底盘 IP；默认从配置读取")
    parser.add_argument("--agv-http-port", type=int, default=None)
    parser.add_argument("--agv-tcp-port", type=int, default=None)
    parser.add_argument("--proto", choices=("auto", "http", "tcp"), default="auto")
    parser.add_argument("--map-yaml-url", default=DEFAULT_MAP_YAML_URL)
    parser.add_argument("--map-image-url", default=None, help="可选；不填则使用 map.yaml 的 image 字段")
    parser.add_argument("--fetch-timeout", type=float, default=15.0)
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出 received_frames 目录")
    parser.add_argument("--camera-index", type=int, default=0, help="兼容旧命令，当前始终按头部相机序列号选择")
    parser.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=DEFAULT_CAMERA_FPS)
    parser.add_argument("--robot-radius", type=float, default=0.30, help="机器人半径，单位 m")
    parser.add_argument("--safety-margin", type=float, default=0.15, help="障碍物额外安全距离，单位 m")
    parser.add_argument("--point-spacing", type=float, default=1.0, help="探索点间距，单位 m")
    parser.add_argument("--max-points", type=int, default=30, help="最多探索点；0 表示不限制")
    parser.add_argument("--start-order", type=int, default=0, help="从路线序号继续执行，序号从 0 开始")
    parser.add_argument("--heading-step-deg", type=float, default=120.0, help="每个点的旋转采样角度")
    parser.add_argument("--settle-time", type=float, default=0.8)
    parser.add_argument("--min-depth-ratio", type=float, default=0.05)
    parser.add_argument("--zmq-server", default=None, help="BoxFusion PUSH 地址，例如 tcp://192.168.1.20:5555")
    parser.add_argument("--distance-tolerance", type=float, default=0.08)
    parser.add_argument("--theta-tolerance", type=float, default=0.08)
    parser.add_argument("--move-timeout", type=float, default=120.0)
    parser.add_argument("--status-interval", type=float, default=0.5)
    args = parser.parse_args()
    host, http_port, tcp_port = load_agv_defaults(args.config)
    args.agv_host = args.agv_host or host
    args.agv_http_port = args.agv_http_port or http_port
    args.agv_tcp_port = args.agv_tcp_port or tcp_port
    if args.heading_step_deg <= 0 or args.heading_step_deg > 360:
        parser.error("--heading-step-deg 必须在 (0, 360] 范围内")
    if args.point_spacing <= 0 or args.robot_radius < 0 or args.safety_margin < 0:
        parser.error("探索距离参数必须为正数")
    return args


def main():
    args = parse_args()
    print(f"[AUTO] AGV={args.agv_host} HTTP={args.agv_http_port} TCP={args.agv_tcp_port} proto={args.proto}")
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
