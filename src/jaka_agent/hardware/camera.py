#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机拍照 + qwen-vl 推理 —— 自包含最小示例(不碰 qwen_planner.py, 也不依赖 demo 的 Camera)。

链路: Orbbec 相机拍一帧彩色(MJPG)→ 存 jpg → qwen-vl 读图回答。
依赖(接相机的机器装): pip install pyorbbecsdk dashscope opencv-python numpy
用法:
    python example_capture_infer.py "桌上有什么东西"
    (不带参数默认问"请描述这张图")
    推理需要 key:  PowerShell  $env:DASHSCOPE_API_KEY="sk-ws-..."
                   bash         export DASHSCOPE_API_KEY=sk-ws-...
    不设 key 时只测拍照, 推理自动跳过。

注意:
- 相机必须插在本机; 关掉 OrbbecViewer 再跑(USB 互斥)。
- 拍照只开彩色流(不开 depth/对齐/同步)——最简配置, 已实测可稳定出真帧。
- 这版 pyorbbecsdk 的 frame.get_data() 返回的是【非连续 ndarray】, 不是 bytes,
  必须先 np.ascontiguousarray 规整成 1D uint8 再 cv2.imdecode; demo 自带的
  frame_to_bgr_image 在这版上是坏的, 所以本文件不依赖它。
"""
import base64
import mimetypes
import os
import sys

import cv2
import dashscope
import numpy as np
from dashscope import MultiModalConversation
from pyorbbecsdk import Config, OBSensorType, Pipeline, OBAlignMode, Context

HERE = os.path.dirname(os.path.abspath(__file__))
# 当前观察/录像统一使用头部相机。DG 是头部相机；VF 备注为手部相机。
HEAD_CAMERA_SN = os.getenv("JAKA_HEAD_CAM_SN", "AY8V74300DG").strip()
HAND_CAMERA_SN = os.getenv("JAKA_HAND_CAM_SN", "AY8V74300VF").strip()

# 和 qwen_planner.py 顶部同一个 key; 没设就只测拍照, 推理自动跳过
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

JPEG_MAGIC = b"\xff\xd8\xff"


def _decode_color_frame(frame):
    """彩色帧 → BGR ndarray; 不是合法 JPEG(全 0xFF 占位帧)返回 None。
    兼容这版 pyorbbecsdk: get_data() 可能返回【非连续 ndarray】(不是 bytes),
    需 np.ascontiguousarray 规整成 1D uint8 再 cv2.imdecode。"""
    raw = frame.get_data()
    if isinstance(raw, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(raw, dtype=np.uint8)
    else:
        buf = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)).ravel()
    if buf[:3].tobytes() != JPEG_MAGIC:          # 全 0xFF 占位帧不是合法 JPEG
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _decode_depth_frame(frame):
    """深度帧 → (H,W) uint16(mm), ×depth_scale; 字节数不符返回 None。
    参 LUMI_DEMO-v2/orbbecCamera.py。关键: 深度是【每像素 2 字节】的缓冲, 必须【按字节重解释】成
    uint16(frombuffer/view) —— 不能 asarray(dtype=uint16)(那是值转换, 会把 uint8 字节数组逐字节
    抬成 uint16, 元素数不变 → 翻倍错位)。这版 get_data() 实测返回 uint8 ndarray(size=2·W·H, 可能
    非连续), 故先规整成连续字节流再 frombuffer(uint16)。"""
    w, h = frame.get_width(), frame.get_height()
    try:
        scale = float(frame.get_depth_scale())   # 多为 1.0(值已是 mm)
    except Exception:
        scale = 1.0
    raw = frame.get_data()
    if isinstance(raw, (bytes, bytearray, memoryview)):
        data = bytes(raw)
    else:                                        # ndarray(实测 uint8, 可能非连续)
        data = np.ascontiguousarray(raw).tobytes()
    if len(data) != w * h * 2:                   # 每像素 2 字节(uint16)
        return None
    depth = np.frombuffer(data, dtype=np.uint16).reshape((h, w)).astype(np.float32) * scale
    return depth.astype(np.uint16)


def _make_pipeline(serial=None, strict_serial=True):
    """只按头部相机序列号构造 Pipeline，避免多相机环境中误切到手部相机。"""
    serial = str(serial or HEAD_CAMERA_SN).strip()
    if not serial:
        raise RuntimeError("未配置头部相机序列号 JAKA_HEAD_CAM_SN")
    try:
        ctx = Context()
        devs = ctx.query_devices()
        for i in range(devs.get_count()):
            d = devs.get_device_by_index(i)
            try:
                if d.get_device_info().get_serial_number() == serial:
                    print(f"[CAM] 按头部相机序列号锁定 {serial}")
                    return Pipeline(d)
            except Exception:
                pass
        message = f"未找到头部相机序列号 {serial}"
        if strict_serial:
            raise RuntimeError(message)
        print(f"[CAM] {message}, 回退默认设备")
    except Exception as e:
        if strict_serial:
            raise RuntimeError(f"头部相机选择失败: {e}") from e
        print(f"[CAM] 设备选择异常({e}), 回退默认")
    return Pipeline()


class ColorCamera:
    """只开彩色流的最小相机封装(底层 pyorbbecsdk)。"""

    def __init__(self, serial=None, strict_serial=True):
        self.pipeline = _make_pipeline(serial, strict_serial=strict_serial)
        dev = self.pipeline.get_device()
        try:
            info = dev.get_device_info()
            print(f"[CAM] {info.get_name()} | sn={info.get_serial_number()}")
        except Exception as e:
            print(f"[CAM] (读设备信息失败 {e})")
        cfg = Config()
        profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self._profile = profiles.get_default_video_stream_profile()
        print(f"[CAM] color {self._profile.get_width()}x{self._profile.get_height()}@"
              f"{self._profile.get_fps()}fps {self._profile.get_format()}")
        cfg.enable_stream(self._profile)
        self.pipeline.start(cfg)

    def grab_color(self, max_tries=30):
        """取一帧合法彩色图(BGR ndarray)。共享 _decode_color_frame(跳占位帧 + 兼容非连续 ndarray)。"""
        for _ in range(max_tries):
            fs = self.pipeline.wait_for_frames(1000)
            if fs is None:
                continue
            cf = fs.get_color_frame()
            if cf is None:
                continue
            img = _decode_color_frame(cf)
            if img is not None and img.size:
                return img
        raise RuntimeError(f"拍照失败: {max_tries} 帧内无合法 JPEG 帧"
                           f"(排查: OrbbecViewer 是否已关 / USB3 直连 / pyorbbecsdk 依赖是否装全)")

    def close(self):
        self.pipeline.stop()


class ColorDepthCamera:
    """彩色 + 深度(align SW + frame sync)。参 LUMI_DEMO-v2/OrbbecSDK/orbbecCamera.py。

    - align SW: 深度对齐到彩色像素 → depth[y,x] 与 color[y,x] 是同一物体(读目标距离的关键)。
    - frame_sync: 彩色/深度时间配对, 同一帧。
    - 深度 uint16 / 单位 mm(scale 通常 1.0); 解析用 _decode_depth_frame(兼容非连续 ndarray)。

    与 ColorCamera 区别: 多开 DEPTH_SENSOR + align + sync。彩色解析逻辑相同(共享 _decode_color_frame)。
    保留 ColorCamera 作 fallback —— 若 align/sync 把彩色搞坏, 退回 ColorCamera 零成本。"""

    def __init__(self, align_mode="SW", enable_sync=True, serial=None, strict_serial=True):
        self.pipeline = _make_pipeline(serial, strict_serial=strict_serial)
        dev = self.pipeline.get_device()
        try:
            info = dev.get_device_info()
            print(f"[CAM] {info.get_name()} | sn={info.get_serial_number()}")
        except Exception as e:
            print(f"[CAM] (读设备信息失败 {e})")
        cfg = Config()
        cprofs = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self._color_profile = cprofs.get_default_video_stream_profile()
        cfg.enable_stream(self._color_profile)
        dprofs = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if dprofs is None:
            raise RuntimeError("相机无深度流 profile(被 OrbbecViewer 占用 / 不支持深度?)")
        self._depth_profile = dprofs.get_default_video_stream_profile()
        cfg.enable_stream(self._depth_profile)
        print(f"[CAM] color {self._color_profile.get_width()}x{self._color_profile.get_height()}@"
              f"{self._color_profile.get_fps()}fps {self._color_profile.get_format()} | "
              f"depth {self._depth_profile.get_width()}x{self._depth_profile.get_height()}@"
              f"{self._depth_profile.get_fps()}fps {self._depth_profile.get_format()}")
        # align: HW首选(快), 不支持(如 Femto Mega)退 SW; 默认 SW 稳。DISABLE 则不对齐(深度分辨率≠彩色)。
        if align_mode == "HW":
            cfg.set_align_mode(OBAlignMode.HW_MODE)
        elif align_mode == "SW":
            cfg.set_align_mode(OBAlignMode.SW_MODE)
        else:
            cfg.set_align_mode(OBAlignMode.DISABLE)
        if enable_sync:
            self.pipeline.enable_frame_sync()
        self.pipeline.start(cfg)

    def grab(self, max_tries=30):
        """取一对 (color_bgr, depth_mm_uint16), 已 align(同分辨率、像素对应)。
        跳过 None / 占位帧(全 0xFF → uint16 65535 → 被 20~10000 滤掉后无有效像素)。"""
        for _ in range(max_tries):
            fs = self.pipeline.wait_for_frames(1000)
            if fs is None:
                continue
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            if cf is None or df is None:
                continue
            color = _decode_color_frame(cf)
            depth = _decode_depth_frame(df)
            if color is None or depth is None:
                continue
            valid = (depth >= 20) & (depth <= 10000)     # 20mm~10m 算有效; 0xFF 占位帧(65535)排除
            if not valid.any():
                continue
            return color, depth
        raise RuntimeError(f"采集失败: {max_tries} 帧内无合法 彩色+深度 对"
                           f"(排查: OrbbecViewer 是否已关 / USB3 直连 / pyorbbecsdk 依赖装全 / 深度是否被挡)")

    def grab_color(self, max_tries=30):
        """只取彩色(CameraBackedDriver.capture 只需彩色时用, 接口与 ColorCamera 一致)。"""
        color, _ = self.grab(max_tries=max_tries)
        return color

    def close(self):
        self.pipeline.stop()


def capture(cam, path=None, max_tries=30):
    """拍照存盘, 返回路径。这就是接进 qwen_planner 时唯一要写的'拍照'那一步。"""
    if path is None:
        path = os.path.join(HERE, "shot.jpg")   # 默认存脚本旁边(Win/Linux 都能写)
    img = cam.grab_color(max_tries=max_tries)
    cv2.imwrite(path, img)
    return path


def _to_data_uri(path):
    """图片 → base64 data URI。比 file:// 更稳(Windows 路径反斜杠不惹事)。"""
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def infer(image_path, question):
    """qwen-vl 读图回答。没设 DASHSCOPE_API_KEY 时自动跳过(只验拍照)。
    注: qwen_planner.py 里的 read_image 是等价的 openai 兼容写法,
        集成进导航栈时直接用那个, 不用重写本函数。"""
    if not dashscope.api_key:
        return "(跳过推理: 没设 DASHSCOPE_API_KEY, 见文件顶部注释)"
    messages = [{"role": "user", "content": [
        {"image": _to_data_uri(image_path)},
        {"text": question},
    ]}]
    resp = MultiModalConversation.call(model="qwen-vl-max", messages=messages)
    if resp.status_code != 200:
        raise RuntimeError(f"qwen-vl 调用失败: {resp.code} {resp.message}")
    return resp.output.choices[0].message.content


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "请描述这张图里有什么"
    cam = ColorCamera()
    try:
        path = capture(cam)
        print(f"[拍照] 已保存 → {path}")
        ans = infer(path, question)
        print(f"[推理] 问: {question}\n  答: {ans}")
    finally:
        cam.close()   # 必须释放(pipeline.stop), 否则下次连不上相机
