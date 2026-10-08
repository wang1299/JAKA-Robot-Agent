"""Bottom chassis protocol commands, navigation drivers and motion helpers."""
from __future__ import annotations
import jaka_agent.models.settings as models_settings
import json
import math
import threading
import time
import socket

def _f(v):
    """浮点格式化: 至多6位小数, 保留一位小数(15.0 而非 15)。"""
    s = f"{float(v):.6f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def cmd_move_location(x, y, theta, distance_tolerance=None, theta_tolerance=None, angle_offset=None):
    """§1.1 单点移动(坐标)。location 优先级高于 marker。"""
    p = [f"location={_f(x)},{_f(y)},{_f(theta)}"]
    if distance_tolerance is not None: p.append(f"distance_tolerance={_f(distance_tolerance)}")
    if theta_tolerance is not None:    p.append(f"theta_tolerance={_f(theta_tolerance)}")
    if angle_offset is not None:       p.append(f"angle_offset={_f(angle_offset)}")
    return "/api/move?" + "&".join(p)


def cmd_move_marker(name, angle_offset=None, distance_tolerance=None, theta_tolerance=None):
    """§1.1 单点移动(点名)。"""
    p = [f"marker={name}"]
    if angle_offset is not None:       p.append(f"angle_offset={_f(angle_offset)}")
    if distance_tolerance is not None: p.append(f"distance_tolerance={_f(distance_tolerance)}")
    if theta_tolerance is not None:    p.append(f"theta_tolerance={_f(theta_tolerance)}")
    return "/api/move?" + "&".join(p)


def cmd_cruise(markers, count=None, distance_tolerance=None, max_continuous_retries=None):
    """§1.2 多点巡游。markers 为点名列表(≥2)。"""
    p = [f"markers={','.join(markers)}"]
    if count is not None:                p.append(f"count={int(count)}")
    if distance_tolerance is not None:   p.append(f"distance_tolerance={_f(distance_tolerance)}")
    if max_continuous_retries is not None: p.append(f"max_continuous_retries={int(max_continuous_retries)}")
    return "/api/move?" + "&".join(p)


def cmd_move_cancel():
    """§2 取消当前移动。"""
    return "/api/move/cancel"


def cmd_estop(flag: bool):
    """§7 软急停(flag=true 进入自由停止/可推动; false 解除)。"""
    return f"/api/estop?flag={'true' if flag else 'false'}"


def cmd_joy_control(linear, angular):
    """§6 直接控制(线速度 m/s, 角速度 rad/s)。线性0+角速度正=原地左转。"""
    return f"/api/joy_control?angular_velocity={_f(angular)}&linear_velocity={_f(linear)}"


def cmd_markers_insert(name, mtype=None, num=None):
    """§5.1 在【当前位置】打 marker。"""
    p = [f"name={name}"]
    if mtype is not None: p.append(f"type={int(mtype)}")
    if num   is not None: p.append(f"num={int(num)}")
    return "/api/markers/insert?" + "&".join(p)


def cmd_markers_insert_by_pose(name, x, y, theta, floor=None, mtype=None, num=None):
    """§5.6 按【指定坐标】打 marker。"""
    p = [f"name={name}", f"x={_f(x)}", f"y={_f(y)}", f"theta={_f(theta)}"]
    if floor is not None: p.append(f"floor={int(floor)}")
    if mtype is not None: p.append(f"type={int(mtype)}")
    if num   is not None: p.append(f"num={int(num)}")
    return "/api/markers/insert_by_pose?" + "&".join(p)


def cmd_accessible_point_query(x, y):
    """§14.5 在目标点附近找可达点。"""
    return f"/api/map/accessible_point_query?x={_f(x)}&y={_f(y)}"


def cmd_distance_probe(x, y):
    """§14.6 查目标点到障碍的距离。"""
    return f"/api/map/distance_probe?x={_f(x)}&y={_f(y)}"


def cmd_robot_status():
    """§3 全局状态(含 move_status / current_pose / current_floor)。"""
    return "/api/robot_status"


def cmd_markers_query_brief():
    """§5.5 点位摘要(名称: 类型-楼层)。"""
    return "/api/markers/query_brief"


class NavDriver:
    """底盘驱动接口(语义层)。Mock 用于 sim/调试; JakaTCPDriver 用于真实。"""

    # --- 移动 (§1.1 / §1.2 / §2) ---
    def move_location(self, x, y, theta, **tol):                       raise NotImplementedError
    def move_marker(self, name, **tol):                                raise NotImplementedError
    def cruise(self, markers, count=None, distance_tolerance=None, max_continuous_retries=None): raise NotImplementedError
    def cancel_move(self):                                             raise NotImplementedError

    # --- 微调 / 安全 (§6 / §7) ---
    def joy_control(self, linear, angular):                            raise NotImplementedError
    def estop(self, flag: bool):                                        raise NotImplementedError

    # --- marker (§5.1 / §5.6 / §5.5) ---
    def insert_marker_here(self, name, mtype=None, num=None):          raise NotImplementedError
    def insert_marker_by_pose(self, name, x, y, theta, floor=None, mtype=None, num=None): raise NotImplementedError
    def query_markers(self):                                           raise NotImplementedError  # → results

    # --- 地图 (§14.5 / §14.6) ---
    def accessible_point_query(self, x, y):        raise NotImplementedError   # → (x,y)
    def distance_probe(self, x, y):                raise NotImplementedError   # → dict

    # --- 状态 (§3) ---
    def robot_status(self):                         raise NotImplementedError  # → results dict
    def get_pose(self):                             raise NotImplementedError  # → (x,y,theta)
    def wait_until_settled(self, timeout=180, target=None, grace=1.0): raise NotImplementedError  # → 终态

    # --- 拍照: 非底盘 API, 走上位机相机 ---
    def capture(self, path) -> str:                 raise NotImplementedError


class MockNavDriver(NavDriver):
    """sim/调试用: 不连机器人, 打印将发送的真实指令串, 返回合理占位值。"""

    def __init__(self, start=(7.9, 33.0, 0.0)):
        self._pose = start

    def move_location(self, x, y, theta, **tol):
        print(f"    [send] {cmd_move_location(x, y, theta, **tol)}")
        self._pose = (x, y, theta)
        return True

    def move_marker(self, name, **tol):
        print(f"    [send] {cmd_move_marker(name, **tol)}")
        return True

    def cruise(self, markers, count=None, distance_tolerance=None, max_continuous_retries=None):
        print(f"    [send] {cmd_cruise(markers, count, distance_tolerance, max_continuous_retries)}")
        return True

    def cancel_move(self):
        print(f"    [send] {cmd_move_cancel()}")
        return True

    def joy_control(self, linear, angular):
        print(f"    [send] {cmd_joy_control(linear, angular)}  (0.5s/条)")
        return True

    def estop(self, flag: bool):
        print(f"    [send] {cmd_estop(flag)}")
        return True

    def insert_marker_here(self, name, mtype=None, num=None):
        print(f"    [send] {cmd_markers_insert(name, mtype, num)}  (当前位置打点)")
        return True

    def insert_marker_by_pose(self, name, x, y, theta, floor=None, mtype=None, num=None):
        print(f"    [send] {cmd_markers_insert_by_pose(name, x, y, theta, floor, mtype, num)}")
        return True

    def query_markers(self):
        print(f"    [send] {cmd_markers_query_brief()}")
        return {}

    def accessible_point_query(self, x, y):
        print(f"    [send] {cmd_accessible_point_query(x, y)}  → mock 原样返回")
        return (x, y)

    def distance_probe(self, x, y):
        print(f"    [send] {cmd_distance_probe(x, y)}")
        return {"obstacle": 0.3, "static": 0.34}

    def robot_status(self):
        return {"move_status": "succeeded", "running_status": "idle",
                "current_pose": {"x": self._pose[0], "y": self._pose[1], "theta": self._pose[2]}}

    def get_pose(self):
        return self._pose

    def wait_until_settled(self, timeout=180, target=None, grace=1.0):
        print("    [poll] robot_status → move_status=succeeded (mock 立即到达)")
        return "succeeded"

    def capture(self, path):
        print(f"    [cam ] 拍照 → {path}  (mock: 未真正存图)")
        return path


class JakaTCPDriver(NavDriver):
    """真实底盘驱动 (对齐 LUMI_DEMO-v2/agv.py 实测):
      - 底盘 192.168.10.10, TCP:31001 / HTTP:9001, 无登录(登录属机械臂 .90, 见 login.py)。
      - TCP 帧: 指令 + CRLF 结束; 返回按 {} 大括号配平, 读到完整 JSON。
      - proto: 'http' / 'tcp' / 'auto'(http 优先, 失败回退 tcp)。
      - 每条指令新建一次连接(与 agv.py 一致, 低频指令足够)。
      capture 不是底盘 API, 走上位机相机 —— 需注入相机客户端。"""

    def __init__(self, host=models_settings.JAKA_HOST, tcp_port=models_settings.JAKA_PORT, http_port=models_settings.JAKA_HTTP_PORT,
                 proto="auto", timeout=3.0):
        self.host, self.tcp_port, self.http_port = host, tcp_port, http_port
        self.proto, self.timeout = proto, timeout

    # --- 传输 ---
    def _tcp_send(self, path):
        """TCP: 发 path+CRLF, 按 {} 配平收完整 JSON (对齐 agv.py.tcp_send)。"""
        line = path if path.endswith("\r\n") else path + "\r\n"
        with socket.create_connection((self.host, self.tcp_port), timeout=self.timeout) as s:
            s.sendall(line.encode("utf-8"))
            s.settimeout(self.timeout)
            buf, depth, started = b"", 0, False
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if not started:
                    i = buf.find(b"{")
                    if i != -1:
                        buf, depth, started = buf[i:], 1, True
                    else:
                        continue
                for b in chunk:
                    if b == 123: depth += 1     # '{'
                    elif b == 125: depth -= 1   # '}'
                if started and depth <= 0:
                    break
        if not buf:
            raise TimeoutError("底盘无响应")
        txt = buf.decode(errors="ignore")
        return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])

    def _http_get(self, path):
        import requests       # 懒导入
        r = requests.get(f"http://{self.host}:{self.http_port}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _send(self, path):
        """按 proto 选传输; auto=http 优先、tcp 兜底。"""
        last = None
        for how in (["http", "tcp"] if self.proto == "auto" else [self.proto]):
            try:
                return self._http_get(path) if how == "http" else self._tcp_send(path)
            except Exception as e:
                last = e
        raise RuntimeError(f"调用失败 {path}: {last}")

    @staticmethod
    def _check(resp, cmd):
        if resp.get("status") not in ("OK", "SUCCEEDED"):
            raise RuntimeError(f"指令失败 {cmd}: status={resp.get('status')} err={resp.get('error_message')}")
        return resp

    # --- 移动 ---
    def move_location(self, x, y, theta, **tol):
        c = cmd_move_location(x, y, theta, **tol); self._check(self._send(c), c)

    def move_marker(self, name, **tol):
        c = cmd_move_marker(name, **tol); self._check(self._send(c), c)

    def cruise(self, markers, count=None, distance_tolerance=None, max_continuous_retries=None):
        c = cmd_cruise(markers, count, distance_tolerance, max_continuous_retries); self._check(self._send(c), c)

    def cancel_move(self):
        c = cmd_move_cancel(); self._check(self._send(c), c)

    def joy_control(self, linear, angular):
        c = cmd_joy_control(linear, angular); self._check(self._send(c), c)

    def estop(self, flag: bool):
        c = cmd_estop(flag); self._check(self._send(c), c)

    # --- marker ---
    def insert_marker_here(self, name, mtype=None, num=None):
        c = cmd_markers_insert(name, mtype, num); self._check(self._send(c), c)

    def insert_marker_by_pose(self, name, x, y, theta, floor=None, mtype=None, num=None):
        c = cmd_markers_insert_by_pose(name, x, y, theta, floor, mtype, num); self._check(self._send(c), c)

    def query_markers(self):
        c = cmd_markers_query_brief(); r = self._check(self._send(c), c); return r.get("results") or {}

    # --- 地图 ---
    def accessible_point_query(self, x, y):
        c = cmd_accessible_point_query(x, y); r = self._check(self._send(c), c)
        pos = r["results"]["position"]; return (pos["x"], pos["y"])

    def distance_probe(self, x, y):
        c = cmd_distance_probe(x, y); r = self._check(self._send(c), c); return r["results"]["env_dist"]

    # --- 状态 ---
    def robot_status(self):
        return self._check(self._send(cmd_robot_status()), cmd_robot_status())["results"]

    def get_pose(self):
        p = self.robot_status()["current_pose"]; return (p["x"], p["y"], p["theta"])

    def wait_until_settled(self, timeout=180, target=None, grace=1.0):
        """轮询 robot_status 直到本次移动到达终态。
        - target(点名)给定时: 用 results.move_target==target 锚定"这次任务"再认 succeeded
          —— 既防"读到上个任务的 succeeded", 也扛得住极短移动(running 一闪而过不误超时)。
        - target 缺省(按坐标移动): 退化为"先见 running 再认 succeeded"; 并加 grace 宽限,
          发令 grace 秒后出现的 succeeded 才采信, 避开起步瞬间残留的旧终态。
        - 任意时刻 estop_state=True → 立即返回 'estop'(手册 §3 results 字段)。"""
        t0 = time.time()
        started = False
        while time.time() - t0 < timeout:
            st = self.robot_status()
            ms = st.get("move_status")
            if st.get("estop_state"):                       # 软/硬件急停 → 不再等
                return "estop"
            if ms in ("failed", "canceled"):                # 被拒/取消, 即便没起步也直接返回
                return ms
            if ms == "running":
                started = True
            elif ms == "succeeded":
                if started:                                 # 见过 running → 必是本次终态
                    return "succeeded"
                if target is not None and st.get("move_target") == target:
                    return "succeeded"                      # 锚定本次目标(极短移动没采到 running)
                if (time.time() - t0) >= grace:             # 过宽限期 → 采信(也兜底 move_target 字段异常)
                    return "succeeded"
            time.sleep(0.5)
        return "timeout"

    def capture(self, path):
        raise NotImplementedError("capture 走上位机相机(192.168.10.90), 非底盘 API —— 注入相机客户端")


class CameraBackedDriver(JakaTCPDriver):
    """真底盘 + 上位机相机: capture() 走【注入的相机客户端】(依赖注入)。

    camera 只需暴露 grab_color() -> BGR ndarray (例如 example_capture_infer.ColorCamera)。
    pyorbbecsdk 由 camera 客户端侧导入, 本类不 import 它 —— 所以在没装相机 SDK 的
    机器(如 Linux 上的规划/dry-run)上 import 本模块不会报错; 真车运行时由调用方
    先构造好相机客户端再传入。相机 pipeline 的生命周期(start/stop)由调用方管理。

    用法(真车入口):
        from example_capture_infer import ColorCamera, ColorDepthCamera
        cam = ColorDepthCamera()                  # 彩色+深度: observe 能做距离修正; 或 ColorCamera() 仅彩色
        driver = CameraBackedDriver(camera=cam)   # 同时能开底盘 + 拍照(+深度)
    """
    def __init__(self, camera=None, camera_factory=None, keep_camera_open=False, warmup_frames=0, **kw):
        super().__init__(**kw)
        self._cam = camera                  # 已开的相机(turn/approach CLI 直接传)
        self._factory = camera_factory      # 或工厂: 首次 capture 时懒开(run 用 —— 避免导航期相机空转 wedged)
        self._owns_cam = camera is None and camera_factory is not None
        self._keep_camera_open = bool(keep_camera_open)
        self._warmup_remaining = max(0, int(warmup_frames))
        self._camera_lock = threading.RLock()
        self.last_depth = None              # 最近一次拍照配对的深度(无深度相机时恒 None)
        self.last_color_shape = (720, 1280)

    def _ensure_cam(self):
        from jaka_agent.tasks.runtime import check_cancelled
        if self._cam is None and self._factory is not None:
            self._cam = self._factory()     # 懒开: 到 observe 才开(此时车已停稳, 不在导航/移动中)
        while self._cam is not None and self._warmup_remaining:
            check_cancelled()
            self._cam.grab_color(max_tries=5)
            self._warmup_remaining -= 1
        check_cancelled()
        return self._cam

    def capture(self, path):
        import cv2   # 懒导入: 仅真拍照时需要
        from jaka_agent.tasks.runtime import check_cancelled
        with self._camera_lock:
            check_cancelled()
            if self._ensure_cam() is None:
                raise NotImplementedError("相机不可用")
            if hasattr(self._cam, "grab"):    # ColorDepthCamera: 返回 (color, depth)
                color, depth = self._cam.grab()
                self.last_depth = depth
            else:                             # ColorCamera: 仅彩色
                color = self._cam.grab_color()
                self.last_depth = None
            self.last_color_shape = color.shape[:2]
            check_cancelled()
        if not cv2.imwrite(path, color):
            raise RuntimeError("相机图片保存失败")
        return path

    def grab_color_frame(self):
        """录像等旁路功能复用同一相机实例，避免另开 pipeline 抢设备。"""
        from jaka_agent.tasks.runtime import check_cancelled
        with self._camera_lock:
            check_cancelled()
            if self._ensure_cam() is None:
                raise NotImplementedError("相机不可用")
            if hasattr(self._cam, "grab"):
                color, depth = self._cam.grab()
                self.last_depth = depth
            else:
                color = self._cam.grab_color()
                self.last_depth = None
            self.last_color_shape = color.shape[:2]
            check_cancelled()
            return color

    def grab_rgbd_frame(self):
        """Atomic paired frame for mapping; never substitute an old depth frame."""
        from jaka_agent.tasks.runtime import check_cancelled
        with self._camera_lock:
            check_cancelled()
            cam = self._ensure_cam()
            if cam is None or not hasattr(cam, "grab"):
                raise RuntimeError("当前共享相机无深度流，无法建图")
            color, depth = cam.grab()
            check_cancelled()
            self.last_color_shape = color.shape[:2]
            self.last_depth = depth
            return color.copy(), depth.copy()

    def close_camera(self, force=False):
        """关相机(底盘驱动保留)。每次 observe 后调 —— 相机只在 observe(车停稳)时开着,
        避免导航/移动期间相机空转导致 USB 流 stall/setXu 报错。"""
        if self._keep_camera_open and not force:
            return
        with self._camera_lock:
            if self._owns_cam and self._cam is not None:
                try: self._cam.close()
                except Exception: pass
                self._cam = None


def rotate_in_place(driver: NavDriver, dyaw: float) -> str:
    """底盘原地转 dyaw(rad): 读当前位姿 → 同(x,y)只改 θ → §1.1 move → §3 等到达。
    - θ 是【绝对朝向】(手册 §1.1: location 的 theta 单位 rad, 范围 [-π,π])。
    - target 包到 [-π, π) —— 手册明确要求该范围, 超出(如 th+dyaw>π)必须归一化。
    - 比 §6 joy_control(只给速度、不定角度)精确: 绝对位姿, 转完即停。
    ⚠ 已知现象: 某些底盘对"原地(同(x,y))换绝对朝向"不按短路径转(实测会转一圈多)。
      若如此, 绝对朝向 move 不可靠 —— 用 `python qwen_planner.py rot <度>` 先诊断底盘行为。"""
    x, y, th = driver.get_pose()
    target = (th + dyaw + math.pi) % (2 * math.pi) - math.pi   # 归一化到 [-π, π)
    driver.move_location(x, y, target)
    return driver.wait_until_settled()


def depth_at(depth, cx_norm, cy_norm, color_w=None, color_h=None, patch=15):
    """读目标距离(米)。纯代码, 不碰大模型 —— 模型只给归一化坐标, 距离由深度传感器读。

    cx_norm/cy_norm∈[0,1] 是【彩色】系坐标(qwen-vl 给)。深度可能和彩色分辨率不同(本机
    深度 1280×800 / 彩色 1280×720, 宽一致高不同) → 先 INTER_NEAREST resize 到彩色尺寸对齐,
    再取目标像素周围 patch×patch 内【有效像素(20~10000mm)】的中位数(抗零点/飞点/小对齐误差)。
    返回距离(米)或 None(目标处无有效深度)。"""
    if cx_norm is None or cy_norm is None:
        return None
    import numpy as np
    import cv2
    dh, dw = depth.shape
    if color_w and color_h and (color_h != dh or color_w != dw):
        depth = cv2.resize(depth, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
        dh, dw = color_h, color_w
    px = int(round(cx_norm * (dw - 1)))
    py = int(round(cy_norm * (dh - 1)))
    y0, y1 = max(0, py - patch), min(dh, py + patch + 1)
    x0, x1 = max(0, px - patch), min(dw, px + patch + 1)
    pd = depth[y0:y1, x0:x1]
    valid = pd[(pd >= 20) & (pd <= 10000)]
    if not valid.size:
        return None
    return float(np.median(valid)) / 1000.0


def forward_move(driver, meters, tol=models_settings.OBSERVE_MOVE_TOL):
    """底盘沿当前朝向平移 meters 米(正=前进, 负=后退): 目标位姿 = 当前 + meters·(cosθ,sinθ), 朝向不变。
    带 distance_tolerance=tol(nudge 实测: 默认容差会吃掉小挪动, 必须显式设小)。
    返回 (wait终态, 沿朝向实际位移): 实际位移 = Δpos 在朝向的投影(前正后负)。若 |actual| 远小于
    |meters| 或反号 → 底盘避障自我保护(掉头/退开, 前方有它看见而相机没看见的障碍), 调用方应据此停车。
    注: forward=(cosθ,sinθ) 已由 nudge 诊断确认(夹角3°)。"""
    x0, y0, th = driver.get_pose()
    tx = x0 + meters * math.cos(th)
    ty = y0 + meters * math.sin(th)
    print(
        f"    [observe-move] start=({x0:.3f},{y0:.3f},{th:.4f}) "
        f"target=({tx:.3f},{ty:.3f},{th:.4f}) requested={meters:+.3f}m tol={tol:.3f}m"
    )
    driver.move_location(tx, ty, th, distance_tolerance=tol)
    st = driver.wait_until_settled()
    x1, y1, _ = driver.get_pose()
    actual = (x1 - x0) * math.cos(th) + (y1 - y0) * math.sin(th)
    print(
        f"    [observe-move] status={st} end=({x1:.3f},{y1:.3f}) "
        f"actual={actual:+.3f}m"
    )
    return st, actual


def forward_clear(depth, step_m, color_w=None, color_h=None, margin=0.05):
    """挪前【深度预检】: 前方(图像中部、身体高度带, 排除底部地面/顶部)是否有物体比本次行程 step_m 更近。
    True=前方净空可挪; False=step 内有障碍, 别动(让调用方停)。看不到有效深度时返 True(由底盘 nav 兜底)。"""
    import numpy as np
    import cv2
    dh, dw = depth.shape
    if color_w and color_h and (color_h != dh or color_w != dw):
        depth = cv2.resize(depth, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
        dh, dw = color_h, color_w
    x0, x1 = int(dw * 0.35), int(dw * 0.65)          # 中间 30% 宽(正前方)
    y0, y1 = int(dh * 0.30), int(dh * 0.75)          # 身体高度带(避底部地面、顶部天花板)
    band = depth[y0:y1, x0:x1]
    valid = band[(band >= 20) & (band <= 10000)]
    if not valid.size:
        return True
    return float(valid.min()) / 1000.0 > step_m + margin   # 最近物 > 行程 → 净空
