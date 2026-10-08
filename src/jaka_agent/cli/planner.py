"""Planning demos and explicit hardware diagnostic subcommands."""
from __future__ import annotations
import jaka_agent.hardware.navigation as hardware_navigation
import jaka_agent.models.settings as models_settings
import jaka_agent.models.vision as models_vision
import jaka_agent.tasks.events as tasks_events
import jaka_agent.tasks.executor as tasks_executor
import jaka_agent.tasks.planning as tasks_planning
import json
import math
import os
import sys

from jaka_agent import paths
SAMPLE_GRAPH = str(paths.RESOURCES_DIR / "maps/scene_graph_sample.json")


def print_command_samples():
    """纯函数: 打印指令串样例, 用来逐条核对 docs/软件API手册.md。"""
    print("=== JAKA 指令串构建器样例 (对照 软件API手册.md) ===")
    print("§1.1 move(location) :", hardware_navigation.cmd_move_location(15.0, 4.0, 1.5707963))
    print("§1.1 move(location)+:", hardware_navigation.cmd_move_location(7.0, 3.0, -1.5708, distance_tolerance=0.3, theta_tolerance=0.1))
    print("§1.1 move(marker)   :", hardware_navigation.cmd_move_marker("meeting_room", angle_offset=0.5))
    print("§1.2 cruise         :", hardware_navigation.cmd_cruise(["m1", "m2", "m3"], count=-1, distance_tolerance=1.0))
    print("§2   cancel         :", hardware_navigation.cmd_move_cancel())
    print("§6   joy_control    :", hardware_navigation.cmd_joy_control(0.2, 0.5))
    print("§5.1 insert(当前)   :", hardware_navigation.cmd_markers_insert("start_point"))
    print("§5.1 insert(type=11):", hardware_navigation.cmd_markers_insert("charge_dock_2", mtype=11))
    print("§5.6 insert_by_pose :", hardware_navigation.cmd_markers_insert_by_pose("205_room", -0.1, 1.0, 0.0, floor=2, mtype=0))
    print("§14.5 accessible    :", hardware_navigation.cmd_accessible_point_query(-0.5, -0.5))
    print("§14.6 distance_probe:", hardware_navigation.cmd_distance_probe(-0.5, -0.5))
    print("§3   robot_status   :", hardware_navigation.cmd_robot_status())


def _print_log(log):
    for i, l in enumerate(log):
        obs = f"  → 读到: {l.observation}" if l.observation else ""
        st = f"  [{l.status}]" if l.status else ""
        print(f"  {i+1}. [{l.step_type}]{st} {l.detail}{obs}")


def run_demo(graph_path=None):
    with open(graph_path or SAMPLE_GRAPH, encoding="utf-8") as f:
        objects = json.load(f)["objects"]

    # --- Demo 1: 单目标 + 观测 (走 Qwen 规划) ---
    instruction = "去看一下电梯现在停在几楼, 然后回来告诉我"
    print("=" * 64)
    print(f"任务指令: {instruction}")
    print("=" * 64)
    plan = tasks_planning.plan_task(instruction, objects)
    print("\n【Qwen 规划输出】")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("\n【执行器 dry-run (MockNavDriver, 打印将发送的真实指令)】")
    _print_log(tasks_executor.PlanExecutor(objects, hardware_navigation.MockNavDriver()).run(plan))

    # --- Demo 2: 多点巡游 (手构计划, 不再调 Qwen; 展示 cruise 真实指令串) ---
    cruise_plan = {"understanding": "巡视沙发、茶几、会议桌",
                   "steps": [{"type": "cruise", "target_ann_ids": [1, 2, 5], "count": 1, "reason": "巡视三处"}],
                   "need_return": True}
    print("\n" + "=" * 64)
    print("Demo 2: 多点巡游 (cruise)")
    print("=" * 64)
    _print_log(tasks_executor.PlanExecutor(objects, hardware_navigation.MockNavDriver()).run(cruise_plan))


def _cli_markers():
    """连真底盘, 列出已标点位(帮你填真实物体索引)。"""
    d = hardware_navigation.JakaTCPDriver()
    print(f"连接 {d.host}:{d.tcp_port}(proto={d.proto}) 查询点位...")
    print(d.query_markers())


def _cli_run(args):
    """真车执行: run "<任务指令>" [graph.json] —— 先规划, 人工确认后再驱动真底盘。"""
    if len(args) < 2:
        print('用法: python qwen_planner.py run "<任务指令>" [graph.json]'); return
    instruction, graph_path = args[1], (args[2] if len(args) > 2 else SAMPLE_GRAPH)
    objects = json.load(open(graph_path, encoding="utf-8"))["objects"]
    print(f"任务: {instruction}\n物体索引: {graph_path} ({len(objects)} 物体)")
    plan = tasks_planning.plan_task(instruction, objects)
    print("\n【Qwen 规划】"); print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("\n⚠ 即将用真底盘执行(有 observe 步会接相机)。回车继续, Ctrl+C 取消。")
    try:
        input()
    except KeyboardInterrupt:
        print("已取消"); return

    # 驱动: 计划含 observe 步 → 接相机(CameraBackedDriver, 拍照走 Orbbec);
    # 否则纯底盘驱动。相机不可用时退回 JakaTCPDriver(observe 会跳过拍照)。
    # 相机懒开: 不在任务开头开(导航期间空转会 wedged), 而是到 observe 才开、observe 完即关。
    need_cam = any(s.get("type") == "observe" for s in plan["steps"])
    factory = None
    if need_cam:
        def factory():
            try:
                from jaka_agent.hardware.camera import ColorDepthCamera
                print("[相机] observe 时接入 彩色+深度")
                return ColorDepthCamera()
            except Exception as e:
                try:
                    from jaka_agent.hardware.camera import ColorCamera
                    print(f"[相机] 深度不可用({e}); 用纯彩色(observe 仅转向, 无距离修正)")
                    return ColorCamera()
                except Exception as e2:
                    print(f"[相机] 不可用({e2}); observe 将跳过拍照")
                    return None
    d = hardware_navigation.CameraBackedDriver(camera_factory=factory) if need_cam else hardware_navigation.JakaTCPDriver()
    ex = tasks_executor.PlanExecutor(objects, d)
    try:
        log = ex.run(plan)
    except KeyboardInterrupt:
        print("\n⚠ 中断: 正在取消当前移动(/api/move/cancel)…")
        try: d.cancel_move()
        except Exception as e: print("cancel 失败, 请按物理急停按钮:", e)
        return
    finally:
        if hasattr(d, "close_camera"):            # 兜底关相机(observe 已各自关, 这里防漏)
            d.close_camera()
    if ex.aborted_reason:
        print(f"\n⚠ 任务中止: {ex.aborted_reason}(已取消残留移动, 剩余步骤未执行)")
    _print_log(log)


WAKE_WORDS = ("小卡", "小咔", "小卡卡")


YES_WORDS  = ("好", "好的", "好吧", "行", "可以", "没问题", "出发", "出发吧",
              "走吧", "执行", "开始", "确认", "是的", "对", "go")


NO_WORDS   = ("取消", "算了", "不要", "不行", "停", "别", "改主意", "no")


def _plan_summary(plan, objs):
    """把 plan 拼成口语化短句, 如"去电梯门，看电梯门，返回"。objs = {ann_id: object}。"""
    parts = []
    for s in plan.get("steps", []):
        t = s.get("type")
        if t == "navigate":
            o = objs.get(s.get("target_ann_id")) or {}
            parts.append("去" + tasks_planning._zh(o))
        elif t == "cruise":
            parts.append("巡视")
        elif t == "observe":
            o = objs.get(s.get("target_ann_id")) or {}
            parts.append("看" + tasks_planning._zh(o))
        elif t == "return":
            parts.append("返回")
    return "，".join(parts) if parts else "执行任务"


def _cli_voice(args):
    """语音模式: voice [graph.json]
    唤醒词"小卡" → 听一句指令 → 语音确认 → 执行(plan+导航+observe+返回, 边走边播)→ 回待机。
    执行期间不监听(安全); TTS 入队不阻塞机器人; 终端输入优先于语音(可免唤醒直接下指令)。"""
    graph_path = next((x for x in args[1:] if x.endswith(".json")), SAMPLE_GRAPH)
    objects = json.load(open(graph_path, encoding="utf-8"))["objects"]
    objs = {o["ann_id"]: o for o in objects}
    try:
        from jaka_agent.hardware.voice import VoiceAssistant
    except Exception as e:
        print(f"语音模块不可用(voice.py 或依赖 sherpa_onnx/sounddevice 缺失): {e}")
        return
    pass  # Voice output is owned by tasks.events.
    va = VoiceAssistant()
    tasks_events._VOICE = va
    va.say("语音控制已就绪", wait=True)
    try:
        while True:
            # ① 终端优先: 有终端输入就直接当指令(免唤醒)
            kb = va.pop_terminal()
            if kb and kb.strip():
                cmd = kb.strip()
                va.say(f"收到终端指令：{cmd}", wait=True)
            else:
                if not va.listen_keyword(WAKE_WORDS):   # ② 语音唤醒(被终端打断→回①)
                    continue
                va.say("我在", wait=True)
                cmd = va.listen_utterance()             # ③ 听一句指令
                if not cmd or len(cmd.replace("小卡", "")) < 2:
                    va.say("没听清，再说一次", wait=True); continue
                cmd = cmd.replace("小卡", "").strip() or cmd
            # 规划 + 复述(规划走云端 API, 单独兜底: 失败则播报后回待机, 不让语音循环崩)
            try:
                plan = tasks_planning.plan_task(cmd, objects)
            except Exception as e:
                tasks_events.announce(f"规划失败：{e}", level="warn"); va.drain_speaker(); continue
            summary = _plan_summary(plan, objs)
            tasks_events.announce("好的，" + summary)
            # ④ 语音确认(终端/语音谁先答都行, 终端优先)
            if not va.confirm(f"确认{summary}吗？", YES_WORDS, NO_WORDS):
                va.say("已取消", wait=True); continue
            # 构造驱动(复用 _cli_run 的相机懒开工厂: 只在 observe 步接相机, 其余纯底盘)
            need_cam = any(s.get("type") == "observe" for s in plan["steps"])
            factory = None
            if need_cam:
                def factory():
                    try:
                        from jaka_agent.hardware.camera import ColorDepthCamera
                        print("[相机] observe 时接入 彩色+深度")
                        return ColorDepthCamera()
                    except Exception as e:
                        try:
                            from jaka_agent.hardware.camera import ColorCamera
                            print(f"[相机] 深度不可用({e}); 用纯彩色")
                            return ColorCamera()
                        except Exception as e2:
                            print(f"[相机] 不可用({e2})")
                            return None
            d = hardware_navigation.CameraBackedDriver(camera_factory=factory) if need_cam else hardware_navigation.JakaTCPDriver()
            ex = tasks_executor.PlanExecutor(objects, d)
            try:
                ex.run(plan)                            # ⑤ 执行(不听语音; TTS 入队不阻塞)
            except KeyboardInterrupt:
                try: d.cancel_move()
                except Exception: pass
                va.say("已停下", wait=True); continue
            finally:
                if hasattr(d, "close_camera"):
                    d.close_camera()
            if not ex.aborted_reason:
                tasks_events.announce("完成了")
            va.drain_speaker()                          # 等最后一句播完再回 IDLE
    except KeyboardInterrupt:
        print("\n退出语音模式")
    finally:
        tasks_events._VOICE = None
        try: va.close()
        except Exception: pass


def _cli_turn(args):
    """只测转向(零平移): 接相机拍一张 → 判取景 → 原地转向找正 → 重拍, 直到取景达标或预算用尽。
    全程只 rotate_in_place(同(x,y)只改θ) —— 不 navigate / 不打 marker / 不返回。
    用法: python qwen_planner.py turn "<目标描述>" "<要回答的问题>"
    例:   python qwen_planner.py turn "桌面上的水杯" "正对水杯, 看清水杯的颜色和上面的字"

    ★怎么提问才会触发转向: 取景模型判 framed=false 才转。若只问"水杯在哪", 即使水杯偏在
      画面边上、模型仍认为"能回答"→framed=true→不转。要让【正对/居中】成为必要条件,
      问题里得带"正对/看清上面的字/看清细节"这类要求, 这样水杯一偏就 framed=false → 原地转。"""
    if len(args) < 3:
        print('用法: python qwen_planner.py turn "<目标>" "<问题>"')
        print('例:   python qwen_planner.py turn "水杯" "正对水杯, 看清水杯的颜色和上面的字"')
        return
    target_desc, question = args[1], args[2]
    try:
        from jaka_agent.hardware.camera import ColorCamera
        cam = ColorCamera()
    except Exception as e:
        print(f"[相机] 接入失败({e}); 无法测试取景转向(检查 pyorbbecsdk / 关掉 OrbbecViewer)。")
        return
    d = hardware_navigation.CameraBackedDriver(camera=cam)
    try:
        x0, y0, th0 = d.get_pose()
    except Exception as e:
        cam.close()
        print(f"[底盘] 连不上({e}); 检查与 192.168.10.10 的网络。")
        return
    print(f"[起点] pose=({x0:.2f},{y0:.2f},θ={th0:.3f} rad≈{math.degrees(th0):+.1f}°)")
    print(f"[目标] '{target_desc}'\n[问题] '{question}'")
    print("[安全] 全程【只原地转向】(同(x,y)改θ), 不会平移。手放物理急停按钮, 回车开始, Ctrl+C 取消。")
    try:
        input()
    except KeyboardInterrupt:
        print("已取消"); cam.close(); return
    path = os.path.join(str(paths.DATA_DIR), "obs_turn.png")
    ex = tasks_executor.PlanExecutor(objects=[], driver=d, mark_start=False)   # mark_start=False: 不打 marker
    try:
        d.capture(path)                            # 先拍第一帧
        ans = ex._reframe_observe(path, target_desc, question)
        x1, y1, th1 = d.get_pose()
    except KeyboardInterrupt:
        print("\n⚠ 中断: 取消残留移动(/api/move/cancel)…")
        try: d.cancel_move()
        except Exception as e: print("cancel 失败, 请按物理急停按钮:", e)
        cam.close(); return
    finally:
        cam.close()
    dth = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi   # 归一化到 (-π, π]
    print(f"\n[结果] {ans}")
    print(f"[终态] pose=({x1:.2f},{y1:.2f},θ={th1:.3f} rad≈{math.degrees(th1):+.1f}°)  "
          f"Δxy=({x1 - x0:+.3f},{y1 - y0:+.3f})m  净Δθ={math.degrees(dth):+.1f}°")
    if abs(x1 - x0) > 0.10 or abs(y1 - y0) > 0.10:
        print("⚠ 平移量 >0.10m, 与预期(只转不移)不符 —— 检查底盘是否把原地转当成了位姿导航。")


def _cli_rot(args):
    """底盘原地转向【诊断】: 只发一条 move_location(当前(x,y), 当前θ+<度>) 看实际转多少。
    用来排查"原地转绝对朝向"是否被底盘按短路径执行 —— qwen_planner 的取景闭环依赖它。
    用法: python qwen_planner.py rot <度数>     正=左/CCW, 负=右/CW
    例:   rot 10   rot -10   rot 30
    判读: 实测 净Δθ ≈ 给的度数 = 正常; 若远大于(尤其 ≈360°±度数) = 底盘没按短路径转,
          绝对朝向 move 不可靠, 取景闭环需改用别的方式(如 joy_control 定时脉冲)。"""
    if len(args) < 2:
        print('用法: python qwen_planner.py rot <度数>  (正=左/CCW, 负=右/CW)')
        print('例:   rot 10   rot -10   rot 30')
        return
    try:
        deg = float(args[1])
    except ValueError:
        print("度数得是数字"); return
    dyaw = math.radians(deg)
    d = hardware_navigation.JakaTCPDriver()
    try:
        x0, y0, th0 = d.get_pose()
    except Exception as e:
        print(f"[底盘] 连不上({e}); 检查与 192.168.10.10 的网络。"); return
    target = (th0 + dyaw + math.pi) % (2 * math.pi) - math.pi
    print(f"[起点] pose=({x0:.2f},{y0:.2f},θ={th0:.3f} rad≈{math.degrees(th0):+.1f}°)")
    print(f"[将发] /api/move?location={hardware_navigation._f(x0)},{hardware_navigation._f(y0)},{hardware_navigation._f(target)}")
    print(f"       目标θ={target:.4f} rad≈{math.degrees(target):+.1f}° (= 当前θ{'+' if deg>=0 else ''}{deg:.1f}°, 已归一化到[-π,π])")
    print("[安全] 手放物理急停。回车执行【单次】原地转向, Ctrl+C 取消。")
    try:
        input()
    except KeyboardInterrupt:
        print("已取消"); return
    try:
        st = hardware_navigation.rotate_in_place(d, dyaw)
        x1, y1, th1 = d.get_pose()
    except KeyboardInterrupt:
        print("\n⚠ 中断: 取消残留移动(/api/move/cancel)…")
        try: d.cancel_move()
        except Exception as e: print("cancel 失败, 按物理急停:", e)
        return
    dth = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi   # 净转向, 归一化
    print(f"\n[终态] status={st}  pose=({x1:.2f},{y1:.2f},θ={th1:.3f} rad≈{math.degrees(th1):+.1f}°)")
    print(f"[对比] 要求 Δθ={deg:+.1f}°  |  实测 净Δθ={math.degrees(dth):+.1f}°  |  Δxy=({x1-x0:+.3f},{y1-y0:+.3f})m")
    if abs(abs(math.degrees(dth)) - abs(deg)) > 15:
        print("⚠ 实测与要求差>15°(若实测≈360°±要求角 → 底盘【没按短路径原地转】, 绝对朝向 move 不可靠)。")


def _cli_depth(args):
    """深度采集【诊断】: 用 ColorDepthCamera 拍彩色+深度, 打印深度统计 + 存伪彩图。
    用来验证深度流能出真帧(非 0xFF、值合理), 再进 Stage B/C。
    用法: python qwen_planner.py depth
    判读: 有效像素占比高、中心距离合理(0.x~几米) = 正常; 有效像素<1% 或全 65535/全 0 = 占位帧
          (SDK 依赖没装全 / USB2 / 被 OrbbecViewer 占用)。"""
    import numpy as np
    import cv2
    try:
        from jaka_agent.hardware.camera import ColorDepthCamera, _decode_color_frame, _decode_depth_frame
    except Exception as e:
        print(f"[相机] 导入失败({e}; 检查 pyorbbecsdk 是否装好)"); return
    cam = ColorDepthCamera()
    n_diag = 10
    try:
        # ── 诊断: 逐帧报告彩色/深度到底卡在哪(不再静默跳过) ──
        print(f"[诊断] 连续抓 {n_diag} 帧, 逐帧报告:")
        got = None
        for i in range(n_diag):
            fs = cam.pipeline.wait_for_frames(1000)
            if fs is None:
                print(f"  #{i}: wait_for_frames → None (超时, 流没出帧)"); continue
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            cs = "None" if cf is None else f"{cf.get_width()}x{cf.get_height()} {cf.get_format()}"
            ds = "None" if df is None else f"{df.get_width()}x{df.get_height()} {df.get_format()}"
            line = f"  #{i}: color={cs} | depth={ds}"
            if df is not None:
                d = _decode_depth_frame(df)
                if d is None:
                    raw = df.get_data()
                    line += (f" | 解析None(buf.size={getattr(raw,'size','?')}, "
                             f"预期={df.get_width()*df.get_height()}×2字节) → Y14 可能非朴素uint16/打包格式")
                else:
                    valid = (d >= 20) & (d <= 10000)
                    line += (f" | 解析OK shape={d.shape} min={int(d.min())} max={int(d.max())}"
                             f" 有效={int(valid.sum())}/{d.size} 0值={int((d==0).sum())} ≥60000={int((d>=60000).sum())}")
            print(line)
            if got is None and cf is not None and df is not None:
                c = _decode_color_frame(cf); d = _decode_depth_frame(df)
                if c is not None and d is not None and ((d >= 20) & (d <= 10000)).any():
                    got = (c, d)
        if got is None:
            print("\n[结论] 10 帧内无合格 彩色+深度 对。按上面分布对号入座:")
            print("  • ≥60000 多 / 有效=0      → 0xFF 占位帧(SDK原生依赖没装全 / USB2 / 被OrbbecViewer占用)")
            print("  • 解析None(buf.size≠W·H·2) → Y14 打包格式, get_data() 非朴素 uint16, 需 SDK 深度处理块/特殊解包")
            print("  • 全 0                     → 深度没起来/被挡/太近(<20cm)/对纯平墙")
            print("  • color=None 或 depth=None → 某条流没出帧(frame_sync/align 卡住, 可试 align=DISABLE 或关 sync)")
            return
        color, depth = got
        # ── 有合格帧了: 统计 + 存图 ──
        h, w = depth.shape
        valid = (depth >= 20) & (depth <= 10000)
        n_valid = int(valid.sum()); ratio = n_valid / (w * h)
        cy, cx = h // 2, w // 2; r = 25
        patch = depth[cy - r:cy + r, cx - r:cx + r]
        pv = patch[(patch >= 20) & (patch <= 10000)]
        center_m = float(np.median(pv)) / 1000.0 if pv.size else float("nan")
        dv = depth[valid]
        print(f"\n[深度] 分辨率 {w}x{h}  |  有效像素 {n_valid}/{w*h} ({ratio*100:.1f}%)")
        if n_valid:
            print(f"[深度] 距离范围 min={dv.min()} max={dv.max()} mm ({dv.min()/1000:.3f}~{dv.max()/1000:.3f} m)")
        print(f"[深度] 中心 {2*r}x{2*r} 距离: {center_m:.3f} m" if not np.isnan(center_m)
              else "[深度] 中心区域无有效距离")
        out_dir = str(paths.DATA_DIR)
        color_path = os.path.join(out_dir, "depth_color.png")
        depth_path = os.path.join(out_dir, "depth_vis.png")
        cv2.imwrite(color_path, color)
        dn = np.clip(depth.astype(np.float32), 20, 10000)
        dvis = ((dn - 20) / (10000 - 20) * 255).astype(np.uint8)
        cv2.imwrite(depth_path, cv2.applyColorMap(dvis, cv2.COLORMAP_JET))
        print(f"[存图] 彩色→{color_path}  深度伪彩→{depth_path}  (二者应像素级对应同一物体)")
    finally:
        cam.close()


def _cli_distance(args):
    """Stage B 验证: 拍彩色+深度 → qwen-vl 定位目标(cx_norm,cy_norm) → 代码读该点深度 → 打印距离。
    用法: python qwen_planner.py distance "<目标>"     例: distance "水杯"
    分工: 大模型只报"目标在哪"(归一化坐标); 距离数值由 depth_at() 从深度传感器读, 不让模型估。"""
    if len(args) < 2:
        print('用法: python qwen_planner.py distance "<目标>"   例: distance "水杯"')
        return
    target = args[1]
    import numpy as np
    import cv2
    try:
        from jaka_agent.hardware.camera import ColorDepthCamera
    except Exception as e:
        print(f"[相机] 导入失败({e})"); return
    cam = ColorDepthCamera()
    try:
        try:
            color, depth = cam.grab()
        except Exception as e:
            print(f"[采集] 失败({e})"); return
        out_dir = str(paths.DATA_DIR)
        path = os.path.join(out_dir, "distance_color.png")
        cv2.imwrite(path, color)
        h, w = color.shape[:2]
        v = models_vision.read_framing(path, target, f"定位目标'{target}', 判断取景并给其归一化中心坐标。")
        cxn, cyn = v.get("cx_norm"), v.get("cy_norm")
        print(f"[取景] framed={v.get('framed')}  cx_norm={cxn}  cy_norm={cyn}  ({v.get('reason','')})")
        if cxn is None or cyn is None:
            print(f"[距离] 目标'{target}' 未定位(cx/cy=null), 无法读距离。"); return
        dist = hardware_navigation.depth_at(depth, cxn, cyn, color_w=w, color_h=h)
        if dist is None:
            print(f"[距离] 目标处无有效深度(零点/玻璃/超量程); 归一化位置 cx={cxn:.2f} cy={cyn:.2f}。"); return
        print(f"[距离] 目标'{target}' 位置(cx={cxn:.2f}, cy={cyn:.2f}) → 距离 {dist:.3f} m")
    finally:
        cam.close()


def _cli_nudge(args):
    """底盘前进/后退【诊断】: 单发一条 move_location 沿当前朝向挪 <米>, 看实际位移与方向。
    用来确认 forward=(cosθ,sinθ) 这台车对不对(世界系↔地图系有坑, 自动距离修正前必验)。
    用法: python qwen_planner.py nudge <米> [距离容差=0.05]
      正=前进, 负=后退; 例: nudge 0.2 / nudge -0.2 / nudge 0.5 0.05
    ★容差: 底盘默认 distance_tolerance 较大(随机型, 可能≥0.2m), 小挪动会被判"已到达"而不动
    (status=succeeded 但位移0)。所以默认显式设 0.05m 逼它真挪。若仍不动 → 底盘有最小行程, 试大点。
    判读: 实际位移≈给定米数、且与朝向夹角<20° = 前向约定✓; 反向/横向 → 约定错, 需翻符号或换轴。"""
    if len(args) < 2:
        print('用法: python qwen_planner.py nudge <米> [距离容差=0.05]  (正=前进, 负=后退)'); return
    try:
        meters = float(args[1])
        tol = float(args[2]) if len(args) > 2 else 0.05
    except ValueError:
        print("参数得是数字"); return
    d = hardware_navigation.JakaTCPDriver()
    try:
        x0, y0, th0 = d.get_pose()
    except Exception as e:
        print(f"[底盘] 连不上({e}); 检查网络。"); return
    tx = x0 + meters * math.cos(th0)     # 假设 forward = (cosθ, sinθ), 待验证
    ty = y0 + meters * math.sin(th0)
    print(f"[起点] pose=({x0:.2f},{y0:.2f},θ={th0:.3f} rad≈{math.degrees(th0):+.1f}°)")
    print(f"[将发] move_location→({hardware_navigation._f(tx)},{hardware_navigation._f(ty)},{hardware_navigation._f(th0)})  挪 {meters:+.2f} m  (distance_tolerance={tol})")
    print("[安全] 手放物理急停, 前方清空。回车执行, Ctrl+C 取消。")
    try:
        input()
    except KeyboardInterrupt:
        print("已取消"); return
    try:
        d.move_location(tx, ty, th0, distance_tolerance=tol)
        st = d.wait_until_settled()
        x1, y1, th1 = d.get_pose()
    except KeyboardInterrupt:
        print("\n⚠ 中断: 取消移动(/api/move/cancel)…")
        try: d.cancel_move()
        except Exception as e: print("cancel 失败, 按物理急停:", e)
        return
    moved = math.hypot(x1 - x0, y1 - y0)
    print(f"\n[终态] status={st}  pose=({x1:.2f},{y1:.2f},θ={th1:.3f} rad≈{math.degrees(th1):+.1f}°)")
    print(f"[对比] 要求 {meters:+.2f} m  |  实际位移 {moved:.3f} m  Δxy=({x1-x0:+.3f},{y1-y0:+.3f})  Δθ={math.degrees(th1-th0):+.1f}°")
    if moved < 0.02:
        print(f"⚠ 仍没动 → 底盘默认容差外可能还有最小行程阈值; 试 nudge {abs(meters)*2:.1f} 0.05 加大距离。")
    else:
        cos_ang = ((x1-x0)*math.cos(th0) + (y1-y0)*math.sin(th0)) / moved   # 与朝向夹角余弦
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cos_ang))))
        verdict = ("前向约定✓(朝向一致)" if ang < 20
                   else "反向(约定反, 要翻符号)" if ang > 160
                   else f"偏 {ang:.0f}°(非纯前进, forward 轴可能不对)")
        print(f"[方向] 实际位移 vs 朝向 夹角≈{ang:.0f}° → {verdict}")
        if abs(moved - abs(meters)) > 0.05:
            print(f"⚠ 位移量 {moved:.3f}m 与要求 {abs(meters):.2f}m 差>5cm。")


def _cli_approach(args):
    """距离修正闭环(会平移车!): 转向找正 + 按模型 approach 靠近/远离, 直到 framed=True。
    与 `run` 的 observe 走【同一个】_observe_full 逻辑(单一实现, 不重复)。
    用法: python qwen_planner.py approach "<目标>" "<问题>"
    例:   python qwen_planner.py approach "水杯" "正对水杯, 读出水杯上的字"
    安全: 前进行程 ≤ min(STEP, dist−ABS_MIN−余量) + 深度预检 + 底盘nav避障 + 避障退即停;
          总位移≤MAX_TOTAL; 手放物理急停, 回车确认才跑。"""
    if len(args) < 3:
        print('用法: python qwen_planner.py approach "<目标>" "<问题>"')
        print('例:   python qwen_planner.py approach "水杯" "正对水杯, 读出水杯上的字"')
        return
    target, question = args[1], args[2]
    try:
        from jaka_agent.hardware.camera import ColorDepthCamera
    except Exception as e:
        print(f"[相机] 导入失败({e})"); return
    cam = ColorDepthCamera()
    d = hardware_navigation.CameraBackedDriver(camera=cam)
    try:
        x0, y0, th0 = d.get_pose()
    except Exception as e:
        cam.close(); print(f"[底盘] 连不上({e})"); return
    print(f"[起点] pose=({x0:.2f},{y0:.2f},θ={math.degrees(th0):+.1f}°)  目标='{target}'")
    print(f"[安全] 会平移! ABS_MIN={models_settings.APPROACH_ABS_MIN}m STEP={models_settings.APPROACH_STEP}m 总上限={models_settings.APPROACH_MAX_TOTAL}m。"
          " 手放物理急停, 前方清空。回车开始, Ctrl+C 取消。")
    try:
        input()
    except KeyboardInterrupt:
        print("已取消"); cam.close(); return
    path = os.path.join(str(paths.DATA_DIR), "approach.png")
    ex = tasks_executor.PlanExecutor(objects=[], driver=d, mark_start=False)
    try:
        ans = ex._observe_full(path, target, question)
        x1, y1, th1 = d.get_pose()
        print(f"\n[结果] {ans}")
        print(f"[终态] pose=({x1:.2f},{y1:.2f},θ={math.degrees(th1):+.1f}°)  "
              f"总位移≈{math.hypot(x1-x0, y1-y0):.3f}m")
    except KeyboardInterrupt:
        print("\n⚠ 中断: 取消移动(/api/move/cancel)…")
        try: d.cancel_move()
        except Exception as e: print("cancel 失败, 按物理急停:", e)
    finally:
        cam.close()


def main():
    a = sys.argv[1:]
    if not a:
        run_demo()                       # 默认 demo (sim sample + Mock dry-run)
    elif a[0] == "cmds":
        print_command_samples()      # 打印指令串构建器样例(纯函数, 不联网)
    elif a[0] == "markers":
        _cli_markers()               # 连真底盘列点位
    elif a[0] == "run":
        _cli_run(a)                  # 真车执行(先规划, 确认后驱动)
    elif a[0] == "voice":
        _cli_voice(a)                # 语音模式: 唤醒词"小卡" + STT 下任务 + TTS 播报(终端优先)
    elif a[0] == "turn":
        _cli_turn(a)                 # 只测转向(零平移): 拍→判景→原地转
    elif a[0] == "rot":
        _cli_rot(a)                  # 转向诊断: 单发一条 move, 看实际转多少
    elif a[0] == "depth":
        _cli_depth(a)                # 深度诊断: 拍彩色+深度, 看能否出真深度帧
    elif a[0] == "distance":
        _cli_distance(a)             # Stage B: 模型定位 + 代码读目标距离
    elif a[0] == "nudge":
        _cli_nudge(a)                # Stage C 前置: 诊断前进方向(沿朝向挪, 看实际位移)
    elif a[0] == "approach":
        _cli_approach(a)             # Stage C: 转正+距离修正闭环(会平移, 带安全限幅)
    elif a[0].endswith(".json"):
        run_demo(a[0])                   # 指定 graph 跑 Mock demo
    else:
        print(__doc__)
