"""Execute validated plans through navigation and observation drivers."""
from __future__ import annotations
import jaka_agent.hardware.navigation as hardware_navigation
import jaka_agent.hardware.observation as hardware_observation
import jaka_agent.models.settings as models_settings
import jaka_agent.models.vision as models_vision
import jaka_agent.tasks.events as tasks_events
import jaka_agent.tasks.planning as tasks_planning
import math
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExecLog:
    step_type: str
    target_ann_id: Optional[int] = None
    detail: str = ""
    observation: Optional[str] = None   # observe 步骤读到的信息
    status: Optional[str] = None        # move 终态(succeeded/failed/...)


class PlanExecutor:
    """把 plan['steps'] 顺序翻译成 driver 调用。"""

    START_MARKER = "task_start"

    def __init__(self, objects: list, driver: hardware_navigation.NavDriver, mark_start: bool = True):
        self.objs = {o["ann_id"]: o for o in objects}
        self.driver = driver
        self.log: list[ExecLog] = []
        self.aborted_reason: Optional[str] = None   # 非 None = 任务在某移动步中止(失败/急停/取消/超时)
        # 出发: 打 marker 便于可靠返回 (§5.1, name 存在则更新)。mark_start=False 跳过(纯转向测试用)
        if mark_start:
            try:
                driver.insert_marker_here(self.START_MARKER)
            except NotImplementedError:
                pass
        self.start_pose = driver.get_pose()

    def run(self, plan: dict) -> list[ExecLog]:
        for i, s in enumerate(plan["steps"]):
            t = s["type"]
            if   t == "navigate": st = self._do_navigate(s)
            elif t == "cruise":   st = self._do_cruise(s)
            elif t == "patrol":   st = self._do_cruise(s)  # 网页执行器会接管拍照对比；命令行保留移动兼容
            elif t == "observe":  st = self._do_observe(s)
            elif t == "return":   st = self._do_return()
            elif t == "cancel":   self._do_cancel();   st = None
            else:                 st = None
            # 移动步(navigate/cruise/return)未成功到达 → 立即中止后续, 保证安全
            if st is not None and st != "succeeded":
                self._abort(i, s, st)
                break
        # 仅在未中止时才兜底返回(中止时车可能处于急停/受阻位, 不应再乱动)
        if not self.aborted_reason and plan.get("need_return") \
                and not any(l.step_type == "return" for l in self.log):
            self._do_return()      # 兜底: 计划说返回但无显式 return
        return self.log

    def _abort(self, step_idx: int, step: dict, reason: str):
        """某移动步未成功到达 → 中止后续步骤并清残留移动状态。
        - failed/timeout: 底盘可能仍在移动或残留任务态 → 发 cancel 清状态。
        - estop: 急停优先, 不再发任何移动指令(手册 §7, 软/硬急停需人工解除)。
        - canceled: 已被取消, 无需再 cancel。"""
        self.aborted_reason = reason
        tasks_events.announce(f"任务中止：{reason}", level="warn")
        if reason in ("failed", "timeout"):
            try:
                self.driver.cancel_move()
            except Exception as e:
                print(f"  (cancel 清残留失败, 请手动按物理急停: {e})")
        tgt = step.get("target_ann_id") or step.get("target_ann_ids") or ""
        self.log.append(ExecLog(
            "aborted", detail=f"第{step_idx + 1}步[{step['type']}] {tgt} 未到达({reason}), 中止后续步骤"))

    # --- 各 step ---
    @staticmethod
    def _convex_hull(points):
        """返回二维点集的逆时针凸包，去掉 3D box 顶/底面投影产生的重复点。"""
        unique = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in points})
        if len(unique) <= 2:
            return unique

        def cross(origin, left, right):
            return ((left[0] - origin[0]) * (right[1] - origin[1])
                    - (left[1] - origin[1]) * (right[0] - origin[0]))

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    def _object_footprint(self, obj):
        """优先读取原始 corners_world；旧地图只有 box3d 时按尺寸和 yaw 还原底面。"""
        geometry = obj.get("geometry") or {}
        box = obj.get("box3d") or {}
        raw_corners = (
            geometry.get("corners_world")
            or geometry.get("bbox_corners")
            or box.get("corners")
            or []
        )
        points3d = []
        for corner in raw_corners:
            try:
                x, y = float(corner[0]), float(corner[1])
                z = float(corner[2]) if len(corner) > 2 else 0.0
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                points3d.append((x, y, z))
        # 标准 3D box 有 8 个顶点，最低的 4 个构成与地面接触的底面。
        if len(points3d) >= 8:
            points3d = sorted(points3d, key=lambda value: value[2])[:4]
        points = [(x, y) for x, y, _ in points3d]
        hull = self._convex_hull(points)
        if len(hull) >= 3:
            return hull

        center = box.get("center") or obj.get("floor_xy") or []
        size = box.get("size") or []
        try:
            cx, cy = float(center[0]), float(center[1])
            half_x, half_y = float(size[0]) / 2, float(size[1]) / 2
            yaw = float(box.get("yaw") or 0.0)
        except (TypeError, ValueError, IndexError):
            return []
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        corners = []
        for local_x, local_y in ((-half_x, -half_y), (-half_x, half_y),
                                 (half_x, half_y), (half_x, -half_y)):
            corners.append((
                cx + local_x * cos_yaw - local_y * sin_yaw,
                cy + local_x * sin_yaw + local_y * cos_yaw,
            ))
        return self._convex_hull(corners)

    def _query_accessible_candidates(
        self, candidates, center, robot_pose, errors, min_center_distance=0.0,
    ):
        """选择安全可达且“行进方向→观察方向”转角最小的候选结果。"""
        results = []
        seen = set()
        for candidate_x, candidate_y, source in candidates:
            key = (round(candidate_x, 3), round(candidate_y, 3))
            if key in seen:
                continue
            seen.add(key)
            try:
                target_x, target_y = self.driver.accessible_point_query(candidate_x, candidate_y)
                target_x, target_y = float(target_x), float(target_y)
                if not (math.isfinite(target_x) and math.isfinite(target_y)):
                    raise ValueError("底盘返回了无效坐标")
                target_distance = math.hypot(target_x - center[0], target_y - center[1])
                if target_distance < min_center_distance:
                    errors.append(
                        f"{source}查询({candidate_x:.2f},{candidate_y:.2f})返回可达点"
                        f"({target_x:.2f},{target_y:.2f})，距物体中心仅"
                        f"{target_distance:.2f}m，低于安全距离{min_center_distance:.2f}m"
                    )
                    continue
                robot_distance = math.hypot(target_x - robot_pose[0], target_y - robot_pose[1])
                face_theta = math.atan2(center[1] - target_y, center[0] - target_x)
                if robot_distance >= 0.12:
                    travel_theta = math.atan2(
                        target_y - robot_pose[1], target_x - robot_pose[0],
                    )
                else:
                    travel_theta = float(robot_pose[2])
                heading_delta = abs(
                    (face_theta - travel_theta + math.pi) % (2 * math.pi) - math.pi
                )
                results.append((target_distance, robot_distance, target_x, target_y, source,
                                candidate_x, candidate_y, heading_delta))
            except Exception as exc:
                from jaka_agent.tasks.runtime import TaskCancelled
                if isinstance(exc, TaskCancelled):
                    raise
                errors.append(f"{source}({candidate_x:.2f},{candidate_y:.2f}): {exc}")
        return min(
            results,
            default=None,
            key=lambda value: (value[7], value[1], value[0]),
        )

    def _find_accessible_target(self, obj, use_viewpoint=False):
        """按 viewpoint、物体中心、3D box 边界、边界外侧的顺序寻找最近可达点。"""
        object_center = (obj.get("box3d") or {}).get("center") or obj.get("floor_xy") or []
        try:
            center_x, center_y = float(object_center[0]), float(object_center[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(f"{tasks_planning._zh(obj)}缺少可用于导航的物体中心坐标") from exc
        center = (center_x, center_y)
        viewpoint = obj.get("viewpoint") or {}
        try:
            robot_pose = tuple(map(float, self.driver.get_pose()))
        except Exception as exc:
            from jaka_agent.tasks.runtime import TaskCancelled
            if isinstance(exc, TaskCancelled):
                raise
            robot_pose = tuple(map(float, self.start_pose))
        errors = []

        if use_viewpoint and all(key in viewpoint for key in ("x", "y")):
            view_candidates = [(float(viewpoint["x"]), float(viewpoint["y"]), "观察点")]
            result = self._query_accessible_candidates(
                view_candidates, center, robot_pose, errors,
                min_center_distance=models_settings.NAV_MIN_CENTER_DISTANCE_M,
            )
            if result:
                distance, _, target_x, target_y, source, _, _, heading_delta = result
                theta = math.atan2(center_y - target_y, center_x - target_x)
                print(
                    f"    [nav ] #{obj.get('ann_id')} 采用{source} → "
                    f"可达点({target_x:.2f},{target_y:.2f})，距物体中心 {distance:.2f}m；"
                    f"终点面向物体中心 θ={math.degrees(theta):.1f}°，"
                    f"预计到点转角{math.degrees(heading_delta):.1f}°"
                )
                return target_x, target_y, theta, source

        footprint = self._object_footprint(obj)
        max_heading_delta = math.radians(models_settings.NAV_APPROACH_MAX_HEADING_DELTA_DEG)
        result = self._query_accessible_candidates(
            [(center_x, center_y, "物体中心")], center, robot_pose, errors,
            min_center_distance=models_settings.NAV_MIN_CENTER_DISTANCE_M,
        )
        # 底盘对“查询物体中心”可能返回目标另一侧的可达点。即使它安全可达，
        # 只要预计到点需要大角度掉头，就继续搜索 box 边界和外侧候选。
        if (not result or result[7] > max_heading_delta) and footprint:
            boundary_result = self._query_accessible_candidates(
                [(x, y, "3D box 顶点") for x, y in footprint],
                center, robot_pose, errors,
                min_center_distance=models_settings.NAV_MIN_CENTER_DISTANCE_M,
            )
            if boundary_result and (
                not result
                or (boundary_result[7], boundary_result[1], boundary_result[0])
                < (result[7], result[1], result[0])
            ):
                result = boundary_result

        if not result or result[7] > max_heading_delta:
            boundary_points = list(footprint)
            if footprint:
                boundary_points.extend(
                    ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
                    for left, right in zip(footprint, footprint[1:] + footprint[:1])
                )
            else:
                box_size = (obj.get("box3d") or {}).get("size") or [1.0, 1.0]
                try:
                    radius = max(0.75, math.hypot(float(box_size[0]), float(box_size[1])) / 2)
                except (TypeError, ValueError, IndexError):
                    radius = 0.75
                boundary_points = [
                    (center_x + radius * math.cos(index * math.pi / 4),
                     center_y + radius * math.sin(index * math.pi / 4))
                    for index in range(8)
                ]
            for clearance in (0.35, 0.7):
                nearby = []
                for x, y in boundary_points:
                    dx, dy = x - center_x, y - center_y
                    length = math.hypot(dx, dy) or 1.0
                    nearby.append((
                        x + dx / length * clearance,
                        y + dy / length * clearance,
                        f"3D box 外侧 {clearance:.2f}m",
                    ))
                nearby_result = self._query_accessible_candidates(
                    nearby, center, robot_pose, errors,
                    min_center_distance=models_settings.NAV_MIN_CENTER_DISTANCE_M,
                )
                if nearby_result and (
                    not result
                    or (nearby_result[7], nearby_result[1], nearby_result[0])
                    < (result[7], result[1], result[0])
                ):
                    result = nearby_result
                if result and result[7] <= max_heading_delta:
                    break

        if not result:
            last_error = errors[-1] if errors else "没有可查询的候选点"
            raise RuntimeError(
                f"无法在{tasks_planning._zh(obj)}(ann_id={obj.get('ann_id')})附近找到可达位置；"
                f"物体中心=({center_x:.2f},{center_y:.2f})，已查询{len(errors)}个候选点。"
                f"最后错误: {last_error}"
            )

        distance, _, target_x, target_y, source, query_x, query_y, heading_delta = result
        # 终点 theta 必须由“可达点 → 物体中心”计算。底盘在规划整段路线时便会把
        # 最终朝向纳入目标位姿，到点后的 _face_object_center 只负责校验和小步补偿。
        theta = math.atan2(center_y - target_y, center_x - target_x)
        print(
            f"    [nav ] #{obj.get('ann_id')} 采用{source}查询"
            f"({query_x:.2f},{query_y:.2f}) → 可达点({target_x:.2f},{target_y:.2f})，"
            f"距物体中心 {distance:.2f}m；终点面向物体中心 θ={math.degrees(theta):.1f}°，"
            f"预计到点转角{math.degrees(heading_delta):.1f}°"
        )
        return target_x, target_y, theta, source

    def _face_object_center(self, obj):
        """到达后校验实际朝向；有误差时分段小角度补偿，并复读位姿确认结果。"""
        center = (obj.get("box3d") or {}).get("center") or obj.get("floor_xy") or []
        try:
            center_x, center_y = float(center[0]), float(center[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError(f"{tasks_planning._zh(obj)}缺少可用于朝向对正的中心坐标") from exc
        tolerance = math.radians(models_settings.NAV_GOAL_THETA_TOLERANCE_DEG)
        max_step = math.radians(models_settings.NAV_FACE_MAX_DELTA_DEG)
        camera_closed = False
        total_adjusted = 0.0
        max_attempts = max(1, math.ceil(math.pi / max(max_step, tolerance)) + 2)

        for attempt in range(1, max_attempts + 1):
            x, y, theta = map(float, self.driver.get_pose())
            distance = math.hypot(center_x - x, center_y - y)
            if distance < models_settings.NAV_MIN_CENTER_DISTANCE_M:
                detail = (
                    f"实际停靠点({x:.2f},{y:.2f})距目标中心仅{distance:.2f}m，"
                    f"低于安全距离{models_settings.NAV_MIN_CENTER_DISTANCE_M:.2f}m，无法可靠计算观察朝向"
                )
                self.log.append(ExecLog(
                    "orient", obj.get("ann_id"), detail, status="failed",
                ))
                print(f"    [orient] {detail}")
                return "failed"

            target_theta = math.atan2(center_y - y, center_x - x)
            delta = (target_theta - theta + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) <= tolerance:
                self.log.append(ExecLog(
                    "orient", obj.get("ann_id"),
                    f"已正对目标中心({center_x:.2f},{center_y:.2f})，"
                    f"最终误差{math.degrees(delta):+.1f}°",
                    status="succeeded",
                ))
                return "succeeded"
            if max_step <= 0:
                detail = (
                    f"当前朝向与目标中心相差{math.degrees(delta):+.1f}°，"
                    "但到点朝向补偿已禁用"
                )
                self.log.append(ExecLog(
                    "orient", obj.get("ann_id"), detail, status="failed",
                ))
                print(f"    [orient] {detail}")
                return "failed"

            step_delta = max(-max_step, min(max_step, delta))
            command_theta = (theta + step_delta + math.pi) % (2 * math.pi) - math.pi
            if not camera_closed and hasattr(self.driver, "close_camera"):
                self.driver.close_camera()
                camera_closed = True
            print(
                f"    [orient] 第{attempt}次补偿：当前"
                f"({x:.2f},{y:.2f},{math.degrees(theta):.1f}°) → "
                f"目标中心({center_x:.2f},{center_y:.2f})，"
                f"本次转{math.degrees(step_delta):+.1f}°"
            )
            self.driver.move_location(
                x, y, command_theta,
                distance_tolerance=0.08, theta_tolerance=tolerance,
            )
            status = self.driver.wait_until_settled()
            if status != "succeeded":
                self.log.append(ExecLog(
                    "orient", obj.get("ann_id"),
                    f"第{attempt}次朝向补偿失败，底盘状态={status}",
                    status=status,
                ))
                return status

            new_x, new_y, new_theta = map(float, self.driver.get_pose())
            new_target_theta = math.atan2(center_y - new_y, center_x - new_x)
            new_delta = (
                new_target_theta - new_theta + math.pi
            ) % (2 * math.pi) - math.pi
            if abs(new_delta) >= abs(delta) - math.radians(1):
                detail = (
                    f"底盘报告成功但朝向未有效改变：补偿前误差"
                    f"{math.degrees(delta):+.1f}°，补偿后"
                    f"{math.degrees(new_delta):+.1f}°"
                )
                self.log.append(ExecLog(
                    "orient", obj.get("ann_id"), detail, status="failed",
                ))
                print(f"    [orient] {detail}")
                return "failed"
            total_adjusted += step_delta

        detail = (
            f"已累计补偿{math.degrees(total_adjusted):+.1f}°，"
            f"但仍未达到{models_settings.NAV_GOAL_THETA_TOLERANCE_DEG:g}°朝向误差要求"
        )
        self.log.append(ExecLog(
            "orient", obj.get("ann_id"), detail, status="failed",
        ))
        print(f"    [orient] {detail}")
        return "failed"

    def _do_navigate(self, s):
        o = self.objs[s["target_ann_id"]]
        navigation_pose = o.get("navigation_pose")
        if isinstance(navigation_pose, dict) and all(
            key in navigation_pose for key in ("x", "y", "theta")
        ):
            try:
                tx = float(navigation_pose["x"])
                ty = float(navigation_pose["y"])
                th = (
                    float(navigation_pose["theta"]) + math.pi
                ) % (2 * math.pi) - math.pi
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{tasks_planning._zh(o)}的 navigation_pose 不是有效数值"
                ) from exc
            if not all(math.isfinite(value) for value in (tx, ty, th)):
                raise RuntimeError(f"{tasks_planning._zh(o)}的 navigation_pose 包含无效数值")
            self.driver.move_location(
                tx,
                ty,
                th,
                theta_tolerance=math.radians(models_settings.NAV_GOAL_THETA_TOLERANCE_DEG),
            )
            on_motion_started = s.get("_on_motion_started")
            if callable(on_motion_started):
                on_motion_started()
            st = self.driver.wait_until_settled()
            self.log.append(ExecLog(
                "navigate",
                o["ann_id"],
                f"到 {tasks_planning._zh(o)} 点位 ({tx:.3f},{ty:.3f},θ={th:.4f})",
                status=st,
            ))
            if st == "succeeded":
                on_translation_arrived = s.get("_on_translation_arrived")
                if callable(on_translation_arrived):
                    on_translation_arrived()
                if s.get("_announce_arrival", True):
                    tasks_events.announce(f"已到达 {tasks_planning._zh(o)}")
            return st

        target = None
        object_center = (o.get("box3d") or {}).get("center") or o.get("floor_xy") or []
        if len(object_center) >= 2:                              # 物体导航优先搜索附近可达点
            tx, ty, th, source = self._find_accessible_target(o, bool(s.get("use_viewpoint")))
            self.driver.move_location(
                tx,
                ty,
                th,
                theta_tolerance=math.radians(models_settings.NAV_GOAL_THETA_TOLERANCE_DEG),
            )
            where = f"({tx:.2f},{ty:.2f},θ={th:.2f}，{source})"
        elif o.get("marker"):                                   # 无物体中心坐标时才退回 marker
            self.driver.move_marker(
                o["marker"],
                theta_tolerance=math.radians(models_settings.NAV_GOAL_THETA_TOLERANCE_DEG),
            )
            where = f"marker={o['marker']}"
            target = o["marker"]                                # 传给 wait 做 move_target 锚定
        else:
            raise RuntimeError(f"{tasks_planning._zh(o)}既没有物体中心坐标，也没有可用 marker")
        # 可达点计算和移动命令发送完成后，才允许 Web 寻物启动行进抓拍，
        # 避免把出发前仍停在上一个目标点的画面误记为“行进中”。
        on_motion_started = s.get("_on_motion_started")
        if callable(on_motion_started):
            on_motion_started()
        st = self.driver.wait_until_settled(target=target)     # §3 轮询
        self.log.append(ExecLog("navigate", o["ann_id"],
                                f"到 {tasks_planning._zh(o)} 附近 {where}", status=st))
        if st == "succeeded":
            # Web 寻物会在这里停止“行进中抓拍”线程。必须先结束后台抓拍，
            # 再做正对目标的朝向修正，避免到点后旋转期间继续产生抓拍。
            on_translation_arrived = s.get("_on_translation_arrived")
            if callable(on_translation_arrived):
                on_translation_arrived()
            st = self._face_object_center(o)
            if st == "succeeded" and s.get("_announce_arrival", True):
                tasks_events.announce(f"已到达 {tasks_planning._zh(o)} 并正对目标")
        return st

    def _do_cruise(self, s):
        ids = s["target_ann_ids"]
        names = []
        for aid in ids:
            o = self.objs[aid]
            if o.get("marker"):                                 # 有现成点名 → 直接用
                names.append(o["marker"])
            else:                                               # 否则 §5.6 按坐标打点(已存在则更新)
                name = f"cruise_{aid}"
                tx, ty, th, _ = self._find_accessible_target(o)
                self.driver.insert_marker_by_pose(name, tx, ty, th)
                names.append(name)
        count = s.get("count", 1)
        self.driver.cruise(names, count=count)                 # §1.2
        st = self.driver.wait_until_settled()
        self.log.append(ExecLog("cruise", detail=f"巡游 {names} count={count}", status=st))
        return st

    def _do_observe(self, s):
        o = self.objs[s["target_ann_id"]]
        q = s.get("question", "")
        target_desc = (f"{tasks_planning._zh(o)} | 功能:{o.get('func_desc','')} "
                       f"| 位置:{o.get('position','')}")
        focus_desc = hardware_observation.observation_focus(target_desc, q, s.get("focus"))
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"obs_ann{o['ann_id']}.png")
        try:
            try:
                self.driver.capture(path)                       # 懒开相机(此时车已停稳, 不在导航中)
            except NotImplementedError:
                self.log.append(ExecLog("observe", o["ann_id"], f"问: {q}",
                                        observation="(相机未接入, 跳过观测 —— 仅完成移动)", status="succeeded"))
                return "succeeded"
            capture_only = bool(s.get("capture_only")) or not str(s.get("focus") or "").strip()
            if not os.path.exists(path):
                ans = "(已拍照, 视觉读取未启用)"
            elif capture_only:
                ans = models_vision.read_image(path, q)
            elif getattr(self.driver, "last_depth", None) is not None:
                ans = self._observe_full(path, target_desc, q, focus_desc)  # 深度闭环：转正 + 前进/后退 + 重拍
            else:
                ans = self._reframe_observe(path, target_desc, q, focus_desc)   # 无深度：只使用到达点当前画面
            ans = hardware_observation.clean_observation_answer(ans)
            if str(ans).startswith("搜索失败："):
                status = "not_found"
            elif str(ans).startswith("调整失败："):
                status = "failed"
            else:
                status = "succeeded"
            self.log.append(ExecLog(
                "observe", o["ann_id"], f"问: {q}", observation=ans, status=status,
            ))
            if s.get("_announce_observation", True) or status != "succeeded":
                tasks_events.announce(f"看到：{(ans or '').split('  (')[0].strip() or ans}")
            return status
        finally:
            if hasattr(self.driver, "close_camera"):
                self.driver.close_camera()                      # observe 完即关: 后续返回/导航不再空转 wedged

    # ─── 到点观察：无深度保持静态；有深度时允许受限的前进/后退闭环 ─────────────
    def _reframe_observe(self, path, target_desc, question, focus_desc=None, **_unused):
        """检查到达点当前画面；不再通过俯仰、偏航或平移寻找目标。"""
        focus_desc = hardware_observation.observation_focus(target_desc, question, focus_desc)
        framing = models_vision.read_framing(path, target_desc, question, focus_desc=focus_desc)
        if not framing.get("focus_visible"):
            return f"搜索失败：到达导航目标后，当前画面仍未发现观察关键区域“{focus_desc}”。"
        return models_vision.read_image(path, question) + "  (使用到达点当前画面)"

    def _observe_full(
        self, path, target_desc, question, focus_desc=None, max_tries=models_settings.APPROACH_MAX_TRIES,
    ):
        """用彩色+深度闭环调整取景，必要时转正、靠近或远离后重新拍摄。"""
        focus_desc = hardware_observation.observation_focus(target_desc, question, focus_desc)
        color_h, color_w = self.driver.last_color_shape
        total_moved = 0.0
        retreat_moved = 0.0
        translation_direction = 0
        tries_left = min(models_settings.APPROACH_MAX_TRIES, max(0, int(max_tries)))

        while tries_left > 0:
            self.driver.capture(path)
            depth = getattr(self.driver, "last_depth", None)
            if depth is None:
                return models_vision.read_image(path, question) + "  (深度丢失，停止距离调整)"

            framing = models_vision.read_framing(path, target_desc, question, focus_desc=focus_desc)
            focus_flag = framing.get("focus_visible")
            focus_visible = (
                bool(focus_flag) if focus_flag is not None else framing.get("cx_norm") is not None
            )
            complete_flag = framing.get("focus_complete")
            focus_complete = (
                bool(complete_flag) if complete_flag is not None else focus_visible
            )
            # 关键区域不完整时，即使模型误报 framed=true，也必须先扩大视野。
            if focus_complete and framing.get("framed"):
                return models_vision.read_image(path, question) + "  (取景达标)"

            if focus_visible:
                cx_norm = framing.get("focus_cx_norm", framing.get("cx_norm"))
                cy_norm = framing.get("focus_cy_norm", framing.get("cy_norm"))
            else:
                cx_norm = framing.get("navigation_cx_norm", framing.get("cx_norm"))
                cy_norm = framing.get("navigation_cy_norm", framing.get("cy_norm"))
            if cx_norm is None:
                return f"搜索失败：导航目标和观察关键区域“{focus_desc}”都未定位，未执行盲目移动。"

            horizontal_offset = float(cx_norm) - 0.5
            # 完整区域才做居中旋转；不完整时平移后退优先，避免调整次数耗在转向上。
            if focus_complete and abs(horizontal_offset) >= models_settings.CENTERED_THRESH:
                tries_left -= 1
                dyaw = max(
                    -models_settings.REFRAME_MAX_DYAW,
                    min(models_settings.REFRAME_MAX_DYAW, horizontal_offset * models_settings.CAM_W / models_settings.CAM_FX),
                ) * models_settings.YAW_SIGN
                status = hardware_navigation.rotate_in_place(self.driver, dyaw)
                if status != "succeeded":
                    return f"调整失败：转向返回 {status}，已停止观察距离调整。"
                continue

            target_distance = hardware_navigation.depth_at(
                depth, cx_norm, cy_norm, color_w=color_w, color_h=color_h,
            )
            decision = hardware_observation.decide_observe_translation(
                approach=framing.get("approach", "none"),
                requested_m=framing.get("move_m", 0.0),
                target_distance_m=target_distance,
                total_moved_m=total_moved,
                focus_visible=focus_visible,
                focus_complete=focus_complete,
                translation_direction=translation_direction,
                retreat_moved_m=retreat_moved,
            )
            print(
                "    [observe-distance] "
                f"focus_visible={focus_visible} focus_complete={focus_complete} "
                f"approach={framing.get('approach', 'none')} "
                f"depth={target_distance if target_distance is not None else 'invalid'}m "
                f"requested={framing.get('move_m', 0.0)}m "
                f"command={decision.meters:+.3f}m total={total_moved:.3f}m "
                f"decision={decision.reason} model_reason={framing.get('reason', '')}"
            )
            if decision.meters == 0:
                return models_vision.read_image(path, question) + f"  ({decision.reason}，按当前画面作答)"

            if decision.meters > 0 and not hardware_navigation.forward_clear(
                depth, decision.meters, color_w=color_w, color_h=color_h,
            ):
                return models_vision.read_image(path, question) + "  (前方存在障碍，停止靠近)"

            tries_left -= 1
            status, actual = hardware_navigation.forward_move(self.driver, decision.meters)
            if status != "succeeded":
                return f"调整失败：底盘距离移动返回 {status}，已停止。"

            minimum_progress = abs(decision.meters) * 0.5
            same_direction = actual * decision.meters > 0
            if not same_direction or abs(actual) < minimum_progress:
                return f"调整失败：底盘实际位移异常 {actual:+.2f}m，已停止。"
            maximum_progress = abs(decision.meters) + models_settings.OBSERVE_MOVE_TOL
            if abs(actual) > maximum_progress:
                return (
                    f"调整失败：底盘实际位移 {actual:+.2f}m 超过命令 "
                    f"{decision.meters:+.2f}m，已停止。"
                )
            if actual < 0 and retreat_moved + abs(actual) > models_settings.OBSERVE_MAX_RETREAT + 1e-9:
                return (
                    f"调整失败：底盘实际累计后退超过 {models_settings.OBSERVE_MAX_RETREAT:.2f}m，已停止。"
                )
            move_direction = 1 if decision.meters > 0 else -1
            if translation_direction == 0:
                translation_direction = move_direction
            if actual < 0:
                retreat_moved += abs(actual)
            total_moved += abs(actual)

        # 最后一次调整也会耗尽循环预算，必须在最终位姿重拍，不能读取移动前旧帧。
        self.driver.capture(path)
        return models_vision.read_image(path, question) + "  (观察调整次数用尽，按最终位姿画面作答)"

    def _do_return(self):
        self.driver.move_marker(self.START_MARKER)             # §1.1 回出发 marker
        st = self.driver.wait_until_settled(target=self.START_MARKER)
        self.log.append(ExecLog("return", detail=f"返回出发 ({self.start_pose[0]:.2f},{self.start_pose[1]:.2f})", status=st))
        tasks_events.announce("正在返回起点")
        return st

    def _do_cancel(self):
        self.driver.cancel_move()
        self.log.append(ExecLog("cancel", detail="取消当前移动"))
