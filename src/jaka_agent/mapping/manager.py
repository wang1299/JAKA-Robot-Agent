"""Plan, monitor and finalize mapping sessions."""
from __future__ import annotations
from jaka_agent import paths
import jaka_agent.mapping.graphs as mapping_graphs
import jaka_agent.tasks.runtime as tasks_runtime
import jaka_agent.web.settings as web_settings
import copy
import json
import math
import re
import shutil
import threading
import time
import uuid
from types import SimpleNamespace
from pathlib import Path

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
        web_settings.MAPPING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_sessions()

    @staticmethod
    def _session_path(session_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(session_id or "")):
            raise ValueError("建图任务 ID 非法")
        return web_settings.MAPPING_RUNS_DIR / str(session_id)

    def _load_sessions(self):
        for path in sorted(web_settings.MAPPING_RUNS_DIR.glob("*/session.json"), reverse=True):
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
        from jaka_agent.mapping.explore import load_agv_defaults

        host, http_port, tcp_port = load_agv_defaults(None)
        return SimpleNamespace(
            execute=True, confirm=True, config=None,
            agv_host=host, agv_http_port=http_port, agv_tcp_port=tcp_port, proto="auto",
            map_yaml_url=web_settings.SLAM_YAML_URL, map_image_url=web_settings.SLAM_IMAGE_URL,
            fetch_timeout=max(5.0, web_settings.SLAM_FETCH_TIMEOUT), out="",
            camera_index=0, camera_width=1280, camera_height=720, camera_fps=30,
            robot_radius=float(options["robot_radius"]), safety_margin=float(options["safety_margin"]),
            point_spacing=float(options["point_spacing"]), max_points=int(options["max_points"]),
            start_order=0, heading_step_deg=float(options["heading_step_deg"]),
            settle_time=float(options["settle_time"]), min_depth_ratio=0.05, zmq_server=None,
            distance_tolerance=0.08, theta_tolerance=0.08,
            move_timeout=120.0, status_interval=0.5,
        )

    def _plan_from_cached_slam(self, args, run_dir, pose):
        from jaka_agent.mapping.explore import SlamMap, build_route, heading_count

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
            "created_at": time.time(), "map_yaml_url": web_settings.SLAM_YAML_URL,
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
            from jaka_agent.mapping.explore import cancel_move
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
        source = next((path for path in (paths.runtime_path("shot.jpg"), paths.runtime_path("obs_ann18.png"), *paths.slam_candidates()) if path.exists()), None)
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
            from jaka_agent.mapping.explore import execute_plan
            from jaka_agent.mapping.bridge import MappingBridgeClient

            self.bridge = MappingBridgeClient(
                session_id, run_dir, web_settings.MAPPING_SERVER,
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
                    camera_factory=lambda *_: tasks_runtime.SharedMappingCamera(
                        self.web.tasks._hardware_camera_driver(), self.stop_event.is_set),
                )
            if self.stop_event.is_set():
                self.bridge.cancel()
                time.sleep(0.2)
                self._update(session_id, status="canceled", error=None)
                return
            self._update(session_id, status="uploading")
            self.bridge.finish()
            deadline = time.time() + max(30.0, web_settings.MAPPING_RESULT_TIMEOUT)
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
        except tasks_runtime.TaskCancelled:
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
        normalized = mapping_graphs._normalize_graph(raw, name)
        raw_objects = raw.get("objects") or []
        raw_count = len(raw_objects)
        functional = raw.get("func_relationships") or (raw.get("relationships") or {}).get("functional") or []
        positional = raw.get("pos_relationships") or (raw.get("relationships") or {}).get("positional") or []
        categories = {str(obj.get("category") or "Unknown") for obj in normalized["objects"]}
        temp = web_settings.HERE / f".{name}.tmp"
        temp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(paths.MAPS_DIR / name)
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
