"""HTTP routes for conversations, tasks, devices and static assets."""
from __future__ import annotations
import jaka_agent.agent.skills as agent_skills
import jaka_agent.diagnostics as diagnostics
import jaka_agent.hardware.audio as hardware_audio
import jaka_agent.mapping.graphs as mapping_graphs
import jaka_agent.tasks.validation as tasks_validation
import jaka_agent.web.document as web_document
import jaka_agent.web.settings as web_settings
import jaka_agent.web.state as web_state
import json
import mimetypes
import re
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urljoin, urlparse

class RobotWebHandler(BaseHTTPRequestHandler):
    server_version = "JakaVision/1.0"

    @property
    def state(self) -> web_state.RobotWebState:
        return self.server.state

    def log_message(self, fmt, *args):
        diagnostics.LOGGER.info("[http] %s %s - %s", self.command, self.path, fmt % args)

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
        if length <= 0 or length > web_settings.MAX_BODY_BYTES:
            raise ValueError("请求体为空或过大")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是合法 JSON") from exc

    def _read_reference_image(self) -> bytes:
        """读取与 /captures 相同的原始图片字节 HTTP body。"""
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type not in {info[0] for info in web_settings.REFERENCE_IMAGE_TYPES.values()}:
            raise ValueError("参考图片 Content-Type 不支持")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("无效 Content-Length") from exc
        if length <= 0 or length > web_settings.MAX_REFERENCE_BYTES:
            raise ValueError("参考图片请求体为空或过大")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("参考图片上传不完整")
        return payload

    def _read_mobile_audio(self) -> bytes:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in web_settings.AUDIO_CONTENT_TYPES:
            raise hardware_audio.AudioRequestError("录音格式不支持", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise hardware_audio.AudioRequestError("无效 Content-Length", HTTPStatus.BAD_REQUEST) from exc
        if length <= 0:
            raise hardware_audio.AudioRequestError("录音内容为空", HTTPStatus.BAD_REQUEST)
        if length > web_settings.MAX_AUDIO_BYTES:
            raise hardware_audio.AudioRequestError(
                f"录音不能超过 {web_settings.MAX_AUDIO_BYTES // (1024 * 1024)} MB",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise hardware_audio.AudioRequestError("录音上传不完整", HTTPStatus.BAD_REQUEST)
        return payload

    def do_GET(self):
        request_url = urlparse(self.path)
        path = request_url.path
        if path == '/api/conversation/media':
            query = parse_qs(request_url.query)
            try:
                cid = (query.get('id') or [''])[0]
                turn_id = (query.get('turn_id') or [None])[0]
                limit = int((query.get('limit') or ['200'])[0])
                offset = int((query.get('offset') or ['0'])[0])
                rows = self.state.media_archive.list(cid, turn_id, limit, offset)
                self._send_json({'media':rows, 'limit':limit, 'offset':offset,
                                 'next_offset':offset+len(rows) if len(rows)==limit else None})
            except ValueError as exc:
                self._error(exc, HTTPStatus.BAD_REQUEST)
            return
        if path in ('/api/conversations', '/api/conversation'):
            try:
                store = self.state.conversation_store
                if path == '/api/conversations':
                    self._send_json({'conversations':store.list()})
                else:
                    cid = (parse_qs(request_url.query).get('id') or [''])[0]
                    self._send_json({'conversation':store.conversation(cid)})
            except ValueError as exc:
                self._error(exc, HTTPStatus.NOT_FOUND)
            return
        if path == "/":
            self._send_bytes(web_document._document(), "text/html; charset=utf-8")
            return
        from jaka_agent.paths import STATIC_DIR
        static_assets = {
            "/static/css/app.css": (STATIC_DIR / "css/app.css", "text/css; charset=utf-8"),
            "/static/css/scene.css": (STATIC_DIR / "css/scene.css", "text/css; charset=utf-8"),
            "/static/js/app.js": (STATIC_DIR / "js/app.js", "text/javascript; charset=utf-8"),
            "/static/js/scene.js": (STATIC_DIR / "js/scene.js", "text/javascript; charset=utf-8"),

            "/manifest.webmanifest": (web_settings.PWA_MANIFEST_PATH, "application/manifest+json; charset=utf-8"),
            "/service-worker.js": (web_settings.SERVICE_WORKER_PATH, "text/javascript; charset=utf-8"),
            "/pwa-icon.svg": (web_settings.PWA_ICON_PATH, "image/svg+xml; charset=utf-8"),
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
                self._send_bytes(web_settings.SCENE_VIEWER_PATH.read_bytes(), "text/html; charset=utf-8")
            except OSError as exc:
                self._error(f"三维拟物体查看页不可用: {exc}", HTTPStatus.NOT_FOUND)
            return
        if path == "/api/health":
            self._send_json({"ok": True, "mock": self.state.mock})
            return
        if path == "/api/skills":
            self._send_json({"skills": agent_skills.skill_catalog()})
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
            cid = (parse_qs(request_url.query).get('conversation_id') or [None])[0]
            self._send_json({
                "robot": self.state.tasks.robot_status(),
                "task": self.state.conversation_task(cid) if getattr(self.state,'conversation_store',None) else self.state.tasks.snapshot(),
                "robot_busy": self.state.tasks.is_busy(),
            })
            return
        if path == "/api/tracks":
            self._send_json(self.state.tasks.tracks_snapshot())
            return
        if path == "/api/task/status":
            cid = (parse_qs(request_url.query).get('conversation_id') or [None])[0]
            task = self.state.tasks.snapshot()
            self._send_json({"task": self.state.conversation_task(cid) if getattr(self.state,'conversation_store',None) else task,
                             "busy": self.state.tasks.is_busy(),
                             "pending": bool(task and task.get('status') in ('planned','needs_clarification'))})
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
            if name not in mapping_graphs.MAP_ICON_NAMES:
                self._error("图标不存在", HTTPStatus.NOT_FOUND)
                return
            icon_path = web_settings.MAP_ICON_DIR / f"{name}.svg"
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
            archive = getattr(self.state, 'media_archive', None)
            video_path = archive.resolve(path) if archive else web_settings.VIDEO_DIR / f"{video_id}.{suffix}"
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
            self._send_bytes(image_path.read_bytes(), web_settings.REFERENCE_IMAGE_TYPES[suffix][0])
            return
        self._error("接口不存在", HTTPStatus.NOT_FOUND)

    def _conversation_plan(self, payload, callback):
        store = getattr(self.state, 'conversation_store', None)
        if not store:
            return callback()
        cid = payload.get('conversation_id')
        user_id = payload.get('user_message_id') or uuid.uuid4().hex
        assistant_id = payload.get('assistant_message_id') or uuid.uuid4().hex
        store.validate_turn_ids(cid, user_id, assistant_id)
        reference = payload.get('reference') or {}
        if not isinstance(reference, dict) or (reference and not store.owns_media(cid, reference.get('image_url', ''))):
            raise ValueError('参考图片未绑定当前会话，请重新上传')
        task = self.state.plan_for_conversation(cid, callback, user_id)
        now = time.time()*1000
        store.message(cid, {'id': user_id,
                           'role':'user','text':payload.get('instruction',''), 'createdAt':now,
                           'referenceStoredImage':(payload.get('reference') or {}).get('image_url','')})
        store.message(cid, {'id':assistant_id,
                           'role':'assistant','text':'任务计划已生成，请检查任务卡并确认后执行。',
                           'state':'done','createdAt':now+1,'taskId':task['id'],'task':task})
        return task

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
                if getattr(self.state, 'conversation_store', None):
                    cid = (parse_qs(request_url.query).get('conversation_id') or [''])[0]
                    self.state.conversation_store.conversation(cid)
                image_bytes = self._read_reference_image()
                if getattr(self.state, 'media_archive', None):
                    turn_id = (parse_qs(request_url.query).get('turn_id') or [uuid.uuid4().hex])[0]
                    result = self.state.save_reference_upload(name, image_bytes,
                        media_scope={'conversation_id':cid,'turn_id':turn_id})
                else:
                    result = self.state.save_reference_upload(name, image_bytes)
                if getattr(self.state, 'conversation_store', None):
                    cid = (parse_qs(request_url.query).get('conversation_id') or [''])[0]
                    self.state.conversation_store.event(cid, 'reference_upload', result)
                self._send_json(result)
                return
            payload = self._read_json()
            if path in ('/api/conversation/create','/api/conversation/import','/api/conversation/delete'):
                store = self.state.conversation_store
                managing_delete = path.endswith('/delete')
                if managing_delete and not self.state.agent_lock.acquire(blocking=False):
                    raise ValueError('对话正在处理，请稍后再管理会话')
                try:
                    if path.endswith('/create'):
                        result = store.create(payload.get('id'), payload.get('title','新对话'))
                    elif path.endswith('/import'):
                        store.import_legacy(payload)
                        result = None
                    else:
                        with self.state.tasks.lock:
                            live_task = self.state.tasks.snapshot() or {}
                            if live_task.get('conversation_id') == payload.get('id') and self.state.tasks.is_busy():
                                raise ValueError('任务仍在运行或清理资源，请稍后再删除会话')
                            store.delete(payload.get('id'))
                        self.state.agent_sessions.pop(payload.get('id'), None)
                        result = None
                    self._send_json({'ok':True,'conversation':result})
                finally:
                    if managing_delete:
                        self.state.agent_lock.release()
                return
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

                disconnected = False
                def emit(event):
                    nonlocal disconnected
                    if disconnected:
                        return
                    try:
                        self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        # Closing a browser does not discard the originating conversation's result.
                        disconnected = True

                try:
                    result = self.state.agent_chat(payload, emit)
                    emit({"type": "final", **result})
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception as exc:
                    diagnostics.LOGGER.warning("[agent] request failed type=%s", type(exc).__name__)
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
                diagnostics.LOGGER.info(
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
                if getattr(self.state, 'conversation_store', None):
                    self.state.conversation_store.conversation(payload.get('conversation_id'))
                if getattr(self.state, 'media_archive', None):
                    result = self.state.save_reference_image(name, encoded, media_scope={
                        'conversation_id':payload.get('conversation_id'), 'turn_id':payload.get('turn_id') or uuid.uuid4().hex})
                else:
                    result = self.state.save_reference_image(name, encoded)
                if getattr(self.state, 'conversation_store', None):
                    self.state.conversation_store.event(payload.get('conversation_id'), 'reference_upload', result)
                self._send_json(result)
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
                self._send_json({"task": self._conversation_plan(payload, lambda: self.state.tasks.plan(instruction))})
                return
            if path == "/api/task/patrol/plan":
                instruction = str(payload.get("instruction") or "").strip()
                snapshot = payload.get("map_snapshot")
                if not instruction or not isinstance(snapshot, str) or not snapshot:
                    raise ValueError("请先加载地图并选择巡逻地点")
                self._send_json({"task": self._conversation_plan(payload, lambda: self.state.tasks.plan_skill(
                    "patrol", instruction, target_ann_ids=payload.get("target_ann_ids"),
                    rounds=payload.get("rounds", -1), expected_map_snapshot=snapshot,
                ))})
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
                content_type = web_settings.REFERENCE_IMAGE_TYPES[suffix][0]
                expected_url = f"/references/{reference_id}{suffix}"
                if reference.get("image_url") != expected_url or reference.get("content_type") != content_type:
                    raise ValueError("参考图片元信息无效")
                safe_reference = {
                    "reference_id": reference_id,
                    "suffix": suffix,
                    "content_type": content_type,
                    "image_url": expected_url,
                }
                self._send_json({"task": self._conversation_plan(payload, lambda: self.state.tasks.plan_skill("find_object", instruction, reference=safe_reference))})
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
                content_type = web_settings.REFERENCE_IMAGE_TYPES[suffix][0]
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
                    "task": self._conversation_plan(payload, lambda: self.state.tasks.plan_skill(
                        "welcome", instruction,
                        reference=safe_reference,
                        pickup_ann_id=pickup_ann_id,
                        return_ann_id=return_ann_id,
                    ))
                })
                return
            if path == "/api/task/execute":
                task_id = str(payload.get("task_id") or "")
                if getattr(self.state, 'conversation_store', None):
                    self.state.require_task_owner(task_id, payload.get('conversation_id'))
                    self.state.conversation_store.event(payload.get('conversation_id'), 'user_confirmed_task', {'task_id':task_id})
                self._send_json({"task": self.state.tasks.execute(task_id)})
                return
            if path == "/api/task/welcome-confirm":
                task_id = str(payload.get("task_id") or "")
                if getattr(self.state, 'conversation_store', None):
                    self.state.require_task_owner(task_id, payload.get('conversation_id'))
                self._send_json({"task": self.state.tasks.confirm_welcome_guest(
                    task_id, payload.get("confirmation_id"), payload.get("accepted"),
                    payload.get("conversation_id"),
                )})
                return
            if path == "/api/task/cancel":
                task_id = str(payload.get("task_id") or "")
                if getattr(self.state, 'conversation_store', None):
                    self.state.require_task_owner(task_id, payload.get('conversation_id'))
                    self.state.conversation_store.event(payload.get('conversation_id'), 'user_canceled_task', {'task_id':task_id})
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
        except tasks_validation.TaskSafetyError as exc:
            diagnostics.LOGGER.warning("[task] execution blocked: %s", exc)
            self._error(exc, HTTPStatus.CONFLICT)
        except hardware_audio.AudioRequestError as exc:
            self._error(exc, exc.status)
        except FileNotFoundError as exc:
            self._error(exc, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._error(exc, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            diagnostics._log_exception(
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
            diagnostics._log_exception(
                "[http] DELETE request failed",
                exc,
                method="DELETE",
                path=request_url.path,
                client=self.address_string(),
            )
            self._error(exc, HTTPStatus.INTERNAL_SERVER_ERROR)
