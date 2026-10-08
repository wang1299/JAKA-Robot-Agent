#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JAKA 自动建图双向通信桥。

树莓派由 robot_web.py 导入 MappingBridgeClient；服务器运行：
  python mapping_bridge.py server --listen tcp://0.0.0.0:5560 \
      --boxfusion-dir /path/to/3D_Code/BoxFusion \
      --boxfusion-command "python demo_online.py online --model-path ./models/cutr_rgbd.pth --config ./config/online.yaml --device cuda"

协议使用 ROUTER/DEALER。RGB-D 帧在树莓派落盘后进入 outbox，收到服务器 ACK
才删除，因此临时断线不会丢帧；服务端按 session_id + frame_id 去重。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import struct
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

try:
    import zmq
except Exception as exc:  # 网页未启用远程建图时允许缺少 pyzmq。
    zmq = None
    ZMQ_IMPORT_ERROR = exc
else:
    ZMQ_IMPORT_ERROR = None


PROTOCOL_VERSION = 1
DEFAULT_ENDPOINT = os.getenv("JAKA_MAPPING_SERVER", "tcp://127.0.0.1:5560")
MAX_IN_FLIGHT = 2


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _read_json(payload: bytes) -> dict:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("协议头必须是 JSON 对象")
    return value


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class MappingBridgeClient:
    """树莓派端可靠发送队列；所有 ZMQ 操作仅发生在后台线程。"""

    def __init__(self, session_id: str, run_dir, endpoint=DEFAULT_ENDPOINT, on_message=None):
        if zmq is None:
            raise RuntimeError(f"当前环境没有 pyzmq: {ZMQ_IMPORT_ERROR}")
        self.session_id = str(session_id)
        self.run_dir = Path(run_dir).resolve()
        self.endpoint = str(endpoint)
        self.outbox_dir = self.run_dir / "bridge_outbox"
        self.preview_dir = self.run_dir / "previews"
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "bridge_state.json"
        self.result_path = self.run_dir / "server_result.json"
        self.on_message = on_message
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.complete_event = threading.Event()
        self.thread = None
        self.finish_requested = False
        self.finish_sent = False
        self.cancel_requested = False
        self.cancel_sent = False
        self.start_acked = False
        self.last_start_send = 0.0
        self.last_finish_send = 0.0
        self.in_flight = {}
        self.state = {
            "session_id": self.session_id,
            "endpoint": self.endpoint,
            "connected": False,
            "status": "offline",
            "queued": len(list(self.outbox_dir.glob("*.json"))),
            "sent": 0,
            "acked": 0,
            "processed": 0,
            "objects": 0,
            "latest_preview": None,
            "error": None,
            "updated_at": time.time(),
        }
        self._load_state()

    def _load_state(self):
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            for key in ("acked", "processed", "objects", "latest_preview"):
                if key in saved:
                    self.state[key] = saved[key]
        except Exception:
            pass

    def _save_state(self):
        with self.lock:
            self.state["queued"] = len(list(self.outbox_dir.glob("*.json")))
            self.state["updated_at"] = time.time()
            value = dict(self.state)
        _atomic_json(self.state_path, value)

    def start(self, options=None):
        if self.thread and self.thread.is_alive():
            return
        self.options = dict(options or {})
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"MappingBridge-{self.session_id[:8]}")
        self.thread.start()

    def enqueue_frame(self, record, rgb_bytes: bytes, depth_bytes: bytes, pose, timestamp: float):
        frame_id = int(record["frame_id"])
        stem = f"{frame_id:08d}"
        header = {
            "protocol": PROTOCOL_VERSION,
            "type": "frame",
            "session_id": self.session_id,
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "point_index": int(record.get("explore_point", -1)),
            "heading_index": int(record.get("heading_index", -1)),
        }
        (self.outbox_dir / f"{stem}.jpg").write_bytes(rgb_bytes)
        (self.outbox_dir / f"{stem}.png").write_bytes(depth_bytes)
        try:
            pose_bytes = pose.astype("float32").tobytes()
        except Exception:
            import struct
            pose_bytes = struct.pack("<16f", *[float(value) for row in pose for value in row])
        (self.outbox_dir / f"{stem}.pose").write_bytes(pose_bytes)
        _atomic_json(self.outbox_dir / f"{stem}.json", header)
        with self.lock:
            self.state["queued"] = len(list(self.outbox_dir.glob("*.json")))
            self.state["status"] = "uploading"
        self.wake.set()

    def finish(self):
        self.finish_requested = True
        self.wake.set()

    def requeue_saved_frames(self):
        """从本地采集记录重建发送队列，用于网页或服务器重启后的完整续传。"""
        metadata_path = self.run_dir / "metadata.jsonl"
        if not metadata_path.exists():
            return 0
        restored = 0
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                frame_id = int(record["frame_id"])
                stem = f"{frame_id:08d}"
                if (self.outbox_dir / f"{stem}.json").exists():
                    continue
                rgb_path = self.run_dir.parent / str(record["rgb_path"])
                depth_path = self.run_dir.parent / str(record["depth_path"])
                pose_values = [float(value) for row in record["pose_4x4"] for value in row]
                header = {
                    "protocol": PROTOCOL_VERSION, "type": "frame",
                    "session_id": self.session_id, "frame_id": frame_id,
                    "timestamp": float(record["timestamp"]),
                    "point_index": int(record.get("explore_point", -1)),
                    "heading_index": int(record.get("heading_index", -1)),
                }
                (self.outbox_dir / f"{stem}.jpg").write_bytes(rgb_path.read_bytes())
                (self.outbox_dir / f"{stem}.png").write_bytes(depth_path.read_bytes())
                (self.outbox_dir / f"{stem}.pose").write_bytes(struct.pack("<16f", *pose_values))
                _atomic_json(self.outbox_dir / f"{stem}.json", header)
                restored += 1
            except Exception as exc:
                print(f"[bridge] 无法恢复采集记录: {exc}")
        if restored:
            self.wake.set()
            self._save_state()
        return restored

    def cancel(self):
        self.cancel_requested = True
        self.wake.set()

    def wait_complete(self, timeout=None):
        return self.complete_event.wait(timeout)

    def snapshot(self):
        with self.lock:
            value = dict(self.state)
        value["queued"] = len(list(self.outbox_dir.glob("*.json")))
        value["in_flight"] = len(self.in_flight)
        value["finish_requested"] = self.finish_requested
        value["result_ready"] = self.result_path.exists()
        return value

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def _emit(self, header, payload=None):
        if self.on_message is not None:
            try:
                self.on_message(dict(header), payload)
            except Exception as exc:
                print(f"[bridge] 消息回调失败: {exc}")

    def _send_start(self, dealer):
        now = time.time()
        if self.start_acked or now - self.last_start_send < 2.0:
            return
        dealer.send_multipart([_json_bytes({
            "protocol": PROTOCOL_VERSION,
            "type": "start",
            "session_id": self.session_id,
            "options": self.options,
        })])
        self.last_start_send = now

    def _outbox_items(self):
        for path in sorted(self.outbox_dir.glob("*.json")):
            try:
                header = json.loads(path.read_text(encoding="utf-8"))
                frame_id = int(header["frame_id"])
                stem = f"{frame_id:08d}"
                yield frame_id, header, stem
            except Exception as exc:
                print(f"[bridge] 忽略损坏 outbox 记录 {path.name}: {exc}")

    def _send_frames(self, dealer):
        now = time.time()
        for frame_id, header, stem in self._outbox_items():
            last_sent = self.in_flight.get(frame_id)
            if last_sent and now - last_sent < 2.0:
                continue
            if len(self.in_flight) >= MAX_IN_FLIGHT and frame_id not in self.in_flight:
                break
            try:
                dealer.send_multipart([
                    _json_bytes(header),
                    (self.outbox_dir / f"{stem}.jpg").read_bytes(),
                    (self.outbox_dir / f"{stem}.png").read_bytes(),
                    (self.outbox_dir / f"{stem}.pose").read_bytes(),
                    struct.pack("<d", float(header["timestamp"])),
                ])
            except FileNotFoundError:
                continue
            self.in_flight[frame_id] = now
            with self.lock:
                self.state["sent"] += 1

    def _ack_frame(self, frame_id):
        stem = f"{int(frame_id):08d}"
        for suffix in (".json", ".jpg", ".png", ".pose"):
            try:
                (self.outbox_dir / f"{stem}{suffix}").unlink()
            except FileNotFoundError:
                pass
        self.in_flight.pop(int(frame_id), None)
        with self.lock:
            self.state["acked"] = max(int(self.state.get("acked") or 0), int(frame_id) + 1)
            self.state["connected"] = True
            self.state["status"] = "uploading"

    def _handle(self, parts):
        header = _read_json(parts[0])
        msg_type = header.get("type")
        if msg_type == "started":
            self.start_acked = True
            with self.lock:
                self.state.update(connected=True, status="ready", error=None)
        elif msg_type == "ack":
            self._ack_frame(header["frame_id"])
        elif msg_type == "progress":
            with self.lock:
                self.state["processed"] = int(header.get("processed") or 0)
                self.state["objects"] = int(header.get("objects") or 0)
                self.state["status"] = str(header.get("status") or "processing")
        elif msg_type == "preview" and len(parts) >= 2:
            frame_id = int(header.get("frame_id") or 0)
            path = self.preview_dir / f"preview_{frame_id:08d}.jpg"
            path.write_bytes(parts[1])
            with self.lock:
                self.state["latest_preview"] = path.name
                self.state["processed"] = max(int(self.state.get("processed") or 0), frame_id + 1)
            header["local_name"] = path.name
        elif msg_type == "complete" and len(parts) >= 2:
            result = _read_json(parts[1])
            _atomic_json(self.result_path, result)
            with self.lock:
                self.state.update(status="complete", connected=True, objects=len(result.get("objects") or []), error=None)
            self.complete_event.set()
        elif msg_type == "error":
            with self.lock:
                self.state.update(status="error", error=str(header.get("message") or "服务器错误"))
        self._save_state()
        self._emit(header, parts[1] if len(parts) >= 2 else None)

    def _run(self):
        context = zmq.Context()
        dealer = context.socket(zmq.DEALER)
        identity = f"{socket.gethostname()}:{self.session_id}:{uuid.uuid4().hex[:8]}".encode("utf-8")
        dealer.setsockopt(zmq.IDENTITY, identity)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.setsockopt(zmq.SNDHWM, 4)
        dealer.setsockopt(zmq.RCVHWM, 8)
        dealer.connect(self.endpoint)
        poller = zmq.Poller()
        poller.register(dealer, zmq.POLLIN)
        try:
            while not self.stop_event.is_set() and not self.complete_event.is_set():
                self._send_start(dealer)
                if self.start_acked:
                    if self.cancel_requested and not self.cancel_sent:
                        dealer.send_multipart([_json_bytes({
                            "protocol": PROTOCOL_VERSION,
                            "type": "cancel",
                            "session_id": self.session_id,
                        })])
                        self.cancel_sent = True
                        with self.lock:
                            self.state["status"] = "canceled"
                    if not self.cancel_requested:
                        self._send_frames(dealer)
                    if not self.cancel_requested and self.finish_requested and not list(self.outbox_dir.glob("*.json")) and not self.in_flight:
                        now = time.time()
                        if not self.finish_sent or now - self.last_finish_send >= 3.0:
                            dealer.send_multipart([_json_bytes({
                                "protocol": PROTOCOL_VERSION,
                                "type": "finish",
                                "session_id": self.session_id,
                                "expected_frames": int(self.state.get("acked") or 0),
                            })])
                            self.finish_sent = True
                            self.last_finish_send = now
                            with self.lock:
                                self.state["status"] = "server_processing"
                events = dict(poller.poll(250))
                if dealer in events:
                    while True:
                        try:
                            self._handle(dealer.recv_multipart(zmq.NOBLOCK))
                        except zmq.Again:
                            break
                self.wake.clear()
        except Exception as exc:
            with self.lock:
                self.state.update(connected=False, status="error", error=str(exc))
            self._save_state()
        finally:
            dealer.close(0)
            context.term()


class MappingBridgeServer:
    """服务器端单任务桥；向 BoxFusion 原始 PULL 端转发四段数据。"""

    def __init__(self, listen, work_root, boxfusion_dir=None, boxfusion_command=None,
                 internal_endpoint="tcp://127.0.0.1:5555", mock=False):
        if zmq is None:
            raise RuntimeError(f"当前环境没有 pyzmq: {ZMQ_IMPORT_ERROR}")
        self.listen = listen
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.boxfusion_dir = Path(boxfusion_dir).resolve() if boxfusion_dir else None
        self.command = shlex.split(boxfusion_command) if boxfusion_command else None
        self.internal_endpoint = internal_endpoint
        self.mock = bool(mock)
        self.active = None

    def _send(self, router, identity, header, payload=None):
        parts = [identity, _json_bytes(header)]
        if payload is not None:
            parts.append(payload)
        router.send_multipart(parts)

    def _session_paths(self, session_id):
        root = self.work_root / session_id
        return root, root / "results", root / "previews"

    def _start(self, router, identity, header, context):
        session_id = str(header.get("session_id") or "")
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("session_id 非法")
        if self.active and self.active["session_id"] != session_id:
            self._send(router, identity, {"type": "error", "message": "建图服务器正在执行其他任务"})
            return
        if self.active:
            self.active["identity"] = identity
            self._send(router, identity, {"type": "started", "session_id": session_id, "resumed": True})
            return
        root, result_dir, preview_dir = self._session_paths(session_id)
        if result_dir.exists():
            shutil.rmtree(result_dir)
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)
        process = None
        forward = None
        if not self.mock:
            if not self.command or not self.boxfusion_dir:
                raise RuntimeError("真实服务器模式必须指定 --boxfusion-dir 和 --boxfusion-command")
            env = os.environ.copy()
            env.update({
                "BOXFUSION_SERVICE": "1",
                "BOXFUSION_SESSION_ID": session_id,
                "BOXFUSION_OUTPUT_DIR": str(result_dir),
                "BOXFUSION_PREVIEW_DIR": str(preview_dir),
            })
            log = (root / "boxfusion.log").open("ab", buffering=0)
            process = subprocess.Popen(self.command, cwd=self.boxfusion_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
            forward = context.socket(zmq.PUSH)
            forward.setsockopt(zmq.LINGER, 0)
            forward.setsockopt(zmq.SNDHWM, 4)
            forward.connect(self.internal_endpoint)
        self.active = {
            "session_id": session_id, "identity": identity, "root": root,
            "result_dir": result_dir, "preview_dir": preview_dir, "process": process,
            "forward": forward, "received": set(), "processed": 0,
            "sent_previews": set(), "finishing": False, "poses": [],
        }
        self._send(router, identity, {"type": "started", "session_id": session_id})
        print(f"[server] 建图任务已开始: {session_id}")

    def _frame(self, router, identity, header, payloads):
        if not self.active or self.active["session_id"] != header.get("session_id"):
            self._send(router, identity, {"type": "error", "message": "请先发送 start"})
            return
        if len(payloads) != 4:
            raise ValueError("frame 必须包含 RGB、depth、pose、timestamp 四段数据")
        frame_id = int(header["frame_id"])
        if frame_id not in self.active["received"]:
            self.active["received"].add(frame_id)
            self.active["poses"].append(payloads[2])
            if self.mock:
                preview = payloads[0]
                path = self.active["preview_dir"] / f"detection_{frame_id}.jpg"
                path.write_bytes(preview)
            else:
                self.active["forward"].send_multipart(payloads)
        self._send(router, identity, {"type": "ack", "session_id": header["session_id"], "frame_id": frame_id})

    def _mock_result(self):
        import struct
        objects = []
        for index, pose_bytes in enumerate(self.active["poses"][:12]):
            try:
                values = struct.unpack("<16f", pose_bytes)
                x, y = float(values[3]), float(values[7])
            except Exception:
                x = y = float(index)
            objects.append({
                "ann_id": index + 1,
                "category": "Mapping preview object",
                "category_zh": "建图测试物体",
                "geometry": {
                    "center_world": [x + 0.5, y, 0.5],
                    "aabb_half_sizes_world": [0.2, 0.2, 0.5],
                    "corners_world": [],
                },
            })
        return {
            "video_id": self.active["session_id"],
            "scene_graph_version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "objects": objects or [{
                "ann_id": 1, "category": "Mapping preview object", "category_zh": "建图测试物体",
                "geometry": {"center_world": [0.0, 0.0, 0.5], "aabb_half_sizes_world": [0.2, 0.2, 0.5]},
            }],
            "relationships": {"functional": [], "positional": []},
        }

    def _finish(self, router, identity, header):
        if not self.active or self.active["session_id"] != header.get("session_id"):
            self._send(router, identity, {"type": "error", "message": "没有对应的活动任务"})
            return
        self.active["identity"] = identity
        if self.active["finishing"]:
            self._send(router, identity, {"type": "progress", "status": "server_processing", "processed": self.active["processed"]})
            return
        self.active["finishing"] = True
        if self.mock:
            path = self.active["result_dir"] / f"{self.active['session_id']}_scene_graph.json"
            _atomic_json(path, self._mock_result())
        else:
            self.active["forward"].send_multipart([_json_bytes({
                "type": "end_session", "session_id": self.active["session_id"]
            })])
        self._send(router, identity, {"type": "progress", "status": "server_processing", "processed": self.active["processed"]})

    def _preview_tick(self, router):
        if not self.active:
            return
        candidates = sorted(self.active["preview_dir"].glob("*.*"), key=lambda path: path.stat().st_mtime_ns)
        for path in candidates:
            if path.name in self.active["sent_previews"] or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            digits = "".join(ch for ch in path.stem if ch.isdigit())
            frame_id = int(digits or self.active["processed"])
            self.active["sent_previews"].add(path.name)
            self.active["processed"] += 1
            self._send(self.router, self.active["identity"], {
                "type": "preview", "session_id": self.active["session_id"],
                "frame_id": frame_id, "processed": self.active["processed"],
            }, path.read_bytes())
            self._send(self.router, self.active["identity"], {
                "type": "progress", "session_id": self.active["session_id"],
                "status": "processing", "processed": self.active["processed"],
            })

    def _complete_tick(self):
        if not self.active:
            return
        process = self.active["process"]
        if not self.active["finishing"]:
            if process is not None and process.poll() is not None:
                self._send(self.router, self.active["identity"], {
                    "type": "error", "session_id": self.active["session_id"],
                    "message": f"BoxFusion 意外退出，exit={process.returncode}，请查看 boxfusion.log",
                })
                self._reset_active()
            return
        if process is not None and process.poll() is None:
            return
        candidates = sorted(self.active["result_dir"].glob("*_scene_graph.json"), key=lambda path: path.stat().st_mtime_ns)
        if not candidates:
            if process is not None:
                self._send(self.router, self.active["identity"], {
                    "type": "error", "session_id": self.active["session_id"],
                    "message": f"BoxFusion 已退出但没有生成场景图，exit={process.returncode}",
                })
                self._reset_active()
            return
        result_path = candidates[-1]
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result.get("objects"), (list, dict)):
                raise ValueError("场景图缺少 objects")
        except Exception as exc:
            self._send(self.router, self.active["identity"], {"type": "error", "message": f"场景图校验失败: {exc}"})
            self._reset_active()
            return
        self._send(self.router, self.active["identity"], {
            "type": "complete", "session_id": self.active["session_id"], "objects": len(result.get("objects") or []),
        }, _json_bytes(result))
        print(f"[server] 建图任务完成: {self.active['session_id']} -> {result_path}")
        self._reset_active()

    def _reset_active(self):
        if not self.active:
            return
        process = self.active.get("process")
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        forward = self.active.get("forward")
        if forward is not None:
            forward.close(0)
        self.active = None

    def serve_forever(self):
        context = zmq.Context()
        router = context.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        router.setsockopt(zmq.RCVHWM, 16)
        router.bind(self.listen)
        self.router = router
        poller = zmq.Poller()
        poller.register(router, zmq.POLLIN)
        print(f"[server] Mapping bridge listening: {self.listen} | mock={self.mock}")
        try:
            while True:
                events = dict(poller.poll(150))
                if router in events:
                    parts = router.recv_multipart()
                    identity, header_bytes, payloads = parts[0], parts[1], parts[2:]
                    try:
                        header = _read_json(header_bytes)
                        if int(header.get("protocol") or PROTOCOL_VERSION) != PROTOCOL_VERSION:
                            raise ValueError("协议版本不兼容")
                        msg_type = header.get("type")
                        if msg_type == "start":
                            self._start(router, identity, header, context)
                        elif msg_type == "frame":
                            self._frame(router, identity, header, payloads)
                        elif msg_type == "finish":
                            self._finish(router, identity, header)
                        elif msg_type == "cancel":
                            self._reset_active()
                        else:
                            raise ValueError(f"未知消息类型: {msg_type}")
                    except Exception as exc:
                        self._send(router, identity, {"type": "error", "message": str(exc)})
                self._preview_tick(router)
                self._complete_tick()
        except KeyboardInterrupt:
            print("\n[server] 正在关闭")
        finally:
            if self.active and self.active.get("process") and self.active["process"].poll() is None:
                self.active["process"].send_signal(signal.SIGINT)
            self._reset_active()
            router.close(0)
            context.term()


def main():
    parser = argparse.ArgumentParser(description="JAKA 自动建图双向通信桥")
    sub = parser.add_subparsers(dest="mode", required=True)
    server = sub.add_parser("server", help="在 BoxFusion 服务器运行")
    server.add_argument("--listen", default="tcp://0.0.0.0:5560")
    server.add_argument("--work-root", default="mapping_server_runs")
    server.add_argument("--boxfusion-dir", default=None)
    server.add_argument("--boxfusion-command", default=None)
    server.add_argument("--internal-endpoint", default="tcp://127.0.0.1:5555")
    server.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mode == "server":
        MappingBridgeServer(
            args.listen, args.work_root, args.boxfusion_dir, args.boxfusion_command,
            args.internal_endpoint, args.mock,
        ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
