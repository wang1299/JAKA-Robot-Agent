"""Load scene graphs and manage site SLAM images and calibration."""
from __future__ import annotations
from jaka_agent import paths
import jaka_agent.agent.map_evidence as agent_map_evidence
import jaka_agent.mapping.graphs as mapping_graphs
import jaka_agent.web.settings as web_settings
import base64
import copy
import json
import math
import os
import struct
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

def _image_size(path: Path) -> tuple[int, int]:
    """只用标准库读取 PNG/JPEG 尺寸，避免树莓派额外安装 Pillow。"""
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if suffix in (".jpg", ".jpeg") and data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            index += 2
            if marker in (0xD8, 0xD9):
                continue
            if index + 2 > len(data):
                break
            length = struct.unpack(">H", data[index:index + 2])[0]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                if index + 7 > len(data):
                    break
                height, width = struct.unpack(">HH", data[index + 3:index + 7])
                return int(width), int(height)
            index += max(2, length)
    raise ValueError("SLAM 图片必须是有效的 PNG 或 JPEG")


def _parse_slam_yaml(text: str) -> dict:
    """解析底盘 map.yaml 的简单键值；仅提取地图配准必需字段。"""
    values = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    try:
        resolution = float(values["resolution"])
        origin = json.loads(values["origin"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("map.yaml 缺少有效的 resolution/origin") from exc
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("map.yaml resolution 必须是正数")
    if not isinstance(origin, list) or len(origin) < 3:
        raise ValueError("map.yaml origin 必须是 [x,y,yaw]")
    try:
        origin = [float(origin[0]), float(origin[1]), float(origin[2])]
    except (TypeError, ValueError) as exc:
        raise ValueError("map.yaml origin 含非数值") from exc
    if not all(math.isfinite(value) for value in origin):
        raise ValueError("map.yaml origin 必须是有限数值")
    return {
        "resolution": resolution,
        "origin": origin,
        "image": values.get("image", "map.png").strip("'\""),
        "map_type": values.get("map_type", ""),
        "create_time": values.get("create_time"),
        "modify_time": values.get("modify_time"),
    }


class MappingCatalogMixin:
    def _default_slam_calibration(self, width: int, height: int) -> dict:
        """没有元数据时先按拟物体范围铺开，用户再在网页中精确标定。"""
        if (width, height) == (2502, 2797):
            # 当前 demo/3 地图的底盘 map.yaml 权威参数；断网使用本地 slam.png 时仍能对齐。
            return {
                "origin_x": -31.062019,
                "origin_y": -39.285198,
                "resolution": 0.03,
                "rotation": 0.0,
                "opacity": 0.58,
            }
        points = [obj["floor_xy"] for obj in self.graph["objects"]]
        min_x = min(float(point[0]) for point in points)
        max_x = max(float(point[0]) for point in points)
        min_y = min(float(point[1]) for point in points)
        max_y = max(float(point[1]) for point in points)
        range_x = max(4.0, max_x - min_x)
        range_y = max(4.0, max_y - min_y)
        resolution = max(range_x * 1.16 / width, range_y * 1.16 / height)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return {
            "origin_x": center_x - width * resolution / 2,
            "origin_y": center_y - height * resolution / 2,
            "resolution": resolution,
            "rotation": 0.0,
            "opacity": 0.58,
        }

    def _load_slam_image(self):
        """Load site maps and saved calibration before falling back to packaged examples."""
        config = {}
        try:
            if web_settings.SLAM_CONFIG_PATH.exists():
                config = json.loads(web_settings.SLAM_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[slam] 忽略无效标定配置: {exc}")
        candidates = []
        env_path = os.getenv("JAKA_WEB_SLAM_IMAGE", "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())
        # 默认优先采用人工确认过、去噪后的固定底图；slam_live 仅作为后备或手动刷新结果。
        candidates.extend(folder / "slam.png" for folder in (paths.MAPS_DIR, paths.LEGACY_DIR))
        configured = str(config.get("image") or "").strip()
        if configured and Path(configured).name == configured:
            candidates.append(paths.resolve_map(configured))
        candidates.extend(paths.slam_candidates())
        candidates.extend(web_settings.HERE / name for name in (
            "slam_live.png", "slam_live.jpg", "slam.jpg", "slam.jpeg",
        ))
        path = next((value.resolve() for value in candidates if value.is_file()), None)
        if path is None:
            return
        try:
            width, height = _image_size(path)
        except Exception as exc:
            print(f"[slam] 跳过 {path.name}: {exc}")
            return
        defaults = self._default_slam_calibration(width, height)
        use_saved = (width, height) != (2502, 2797) or config.get("calibration_version") == 3
        calibration = {}
        for key, default in defaults.items():
            try:
                value = float(config.get(key, default) if use_saved else default)
                calibration[key] = value if math.isfinite(value) else default
            except (TypeError, ValueError):
                calibration[key] = default
        if calibration["resolution"] <= 0:
            calibration["resolution"] = defaults["resolution"]
        calibration["opacity"] = min(1.0, max(0.05, calibration["opacity"]))
        self.slam_image_path = path
        self.slam_calibration = calibration
        print(f"[slam] 已加载 {path.name} ({width}x{height})，等待/沿用网页标定")

    def slam_snapshot(self) -> dict:
        with self.slam_lock:
            path = self.slam_image_path
            if path is None or not path.exists() or self.slam_calibration is None:
                return {
                    "available": False,
                    "source_url": web_settings.SLAM_IMAGE_URL,
                    "metadata_url": web_settings.SLAM_YAML_URL,
                    "fetch_error": self.slam_fetch_error,
                }
            width, height = _image_size(path)
            return {
                "available": True,
                "name": path.name,
                "width": width,
                "height": height,
                "image_url": f"/slam-map/image?v={path.stat().st_mtime_ns}",
                "calibration": copy.deepcopy(self.slam_calibration),
                "source_url": web_settings.SLAM_IMAGE_URL,
                "metadata_url": web_settings.SLAM_YAML_URL,
                "metadata": copy.deepcopy(self.slam_metadata),
                "fetched_at": self.slam_fetched_at,
                "fetch_error": self.slam_fetch_error,
            }

    def reload_local_slam(self) -> dict:
        """重新读取脚本目录中的固定底图，避免网页刷新按钮切回未去噪的底盘图片。"""
        with self.slam_lock:
            self.slam_metadata = None
            self.slam_fetched_at = None
            self.slam_fetch_error = None
            self._load_slam_image()
        return self.slam_snapshot()

    def slam_image(self) -> tuple[bytes, str]:
        with self.slam_lock:
            if self.slam_image_path is None or not self.slam_image_path.exists():
                raise FileNotFoundError("尚未加载 SLAM 地图图片")
            content_type = web_settings.SLAM_IMAGE_TYPES.get(self.slam_image_path.suffix.lower())
            if not content_type:
                raise ValueError("不支持的 SLAM 图片格式")
            return self.slam_image_path.read_bytes(), content_type

    def _auto_refresh_slam(self):
        try:
            result = self.refresh_slam_from_chassis()
            print(f"[slam] 已从底盘刷新 {result['name']} ({result['width']}x{result['height']})")
        except Exception as exc:
            self.slam_fetch_error = str(exc)
            print(f"[slam] 底盘地图抓取失败，继续使用本地图片: {exc}")

    def refresh_slam_from_chassis(self) -> dict:
        """成对抓取 map.yaml + 图片，以底盘 origin/resolution 做权威配准。"""
        parsed = urlparse(web_settings.SLAM_IMAGE_URL)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("JAKA_SLAM_IMAGE_URL 必须是有效的 HTTP/HTTPS 地址")
        yaml_parsed = urlparse(web_settings.SLAM_YAML_URL)
        if yaml_parsed.scheme not in ("http", "https") or not yaml_parsed.hostname:
            raise ValueError("JAKA_SLAM_YAML_URL 必须是有效的 HTTP/HTTPS 地址")
        yaml_request = Request(
            web_settings.SLAM_YAML_URL,
            headers={"User-Agent": "JAKA-Vision/1.0", "Accept": "text/yaml,text/plain,*/*;q=0.5"},
        )
        with urlopen(yaml_request, timeout=max(1.0, web_settings.SLAM_FETCH_TIMEOUT)) as response:
            yaml_payload = response.read(64 * 1024 + 1)
        if not yaml_payload or len(yaml_payload) > 64 * 1024:
            raise ValueError("底盘返回的 map.yaml 为空或过大")
        try:
            metadata = _parse_slam_yaml(yaml_payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("底盘 map.yaml 不是 UTF-8 文本") from exc
        image_url = urljoin(web_settings.SLAM_YAML_URL, metadata["image"])
        request = Request(
            image_url,
            headers={"User-Agent": "JAKA-Vision/1.0", "Accept": "image/png,image/jpeg;q=0.9"},
        )
        with urlopen(request, timeout=max(1.0, web_settings.SLAM_FETCH_TIMEOUT)) as response:
            payload = response.read(14 * 1024 * 1024 + 1)
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
        if not payload or len(payload) > 14 * 1024 * 1024:
            raise ValueError("底盘返回的 SLAM 图片为空或超过 14 MB")
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            suffix = ".png"
        elif payload.startswith(b"\xff\xd8"):
            suffix = ".jpg"
        else:
            raise ValueError(f"底盘返回的不是 PNG/JPEG（Content-Type={content_type or '未知'}）")
        temp = paths.MAPS_DIR / f".slam_fetch_{uuid.uuid4().hex}{suffix}"
        final = paths.MAPS_DIR / f"slam_live{suffix}"
        try:
            temp.write_bytes(payload)
            width, height = _image_size(temp)
            with self.slam_lock:
                temp.replace(final)
                self.slam_image_path = final.resolve()
                opacity = float((self.slam_calibration or {}).get("opacity", 0.58))
                origin = metadata["origin"]
                self.slam_calibration = {
                    "origin_x": origin[0],
                    "origin_y": origin[1],
                    "resolution": metadata["resolution"],
                    "rotation": math.degrees(origin[2]),
                    "opacity": min(1.0, max(0.05, opacity)),
                }
                self.slam_metadata = metadata
                self.slam_fetched_at = time.time()
                self.slam_fetch_error = None
                self._save_slam_config()
        finally:
            if temp.exists():
                temp.unlink()
        return self.slam_snapshot()

    def upload_slam_image(self, name: str, encoded: str) -> dict:
        suffix = Path(name).suffix.lower()
        if suffix not in web_settings.SLAM_IMAGE_TYPES:
            raise ValueError("SLAM 图片仅支持 PNG 或 JPEG")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise ValueError("SLAM 图片不是合法 Base64") from exc
        if not payload or len(payload) > 14 * 1024 * 1024:
            raise ValueError("SLAM 图片为空或超过 14 MB")
        temp = paths.MAPS_DIR / f".slam_upload_{uuid.uuid4().hex}{suffix}"
        final = paths.MAPS_DIR / f"slam_uploaded{suffix}"
        try:
            temp.write_bytes(payload)
            width, height = _image_size(temp)
            temp.replace(final)
        finally:
            if temp.exists():
                temp.unlink()
        self.slam_image_path = final.resolve()
        self.slam_calibration = self._default_slam_calibration(width, height)
        self.slam_metadata = None
        self._save_slam_config()
        return self.slam_snapshot()

    def _save_slam_config(self):
        if self.slam_image_path is None or self.slam_calibration is None:
            return
        value = {"calibration_version": 3, "image": self.slam_image_path.name, **self.slam_calibration}
        web_settings.SLAM_CONFIG_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_slam_calibration(self, value: dict) -> dict:
        if self.slam_image_path is None or self.slam_calibration is None:
            raise RuntimeError("请先加载 SLAM 地图图片")
        calibration = {}
        for key in ("origin_x", "origin_y", "resolution", "rotation", "opacity"):
            try:
                number = float(value.get(key))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"SLAM 标定参数 {key} 无效") from exc
            if not math.isfinite(number):
                raise ValueError(f"SLAM 标定参数 {key} 必须是有限数值")
            calibration[key] = number
        if not 0.0001 <= calibration["resolution"] <= 10:
            raise ValueError("米/像素必须在 0.0001 到 10 之间")
        if not 0.05 <= calibration["opacity"] <= 1:
            raise ValueError("透明度必须在 0.05 到 1 之间")
        calibration["rotation"] = ((calibration["rotation"] + 180) % 360) - 180
        self.slam_calibration = calibration
        self._save_slam_config()
        return self.slam_snapshot()

    def reset_slam_calibration(self) -> dict:
        if self.slam_image_path is None:
            raise RuntimeError("请先加载 SLAM 地图图片")
        if self.slam_metadata:
            origin = self.slam_metadata["origin"]
            opacity = float((self.slam_calibration or {}).get("opacity", 0.58))
            self.slam_calibration = {
                "origin_x": origin[0],
                "origin_y": origin[1],
                "resolution": self.slam_metadata["resolution"],
                "rotation": math.degrees(origin[2]),
                "opacity": min(1.0, max(0.05, opacity)),
            }
        else:
            width, height = _image_size(self.slam_image_path)
            self.slam_calibration = self._default_slam_calibration(width, height)
        self._save_slam_config()
        return self.slam_snapshot()

    def _load_initial_graph(self):
        preferred = os.getenv("JAKA_WEB_GRAPH", "zmq_scene_graph.json")
        candidates = []
        for name in (preferred, "zmq_scene_graph.json", "scene_graph_real.json", "scene_graph_sample.json"):
            path = paths.resolve_map(name)
            if path not in candidates:
                candidates.append(path)
        for path in candidates:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return mapping_graphs._normalize_graph(json.load(handle), path.name)
            except Exception as exc:
                print(f"[map] 跳过 {path.name}: {exc}")
        raise RuntimeError("找不到可用场景图; 请放置 zmq_scene_graph.json 或设置 JAKA_WEB_GRAPH")

    def graph_snapshot(self):
        with self.graph_lock:
            graph = copy.deepcopy(self.graph)
        graph["snapshot"] = agent_map_evidence.MapEvidence(graph).version
        return graph

    def map_files(self):
        """列出脚本目录下可转换为导航地图的 JSON, 大文件只在选择时完整返回。"""
        names = []
        for path in sorted({p for folder in paths.map_directories() for p in folder.glob("*.json")}):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    mapping_graphs._normalize_graph(json.load(handle), path.name)

                if path.name not in names:
                    names.append(path.name)
            except Exception:
                continue
        return names

    def select_map(self, name: str):
        task = self.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            raise RuntimeError("任务执行期间不能切换地图")
        safe_name = Path(name).name
        if safe_name != name or not safe_name.lower().endswith(".json"):
            raise ValueError("地图文件名非法")
        path = paths.resolve_map(safe_name)
        with path.open("r", encoding="utf-8") as handle:
            graph = mapping_graphs._normalize_graph(json.load(handle), safe_name)
        with self.graph_lock:
            self.graph = graph
        return self.graph_snapshot()

    def upload_map(self, name: str, data: dict):
        task = self.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            raise RuntimeError("任务执行期间不能切换地图")
        graph = mapping_graphs._normalize_graph(data, Path(name or "浏览器地图.json").name)
        with self.graph_lock:
            self.graph = graph
        return self.graph_snapshot()
