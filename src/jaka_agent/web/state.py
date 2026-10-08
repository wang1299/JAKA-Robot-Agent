"""Compose robot, model, map and conversation services for the web server."""
from __future__ import annotations
import jaka_agent.mapping.manager as mapping_manager
import jaka_agent.tasks.events as tasks_events
import jaka_agent.tasks.manager as tasks_manager
import jaka_agent.web.settings as web_settings
from jaka_agent import paths
import os
import threading
from jaka_agent.mapping.catalog import MappingCatalogMixin
from jaka_agent.hardware.service import HardwareServiceMixin
from jaka_agent.agent.service import AgentServiceMixin
from jaka_agent.models.perception import ModelsPerceptionMixin

class RobotWebState(MappingCatalogMixin, HardwareServiceMixin, AgentServiceMixin, ModelsPerceptionMixin):
    def __init__(self, mock=False):
        paths.ensure_data_directories()
        self.mock = mock
        self.camera_lock = threading.Lock()
        self.infer_lock = threading.Lock()
        self.voice_lock = threading.Lock()
        self.agent_lock = threading.Lock()
        self.agent_sessions = {}
        from jaka_agent.storage.memory import ConversationStore
        self.conversation_store = ConversationStore(os.getenv('JAKA_CONVERSATION_DB', str(paths.runtime_path('conversation_data') / 'conversations.sqlite3')))
        self.conversation_store.claim_runtime()
        self.conversation_store.recover()
        from jaka_agent.storage.media import MediaArchive
        self.media_archive = MediaArchive(self.conversation_store,
            os.getenv('JAKA_MEDIA_ROOT', str(self.conversation_store.path.parent / 'media')),
            web_settings.CAPTURE_DIR, web_settings.VIDEO_DIR)
        self.graph_lock = threading.RLock()
        self.slam_lock = threading.RLock()
        self.voice = None
        web_settings.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        web_settings.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        # Browser-only historical conversations may be imported on a later visit.
        # Do not age-delete their photos before their ownership has been migrated.
        # Media retention is explicit maintenance now, not a startup side effect.
        self.graph = self._load_initial_graph()
        self.slam_image_path = None
        self.slam_calibration = None
        self.slam_metadata = None
        self.slam_fetched_at = None
        self.slam_fetch_error = None
        self._load_slam_image()
        self.tasks = tasks_manager.RobotTaskManager(self)
        self.mapping = mapping_manager.MappingManager(self)
        # 默认固定使用脚本目录中的去噪 slam.png；确需底盘原图时显式设置 JAKA_SLAM_AUTO_FETCH=1。
        if not self.mock and web_settings.SLAM_IMAGE_URL and os.getenv("JAKA_SLAM_AUTO_FETCH", "0") == "1":
            threading.Thread(target=self._auto_refresh_slam, daemon=True, name="SlamMapFetch").start()

    def close(self):
        task = self.tasks.snapshot()
        if task and task.get("status") in ("running", "canceling"):
            try:
                self.tasks.cancel(task["id"])
            except Exception:
                pass
        self.tasks.close()
        if self.voice is not None:
            voice = self.voice
            try:


                if tasks_events._VOICE is voice:
                    tasks_events._VOICE = None
            except Exception:
                pass
            try:
                voice.close()
            except Exception:
                pass
            self.voice = None
