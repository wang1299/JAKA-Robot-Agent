"""Agent-to-robot task contracts with synthetic observations and no external access."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jaka_agent.agent.runner import AgentRunner, execution_snapshot
from jaka_agent.replay.fixtures import CASES, FEEDBACK_QUESTION, scripted_completion
from jaka_agent.replay.state import ReplayWebState
from jaka_agent.web import settings
from jaka_agent.web.handler import RobotWebHandler
from jaka_agent.web.server import RobotWebServer


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict(os.environ, {"JAKA_CONVERSATION_DB": str(root / "memory.sqlite3"),
                                                       "JAKA_MEDIA_ROOT": str(root / "media")}))
        for name, value in {"CAPTURE_DIR": root / "captures", "VIDEO_DIR": root / "videos",
                            "TRACK_HISTORY_PATH": root / "tracks.json"}.items():
            self.stack.enter_context(patch.object(settings, name, value))
        self.state = ReplayWebState(tick=0.001)
        self.addCleanup(self.state.close)
        self.cid = "replay-session-A"

    def plan(self, case="find_object", cid=None):
        return self.state.begin_replay(case, cid or self.cid)["result"]

    def execute(self, task):
        self.state.tasks.execute(task["id"])
        self.state.tasks.thread.join(3)
        self.assertFalse(self.state.tasks.is_busy())
        return self.state.tasks.snapshot()

    def wait_guest(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            task = self.state.tasks.snapshot()
            if (task.get("guest_confirmation") or {}).get("status") == "pending":
                return task
            time.sleep(0.01)
        self.fail("Guest confirmation was not requested")

    def test_agent_uses_actual_tools_and_waits_for_confirmation(self):
        with patch("jaka_agent.hardware.navigation.JakaTCPDriver", side_effect=AssertionError("hardware forbidden")):
            answer = self.plan()
        self.assertEqual([item["tool"] for item in answer["tool_trace"]], ["get_skill_context", "plan_find_object"])
        self.assertEqual(answer["task"]["status"], "planned")
        self.assertFalse(self.state.tasks.is_busy())
        self.assertEqual(self.state.tasks.mock_pose, [0, 0, 0])
        self.assertEqual(self.state.tasks.snapshot()["observations"], [])

    def test_search_uses_observations_and_reports_recorded_result(self):
        task = self.execute(self.plan()["task"])
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual([item["found"] for item in task["observations"]], [False, True])
        self.assertEqual(task["match"]["ann_id"], 22)
        feedback = self.state.replay_feedback(self.cid)
        self.assertEqual(feedback["execution"]["state"], "succeeded")
        self.assertIn(task["result_text"], feedback["text"])
        self.assertTrue(all(item["source_type"] == "synthetic_replay" for item in task["observations"]))

    def test_guest_requires_photo_specific_owner_confirmation(self):
        task = self.plan("welcome")["task"]
        self.state.tasks.execute(task["id"])
        pending = self.wait_guest()
        position = self.state.tasks.mock_pose[:]
        time.sleep(0.04)
        self.assertEqual(position, self.state.tasks.mock_pose)
        token = pending["guest_confirmation"]["id"]
        with self.assertRaises(ValueError):
            self.state.tasks.confirm_welcome_guest(task["id"], token, True, "replay-session-B")
        with self.assertRaises(ValueError):
            self.state.tasks.confirm_welcome_guest(task["id"], "stale-photo", True, self.cid)
        self.state.tasks.confirm_welcome_guest(task["id"], token, True, self.cid)
        self.state.tasks.thread.join(3)
        self.assertEqual(self.state.tasks.snapshot()["status"], "succeeded")
        self.assertEqual(self.state.tasks.mock_pose[:2], [6, 4])

    def test_guest_rejection_waits_for_a_new_photo_and_can_cancel(self):
        task = self.plan("welcome")["task"]
        self.state.tasks.execute(task["id"])
        pending = self.wait_guest()
        token = pending["guest_confirmation"]["id"]
        self.state.tasks.confirm_welcome_guest(task["id"], token, False, self.cid)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            newer = self.state.tasks.snapshot().get("guest_confirmation", {})
            if newer.get("status") == "pending" and newer.get("id") != token:
                break
            time.sleep(0.01)
        else:
            self.fail("Rejection did not request another photo")
        self.assertEqual(self.state.tasks.mock_pose[:2], [1, 4])
        self.state.tasks.cancel(task["id"])
        self.state.tasks.thread.join(3)
        self.assertEqual(self.state.tasks.snapshot()["status"], "canceled")

    def test_navigation_failure_prevents_observation_and_remains_visible(self):
        task = self.execute(self.plan("blocked")["task"])
        self.assertEqual(task["status"], "aborted")
        self.assertEqual(task["observations"], [])
        feedback = self.state.replay_feedback(self.cid)
        self.assertEqual(feedback["execution"]["state"], "aborted")
        self.assertIn("中止", feedback["text"])
        self.assertNotIn("已完成", feedback["text"])

    def test_missing_reference_clarifies_without_creating_task(self):
        answer = self.plan("missing_reference")
        self.assertTrue(answer["requires_clarification"])
        self.assertIsNone(self.state.tasks.snapshot())
        self.assertEqual(self.state.tasks.mock_pose, [0, 0, 0])

    def test_cancel_during_navigation_discards_later_observations(self):
        task = self.plan()["task"]
        entered, release = threading.Event(), threading.Event()
        original = self.state.tasks._mock_move
        def paused_move(target):
            entered.set()
            if not release.wait(3):
                raise AssertionError("Test barrier timed out")
            return original(target)
        with patch.object(self.state.tasks, "_mock_move", paused_move):
            try:
                self.state.tasks.execute(task["id"])
                self.assertTrue(entered.wait(2))
                self.state.tasks.cancel(task["id"])
            finally:
                release.set()
                self.state.tasks.thread.join(3)
        self.assertEqual(self.state.tasks.snapshot()["status"], "canceled")
        self.assertEqual(self.state.tasks.snapshot()["observations"], [])

    def test_map_change_invalidates_plan_before_motion(self):
        task = self.plan()["task"]
        self.state.graph["objects"][0]["floor_xy"][0] += 0.5
        with self.assertRaises(RuntimeError):
            self.state.tasks.execute(task["id"])
        self.assertFalse(self.state.tasks.is_busy())

    def test_unknown_free_text_never_dispatches_scripted_action(self):
        self.state.conversation_store.create(self.cid)
        answer = self.state.agent_chat({"question": "随便去一个地方", "conversation_id": self.cid})
        self.assertFalse(answer.get("task"))
        self.assertIsNone(self.state.tasks.snapshot())
        self.assertIn("固定案例", answer["text"])

    def test_trace_is_scoped_to_conversation(self):
        self.plan()
        self.assertTrue(self.state.replay_snapshot(self.cid)["events"])
        self.assertEqual(self.state.replay_snapshot("replay-session-B")["events"], [])
        self.assertIsNone(self.state.replay_snapshot("replay-session-B")["task"])

    def test_model_mode_uses_same_provider_without_enabling_hardware(self):
        from jaka_agent.agent.service import AgentServiceMixin
        self.state.decision_model = True
        with patch.object(AgentServiceMixin, "agent_complete", return_value="provider-result") as complete:
            self.assertEqual(self.state.agent_complete([], []), "provider-result")
        complete.assert_called_once()
        self.assertTrue(self.state.mock)

    def test_evaluation_cleans_up_unexpected_pending_plan(self):
        from tools.evaluate_agent import evaluate_case
        original = self.state.begin_replay
        def wrong_case(case, cid):
            return original("blocked", cid)
        with patch.object(self.state, "begin_replay", side_effect=wrong_case):
            result = evaluate_case(self.state, "find_object", CASES["find_object"])
        self.assertFalse(result["successful"])
        self.assertEqual(self.state.tasks.snapshot()["status"], "canceled")
        self.assertEqual(self.state.tasks.mock_pose, [0, 0, 0])

    def test_evaluation_redacts_provider_error_detail(self):
        from tools.evaluate_agent import evaluate_case
        with patch.object(self.state, "begin_replay", side_effect=RuntimeError("private-endpoint-and-token")):
            result = evaluate_case(self.state, "find_object", CASES["find_object"])
        self.assertFalse(result["successful"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertNotIn("private-endpoint", json.dumps(result))

    def test_http_routes_reuse_ownership_guards(self):
        server = RobotWebServer(("127.0.0.1", 0), RobotWebHandler, self.state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def request(path, payload=None):
            data = None if payload is None else json.dumps(payload).encode()
            return urlopen(Request(base + path, data=data, headers={"Content-Type": "application/json"}), timeout=5)
        try:
            with request("/replay") as response:
                self.assertIn("Agent".encode(), response.read())
            with request("/static/js/replay.js") as response:
                self.assertIn(b"textContent", response.read())
            with request("/api/replay/plan", {"case_id": "find_object", "conversation_id": self.cid}) as response:
                task = json.load(response)["result"]["task"]
            with self.assertRaises(HTTPError):
                request("/api/task/execute", {"task_id": task["id"], "conversation_id": "replay-session-B"})
            self.assertEqual(self.state.tasks.snapshot()["status"], "planned")
            self.assertFalse(self.state.tasks.is_busy())
            self.state.replay = False
            with self.assertRaises(HTTPError) as failure:
                request("/api/replay")
            self.assertEqual(failure.exception.code, 404)
        finally:
            server.shutdown(); server.server_close(); thread.join(3)
            self.state.replay = True


class ExecutionFeedbackTests(unittest.TestCase):
    def test_every_runtime_terminal_status_has_an_explicit_label(self):
        for status in ("succeeded", "aborted", "error", "not_found", "inconclusive", "canceled", "interrupted"):
            with self.subTest(status=status):
                state = execution_snapshot({"id": "task-123", "status": status})
                self.assertEqual(state["state"], status)
                self.assertNotIn("未知", state["label"])


if __name__ == "__main__":
    unittest.main()
