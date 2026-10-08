"""Evaluate fixed Agent/task contracts with synthetic robot feedback; no hardware."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def wait_for(state, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = state.tasks.snapshot() or {}
        if predicate(task):
            return task
        time.sleep(0.01)
    raise RuntimeError("Replay task did not reach the expected state")


def stop_pending(state):
    task = state.tasks.snapshot() or {}
    if task.get("status") in {"planned", "running", "canceling"}:
        state.tasks.cancel(task["id"])
    if state.tasks.thread and state.tasks.thread.is_alive():
        state.tasks.thread.join(8)
    if state.tasks.is_busy():
        raise RuntimeError("Replay task did not stop")


def evaluate_case(state, name, case):
    cid = uuid.uuid4().hex
    started = time.perf_counter()
    checks = {}
    failure = None
    try:
        answer = state.begin_replay(name, cid)["result"]
        elapsed = (time.perf_counter() - started) * 1000
        task = answer.get("task") or {}
        planned = (task.get("skill") or {}).get("id") == case["skill"] and task.get("status") == "planned"
        checks = {"planning_contract": bool(answer.get("requires_clarification")) and not task
                  if name == "missing_reference" else planned,
                  "confirmation_gate": not state.tasks.is_busy()}
        if planned:
            state.require_task_owner(task["id"], cid)
            state.tasks.execute(task["id"])
            if name == "welcome":
                pending = wait_for(state, lambda t: (t.get("guest_confirmation") or {}).get("status") == "pending")
                checks["guest_confirmation_gate"] = pending.get("current_step") == 1
                state.tasks.confirm_welcome_guest(task["id"], pending["guest_confirmation"]["id"], True, cid)
            state.tasks.thread.join(8)
            finished = state.tasks.snapshot()
            checks["execution_status"] = finished["status"] == ("aborted" if name == "blocked" else "succeeded")
            if name == "find_object":
                checks["observation_changes_search"] = [o["found"] for o in finished["observations"]] == [False, True]
            feedback = state.replay_feedback(cid)
            checks["feedback_uses_task_status"] = (feedback.get("execution") or {}).get("state") == finished["status"]
    except Exception as exc:
        # Provider exception text can include private endpoint URLs or credentials.
        failure = type(exc).__name__
        checks["request_completed"] = False
        elapsed = (time.perf_counter() - started) * 1000
    finally:
        stop_pending(state)
        state.tasks.mock_pose[:] = [0.0, 0.0, 0.0]
        state.tasks.track.clear()
    result = {"case": name, "decision_latency_ms": round(elapsed, 2), "checks": checks,
              "successful": all(checks.values())}
    if failure:
        result["error_type"] = failure
    return result


def evaluate(state):
    """The harness explicitly confirms replay tasks; UI confirmation stays manual."""
    from jaka_agent.replay.fixtures import CASES
    results = [evaluate_case(state, name, case) for name, case in CASES.items()]
    cid = uuid.uuid4().hex
    checks, failure = {}, None
    try:
        task = state.begin_replay("find_object", cid)["result"].get("task")
        checks["planned"] = bool(task) and task.get("status") == "planned" and (task.get("skill") or {}).get("id") == "find_object"
        if checks["planned"]:
            state.tasks.execute(task["id"])
            state.tasks.cancel(task["id"])
            state.tasks.thread.join(8)
            checks["canceled_without_later_observation"] = state.tasks.snapshot()["status"] == "canceled" and not state.tasks.snapshot()["observations"]
    except Exception as exc:
        checks["request_completed"] = False
        failure = type(exc).__name__
    finally:
        stop_pending(state)
    results.append({"case": "cancel_during_navigation", "checks": checks, "successful": all(checks.values()),
                    **({"error_type": failure} if failure else {})})
    timings = [item["decision_latency_ms"] for item in results if "decision_latency_ms" in item]
    return {"mode": "model_decisions_on_replay" if state.decision_model else "scripted_replay",
            "scope": "Fixed task-contract cases with synthetic observations; no physical robot or visual accuracy measurement",
            "timing_scope": "Decision-model requests" if state.decision_model else "Scripted tool replay overhead; not model latency",
            "passed": sum(item["successful"] for item in results), "total": len(results),
            "median_decision_latency_ms": round(statistics.median(timings), 2), "cases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replay", "model"], default="replay",
                        help="model contacts the configured decision endpoint; all robot feedback remains synthetic")
    parser.add_argument("--config", type=Path, help="Private model JSON used with --mode model")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/artifacts/agent-evaluation.json")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    with tempfile.TemporaryDirectory(prefix="jaka-agent-eval-") as temporary:
        os.environ["JAKA_DATA_DIR"] = temporary
        os.environ["JAKA_CONVERSATION_DB"] = str(Path(temporary) / "conversations.sqlite3")
        os.environ["JAKA_MEDIA_ROOT"] = str(Path(temporary) / "media")
        if args.mode == "model":
            from jaka_agent.models.config import load_model_config
            load_model_config(args.config)
        from jaka_agent.replay.state import ReplayWebState
        state = ReplayWebState(decision_model=args.mode == "model", tick=0.02)
        try:
            report = evaluate(state)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Task-contract evaluation ({report['mode']}): {report['passed']}/{report['total']}")
            print(f"Report: {args.output}")
            return 0 if report["passed"] == report["total"] else 1
        finally:
            state.close()
            logging.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
