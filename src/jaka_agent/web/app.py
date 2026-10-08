"""Load configuration and start the robot web service."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="JAKA Robot Agent Web 服务")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mock", action="store_true", help="不连接相机/模型/麦克风")
    parser.add_argument("--replay", action="store_true", help="完整任务回放；预设决策与合成观察，不连接硬件")
    parser.add_argument("--replay-model", action="store_true", help="回放中使用配置的真实决策模型；需同时指定 --replay")
    args = parser.parse_args()
    if args.mock and args.replay:
        parser.error("--mock 和 --replay 请选择一种模式")
    if args.replay_model and not args.replay:
        parser.error("--replay-model 需要 --replay")
    args.host = args.host or ("127.0.0.1" if args.replay else "0.0.0.0")
    if args.replay:
        os.environ.setdefault("JAKA_DATA_DIR", str(Path.cwd() / "data" / "replay"))
    import jaka_agent.diagnostics as diagnostics

    from jaka_agent.models.config import load_model_config
    model_config = load_model_config()
    for key, value in sorted(model_config.items()):
        diagnostics.LOGGER.info("[models] %s=%s", key, value)

    import jaka_agent.web.handler as web_handler
    import jaka_agent.web.server as web_server
    import jaka_agent.web.state as web_state
    if args.replay:
        from jaka_agent.replay.state import ReplayWebState
        state = ReplayWebState(decision_model=args.replay_model)
    else:
        state = web_state.RobotWebState(mock=args.mock)
    server = web_server.RobotWebServer((args.host, args.port), web_handler.RobotWebHandler, state)
    print(f"[web] JAKA Robot Agent 正在监听: http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print(f"[web] 远程浏览器请访问: http://<树莓派IP>:{args.port}")
        print(f"[web] SSH 转发后请访问: http://127.0.0.1:{args.port}")
    print(f"[web] mode={'replay' if args.replay else 'mock' if args.mock else 'hardware'} | Ctrl+C 退出")
    if args.replay:
        print(f"[web] Agent 与机器人任务回放: http://127.0.0.1:{args.port}/replay")
    diagnostics.LOGGER.info("[web] backend log file: %s", diagnostics.LOG_PATH)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 正在关闭")
    finally:
        server.server_close()
        state.close()
