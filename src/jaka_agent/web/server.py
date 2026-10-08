"""Threaded HTTP server with a shared robot service state."""
from __future__ import annotations
import jaka_agent.diagnostics as diagnostics
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class RobotWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, state):
        super().__init__(address, handler)
        self.state = state

    def handle_error(self, request, client_address):
        """记录未被请求处理器捕获的异常，避免只在终端看到连接断开。"""
        exc_type, exc, _ = sys.exc_info()
        if exc is not None:
            diagnostics._log_exception(
                "[http] unhandled request exception",
                exc,
                method=getattr(request, "command", None),
                client=client_address,
                exception_type=getattr(exc_type, "__name__", None),
            )
        else:
            diagnostics.LOGGER.error("[http] unhandled request exception client=%r", client_address)
