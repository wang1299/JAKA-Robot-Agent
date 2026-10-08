"""Read the packaged web page template."""
from __future__ import annotations
import jaka_agent.web.settings as web_settings


def _document() -> bytes:
    """读取前端片段并组装完整 HTML 文档。"""
    try:
        page_html = web_settings.PAGE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"前端页面文件不可用: {web_settings.PAGE_PATH}") from exc
    head = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#f6f7f8">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="default">
  <meta name="apple-mobile-web-app-title" content="JAKA Vision">
  <link rel="manifest" href="/manifest.webmanifest">
  <link rel="icon" href="/pwa-icon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/pwa-icon.svg">
  <title>JAKA Vision</title>
</head>
<body>
"""
    return (head + page_html + "\n</body>\n</html>\n").encode("utf-8")
