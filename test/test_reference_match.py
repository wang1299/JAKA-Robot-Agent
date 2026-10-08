#!/usr/bin/env python3
"""
用两张独立图片测试 MiniCPM-V 是否能在现场图中找到参考物。

默认图片：
  图片1（参考图）：98928594713c4d919b2f17e31c526c8a.jpg
  图片2（现场图）：84863d47f4b0478abe3ceccb8f0e072e.jpg

本脚本不会拼接图片。每次请求始终是：
  image_url(参考图) + image_url(现场图) + text(问题)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "fixtures" / "98928594713c4d919b2f17e31c526c8a.jpg"
DEFAULT_SCENE = ROOT / "fixtures" / "84863d47f4b0478abe3ceccb8f0e072e.jpg"
TARGET_SIDE = 1280

PROMPT = """
图片1是参考物近照，图片2是现场图。
只有图片2中清楚可见一个独立真实物体，并且足以确认其整体外观设计与图片1目标一致时，
才可 found=true、confidence=high。相同类别、相似形状或相似颜色不能单独证明匹配。
如果目标太小、模糊、被遮挡，或者无法排除只是相似物品，必须 found=false，
confidence=medium 或 low。不得根据位置、背景或常识猜测，也不得补充现场看不清的部件。
不要逐步分析；只要不能明确确认，就输出 found=false。
只输出一行合法 JSON，不要输出 Markdown 或解释文字：
{"found":true或false,"confidence":"high或medium或low","reason":"简短中文依据"}
""".strip()


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"不支持的图片格式: {path.suffix}")


def _raw_data_url(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{_mime_type(path)};base64,{encoded}", len(payload)


def _padded_jpeg(path: Path) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """与主程序一致：不缩放、不裁剪，居中补黑边到固定 1280×1280。"""
    with Image.open(path) as opened:
        image = opened.convert("RGB")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸无效: {path} ({width}x{height})")
    if width > TARGET_SIDE or height > TARGET_SIDE:
        raise ValueError(
            f"图片 {path.name} 为 {width}x{height}，超过固定 "
            f"{TARGET_SIDE}x{TARGET_SIDE} 画布；测试禁止缩放和裁剪"
        )

    offset = ((TARGET_SIDE - width) // 2, (TARGET_SIDE - height) // 2)
    canvas = Image.new("RGB", (TARGET_SIDE, TARGET_SIDE), (0, 0, 0))
    canvas.paste(image, offset)

    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue(), (width, height), offset


def _padded_data_url(path: Path) -> tuple[str, int, tuple[int, int], tuple[int, int]]:
    payload, source_size, offset = _padded_jpeg(path)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", len(payload), source_size, offset


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _save_padded_preview(path: Path, output_path: Path) -> None:
    payload, _, _ = _padded_jpeg(path)
    output_path.write_bytes(payload)


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    if not (candidate.startswith("{") and candidate.endswith("}")):
        embedded = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not embedded:
            return None
        candidate = embedded.group(0)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _build_images(
    reference: Path,
    scene: Path,
    mode: str,
) -> tuple[str, str]:
    if mode == "original":
        reference_url, reference_bytes = _raw_data_url(reference)
        scene_url, scene_bytes = _raw_data_url(scene)
        print(
            f"输入模式: original（原图直传）\n"
            f"  图片1/参考图: {_image_size(reference)[0]}x{_image_size(reference)[1]}, "
            f"{reference_bytes} bytes\n"
            f"  图片2/现场图: {_image_size(scene)[0]}x{_image_size(scene)[1]}, "
            f"{scene_bytes} bytes"
        )
        return reference_url, scene_url

    reference_url, reference_bytes, reference_size, reference_offset = (
        _padded_data_url(reference)
    )
    scene_url, scene_bytes, scene_size, scene_offset = _padded_data_url(scene)
    print(
        f"输入模式: padded（不缩放，居中补黑边到 {TARGET_SIDE}x{TARGET_SIDE}）\n"
        f"  图片1/参考图: {reference_size[0]}x{reference_size[1]} "
        f"→ {TARGET_SIDE}x{TARGET_SIDE}, offset={reference_offset}, "
        f"{reference_bytes} bytes\n"
        f"  图片2/现场图: {scene_size[0]}x{scene_size[1]} "
        f"→ {TARGET_SIDE}x{TARGET_SIDE}, offset={scene_offset}, "
        f"{scene_bytes} bytes"
    )
    return reference_url, scene_url


def _ask_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    reference_url: str,
    scene_url: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包，请先执行: pip install openai") from exc

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": reference_url}},
                    {"type": "image_url", "image_url": {"url": scene_url}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        temperature=0,
        max_tokens=128,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用两张独立图片测试 MiniCPM-V 参考物存在性判断"
    )
    parser.add_argument(
        "reference",
        nargs="?",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"图片1，参考物近照（默认: {DEFAULT_REFERENCE.name}）",
    )
    parser.add_argument(
        "scene",
        nargs="?",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"图片2，现场图（默认: {DEFAULT_SCENE.name}）",
    )
    parser.add_argument(
        "--mode",
        choices=("original", "padded", "both"),
        default="both",
        help="测试原图、补齐图，或两者都测试（默认: both）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="每种输入模式重复请求次数（默认: 1）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "MINICPM_BASE_URL",
            os.getenv("DASHSCOPE_BASE_URL", "http://localhost:8000/v1"),
        ),
        help="OpenAI 兼容服务地址",
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "MINICPM_MODEL",
            os.getenv("QWEN_VISION_MODEL", "/home/admin1/MiniCPM/Model/MiniCPM-V-4.6"),
        ),
        help="模型名称或路径",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "MINICPM_API_KEY",
            os.getenv("DASHSCOPE_API_KEY", "not-needed"),
        ),
        help="API Key；本地服务通常无需真实 Key",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="请求超时秒数")
    parser.add_argument(
        "--save-previews",
        nargs="?",
        const=ROOT / "artifacts" / "reference_match_previews",
        type=Path,
        metavar="DIR",
        help="保存实际补齐后的两张输入图；可选指定目录",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成并检查输入图，不请求模型",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat 必须大于等于 1")
    return args


def main() -> int:
    args = _parse_args()
    reference = args.reference.resolve()
    scene = args.scene.resolve()

    for role, path in (("参考图", reference), ("现场图", scene)):
        if not path.is_file():
            raise FileNotFoundError(f"{role}不存在: {path}")

    print(f"图片数量: 2（两张图片分开输入，未拼接）")
    print(f"图片1/参考图: {reference}")
    print(f"图片2/现场图: {scene}")

    if args.save_previews is not None:
        preview_dir = args.save_previews.resolve()
        preview_dir.mkdir(parents=True, exist_ok=True)
        reference_preview = preview_dir / "reference_padded_1280x1280.jpg"
        scene_preview = preview_dir / "scene_padded_1280x1280.jpg"
        _save_padded_preview(reference, reference_preview)
        _save_padded_preview(scene, scene_preview)
        print(f"补齐图已保存: {reference_preview}")
        print(f"补齐图已保存: {scene_preview}")

    modes = ("original", "padded") if args.mode == "both" else (args.mode,)
    for mode in modes:
        print(f"\n{'=' * 18} {mode} {'=' * 18}")
        reference_url, scene_url = _build_images(reference, scene, mode)
        if args.prepare_only:
            print("prepare-only: 跳过模型请求")
            continue

        print(f"服务: {args.base_url}")
        print(f"模型: {args.model}")
        for index in range(1, args.repeat + 1):
            raw = _ask_model(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                reference_url=reference_url,
                scene_url=scene_url,
            )
            parsed = _extract_json(raw)
            print(f"\n[{mode} 第 {index}/{args.repeat} 次] 原始回答:")
            print(raw)
            if parsed is None:
                print("解析结果: 非法 JSON（不替模型修正结论）")
            else:
                print(
                    "解析结果: "
                    f"found={parsed.get('found')!r}, "
                    f"confidence={parsed.get('confidence')!r}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
