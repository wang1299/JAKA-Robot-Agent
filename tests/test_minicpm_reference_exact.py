#!/usr/bin/env python3
"""
在树莓派上使用与 robot_web.py 相同的图片处理和请求结构测试参考图寻物。

严格保持生产结构：
  1. 参考图在前、现场图在后，作为两个独立 image_url；
  2. 两张图都不缩放、不裁剪，居中补黑边到 1280x1280；
  3. 使用与生产代码相同的瘦身 CoT 提示词；
  4. temperature=0、max_tokens=300；
  5. 使用 robot_web_routing._parse_find_object_result 解析结果。

示例：
  python test/test_minicpm_reference_exact.py \
    --reference test/fixtures/98928594713c4d919b2f17e31c526c8a.jpg \
    --scene test/fixtures/84863d47f4b0478abe3ceccb8f0e072e.jpg \
    --repeat 5 \
    --save-prepared
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jaka_agent.agent.routing import FIND_TARGET_COT_MAX_TOKENS, FIND_TARGET_COT_SYSTEM, FIND_TARGET_COT_USER, _parse_find_object_result


HERE = Path(__file__).resolve().parent / "fixtures"
TARGET_SIDE = 1280
DEFAULT_REFERENCE = HERE / "98928594713c4d919b2f17e31c526c8a.jpg"
DEFAULT_SCENE = HERE / "84863d47f4b0478abe3ceccb8f0e072e.jpg"

def prepare_image(path: Path) -> tuple[str, bytes, tuple[int, int], tuple[int, int]]:
    """复刻 qwen_planner._to_square_data_url。"""
    with Image.open(path) as opened:
        image = opened.convert("RGB")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸无效: {path} ({width}x{height})")
    if width > TARGET_SIDE or height > TARGET_SIDE:
        raise ValueError(
            f"图片 {path.name} 为 {width}x{height}，超过固定 "
            f"{TARGET_SIDE}x{TARGET_SIDE} 画布；生产配置禁止缩放和裁剪"
        )

    offset = ((TARGET_SIDE - width) // 2, (TARGET_SIDE - height) // 2)
    canvas = Image.new("RGB", (TARGET_SIDE, TARGET_SIDE), (0, 0, 0))
    canvas.paste(image, offset)

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=90, optimize=True)
    payload = output.getvalue()
    encoded = base64.b64encode(payload).decode("ascii")
    return (
        f"data:image/jpeg;base64,{encoded}",
        payload,
        (width, height),
        offset,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用与 robot_web.py 相同的图片处理和请求结构测试 MiniCPM 参考物匹配"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"图片1，参考图（默认: {DEFAULT_REFERENCE.name}）",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help=f"图片2，现场图（默认: {DEFAULT_SCENE.name}）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="对同一请求重复推理次数（默认: 5）",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("DASHSCOPE_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI 兼容服务地址",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("QWEN_VISION_MODEL", "/home/admin1/MiniCPM/Model/MiniCPM-V-4.6"),
        help="模型名称或路径",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY", "not-needed"),
        help="本地服务通常使用 not-needed",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="请求超时秒数")
    parser.add_argument(
        "--save-prepared",
        action="store_true",
        help="将实际发送给模型的两张 1280x1280 JPEG 保存到当前工作目录",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只检查模型输入、不发送推理请求；配合 --save-prepared 可保存处理后图片",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须大于等于 1")
    return args


def main() -> int:
    args = parse_args()
    reference = args.reference.resolve()
    scene = args.scene.resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"参考图不存在: {reference}")
    if not scene.is_file():
        raise FileNotFoundError(f"现场图不存在: {scene}")

    reference_url, reference_payload, reference_size, reference_offset = prepare_image(reference)
    scene_url, scene_payload, scene_size, scene_offset = prepare_image(scene)

    if args.save_prepared:
        output_dir = Path.cwd()
        reference_output = output_dir / "prepared_01_reference_1280x1280.jpg"
        scene_output = output_dir / "prepared_02_scene_1280x1280.jpg"
        reference_output.write_bytes(reference_payload)
        scene_output.write_bytes(scene_payload)
        print(f"处理后参考图: {reference_output}")
        print(f"处理后现场图: {scene_output}")

    print("请求结构: image_url(参考图) + image_url(现场图) + text(生产提示词)")
    print(f"图片数量: 2（独立输入，未拼接）")
    print(
        f"图片1/参考图: {reference}\n"
        f"  原始={reference_size[0]}x{reference_size[1]}，"
        f"补齐=1280x1280，offset={reference_offset}，"
        f"发送JPEG={len(reference_payload)} bytes"
    )
    print(
        f"图片2/现场图: {scene}\n"
        f"  原始={scene_size[0]}x{scene_size[1]}，"
        f"补齐=1280x1280，offset={scene_offset}，"
        f"发送JPEG={len(scene_payload)} bytes"
    )
    print(f"服务: {args.base_url}")
    print(f"模型: {args.model}")
    print(f"temperature=0, max_tokens={FIND_TARGET_COT_MAX_TOKENS}")
    print(
        "prompt_sha256="
        + hashlib.sha256(
            (FIND_TARGET_COT_SYSTEM + "\n" + FIND_TARGET_COT_USER).encode("utf-8")
        ).hexdigest()
    )
    if args.prepare_only:
        print("prepare-only: 已跳过模型请求")
        return 0

    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    production_matches = 0
    for index in range(1, args.repeat + 1):
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": FIND_TARGET_COT_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": reference_url}},
                        {"type": "image_url", "image_url": {"url": scene_url}},
                        {"type": "text", "text": FIND_TARGET_COT_USER},
                    ],
                }
            ],
            max_tokens=FIND_TARGET_COT_MAX_TOKENS,
            temperature=0.0,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = _parse_find_object_result(raw)
        matched = (
            bool(parsed.get("found"))
            and str(parsed.get("confidence") or "").lower() == "high"
        )
        production_matches += int(matched)

        print(f"\n========== 第 {index}/{args.repeat} 次 ==========")
        print("模型原始输出:")
        print(raw)
        print("生产解析结果:")
        print(parsed)
        print(f"生产单帧是否判定命中: {matched}")

    print("\n========== 汇总 ==========")
    print(f"单帧 high/true 次数: {production_matches}/{args.repeat}")
    print(
        "说明：到达点只使用第1次单帧结果；行进中需要真实连续抓拍的两帧都为 "
        "high/true，本脚本重复同一静态图片仅用于观察模型稳定性。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
