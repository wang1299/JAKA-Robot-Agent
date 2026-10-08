#!/usr/bin/env python3
"""
测试 MiniCPM-V 4.6 (transformers serve) 服务是否正常.
依赖: requests (transformers 已自带, 无需额外安装).

注意: 服务以 FORCE_MODEL(pin 模式) 启动时, 请求里的 model 字段必须
      精确等于启动时传入的模型路径
      (这里是 /home/admin1/MiniCPM/Model/MiniCPM-V-4.6),
      否则会报 "Server is pinned to ...". 且 pin 模式下 /v1/models 返回空列表.

用法:
    python test/test_service.py                                            # 全默认
    python test/test_service.py http://localhost:8000                      # 指定服务地址
    python test/test_service.py http://localhost:8000 ./a.jpg              # 用本地图测试
    python test/test_service.py http://localhost:8000 ./a.jpg /home/admin1/MiniCPM/Model/MiniCPM-V-4.6
"""
import sys, base64, io, time, requests
from pathlib import Path
from PIL import Image

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else \
    str(Path(__file__).resolve().parent / "fixtures" / "84863d47f4b0478abe3ceccb8f0e072e.jpg")
MODEL_ID = sys.argv[3] if len(sys.argv) > 3 else \
    "/home/admin1/MiniCPM/Model/MiniCPM-V-4.6"
H = {"Content-Type": "application/json"}


def wait_ready(timeout_s=300):
    """等模型加载完成(首次启动约 1 分钟)."""
    print("等待服务就绪(模型加载中, 首次约 1 分钟)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(f"{BASE}/health", timeout=5).status_code == 200:
                print("✓ 服务已就绪\n")
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def chat(content, max_tokens=512, timeout=300, extra=None):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if extra:
        payload.update(extra)
    r = requests.post(f"{BASE}/v1/chat/completions", headers=H, json=payload, timeout=timeout)
    if r.status_code != 200:
        sys.exit(
            f"❌ 请求失败 HTTP {r.status_code}: {r.text[:300]}\n"
            "   若提示 'Server is pinned to ...', 说明 MODEL_ID 和启动路径对不上 —— "
            "用第 3 个参数传入正确的模型名(即启动脚本里的 MODEL_PATH)。"
        )
    return r.json()["choices"][0]["message"]["content"]


def to_image_url(img):
    """URL 原样返回; 本地图补黑边成正方形再转 base64."""
    if img.startswith(("http://", "https://")):
        return img
    im = Image.open(img).convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


if __name__ == "__main__":
    if not wait_ready():
        sys.exit("❌ 服务迟迟未就绪, 请看 transformers_serve.log 排查(依赖/显存等)")

    print(f"服务地址: {BASE}")
    print(f"model 字段: {MODEL_ID}\n")

    # 1) 纯文本
    print("=== [1/3] 纯文本测试 ===")
    print("问: 用一句话介绍你自己, 并说出你是哪个模型。")
    print("答:", chat("用一句话介绍你自己, 并说出你是哪个模型。", max_tokens=128))
    print()

    # 2) 图像理解
    print("=== [2/3] 单图测试 ===")
    print(f"图: {IMAGE}")
    # 图已在 to_image_url 里补边成正方形, 默认处理就会自动多分块(2×2/3×3), 细节远好于单块.
    ans = chat(
        [
            {"type": "image_url", "image_url": {"url": to_image_url(IMAGE)}},
            {"type": "text", "text": "这张图里有什么?请简短描述。"},
        ],
        max_tokens=512,
    )
    print("答:", ans)
    print()

    # 3) 双图理解：直接验证 robot_web 寻物/巡逻所依赖的请求形态。
    # 默认把同一张图作为 image 1/2；重点是确认两个独立 image_url 不会 HTTP 500。
    print("=== [3/3] 双图测试 ===")
    ans = chat(
        [
            {"type": "image_url", "image_url": {"url": to_image_url(IMAGE)}},
            {"type": "image_url", "image_url": {"url": to_image_url(IMAGE)}},
            {
                "type": "text",
                "text": "图片1是参考图，图片2是现场图。请比较两张独立图片是否相同。",
            },
        ],
        max_tokens=256,
    )
    print("答:", ans)
    print("\n✅ 文本、单图和双图服务均正常可用。")
