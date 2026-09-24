#!/usr/bin/env python3
"""
机器人端调用 MiniCPM-V 4.6 (transformers serve, OpenAI 兼容) 的客户端.
用官方 openai SDK 调 —— 这样以后换成任何 OpenAI 兼容服务(真正的 OpenAI / vLLM / sglang ...)
几乎不用改代码, 只动 base_url 和 model.

前置(机器人/上位机上, 只需一次):
    pip install openai pillow
    先把隧道建好(见 tunnel.sh): ./tunnel.sh start
    保证 http://localhost:8000/health 能通.

几个关键点(都是之前踩过的坑):
    - base_url 指向本地隧道口; api_key 随便填 —— serve 不校验, 但 SDK 要求非空.
    - model 必须**精确等于**启动路径 "/root/Model/MiniCPM-V-4.6"(pin 模式, 写别的会被拒).
    - 本地图先补黑边成正方形再发: MiniCPM-V 4.6 多分块时非方形图各块尺寸不等, 会触发
      形状 bug(HTTP 500); 正方形图自动 2×2/3×3 等大切块, 稳定且保细节(见项目记忆).

可用环境变量覆盖默认值:
    MINICPM_BASE_URL  默认 http://localhost:8000/v1
    MINICPM_MODEL      默认 /root/Model/MiniCPM-V-4.6
    MINICPM_API_KEY    默认 not-needed
"""
import base64
import io
import json
import os
import re

from openai import OpenAI
from PIL import Image

BASE_URL = os.environ.get("MINICPM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MINICPM_MODEL", "/root/Model/MiniCPM-V-4.6")

client = OpenAI(
    base_url=BASE_URL,
    api_key=os.environ.get("MINICPM_API_KEY", "not-needed"),
)


def _pad_to_square(im):
    """补黑边成正方形(保比例, 不拉伸)."""
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def _encode_jpeg_data_url(im):
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _image_data_urls(paths, common_size=False):
    """一批本地图 -> 补方 -> base64 data URL 列表.

    common_size=True 时(多图), 把所有图 resize 到同一正方形尺寸(取最大边长),
    使每张图切片数 / token 数一致 —— 这是多图推理的关键:
    transformers 5.12.1 的 vit_merger 假设打包的多图每张 token 数相同, 尺寸不一会触发
    .view 形状 bug(HTTP 500). 单图时各自补方即可.
    """
    squares = [_pad_to_square(Image.open(p).convert("RGB")) for p in paths]
    if common_size and len(squares) > 1:
        target = max(c.width for c in squares)  # 用最大边长, 不丢细节
        squares = [c.resize((target, target), Image.LANCZOS) if c.width != target else c
                   for c in squares]
    return [_encode_jpeg_data_url(c) for c in squares]


def _build_content(question, image=None, images=None):
    paths = []
    if images:
        paths.extend(images)
    if image is not None:
        paths.append(image)
    if not paths:
        return [{"type": "text", "text": question}]
    urls = _image_data_urls(paths, common_size=len(paths) > 1)
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": question})
    return content


def ask(question, image=None, images=None, max_tokens=512, temperature=0.0):
    """问一个问题, 返回完整文本.

    传图方式:
      image=path            单图
      images=[path1, ...]   多图(自动统一到同一正方形尺寸, 避开跨图 bug)
      都不传                 纯文本

    temperature 默认 0(贪心解码, 输出确定): 模型 generation_config 默认是 0.7 采样,
    会随机、格式不稳; 机器人/格式化输出务必压到 0(或 0.1~0.2), 同一指令才稳定给同样结果.
    想要有点发散(头脑风暴之类)再调高.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": _build_content(question, image, images)}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def ask_stream(question, image=None, images=None, max_tokens=512, temperature=0.0):
    """流式版: 逐段 yield 文本. 机器人想"边出字边动作"时用这个, 首字更快. (参数同 ask)"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": _build_content(question, image, images)}],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ---------- 目标物检索(多图): scene 里有没有 target ----------
FIND_TARGET_SYSTEM = (
    "图片1是目标物参考图(近景),图片2是机器人现场图(远景)。判断图片2里有没有图片1的目标物。"
    "先一句话指出图片2里最像图片1目标物的具体物体(位置+形状+主色),或说明没有相似物体。"
    "标准:只有该物体在整体形状+比例+主颜色上都和图片1明显一致才found=true;"
    "仅类别/颜色相似或拿不准就found=false,不许凭空猜。"
    "然后用一行JSON总结(不要markdown、不要代码块),5字段全:"
    '{"found":false,"candidate_visible":false,"candidate_region":"方位或空串",'
    '"confidence":"low","reason":"比对依据"}'
)


def parse_result(text):
    """从模型输出抠结果:容忍前缀散文、```json 代码块、轻微格式瑕疵。found 必出。"""
    text = re.sub(r"```json|```", "", text)
    for block in reversed(re.findall(r"\{[^{}]*\}", text, re.S)):
        try:
            d = json.loads(block)
            if "found" in d:
                return d
        except json.JSONDecodeError:
            continue
    # 兜底:JSON 没解析出来就正则抠 found
    m = re.search(r'found["\']?\s*:\s*["\']?(true|false)', text, re.I)
    return {"found": (m.group(1) == "true") if m else None}


def find_target(target_img, scene_img, temperature=0.0, max_tokens=300):
    """判断 scene_img 里有没有 target_img 所示的目标物。

    返回解析后的 dict,至少含 found(True/False/None);顺利时还有
    candidate_visible / candidate_region / confidence / reason。
    用瘦身 CoT(一句话 grounding + JSON),实测正负例都对、~2.3s、准确率接近完整 CoT。
    多图会自动统一到同一正方形尺寸(避开跨图形状 bug)。
    """
    content = _build_content("请判断并输出JSON。", images=[target_img, scene_img])
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": FIND_TARGET_SYSTEM},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return parse_result(resp.choices[0].message.content)


if __name__ == "__main__":
    import sys

    # 用法:
    #   python robot_client.py                              # 纯文本自报家门
    #   python robot_client.py "图里有什么?" ./a.jpg         # 看单图
    #   python robot_client.py find ./target.jpg ./scene.jpg  # 目标物检索(多图)
    #   多图问答在代码里: ask("比较这两张图", images=["a.jpg", "b.jpg"])
    args = sys.argv[1:]
    print(f"服务: {BASE_URL}  模型: {MODEL}\n")

    if args and args[0] == "find" and len(args) >= 3:
        print(f"target={args[1]}  scene={args[2]}")
        print("结果:", find_target(args[1], args[2]))
    else:
        q = args[0] if args else "用一句话介绍你自己。"
        img = args[1] if len(args) > 1 else None
        if img:
            print(f"图: {img}")
        print("答:", ask(q, image=img))
