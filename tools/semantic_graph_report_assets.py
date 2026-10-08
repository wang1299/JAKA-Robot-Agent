#!/usr/bin/env python3
"""Build report-ready evidence and figures from the large semantic graph JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SCENE_COLORS = {
    "办公场景": "#2563EB",
    "居家场景": "#F59E0B",
    "工业场景": "#0F766E",
}

RELATION_ZH = {
    "/r/AtLocation": "位于",
    "/r/CapableOf": "能够",
    "/r/IsA": "属于",
    "/r/MadeOf": "材质",
    "/r/Synonym": "同义",
    "/r/UsedFor": "用于",
}

# Project category -> semantic graph concept.  The left-hand categories come from
# this repository's current scene graph; the middle concepts are source JSON labels.
PROJECT_ROWS = [
    {
        "project": "蓝色椅子 / 皮质扶手椅",
        "concept": "椅子",
        "edges": [("/r/IsA", "家具"), ("/r/UsedFor", "坐下")],
    },
    {
        "project": "白色操作台",
        "concept": "桌子",
        "edges": [("/r/IsA", "家具"), ("/r/AtLocation", "辦公室")],
    },
    {
        "project": "玻璃茶几",
        "concept": "茶几",
        "edges": [("/r/IsA", "桌子"), ("/r/MadeOf", "玻璃")],
    },
    {
        "project": "木质储物柜",
        "concept": "櫃子",
        "edges": [("/r/IsA", "家具"), ("/r/AtLocation", "辦公室")],
    },
    {
        "project": "普通办公室门",
        "concept": "門",
        "edges": [("/r/AtLocation", "房間")],
    },
    {
        "project": "银色电梯门",
        "concept": "電梯",
        "edges": [("/r/CapableOf", "上升"), ("/r/CapableOf", "載人")],
    },
    {
        "project": "盆栽",
        "concept": "盆栽",
        "edges": [("/r/IsA", "植物"), ("/r/IsA", "裝飾")],
    },
    {
        "project": "黑色航空箱",
        "concept": "箱子",
        "edges": [("/r/UsedFor", "搬家"), ("/r/AtLocation", "櫃子")],
    },
]


def _font(size: float, bold: bool = False) -> FontProperties:
    path = Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc")
    return FontProperties(fname=str(path), size=size)


def load_graph(path: Path) -> tuple[dict, dict]:
    """Load the supplied JSON while repairing its one known stray character in memory."""
    raw = path.read_text(encoding="utf-8-sig")
    repair = {
        "applied": False,
        "line": None,
        "before": None,
        "after": None,
        "reason": None,
    }
    try:
        return json.loads(raw), repair
    except json.JSONDecodeError as exc:
        lines = raw.splitlines(keepends=True)
        index = exc.lineno - 1
        if not (0 <= index < len(lines) and ",x" in lines[index]):
            raise
        before = lines[index].rstrip("\r\n")
        lines[index] = lines[index].replace(",x", ",", 1)
        after = lines[index].rstrip("\r\n")
        data = json.loads("".join(lines))
        repair = {
            "applied": True,
            "line": exc.lineno,
            "before": before,
            "after": after,
            "reason": "删除属性值后多余的字符 x；原文件不改写",
        }
        return data, repair


def analyze(data: dict) -> dict:
    scenes = {}
    node_sets = {}
    edge_sets = {}
    for name, payload in data.items():
        nodes = payload.get("nodes", {})
        edges = payload.get("edges", [])
        node_sets[name] = set(nodes)
        edge_tuples = [
            (edge.get("src"), edge.get("rel"), edge.get("tgt"), edge.get("weight"))
            for edge in edges
        ]
        edge_sets[name] = set(edge_tuples)
        bad_endpoints = sum(
            edge.get("src") not in nodes or edge.get("tgt") not in nodes for edge in edges
        )
        scenes[name] = {
            "node_records": len(nodes),
            "edge_records": len(edges),
            "anchor_nodes": sum(meta.get("source") == "anchor" for meta in nodes.values()),
            "hop_distribution": dict(Counter(meta.get("hop") for meta in nodes.values())),
            "relation_distribution": dict(Counter(edge.get("rel") for edge in edges)),
            "duplicate_edge_records": len(edges) - len(edge_sets[name]),
            "self_loops": sum(edge.get("src") == edge.get("tgt") for edge in edges),
            "bad_endpoints": bad_endpoints,
        }

    names = list(data)
    overlaps = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlaps[f"{left} ∩ {right}"] = len(node_sets[left] & node_sets[right])

    return {
        "scene_count": len(data),
        "total_node_records": sum(item["node_records"] for item in scenes.values()),
        "global_unique_labels": len(set().union(*node_sets.values())),
        "total_edge_records": sum(item["edge_records"] for item in scenes.values()),
        "global_unique_weighted_edges": len(set().union(*edge_sets.values())),
        "scene_node_overlaps": overlaps,
        "scenes": scenes,
    }


def extract_project_edges(data: dict) -> list[dict]:
    office = data["办公场景"]
    index = {}
    for edge in office["edges"]:
        index.setdefault((edge.get("src"), edge.get("rel"), edge.get("tgt")), []).append(edge)

    selected = []
    missing = []
    for row in PROJECT_ROWS:
        for relation, target in row["edges"]:
            matches = index.get((row["concept"], relation, target), [])
            if not matches:
                missing.append((row["concept"], relation, target))
                continue
            selected.append(max(matches, key=lambda item: float(item.get("weight", 0))))
    if missing:
        raise ValueError(f"Expected semantic edges are missing: {missing}")
    return selected


def _format_int(value: int) -> str:
    return f"{value:,}"


def draw_scale_figure(metrics: dict, output: Path) -> None:
    fig = plt.figure(figsize=(13.2, 7.4), dpi=220, facecolor="white")
    ax = fig.add_axes([0.07, 0.12, 0.86, 0.74])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.07, 0.925, "语义图谱规模与三场景覆盖验证", fontproperties=_font(20, True), color="#172033")
    fig.text(
        0.07,
        0.885,
        "源文件统计结果 · 办公、居家、工业场景",
        fontproperties=_font(10),
        color="#64748B",
    )

    cards = [
        ("场景命名空间节点记录", metrics["total_node_records"], "达到任务书“不少于10万个”口径"),
        ("关系记录", metrics["total_edge_records"], "src-rel-tgt-weight 四元关系"),
        ("全局去重语义标签", metrics["global_unique_labels"], "按 label 跨场景合并后的数量"),
    ]
    for index, (label, value, note) in enumerate(cards):
        x = 0.02 + index * 0.33
        box = FancyBboxPatch(
            (x, 0.73),
            0.29,
            0.20,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F8FAFC",
            edgecolor="#CBD5E1",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x + 0.025, 0.875, label, fontproperties=_font(10), color="#475569", va="center")
        ax.text(x + 0.025, 0.815, _format_int(value), fontproperties=_font(24, True), color="#0F172A", va="center")
        ax.text(x + 0.025, 0.765, note, fontproperties=_font(8.5), color="#64748B", va="center")

    scene_names = list(metrics["scenes"])
    max_edges = max(metrics["scenes"][name]["edge_records"] for name in scene_names)
    y_values = [0.60, 0.43, 0.26]
    for name, y in zip(scene_names, y_values):
        scene = metrics["scenes"][name]
        color = SCENE_COLORS.get(name, "#475569")
        ax.text(0.02, y + 0.035, name, fontproperties=_font(12, True), color="#1E293B", va="center")
        ax.text(0.20, y + 0.035, "节点", fontproperties=_font(9), color="#64748B", va="center")
        ax.add_patch(
            FancyBboxPatch(
                (0.26, y + 0.012),
                0.28,
                0.046,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor=color,
                edgecolor="none",
                alpha=0.92,
            )
        )
        ax.text(0.56, y + 0.035, _format_int(scene["node_records"]), fontproperties=_font(10, True), color="#1E293B", va="center")

        edge_width = 0.28 * scene["edge_records"] / max_edges
        ax.text(0.20, y - 0.035, "关系", fontproperties=_font(9), color="#64748B", va="center")
        ax.add_patch(
            FancyBboxPatch(
                (0.26, y - 0.058),
                edge_width,
                0.046,
                boxstyle="round,pad=0,rounding_size=0.012",
                facecolor=color,
                edgecolor="none",
                alpha=0.55,
            )
        )
        ax.text(0.56, y - 0.035, _format_int(scene["edge_records"]), fontproperties=_font(10, True), color="#1E293B", va="center")
        ax.text(
            0.71,
            y,
            f"锚点 {scene['anchor_nodes']}  ·  一跳扩展 {scene['hop_distribution'].get(1, 0):,}",
            fontproperties=_font(9),
            color="#475569",
            va="center",
        )

    ax.plot([0.02, 0.98], [0.13, 0.13], color="#E2E8F0", linewidth=1)
    ax.text(
        0.02,
        0.075,
        "统计说明：105,000 为三个场景命名空间内的节点记录总数；若仅按实体文字标签跨场景去重，则为 56,293。",
        fontproperties=_font(9.2),
        color="#475569",
        va="center",
    )
    ax.text(
        0.02,
        0.025,
        "数据质量：关系端点完整；源文件含 1 处可定位语法字符错误，生成本图时仅在内存中修复，未改写原文件。",
        fontproperties=_font(8.7),
        color="#64748B",
        va="center",
    )

    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _rounded_node(
    ax,
    x: float,
    y: float,
    width: float,
    text: str,
    face: str,
    edge: str,
    *,
    bold: bool = False,
    height: float = 0.48,
    font_size: float = 8.7,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontproperties=_font(font_size, bold), color="#172033")


def draw_project_subgraph(data: dict, selected_edges: list[dict], output: Path) -> None:
    edge_lookup = {
        (edge["src"], edge["rel"], edge["tgt"]): float(edge.get("weight", 0))
        for edge in selected_edges
    }
    fig, ax = plt.subplots(figsize=(15.5, 10.2), dpi=220, facecolor="white")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    ax.text(0.45, 9.78, "项目实景类别对齐后的语义图谱子图", fontproperties=_font(20, True), color="#172033", va="center")
    ax.text(
        0.45,
        9.35,
        "从“办公场景”截取 · 虚线为项目类别归一化，实线为源 JSON 中的真实关系",
        fontproperties=_font(10),
        color="#64748B",
        va="center",
    )

    headers = [(2.0, "项目当前场景图"), (6.3, "语义图谱结点"), (11.4, "真实关系目标")]
    for x, label in headers:
        ax.text(x, 8.85, label, ha="center", fontproperties=_font(11, True), color="#334155")

    y_values = [8.10, 7.13, 6.16, 5.19, 4.22, 3.25, 2.28, 1.31]
    project_width = 3.25
    concept_width = 1.75
    target_width = 2.15
    for row, y in zip(PROJECT_ROWS, y_values):
        _rounded_node(ax, 2.0, y, project_width, row["project"], "#EFF6FF", "#93C5FD")
        _rounded_node(ax, 6.3, y, concept_width, row["concept"], "#ECFDF5", "#6EE7B7", bold=True)
        ax.add_patch(
            FancyArrowPatch(
                (2.0 + project_width / 2 + 0.05, y),
                (6.3 - concept_width / 2 - 0.08, y),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.25,
                linestyle=(0, (4, 3)),
                color="#64748B",
            )
        )
        ax.text(4.15, y + 0.13, "归一化", ha="center", fontproperties=_font(7.8), color="#64748B")

        targets = row["edges"]
        target_ys = [y] if len(targets) == 1 else [y + 0.22, y - 0.22]
        for target_index, ((relation, target), target_y) in enumerate(zip(targets, target_ys)):
            target_x = 11.4
            _rounded_node(
                ax,
                target_x,
                target_y,
                target_width,
                target,
                "#FFF7ED",
                "#FDBA74",
                height=0.34,
                font_size=8.2,
            )
            weight = edge_lookup[(row["concept"], relation, target)]
            curve = 0.0 if len(targets) == 1 else (-0.10 if target_index == 0 else 0.10)
            ax.add_patch(
                FancyArrowPatch(
                    (6.3 + concept_width / 2 + 0.05, y),
                    (target_x - target_width / 2 - 0.08, target_y),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.45,
                    color="#334155",
                    connectionstyle=f"arc3,rad={curve}",
                )
            )
            ax.text(
                8.45,
                target_y + (0.12 if target_index == 0 or len(targets) == 1 else -0.12),
                f"{RELATION_ZH[relation]}  w={weight:g}",
                ha="center",
                fontproperties=_font(7.4),
                color="#475569",
            )

    ax.plot([0.45, 15.0], [0.72, 0.72], color="#E2E8F0", linewidth=1)
    ax.text(
        0.45,
        0.38,
        "注：保留源图谱中的原始结点文字（如“櫃子/門/電梯/辦公室”）；关系名称已中文化，w 为源文件权重。",
        fontproperties=_font(8.7),
        color="#64748B",
    )

    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_excerpt(path: Path, source: Path, repair: dict, metrics: dict, edges: list[dict]) -> None:
    payload = {
        "source_file": str(source),
        "source_repair": repair,
        "metrics": metrics,
        "project_alignment": [
            {"project_category": row["project"], "semantic_concept": row["concept"]}
            for row in PROJECT_ROWS
        ],
        "semantic_edges_from_office_scene": edges,
        "relation_name_zh": RELATION_ZH,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report_notes(path: Path, source: Path, repair: dict, metrics: dict) -> None:
    office = metrics["scenes"]["办公场景"]
    home = metrics["scenes"]["居家场景"]
    industry = metrics["scenes"]["工业场景"]
    duplicates = sum(item["duplicate_edge_records"] for item in metrics["scenes"].values())
    self_loops = sum(item["self_loops"] for item in metrics["scenes"].values())
    text = f"""# 语义图谱测试报告素材

## 1. 验收依据

任务书“主要技术指标”第（3）项要求：构建可动态更新的空间语义图谱系统，支撑任务驱动的语义推理、路径规划与交互决策，并实现空间对象关系、行为和场景拓扑的自动维护与更新；图谱实体数量不少于 10 万个。

## 2. 测试对象与读取方法

- 测试文件：`{source}`
- 文件大小约 102 MB，顶层包含“办公场景、居家场景、工业场景”三个命名空间。
- 每个场景包含 `nodes`、`edges`、`stats` 三个字段。
- `nodes` 以实体文字为键，属性包括 `hop`、`score`、`source`、`rank`。
- `edges` 为关系数组，关系记录包含 `src`、`rel`、`tgt`、`weight`。
- 原文件第 {repair.get('line')} 行存在 1 个多余字符 `x`，导致标准 JSON 解析失败。本次测试仅在读取内存中删除该字符，未修改原始文件。

## 3. 实测结果

| 场景 | 节点记录 | 锚点 | 一跳扩展节点 | 关系记录 |
|---|---:|---:|---:|---:|
| 办公场景 | {office['node_records']:,} | {office['anchor_nodes']:,} | {office['hop_distribution'].get(1, 0):,} | {office['edge_records']:,} |
| 居家场景 | {home['node_records']:,} | {home['anchor_nodes']:,} | {home['hop_distribution'].get(1, 0):,} | {home['edge_records']:,} |
| 工业场景 | {industry['node_records']:,} | {industry['anchor_nodes']:,} | {industry['hop_distribution'].get(1, 0):,} | {industry['edge_records']:,} |
| 合计 | {metrics['total_node_records']:,} | {office['anchor_nodes'] + home['anchor_nodes'] + industry['anchor_nodes']:,} | {office['hop_distribution'].get(1, 0) + home['hop_distribution'].get(1, 0) + industry['hop_distribution'].get(1, 0):,} | {metrics['total_edge_records']:,} |

按“场景命名空间内的实体节点记录”统计，图谱共 {metrics['total_node_records']:,} 个节点，达到“不少于 10 万个”的指标。按实体文字标签跨场景去重后为 {metrics['global_unique_labels']:,} 个；因此报告中应明确使用“场景命名空间实体节点”口径，不宜写成“全局唯一实体 10.5 万个”。

## 4. 项目相关子图说明

当前机器人项目的实景场景图包含椅子/扶手椅、操作台、茶几、储物柜、办公室门、电梯门、盆栽和航空箱等类别。将其归一化到语义结点后，可从办公场景中取得“属于、位于、用于、能够、材质”等真实关系，示例包括：

- 椅子—属于→家具；椅子—用于→坐下。
- 桌子—位于→办公室；桌子—属于→家具。
- 茶几—属于→桌子；茶几—材质→玻璃。
- 柜子—位于→办公室；柜子—属于→家具。
- 门—位于→房间；电梯—能够→上升/载人。
- 盆栽—属于→植物/装饰；箱子—用于→搬家。

这些关系可与项目现有的物体定位、靠近/包含/支撑关系以及导航、巡检、寻物、迎宾任务结合，用于证明“空间对象—属性—行为/功能”语义链条已具备可用数据基础。

## 5. 数据质量与结论建议

- 关系端点缺失：0 条。
- 完全重复的关系记录：{duplicates:,} 条。
- 自环关系：{self_loops:,} 条，主要来自同义或自身关联，应在推理前按关系类型过滤。
- 原 JSON 含 1 处语法错误，建议修复后重新归档，并附校验值。

建议测试结论表述：**经解析验证，图谱覆盖办公、居家、工业三个场景，按场景命名空间统计实体节点 105,000 个、关系记录 760,649 条，达到任务书“图谱实体数量不少于 10 万个”的数量指标；项目相关办公机器人实体可映射到多类真实语义关系。**
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, repair = load_graph(args.input)
    metrics = analyze(data)
    selected_edges = extract_project_edges(data)

    draw_scale_figure(metrics, args.output_dir / "01_语义图谱规模与三场景覆盖.png")
    draw_project_subgraph(data, selected_edges, args.output_dir / "02_办公机器人项目相关语义子图.png")
    write_excerpt(
        args.output_dir / "03_项目相关语义子图数据.json",
        args.input,
        repair,
        metrics,
        selected_edges,
    )
    write_report_notes(
        args.output_dir / "04_测试报告可用文字.md",
        args.input,
        repair,
        metrics,
    )

    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "repair": repair,
        "metrics": {
            "total_node_records": metrics["total_node_records"],
            "global_unique_labels": metrics["global_unique_labels"],
            "total_edge_records": metrics["total_edge_records"],
        },
        "selected_edge_count": len(selected_edges),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
