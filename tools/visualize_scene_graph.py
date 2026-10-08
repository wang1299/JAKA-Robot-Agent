#!/usr/bin/env python3
"""Render a ZMQ scene graph JSON as a 2D world-coordinate overview."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon


CATEGORY_COLORS = {
    "Silver elevator door": "#3b82f6",
    "Potted plant": "#3fa34d",
    "Ceiling spotlights": "#f59e0b",
    "Blue chair": "#14b8a6",
    "Wooden storage cabinet": "#8b5cf6",
    "Paper wall poster": "#ec4899",
    "Leather armchair": "#ef4444",
    "White utility table": "#6366f1",
    "Black flight case": "#64748b",
    "Ordinary office door": "#f59e0b",
    "Window": "#38bdf8",
    "Shrub or bush": "#16a34a",
    "Glass coffee table": "#a855f7",
    "office door": "#e11d48",
    "door": "#8b5cf6",
    "door_frame": "#06b6d4",
    "door frame": "#06b6d4",
}

CATEGORY_LABELS = {
    "office door": "办公室门 (office door)",
    "door": "门 (door)",
    "door_frame": "门框 (door_frame)",
    "door frame": "门框 (door frame)",
    "Ordinary office door": "普通办公室门 (Ordinary office door)",
}


def _load_objects(path: Path) -> tuple[dict, list[dict], int]:
    with path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)

    raw_objects = graph.get("objects", {})
    objects = list(raw_objects.values()) if isinstance(raw_objects, dict) else list(raw_objects)
    valid = []
    skipped = 0
    for obj in objects:
        geometry = obj.get("geometry", {})
        center = geometry.get("center_world")
        half_sizes = geometry.get("aabb_half_sizes_world")
        values = (center or []) + (half_sizes or [])
        if (
            len(center or []) >= 2
            and len(half_sizes or []) >= 2
            and all(math.isfinite(float(value)) for value in values)
            and float(half_sizes[0]) > 0
            and float(half_sizes[1]) > 0
        ):
            valid.append(obj)
        else:
            skipped += 1
    return graph, valid, skipped


def _rotate_xy(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return x * cosine - y * sine, x * sine + y * cosine


def _vertical_alignment_angle(objects: list[dict]) -> float:
    """Rotate the dominant center-line axis onto the positive/negative Y axis."""
    points = [tuple(map(float, obj["geometry"]["center_world"][:2])) for obj in objects]
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    xx = yy = xy = 0.0
    for x, y in points:
        dx, dy = x - mean_x, y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    principal_axis = 0.5 * math.degrees(math.atan2(2.0 * xy, xx - yy))
    rotation = 90.0 - principal_axis
    while rotation > 90.0:
        rotation -= 180.0
    while rotation <= -90.0:
        rotation += 180.0
    return rotation


def render(
    input_path: Path,
    output_path: Path,
    dpi: int = 180,
    rotation: float = 0.0,
    auto_align_corridor: bool = False,
    focus_categories: list[str] | None = None,
    show_visual_doorways: bool = False,
) -> float:
    graph, objects, skipped = _load_objects(input_path)
    if not objects:
        raise ValueError("JSON 中没有可绘制的 center_world/aabb_half_sizes_world")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    if auto_align_corridor:
        rotation = _vertical_alignment_angle(objects)

    xs = []
    ys = []
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []
    for obj in objects:
        geometry = obj["geometry"]
        source_x, source_y = map(float, geometry["center_world"][:2])
        x, y = _rotate_xy(source_x, source_y, rotation)
        half_x, half_y = map(float, geometry["aabb_half_sizes_world"][:2])
        corners = [
            _rotate_xy(source_x + dx, source_y + dy, rotation)
            for dx, dy in ((-half_x, -half_y), (half_x, -half_y),
                           (half_x, half_y), (-half_x, half_y))
        ]
        xs.append(x)
        ys.append(y)
        x_mins.append(min(point[0] for point in corners))
        x_maxs.append(max(point[0] for point in corners))
        y_mins.append(min(point[1] for point in corners))
        y_maxs.append(max(point[1] for point in corners))
    x_extent = max(x_maxs) - min(x_mins)
    y_extent = max(y_maxs) - min(y_mins)
    aspect = max(0.72, min(1.45, x_extent / max(y_extent, 1e-6)))
    fig, ax = plt.subplots(figsize=(11.5 * aspect, 11.5), constrained_layout=False)

    all_counts = Counter(obj.get("category") or "Unknown" for obj in objects)
    focus_categories = list(dict.fromkeys(focus_categories or []))
    if focus_categories:
        unknown_focus = [name for name in focus_categories if name not in all_counts]
        if unknown_focus:
            raise ValueError("地图中没有指定类别: " + ", ".join(unknown_focus))
        counts = Counter({name: all_counts[name] for name in focus_categories})
    else:
        counts = all_counts
    fallback_colors = plt.get_cmap("tab20")
    unknown_categories = [name for name in all_counts if name not in CATEGORY_COLORS]
    extra_colors = {
        name: fallback_colors(index % 20) for index, name in enumerate(unknown_categories)
    }
    category_colors = {**extra_colors, **CATEGORY_COLORS}

    # Large boxes first so small objects and their IDs remain visible.
    objects.sort(
        key=lambda obj: float(obj["geometry"]["aabb_half_sizes_world"][0])
        * float(obj["geometry"]["aabb_half_sizes_world"][1]),
        reverse=True,
    )
    focus_labels = []
    for obj in objects:
        geometry = obj["geometry"]
        source_x, source_y = map(float, geometry["center_world"][:2])
        x, y = _rotate_xy(source_x, source_y, rotation)
        half_x, half_y = map(float, geometry["aabb_half_sizes_world"][:2])
        corners = [
            _rotate_xy(source_x + dx, source_y + dy, rotation)
            for dx, dy in ((-half_x, -half_y), (half_x, -half_y),
                           (half_x, half_y), (-half_x, half_y))
        ]
        category = obj.get("category") or "Unknown"
        color = category_colors[category]
        is_elevator = category == "Silver elevator door"
        is_focused = not focus_categories or category in focus_categories

        rect = Polygon(
            corners,
            closed=True,
            facecolor=("none" if is_elevator else color) if is_focused else "none",
            edgecolor=color if is_focused else "#cbd5e1",
            linewidth=1.35 if is_focused and focus_categories else (0.75 if is_focused else 0.45),
            alpha=0.82 if is_focused else 0.38,
            hatch="////" if is_focused and is_elevator else None,
            zorder=3 if is_focused else 1,
        )
        ax.add_patch(rect)
        if is_focused and focus_categories:
            focus_labels.append({
                "x": x,
                "y": y,
                "text": str(obj.get("ann_id", "?")),
                "color": color,
            })
        elif is_focused:
            ax.text(
                x,
                y,
                str(obj.get("ann_id", "?")),
                ha="center",
                va="center",
                fontsize=6.2 if focus_categories else 5.2,
                fontweight="bold" if focus_categories else "normal",
                color="#111827",
                zorder=4,
                clip_on=True,
            )

    visual_doorways = []
    if show_visual_doorways:
        raw_doorways = graph.get("doorways") or {}
        doorway_values = raw_doorways.values() if isinstance(raw_doorways, dict) else raw_doorways
        for doorway in doorway_values:
            if (not isinstance(doorway, dict)
                    or doorway.get("category") != "Glass doorway"
                    or doorway.get("anchor_ann_id") is not None):
                continue
            geometry = doorway.get("geometry") or {}
            center = geometry.get("center_world") or []
            half_sizes = geometry.get("aabb_half_sizes_world") or []
            if len(center) < 2 or len(half_sizes) < 2:
                continue
            source_x, source_y = map(float, center[:2])
            half_x, half_y = map(float, half_sizes[:2])
            x, y = _rotate_xy(source_x, source_y, rotation)
            corners = [
                _rotate_xy(source_x + dx, source_y + dy, rotation)
                for dx, dy in ((-half_x, -half_y), (half_x, -half_y),
                               (half_x, half_y), (-half_x, half_y))
            ]
            portal_id = str(doorway.get("portal_id") or "visual_doorway:?")
            visual_doorways.append((portal_id, x, y, corners))
            ax.add_patch(Polygon(
                corners,
                closed=True,
                facecolor="#f0abfc",
                edgecolor="#c026d3",
                linewidth=1.8,
                linestyle="--",
                alpha=0.38,
                zorder=5,
            ))
            ax.scatter([x], [y], marker="X", s=38, color="#a21caf", zorder=6)
            ax.annotate(
                portal_id,
                xy=(x, y),
                xytext=(x + 0.85, y + 0.38),
                ha="left",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                color="#86198f",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fdf4ff",
                      "edgecolor": "#c026d3", "linewidth": 0.9, "alpha": 0.96},
                arrowprops={"arrowstyle": "-", "color": "#c026d3", "linewidth": 0.8},
                zorder=7,
                clip_on=False,
            )

    if focus_labels:
        median_x = sorted(label["x"] for label in focus_labels)[len(focus_labels) // 2]
        for side in (-1, 1):
            lane = sorted(
                (label for label in focus_labels
                 if (-1 if label["x"] <= median_x else 1) == side),
                key=lambda label: label["y"],
            )
            previous_y = -math.inf
            for label in lane:
                label_y = max(label["y"], previous_y + 0.58)
                previous_y = label_y
                ax.annotate(
                    label["text"],
                    xy=(label["x"], label["y"]),
                    xytext=(label["x"] + side * 0.72, label_y),
                    ha="right" if side < 0 else "left",
                    va="center",
                    fontsize=6.0,
                    fontweight="bold",
                    color="#111827",
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "white",
                          "edgecolor": label["color"], "linewidth": 0.8, "alpha": 0.96},
                    arrowprops={"arrowstyle": "-", "color": label["color"],
                                "linewidth": 0.65, "alpha": 0.9},
                    zorder=5,
                    clip_on=True,
                )

    pad_x = max(1.0, x_extent * 0.04)
    pad_y = max(1.0, y_extent * 0.04)
    ax.set_xlim(min(x_mins) - pad_x, max(x_maxs) + pad_x)
    ax.set_ylim(min(y_mins) - pad_y, max(y_maxs) + pad_y)
    ax.set_aspect("equal", adjustable="box")
    rotated = not math.isclose(rotation, 0.0, abs_tol=1e-9)
    ax.set_xlabel("X′ / m" if rotated else "X / m")
    ax.set_ylabel("Y′ / m" if rotated else "Y / m")
    ax.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#c7cdd4")

    version = graph.get("scene_graph_version", "?")
    saved_at = graph.get("saved_at", "")
    dedup = graph.get("deduplication") or {}
    rotation_label = f"；旋转 {rotation:+.1f}°" if rotated else ""
    dedup_label = "去重后" if dedup else ""
    if focus_categories:
        portal_label = f" + {len(visual_doorways)} 个独立玻璃门洞" if visual_doorways else ""
        title = f"门位置俯视图 - {sum(counts.values())} 个门类对象{portal_label}（scene graph v{version}{rotation_label}）"
        subtitle = "彩色方框=门类对象；洋红虚线框=独立玻璃门洞；灰色轮廓=其他物体；数字=ann_id"
    else:
        title = f"拟物体俯视图{dedup_label} - {len(objects)} 个有效物体（scene graph v{version}{rotation_label}）"
        subtitle = "BoxFusion 世界坐标；方框=AABB 占地；数字=ann_id"
    if dedup:
        subtitle += (
            f"；{dedup.get('source_objects', '?')}→{dedup.get('result_objects', '?')} 条"
            f"（阈值 {dedup.get('xy_distance_m', '?')}m）"
        )
    if skipped:
        subtitle += f"；跳过 {skipped} 个无效几何物体"
    narrow_layout = x_extent < y_extent * 0.45
    ax.set_title(
        title,
        loc="left",
        fontsize=11.5 if narrow_layout else 14,
        fontweight="bold",
        pad=42 if narrow_layout else 28,
    )
    ax.text(
        0,
        1.025 if narrow_layout else 1.012,
        subtitle,
        transform=ax.transAxes,
        fontsize=7 if narrow_layout else 8,
        color="#4b5563",
    )
    if saved_at and not dedup:
        ax.text(
            0 if narrow_layout else 1,
            1.011 if narrow_layout else 1.012,
            saved_at,
            transform=ax.transAxes,
            ha="left" if narrow_layout else "right",
            fontsize=7,
            color="#6b7280",
        )

    legend_handles = []
    legend_categories = focus_categories or [name for name, _ in counts.most_common()]
    for category in legend_categories:
        count = counts[category]
        color = category_colors[category]
        legend_handles.append(
            Patch(
                facecolor="none" if category == "Silver elevator door" else color,
                edgecolor=color,
                hatch="////" if category == "Silver elevator door" else None,
                label=f"{CATEGORY_LABELS.get(category, category)} x{count}",
            )
        )
    if focus_categories:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#cbd5e1", label="其他物体（位置参照）")
        )
    if visual_doorways:
        legend_handles.append(Patch(
            facecolor="#f0abfc", edgecolor="#c026d3", linestyle="--",
            label=f"独立玻璃门洞 x{len(visual_doorways)}",
        ))
    ax.legend(
        handles=legend_handles,
        title="类别",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        handlelength=1.3,
    )

    fig.subplots_adjust(left=0.08, right=0.77, bottom=0.07, top=0.91)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="zmq_scene_graph.json")
    parser.add_argument("-o", "--output", default="zmq_scene_graph_topdown.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--rotation", type=float, default=0.0,
                        help="逆时针旋转角度（度），可为任意小数")
    parser.add_argument("--auto-align-corridor", action="store_true",
                        help="根据物体中心主轴自动把走廊方向旋转为竖直")
    parser.add_argument("--focus-category", action="append", default=[],
                        help="只重点显示指定类别；可重复提供，其他对象显示为灰色轮廓")
    parser.add_argument("--show-visual-doorways", action="store_true",
                        help="标出 doorways 中没有 anchor_ann_id 的多视角独立玻璃门洞")
    args = parser.parse_args()
    if args.auto_align_corridor and not math.isclose(args.rotation, 0.0, abs_tol=1e-9):
        parser.error("--rotation 与 --auto-align-corridor 不能同时使用")
    used_rotation = render(
        Path(args.input), Path(args.output), args.dpi, args.rotation,
        args.auto_align_corridor, args.focus_category, args.show_visual_doorways,
    )
    print(f"已生成: {Path(args.output).resolve()}（旋转 {used_rotation:+.3f}°）")


if __name__ == "__main__":
    main()
