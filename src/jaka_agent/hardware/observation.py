"""Camera geometry and visual observation cleanup."""
from __future__ import annotations
import jaka_agent.models.settings as models_settings
import jaka_agent.models.vision as models_vision
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class ObserveTranslation:
    """一次观察距离调整的有符号位移；正数前进，负数后退，0 表示停止。"""

    meters: float
    reason: str


def observation_focus(navigation_target, question, explicit_focus=None):
    """优先使用规划器给出的关键区域；旧计划回退到观察问题，不按类别特判。"""
    return str(explicit_focus or question or navigation_target).strip()


def clean_observation_answer(value):
    """移除距离调整等执行器内部说明，只把自然语言观察结果交给用户。"""
    text = models_vision.format_visual_answer(value)
    internal_notes = (
        r"距离无需调整，按当前画面作答",
        r"使用到达点当前画面",
        r"取景达标",
        r"深度丢失，停止距离调整",
        r"前方存在障碍，停止靠近",
    )
    for note in internal_notes:
        text = re.sub(rf"\s*[（(]{note}[）)]", "", text)
    return text.strip()


def decide_observe_translation(
    approach, requested_m, target_distance_m, total_moved_m, focus_visible=True,
    focus_complete=None, translation_direction=0, retreat_moved_m=0.0,
):
    """将视觉距离建议转换为受安全距离、单步和累计预算约束的底盘位移。"""
    remaining = max(0.0, models_settings.APPROACH_MAX_TOTAL - abs(float(total_moved_m or 0.0)))
    if remaining <= models_settings.OBSERVE_MOVE_TOL:
        return ObserveTranslation(0.0, "已达到观察调整累计位移上限")
    if focus_complete is None:
        focus_complete = bool(focus_visible)

    def guarded(meters, reason):
        move_direction = 1 if meters > 0 else -1 if meters < 0 else 0
        if translation_direction and move_direction and move_direction != translation_direction:
            return ObserveTranslation(0.0, "检测到距离调整方向反转，为避免前后振荡而保持当前位置")
        return ObserveTranslation(meters, reason)

    if focus_complete is False:
        near_missing_focus = (
            focus_visible is False
            and target_distance_m is not None
            and float(target_distance_m) <= models_settings.OBSERVE_NEAR_CROP_DISTANCE
        )
        continuing_retreat = translation_direction == -1 and retreat_moved_m > 0
        if focus_visible is False and not near_missing_focus and not continuing_retreat:
            return ObserveTranslation(0.0, "关键区域未定位且导航目标不近，不执行首次盲目平移")
        retreat_remaining = max(0.0, models_settings.OBSERVE_MAX_RETREAT - abs(float(retreat_moved_m or 0.0)))
        retreat = min(models_settings.OBSERVE_CROP_RETREAT_STEP, models_settings.APPROACH_STEP, remaining, retreat_remaining)
        if retreat <= models_settings.OBSERVE_MOVE_TOL:
            return ObserveTranslation(0.0, "关键区域仍未完整入画，但已达到后退预算")
        if focus_visible:
            reason = "关键区域未完整入画，后退扩大视野"
        elif continuing_retreat:
            reason = "关键区域仍未入画，沿已确认方向继续后退扩大视野"
        else:
            reason = "关键区域未入画且导航目标较近，后退扩大视野"
        return guarded(-retreat, reason)

    direction = str(approach or "none").lower()
    if direction not in ("closer", "farther"):
        return ObserveTranslation(0.0, "距离无需调整")

    try:
        requested = float(requested_m or 0.0)
    except (TypeError, ValueError):
        requested = 0.0
    requested = requested if requested > 0 else models_settings.APPROACH_STEP
    step = min(requested, models_settings.APPROACH_STEP, remaining)

    if direction == "closer":
        if target_distance_m is None:
            return ObserveTranslation(0.0, "目标深度无效，不能安全靠近")
        safe_forward = float(target_distance_m) - models_settings.APPROACH_ABS_MIN - models_settings.APPROACH_MARGIN
        step = min(step, safe_forward)
        if step <= models_settings.OBSERVE_MOVE_TOL:
            return ObserveTranslation(0.0, "已到达目标最小安全距离")
        return guarded(step, "目标太远，向前靠近")

    retreat_remaining = max(0.0, models_settings.OBSERVE_MAX_RETREAT - abs(float(retreat_moved_m or 0.0)))
    step = min(step, retreat_remaining)
    if step <= models_settings.OBSERVE_MOVE_TOL:
        return ObserveTranslation(0.0, "已达到观察后退预算")
    return guarded(-step, "目标太近，向后远离")
