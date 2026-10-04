"""二次网循环的分步收尾。

收尾方式一次定死为**分步收**：不做"一把收干净"，而是按固定顺序逐步执行，每一步都
单独落步骤、单独判定，顺序不许乱：

- 正常收尾（``completed``）：停泵 → 等泵停稳 → 阀门保持当前位（正常结束不关阀）；
- 就地中断收尾（``in_place``）：停泵 → 等泵停稳 → 关阀（先停泵后关阀，阀先关会把
  泵憋住）；
- 退回安全阀位的中断收尾（``retreat``）：泵还在转时先把阀退到安全（非关闭）阀位 →
  停泵 → 等泵停稳 → 再关阀。退阀位不是关阀，不违反"先停泵后关阀"。

任何一步自身抛错都只记入 ``reasons``，不中断后续收尾——泵停稳是硬保证。泵最终没
停稳时关阀联锁拒绝执行，本次收尾结论必然是中断。
"""

from __future__ import annotations

from ..pump.pump import Pump
from ..valve.valve import Valve

CLOSEOUT_IN_PLACE = "in_place"
CLOSEOUT_RETREAT = "retreat"
CLOSEOUT_MODES = (CLOSEOUT_IN_PLACE, CLOSEOUT_RETREAT)

MAX_STEPS = 600

# 退回安全阀位的默认开度：保持回路打通，避免泵减速时被关小的阀抽成负压。
DEFAULT_SAFE_VALVE_PCT = 100.0


def close_out(
    pump: Pump,
    valve: Valve,
    *,
    completed: bool,
    mode: str = CLOSEOUT_IN_PLACE,
    safe_valve_pct: float = DEFAULT_SAFE_VALVE_PCT,
    dt: float = 1.0,
) -> dict:
    """按固定步骤收尾，返回泵状态、阀门状态与每一步的记录。

    ``completed`` 为真表示循环正常走完，只停泵不关阀；为假表示中途出错（含读数
    缺失的部分失败），按 ``mode`` 指定的工况就地停泵或先退安全阀位再停。
    """

    if dt <= 0:
        raise ValueError("步长必须为正")
    if mode not in CLOSEOUT_MODES:
        raise ValueError(f"未知收尾方式：{mode!r}")
    safe_valve_pct = max(0.0, min(100.0, float(safe_valve_pct)))

    steps: list[dict] = []
    reasons: list[str] = []

    # 工况一（retreat）：先退回安全阀位，再停泵。泵在转的时候绝不能把阀往关闭位
    # 推，因此安全阀位不允许为 0。
    if not completed and mode == CLOSEOUT_RETREAT:
        if safe_valve_pct <= valve.limits.settle_band_pct:
            reasons.append("closeout_safe_valve_closed")
        else:
            target = valve.set_target(safe_valve_pct)
            steps.append(
                {
                    "component": "valve",
                    "action": "retreat_to_safe",
                    "detail": {
                        "target_pct": round(target, 2),
                        "position_pct": round(valve.position_pct(), 2),
                    },
                }
            )
            for _ in range(MAX_STEPS):
                try:
                    valve.step(dt)
                except Exception:  # noqa: BLE001 - 退阀失败也要继续停泵
                    reasons.append("closeout_valve_error")
                    break
                if valve.at_target():
                    break
            steps.append(
                {
                    "component": "valve",
                    "action": "safe_position_reached",
                    "detail": {
                        "position_pct": round(valve.position_pct(), 2),
                        "target_pct": round(valve.target_pct(), 2),
                    },
                }
            )

    # 固定第一步：停泵。
    try:
        pump.stop()
    except Exception as error:  # noqa: BLE001 - 停泵指令异常也要尝试等停
        reasons.append("closeout_pump_error")
        steps.append(
            {
                "component": "pump",
                "action": "stop_error",
                "detail": {"error": str(error)},
            }
        )
    else:
        steps.append(
            {
                "component": "pump",
                "action": "stop",
                "detail": {"running": pump.running(), "flow_t_h": round(pump.flow_t_h(), 2)},
            }
        )

    # 固定第二步：等泵停稳（流量归零）。
    for _ in range(MAX_STEPS):
        try:
            pump.step(dt)
        except Exception:  # noqa: BLE001 - 推进异常也要继续判定停稳
            reasons.append("closeout_pump_error")
            break
        if pump.idle():
            break

    pump_idle = pump.idle()
    if not pump_idle:
        # 泵没停稳是最严重的收尾失败：关阀联锁必须拒绝，绝不能带着在转的泵关阀。
        reasons.append("pump_not_settled")
        steps.append(
            {
                "component": "pump",
                "action": "pump_not_settled",
                "detail": {"running": pump.running(), "flow_t_h": round(pump.flow_t_h(), 2)},
            }
        )

    valve_target_pct = round(valve.target_pct(), 2)
    valve_closed = False

    if completed:
        # 正常收尾：阀门保持当前位，不执行关阀。
        steps.append(
            {
                "component": "valve",
                "action": "hold_position",
                "detail": {
                    "position_pct": round(valve.position_pct(), 2),
                    "target_pct": valve_target_pct,
                },
            }
        )
    elif pump_idle:
        # 中断收尾且泵已停稳：先停泵后关阀，关到全关位。
        valve.set_target(0.0)
        steps.append(
            {
                "component": "valve",
                "action": "close_after_pump",
                "detail": {"flow_t_h": round(pump.flow_t_h(), 2), "target_pct": 0.0},
            }
        )
        for _ in range(MAX_STEPS):
            try:
                valve.step(dt)
            except Exception:  # noqa: BLE001 - 关阀异常记入收尾原因
                reasons.append("closeout_valve_error")
                break
            if valve.at_target():
                break
        valve_closed = valve.at_target()
        steps.append(
            {
                "component": "valve",
                "action": "valve_closed" if valve_closed else "valve_not_closed",
                "detail": {
                    "position_pct": round(valve.position_pct(), 2),
                    "target_pct": round(valve.target_pct(), 2),
                },
            }
        )
        if not valve_closed:
            reasons.append("closeout_valve_not_closed")

    return {
        "pump_idle": pump_idle,
        "pump_running": pump.running(),
        "valve_position_pct": round(valve.position_pct(), 2),
        "valve_target_pct": valve_target_pct,
        "valve_closed": valve_closed,
        "closeout_mode": mode,
        "reasons": reasons,
        "steps": steps,
    }
