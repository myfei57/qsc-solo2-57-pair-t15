"""二次网循环编排。

一个循环里先启泵、再推进流量与回水，任何一步越限、抛错或读数缺失都算中途失败。
无论循环正常走完还是中途失败，收尾都在 ``finally`` 里走**分步收尾**（见
:mod:`~heatnet.secondary.closeout`）：异常短路和提前返回都绕不过去，泵一定在离开
前停稳，避免泵空转把二次网抽空。

收尾结论固定为两档，写进返回报告、持久态与审计事件，三处口径一致：

- ``completed``：循环正常走完、无任何失败原因、且泵已停稳；
- ``interrupted``：中途越限、读数缺失（部分失败）、抛错，或泵最终没停稳。

中断时按工况选择收尾方式：默认就地停泵再关阀（``in_place``），存在抽空风险时
先退回安全阀位再停泵（``retreat``）；两种方式都遵守"先停泵、后关阀"的顺序。
"""

from __future__ import annotations

import math
from typing import Callable

from ..eventbus import Event, EventBus
from ..pump.pump import Pump
from ..store.state import DurableState
from ..valve.valve import Valve
from .closeout import CLOSEOUT_IN_PLACE, CLOSEOUT_MODES, close_out
from .loop import SecondaryLoop

MAX_STEPS = 600

CONCLUSION_COMPLETED = "completed"
CONCLUSION_INTERRUPTED = "interrupted"

# 一个控制步必须读到的现场量；任一缺失都属于部分失败，按中断收尾，不得当成正常完成。
REQUIRED_READINGS = ("flow_t_h", "return_c", "pressure_kpa")

Readings = dict | None
ReadingsReader = Callable[[float, float], Readings]


def _default_readings(flow_t_h: float, loop: SecondaryLoop) -> Readings:
    """正常情况下直接取泵与回路的当前读数。"""

    return {
        "flow_t_h": flow_t_h,
        "return_c": loop.return_c(),
        "pressure_kpa": loop.pressure_kpa(),
    }


def _reading_missing(readings: Readings) -> bool:
    """读数缺失或非有限值即判缺失（部分失败）。"""

    if not isinstance(readings, dict):
        return True
    for signal in REQUIRED_READINGS:
        value = readings.get(signal)
        if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            return True
        if not math.isfinite(float(value)):
            return True
    return False


def run_cycle(
    loop: SecondaryLoop,
    pump: Pump,
    valve: Valve,
    durable: DurableState,
    dt: float = 1.0,
    speed_pct: float = 85.0,
    *,
    mode: str = CLOSEOUT_IN_PLACE,
    safe_valve_pct: float = 100.0,
    reader: ReadingsReader | None = None,
    bus: EventBus | None = None,
) -> dict:
    """跑完一个二次网循环，并保证泵在返回前停稳、给出收尾结论。"""

    if dt <= 0:
        raise ValueError("步长必须为正")
    if mode not in CLOSEOUT_MODES:
        raise ValueError(f"未知收尾方式：{mode!r}")

    read: ReadingsReader = reader or (lambda flow, _dt: _default_readings(flow, loop))

    steps: list[dict] = []
    reasons: list[str] = []
    flow_t_h = 0.0
    last: dict = {}
    completed = False

    try:
        durable.save(f"secondary/{loop.limits.makeup_target_kpa:.0f}", {"running": True})
        pump.start(speed_pct)
        steps.append(
            {
                "component": "pump",
                "action": "start",
                "detail": {"target_flow_t_h": round(pump.target_flow_t_h(), 2)},
            }
        )
        pump.set_speed(speed_pct)
        for _ in range(MAX_STEPS):
            flow_t_h = pump.step(dt)
            try:
                readings = read(flow_t_h, dt)
            except Exception:  # noqa: BLE001 - 读表异常与读不到数同口径处理
                readings = None
            if _reading_missing(readings):
                # 部分失败：读数缺失不能当成正常完成，按中断收尾。
                reasons.append("reading_missing")
                steps.append(
                    {
                        "component": "secondary",
                        "action": "reading_missing",
                        "detail": {"required": list(REQUIRED_READINGS)},
                    }
                )
                raise _InterruptedCycle()
            last = loop.step(dt, flow_t_h)
            if last["overheat"]:
                reasons.append("return_overtemperature")
                steps.append(
                    {
                        "component": "secondary",
                        "action": "abort",
                        "detail": {
                            "return_c": round(loop.return_c(), 2),
                            "ceiling_c": round(loop.return_ceiling_c(), 2),
                        },
                    }
                )
                raise _InterruptedCycle()
            if pump.target_flow_t_h() > 0 and abs(pump.flow_t_h() - pump.target_flow_t_h()) < 1e-6:
                break
        steps.append(
            {
                "component": "secondary",
                "action": "settle",
                "detail": {"flow_t_h": round(pump.flow_t_h(), 2), "return_c": round(loop.return_c(), 2)},
            }
        )
        valve.set_target(valve.position_pct())
        completed = True
    except _InterruptedCycle:
        pass
    except Exception as error:  # noqa: BLE001 - 任何未预期错误也要走完收尾，不把泵挂在那里
        reasons.append("unexpected_error")
        steps.append(
            {
                "component": "secondary",
                "action": "unexpected_error",
                "detail": {"error": str(error)},
            }
        )
    finally:
        closure = close_out(
            pump,
            valve,
            completed=completed and not reasons,
            mode=mode,
            safe_valve_pct=safe_valve_pct,
            dt=dt,
        )
        steps.extend(closure["steps"])
        reasons.extend(closure["reasons"])

        # 结论唯一口径：无失败原因且泵停稳才算正常收尾。
        pump_idle = bool(closure["pump_idle"])
        partial = "reading_missing" in reasons
        conclusion = (
            CONCLUSION_COMPLETED
            if completed and not reasons and pump_idle
            else CONCLUSION_INTERRUPTED
        )

        state_payload = {
            "running": False,
            "flow_t_h": round(pump.flow_t_h(), 4),
            "pump_idle": pump_idle,
            "conclusion": conclusion,
            "partial": partial,
        }
        try:
            durable.save(f"secondary/{loop.limits.makeup_target_kpa:.0f}", state_payload)
        except Exception:  # noqa: BLE001 - 落盘失败也不能改变泵已停稳这一事实
            reasons.append("closeout_state_persist_failed")

        outcome: dict = {
            "ok": conclusion == CONCLUSION_COMPLETED,
            "conclusion": conclusion,
            "partial": partial,
            "closeout_mode": closure["closeout_mode"],
            "steps": steps,
            "reasons": reasons,
            "violations": list(reasons),
            "flow_t_h": round(pump.flow_t_h(), 2),
            "pump_idle": pump_idle,
            "pump_running": pump.running(),
            "valve_position_pct": closure["valve_position_pct"],
            "valve_target_pct": closure["valve_target_pct"],
            "valve_closed": closure["valve_closed"],
        }
        if last:
            outcome["balance"] = last

        if bus is not None:
            if conclusion == CONCLUSION_COMPLETED:
                severity, message = "notice", "二次网循环正常收尾，泵已停稳"
            elif partial:
                severity, message = "warning", "二次网循环部分失败（读数缺失），按中断收尾"
            elif not pump_idle:
                severity, message = "alarm", "二次网循环中断收尾且泵未停稳"
            else:
                severity, message = "warning", "二次网循环中断收尾，泵已停稳"
            bus.publish(
                Event.of(
                    "secondary",
                    "secondary",
                    "cycle_closed",
                    severity=severity,
                    message=message,
                    conclusion=conclusion,
                    partial=partial,
                    pump_idle=pump_idle,
                    closeout_mode=outcome["closeout_mode"],
                    valve_closed=outcome["valve_closed"],
                    reasons=list(reasons),
                )
            )

    return outcome


class _InterruptedCycle(Exception):
    """循环内部已知的中途失败信号，用于跳出推进进入收尾。"""
