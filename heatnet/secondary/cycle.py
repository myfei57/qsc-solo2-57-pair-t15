"""二次网循环编排。

一个循环里先启泵、再推进流量与回水；不管走的是正常路径、越限路径还是异常路径，
离开循环之前泵都必须停稳——泵没停稳就返回，循环会一直空转，把二次网抽空。

收尾固定为分步收，顺序定死、不允许调换：

1. 按工况选定收尾方式：回水超温先把混水阀退回安全开度（泵还在转，只能调制、
   不能关死，阀先关死会把泵憋住）；读数缺失或异常就就地停泵，阀保持原位；
2. 停泵并等流量归零；
3. 泵停稳之后才允许动阀——先停泵、后关阀，顺序反了会把泵憋住；
4. 把收尾结论与泵的最终状态落盘。

中途任何环节抛错都按中断收尾：报告照常返回，收尾照样执行，泵不会挂着空转。
中途读数缺失属于部分失败，结论只能是中断收尾，不能算正常完成。报告里的结论、
泵状态与落盘记录出自同一份终态，控制台展示与审计落盘的口径一致。
"""

from __future__ import annotations

import math

from ..errors import PlantError
from ..pump.pump import Pump
from ..store.state import DurableState
from ..valve.valve import Valve
from .loop import SecondaryLoop

MAX_STEPS = 600

# 超温收尾时混水阀的安全开度：压低一次侧进热但不断流，开度为零会把运行中的泵憋住。
SAFE_VALVE_PCT = 20.0

# 收尾方式：正常收尾阀保持当前位；超温先退安全阀位再停泵；读数缺失或异常就地停泵。
TEARDOWN_HOLD = "hold"
TEARDOWN_SAFE_VALVE = "safe_valve"
TEARDOWN_IN_PLACE = "in_place"

# 收尾结论：正常收尾 / 中断收尾。
CONCLUSION_COMPLETED = "completed"
CONCLUSION_INTERRUPTED = "interrupted"

# 每个推进步必须回传的读数：数值读数要有限，标志位要是布尔。缺键、错型、
# 非有限都按读数缺失处理——读数不可信时绝不能当成工况正常继续走。
_NUMERIC_READINGS = ("flow_t_h", "delta_c", "load_kw")
_FLAG_READINGS = ("overheat", "needs_makeup")


def _state_doc(loop: SecondaryLoop) -> str:
    """二次网当前态文档标识。"""

    return f"secondary/{loop.limits.makeup_target_kpa:.0f}"


def _missing_readings(payload: dict) -> list[str]:
    """检查本步读数，返回缺失项；全部可信时返回空列表。"""

    missing: list[str] = []
    for key in _NUMERIC_READINGS:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            missing.append(key)
    for key in _FLAG_READINGS:
        if not isinstance(payload.get(key), bool):
            missing.append(key)
    return missing


def _circulate(
    loop: SecondaryLoop,
    pump: Pump,
    durable: DurableState,
    dt: float,
    speed_pct: float,
    steps: list[dict],
    violations: list[str],
) -> tuple[dict, str, bool]:
    """循环本体：启泵、建立流量、推进回水，返回 (结果, 收尾方式, 是否部分失败)。"""

    durable.save(_state_doc(loop), {"running": True})
    pump.start(speed_pct)
    steps.append(
        {
            "component": "pump",
            "action": "start",
            "detail": {"target_flow_t_h": round(pump.target_flow_t_h(), 2)},
        }
    )
    pump.set_speed(speed_pct)

    last: dict = {}
    for _ in range(MAX_STEPS):
        flow = pump.step(dt)
        last = loop.step(dt, flow)
        missing = _missing_readings(last)
        if missing:
            violations.append("readings_missing")
            steps.append(
                {
                    "component": "secondary",
                    "action": "abort",
                    "detail": {"reason": "readings_missing", "missing": ",".join(missing)},
                }
            )
            return {"flow_t_h": round(flow, 2)}, TEARDOWN_IN_PLACE, True
        if last["overheat"]:
            violations.append("return_overtemperature")
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
            return {"flow_t_h": round(flow, 2)}, TEARDOWN_SAFE_VALVE, False
        if pump.target_flow_t_h() > 0 and abs(pump.flow_t_h() - pump.target_flow_t_h()) < 1e-6:
            break
    steps.append(
        {
            "component": "secondary",
            "action": "settle",
            "detail": {"flow_t_h": round(pump.flow_t_h(), 2), "return_c": round(loop.return_c(), 2)},
        }
    )
    return {"flow_t_h": round(pump.flow_t_h(), 2), "balance": last}, TEARDOWN_HOLD, False


def _teardown(
    mode: str,
    pump: Pump,
    valve: Valve,
    dt: float,
    steps: list[dict],
    violations: list[str],
) -> None:
    """分步定序收尾：退安全阀位 → 停泵 → 泵停稳后才动阀。收尾本身不再抛错。"""

    # 第一步：超温工况先把混水阀退回安全开度，压低一次侧进热；泵还在转，
    # 只能调制不能关死。
    if mode == TEARDOWN_SAFE_VALVE:
        valve.set_target(SAFE_VALVE_PCT)
        for _ in range(MAX_STEPS):
            valve.step(dt)
            if valve.at_target():
                break
        steps.append(
            {
                "component": "valve",
                "action": "safe_position",
                "detail": {"position_pct": round(valve.position_pct(), 2), "settled": valve.at_target()},
            }
        )

    # 第二步：停泵并等流量归零。
    pump.stop()
    for _ in range(MAX_STEPS):
        pump.step(dt)
        if pump.idle():
            break
    steps.append(
        {
            "component": "pump",
            "action": "stop",
            "detail": {"flow_t_h": round(pump.flow_t_h(), 4), "idle": pump.idle()},
        }
    )
    if not pump.idle():
        violations.append("pump_not_idle")

    # 第三步：泵停稳之后才允许动阀——先停泵、后关阀，顺序不能反。
    if not pump.idle():
        steps.append(
            {
                "component": "valve",
                "action": "hold",
                "detail": {"position_pct": round(valve.position_pct(), 2), "blocked": "pump_not_idle"},
            }
        )
        return
    if mode == TEARDOWN_SAFE_VALVE:
        valve.set_target(0.0)
        for _ in range(MAX_STEPS):
            valve.step(dt)
            if valve.at_target():
                break
        steps.append(
            {
                "component": "valve",
                "action": "close",
                "detail": {"position_pct": round(valve.position_pct(), 2), "settled": valve.at_target()},
            }
        )
    else:
        valve.set_target(valve.position_pct())
        steps.append(
            {
                "component": "valve",
                "action": "hold",
                "detail": {"position_pct": round(valve.position_pct(), 2)},
            }
        )


def _persist(
    durable: DurableState,
    loop: SecondaryLoop,
    pump: Pump,
    mode: str,
    conclusion: str,
    partial: bool,
    steps: list[dict],
    violations: list[str],
) -> bool:
    """把收尾结论与泵的最终状态落盘；写盘失败如实记进报告，不装成功。"""

    doc = _state_doc(loop)
    payload = {
        "running": pump.running(),
        "flow_t_h": round(pump.flow_t_h(), 4),
        "conclusion": conclusion,
        "teardown": mode,
        "partial": partial,
    }
    try:
        durable.save(doc, payload)
        persisted = True
    except PlantError:
        violations.append("state_persist_failed")
        persisted = False
    steps.append(
        {
            "component": "secondary",
            "action": "persist_state",
            "detail": {"doc": doc, "persisted": persisted, "conclusion": conclusion},
        }
    )
    return persisted


def run_cycle(
    loop: SecondaryLoop,
    pump: Pump,
    valve: Valve,
    durable: DurableState,
    dt: float = 1.0,
    speed_pct: float = 85.0,
) -> dict:
    """跑完一个二次网循环：任何路径离开前泵都停稳，收尾结论落盘并写进报告。"""

    if dt <= 0:
        raise ValueError("步长必须为正")

    steps: list[dict] = []
    violations: list[str] = []
    partial = False
    try:
        outcome, mode, partial = _circulate(loop, pump, durable, dt, speed_pct, steps, violations)
    except Exception as error:  # 中途抛错按中断收尾：报告照常返回，泵不挂着空转
        mode = TEARDOWN_IN_PLACE
        outcome = {"flow_t_h": round(pump.flow_t_h(), 2)}
        violations.append("cycle_exception")
        steps.append(
            {
                "component": "secondary",
                "action": "abort",
                "detail": {"error": getattr(error, "message", str(error)), "type": type(error).__name__},
            }
        )

    _teardown(mode, pump, valve, dt, steps, violations)
    if violations or partial or not pump.idle():
        conclusion = CONCLUSION_INTERRUPTED
    else:
        conclusion = CONCLUSION_COMPLETED
    persisted = _persist(durable, loop, pump, mode, conclusion, partial, steps, violations)

    report = {
        "ok": not violations,
        "conclusion": conclusion,
        "teardown": mode,
        "partial": partial,
        "steps": steps,
        "violations": violations,
        "flow_t_h": outcome.get("flow_t_h", 0.0),
        "pump_idle": pump.idle(),
        "state_persisted": persisted,
    }
    if "balance" in outcome:
        report["balance"] = outcome["balance"]
    return report
