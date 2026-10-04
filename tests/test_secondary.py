"""二次网循环的收尾保证与水力平衡。"""

from __future__ import annotations

import pytest

from heatnet.config import default_config
from heatnet.eventbus import EventBus
from heatnet.pump.curve import PumpCurve
from heatnet.pump.pump import Pump
from heatnet.pump.start import start_pump_after_valve, stop_pump_before_valve
from heatnet.secondary.balance import hydraulic_balance
from heatnet.secondary.cycle import run_cycle
from heatnet.secondary.loop import SecondaryLoop
from heatnet.secondary.makeup import makeup_then_pressurise
from heatnet.store.jsonfile import JsonStore
from heatnet.store.state import DurableState
from heatnet.valve.valve import Valve


def _pump() -> Pump:
    config = default_config()
    baseline = config.primary.flow_baseline_t_h
    curve = PumpCurve(config.pump.min_speed_pct, config.pump.max_speed_pct, baseline * 0.25, baseline)
    return Pump("secondary.circulating", config.pump, curve)


def _loop() -> SecondaryLoop:
    return SecondaryLoop(default_config().secondary)


def _durable(tmp_path) -> DurableState:
    return DurableState(JsonStore(str(tmp_path)))


def _readings(loop: SecondaryLoop) -> dict:
    def read(flow_t_h: float, dt: float) -> dict:
        return {
            "flow_t_h": flow_t_h,
            "return_c": loop.return_c(),
            "pressure_kpa": loop.pressure_kpa(),
        }

    return read


def test_hydraulic_balance_scales_with_flow_and_delta():
    report = hydraulic_balance(2000.0, 60.0, 45.0)
    assert report.delta_c == pytest.approx(15.0)
    assert report.load_kw == pytest.approx(2000.0 * 15.0 / 1.163)
    assert report.flow_ratio == pytest.approx(2000.0 / 2400.0)
    assert report.as_dict()["load_kw"] == pytest.approx(round(report.load_kw, 2))
    assert hydraulic_balance(0.0, 60.0, 45.0).load_kw == pytest.approx(0.0)


def test_cycle_stops_pump_even_on_success(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is True
    assert report["conclusion"] == "completed"
    assert report["partial"] is False
    assert report["pump_idle"] is True
    assert pump.idle() is True
    assert pump.running() is False
    assert report["valve_closed"] is False
    state = durable.load("secondary/320")
    assert state["running"] is False
    assert state["conclusion"] == "completed"
    assert state["pump_idle"] is True


def test_cycle_error_path_still_stops_pump(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    loop.disturb(3.0)
    assert loop.overheat() is True
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is False
    assert report["conclusion"] == "interrupted"
    assert report["violations"] == ["return_overtemperature"]
    assert report["pump_idle"] is True
    assert pump.idle() is True
    assert pump.starts() == 1
    assert pump.stops() == 1


def test_cycle_interrupted_closes_valve_after_pump_settles(tmp_path):
    """就地中断收尾：先停泵停稳，再关阀，顺序不能乱。"""

    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    loop.disturb(3.0)
    report = run_cycle(loop, pump, valve, durable, 1.0, mode="in_place")
    actions = [step["action"] for step in report["steps"]]
    assert actions.index("stop") < actions.index("close_after_pump")
    assert "retreat_to_safe" not in actions
    assert report["valve_closed"] is True
    assert valve.position_pct() == pytest.approx(0.0)


def test_cycle_retreat_mode_repositions_valve_before_pump_stop(tmp_path):
    """退回安全阀位：先退阀（泵仍在转）→ 停泵停稳 → 再关阀。"""

    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    loop.disturb(3.0)
    report = run_cycle(loop, pump, valve, durable, 1.0, mode="retreat", safe_valve_pct=80.0)
    assert report["conclusion"] == "interrupted"
    actions = [step["action"] for step in report["steps"]]
    assert (
        actions.index("retreat_to_safe")
        < actions.index("stop")
        < actions.index("close_after_pump")
    )
    assert report["valve_closed"] is True
    assert valve.position_pct() == pytest.approx(0.0)


def test_missing_reading_is_partial_failure_and_interrupts(tmp_path):
    """中途读数缺失属于部分失败：按中断收尾，不能当成正常完成。"""

    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    calls = {"n": 0}

    def reader(flow_t_h: float, dt: float):
        calls["n"] += 1
        if calls["n"] == 3:
            return {"flow_t_h": flow_t_h, "return_c": None, "pressure_kpa": loop.pressure_kpa()}
        return _readings(loop)(flow_t_h, dt)

    report = run_cycle(loop, pump, valve, durable, 1.0, reader=reader)
    assert report["conclusion"] == "interrupted"
    assert report["partial"] is True
    assert "reading_missing" in report["reasons"]
    assert report["ok"] is False
    assert report["pump_idle"] is True
    assert report["valve_closed"] is True
    assert durable.load("secondary/320")["conclusion"] == "interrupted"


def test_reader_exception_is_treated_as_missing_reading(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)

    def reader(flow_t_h: float, dt: float):
        raise RuntimeError("sensor bus down")

    report = run_cycle(loop, pump, valve, durable, 1.0, reader=reader)
    assert report["conclusion"] == "interrupted"
    assert report["partial"] is True
    assert "reading_missing" in report["reasons"]
    assert pump.idle() is True


def test_unexpected_error_during_cycle_still_closes_out(tmp_path):
    class BrokenLoop(SecondaryLoop):
        def step(self, dt: float, flow_t_h: float) -> dict:
            raise RuntimeError("hydraulics exploded")

    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    report = run_cycle(BrokenLoop(default_config().secondary), pump, valve, durable, 1.0)
    assert report["conclusion"] == "interrupted"
    assert "unexpected_error" in report["reasons"]
    assert pump.idle() is True


def test_pump_that_never_settles_marks_interrupted_and_withholds_valve(tmp_path):
    """泵没停稳：结论必为中断、关阀联锁拒绝、审计告警。"""

    class StuckPump(Pump):
        def idle(self) -> bool:
            return False

    config = default_config()
    curve = PumpCurve(
        config.pump.min_speed_pct,
        config.pump.max_speed_pct,
        config.primary.flow_baseline_t_h * 0.25,
        config.primary.flow_baseline_t_h,
    )
    pump = StuckPump("secondary.circulating", config.pump, curve)
    valve = Valve("st01.mix", config.valve)
    durable = _durable(tmp_path)
    bus = EventBus()
    report = run_cycle(_loop(), pump, valve, durable, 1.0, bus=bus)
    assert report["conclusion"] == "interrupted"
    assert report["pump_idle"] is False
    assert "pump_not_settled" in report["reasons"]
    assert report["valve_closed"] is False
    actions = [step["action"] for step in report["steps"]]
    # 泵没停稳时关阀联锁拒绝执行：不允许出现"泵后关阀"这一步。
    assert "close_after_pump" not in actions
    assert "pump_not_settled" in actions
    events = [event for event in bus.history() if event.action == "cycle_closed"]
    assert len(events) == 1
    assert events[0].payload["conclusion"] == "interrupted"
    assert events[0].payload["pump_idle"] is False
    assert events[0].payload["severity"] == "alarm"


def test_cycle_publishes_closeout_event_matching_report(tmp_path):
    """收尾结论、泵状态与审计三者口径一致（正常 + 读数缺失各一条）。"""

    # 正常
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _durable(tmp_path)
    bus = EventBus()
    report = run_cycle(_loop(), pump, valve, durable, 1.0, bus=bus)
    events = [event for event in bus.history() if event.action == "cycle_closed"]
    assert len(events) == 1
    assert events[0].payload["severity"] == "notice"
    assert events[0].payload["conclusion"] == report["conclusion"] == "completed"
    assert events[0].payload["pump_idle"] == report["pump_idle"] is True

    # 读数缺失（部分失败）
    loop = _loop()
    calls = {"n": 0}

    def reader(flow_t_h: float, dt: float):
        calls["n"] += 1
        if calls["n"] == 2:
            return None
        return _readings(loop)(flow_t_h, dt)

    report = run_cycle(loop, _pump(), Valve("v2", default_config().valve), durable, 1.0,
                       reader=reader, bus=bus)
    events = [event for event in bus.history() if event.action == "cycle_closed"]
    assert len(events) == 2
    assert events[-1].payload["conclusion"] == "interrupted"
    assert events[-1].payload["partial"] is True
    assert events[-1].payload["severity"] == "warning"


def test_cycle_rejects_unknown_closeout_mode(tmp_path):
    with pytest.raises(ValueError):
        run_cycle(_loop(), _pump(), Valve("st01.mix", default_config().valve),
                  _durable(tmp_path), 1.0, mode="fly-away")


def test_makeup_runs_before_pressurising():
    loop = _loop()
    loop.bleed_pressure(loop.limits.makeup_min_kpa)
    assert loop.needs_makeup() is True
    report = makeup_then_pressurise(loop, 1.0)
    assert report["ok"] is True
    assert loop.needs_makeup() is False
    assert [step["action"] for step in report["steps"]] == ["makeup"]


def test_makeup_bleeds_when_overpressurised():
    loop = _loop()
    loop.add_pressure(200.0)
    assert loop.overpressure() is True
    report = makeup_then_pressurise(loop, 1.0)
    assert report["ok"] is True
    assert [step["action"] for step in report["steps"]] == ["makeup", "bleed"]
    assert loop.overpressure() is False


def test_pump_start_requires_settled_valve(tmp_path):
    config = default_config()
    valve = Valve("st01.mix", config.valve)
    durable = DurableState(JsonStore(str(tmp_path)))
    pump = _pump()
    report = start_pump_after_valve(valve, durable, pump, 60.0, 1.0)
    assert report["ok"] is True
    assert valve.at_target() is True
    assert pump.running() is True
    assert pump.target_flow_t_h() > 0.0

    stopped = stop_pump_before_valve(pump, valve, 1.0)
    assert stopped["ok"] is True
    assert pump.idle() is True
    assert valve.target_pct() == pytest.approx(0.0)
