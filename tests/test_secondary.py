"""二次网循环的收尾保证与水力平衡。"""

from __future__ import annotations

import pytest

from heatnet.config import default_config
from heatnet.errors import StorageError
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


class _SensorDroppedLoop(SecondaryLoop):
    """回传丢读的回路替身：平衡读数缺值，模拟传感器掉线。"""

    def step(self, dt: float, flow_t_h: float) -> dict:
        payload = super().step(dt, flow_t_h)
        for key in ("flow_t_h", "delta_c", "load_kw"):
            payload[key] = None
        return payload


class _FailingLoop(SecondaryLoop):
    """第一步推进就抛错的回路替身。"""

    def step(self, dt: float, flow_t_h: float) -> dict:
        raise RuntimeError("回读总线掉线")


class _StuckPump(Pump):
    """永远报未停稳的泵替身，验证泵未停稳时阀门不被允许动作。"""

    def idle(self) -> bool:
        return False


class _FlakyDurable(DurableState):
    """第二次写盘必失败的当前态替身。"""

    def __init__(self, store: JsonStore) -> None:
        super().__init__(store)
        self._saves = 0

    def save(self, component: str, payload):
        self._saves += 1
        if self._saves > 1:
            raise StorageError("磁盘写失败", component="store")
        return super().save(component, payload)


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
    durable = DurableState(JsonStore(str(tmp_path)))
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is True
    assert report["pump_idle"] is True
    assert pump.idle() is True
    assert pump.running() is False
    assert durable.load("secondary/320")["running"] is False


def test_cycle_error_path_still_stops_pump(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = DurableState(JsonStore(str(tmp_path)))
    loop.disturb(3.0)
    assert loop.overheat() is True
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is False
    assert report["violations"] == ["return_overtemperature"]
    assert report["pump_idle"] is True
    assert pump.idle() is True
    assert pump.starts() == 1
    assert pump.stops() == 1


def test_cycle_success_concludes_completed_and_persists(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = DurableState(JsonStore(str(tmp_path)))
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is True
    assert report["conclusion"] == "completed"
    assert report["teardown"] == "hold"
    assert report["partial"] is False
    assert report["state_persisted"] is True
    saved = durable.load("secondary/320")
    assert saved["running"] is False
    assert saved["conclusion"] == "completed"
    assert saved["partial"] is False


def test_cycle_overheat_teardown_orders_safe_valve_then_pump_then_close(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    valve.apply(65.0)
    durable = DurableState(JsonStore(str(tmp_path)))
    loop.disturb(3.0)
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["conclusion"] == "interrupted"
    assert report["teardown"] == "safe_valve"
    actions = [step["action"] for step in report["steps"]]
    assert actions == ["start", "abort", "safe_position", "stop", "close", "persist_state"]
    assert actions.index("safe_position") < actions.index("stop") < actions.index("close")
    safe = report["steps"][actions.index("safe_position")]
    assert safe["detail"]["position_pct"] == pytest.approx(20.0)
    assert valve.position_pct() == pytest.approx(0.0)
    assert pump.idle() is True
    assert durable.load("secondary/320")["conclusion"] == "interrupted"


def test_cycle_missing_readings_is_partial_failure(tmp_path):
    loop = _SensorDroppedLoop(default_config().secondary)
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = DurableState(JsonStore(str(tmp_path)))
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is False
    assert report["partial"] is True
    assert report["conclusion"] == "interrupted"
    assert report["teardown"] == "in_place"
    assert report["violations"] == ["readings_missing"]
    assert report["pump_idle"] is True
    assert pump.idle() is True
    saved = durable.load("secondary/320")
    assert saved["running"] is False
    assert saved["conclusion"] == "interrupted"
    assert saved["partial"] is True


def test_cycle_exception_is_interrupted_and_still_stops_pump(tmp_path):
    loop = _FailingLoop(default_config().secondary)
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = DurableState(JsonStore(str(tmp_path)))
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is False
    assert report["conclusion"] == "interrupted"
    assert report["teardown"] == "in_place"
    assert report["violations"] == ["cycle_exception"]
    assert report["pump_idle"] is True
    assert pump.idle() is True
    assert pump.starts() == 1
    assert pump.stops() == 1
    assert durable.load("secondary/320")["conclusion"] == "interrupted"


def test_persist_failure_is_reported_not_hidden(tmp_path):
    loop = _loop()
    pump = _pump()
    valve = Valve("st01.mix", default_config().valve)
    durable = _FlakyDurable(JsonStore(str(tmp_path)))
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is False
    assert report["state_persisted"] is False
    assert "state_persist_failed" in report["violations"]
    assert report["pump_idle"] is True
    assert pump.idle() is True


def test_valve_stays_put_when_pump_never_idles(tmp_path):
    loop = _loop()
    config = default_config()
    curve = PumpCurve(
        config.pump.min_speed_pct,
        config.pump.max_speed_pct,
        config.primary.flow_baseline_t_h * 0.25,
        config.primary.flow_baseline_t_h,
    )
    pump = _StuckPump("secondary.circulating", config.pump, curve)
    valve = Valve("st01.mix", config.valve)
    valve.apply(65.0)
    durable = DurableState(JsonStore(str(tmp_path)))
    report = run_cycle(loop, pump, valve, durable, 1.0)
    assert report["ok"] is False
    assert report["conclusion"] == "interrupted"
    assert "pump_not_idle" in report["violations"]
    assert report["pump_idle"] is False
    hold = [step for step in report["steps"] if step["action"] == "hold"]
    assert hold and hold[-1]["detail"]["blocked"] == "pump_not_idle"
    assert not any(step["action"] == "close" for step in report["steps"])
    assert valve.position_pct() == pytest.approx(65.0)


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
