"""HTTP 接口端到端行为：健康检查、算例、直流、瞬态与交流频响的响应形状。"""

import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

DIVIDER = {
    "elements": [
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
        {"name": "R2", "type": "resistor", "nodes": ["out", "0"], "value": 2000.0},
    ]
}

RC = {
    "elements": [
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": 1e-6},
    ]
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dc_endpoint_happy_path():
    resp = client.post("/api/dc", json={"netlist": DIVIDER})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert math.isclose(body["node_voltages"]["in"], 5.0, rel_tol=1e-12)
    assert math.isclose(body["node_voltages"]["out"], 10.0 / 3.0, rel_tol=1e-12)
    assert body["node_voltages"]["0"] == 0.0
    assert math.isclose(body["voltage_source_currents"]["V1"], -5.0 / 3000.0, rel_tol=1e-12)


def test_transient_endpoint_response_shape():
    resp = client.post("/api/transient", json={"netlist": RC, "dt": 1e-5, "tstop": 1e-3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["integration"] == "backward_euler"
    assert body["steps"] == 100
    assert len(body["time"]) == body["steps"] + 1
    assert body["time"][0] == 0.0
    assert math.isclose(body["time"][-1], 1e-3, rel_tol=1e-9)
    for node in ("in", "out", "0"):
        assert len(body["node_voltages"][node]) == len(body["time"])
    assert body["node_voltages"]["in"][-1] == 5.0
    # 1ms = 1τ 处应约为 63.2% 的电源电压
    assert math.isclose(body["node_voltages"]["out"][-1], 5 * (1 - math.exp(-1)), rel_tol=2e-2)


def test_examples_available_and_hand_checkable():
    resp = client.get("/api/examples")
    assert resp.status_code == 200
    examples = resp.json()
    assert set(examples) >= {"voltage_divider", "rc_charge"}

    # 算例一：分压网络，直流解可笔算验证
    r = client.post("/api/dc", json={"netlist": examples["voltage_divider"]["netlist"]})
    assert r.status_code == 200
    assert math.isclose(r.json()["node_voltages"]["out"], 10.0 / 3.0, rel_tol=1e-9)

    # 算例二：RC 充电，瞬态末态逼近电源电压
    rc = examples["rc_charge"]
    r = client.post("/api/transient", json={
        "netlist": rc["netlist"],
        "dt": rc["suggested_dt"],
        "tstop": rc["suggested_tstop"],
    })
    assert r.status_code == 200
    v_end = r.json()["node_voltages"]["out"][-1]
    assert math.isclose(v_end, 5 * (1 - math.exp(-5)), rel_tol=1e-2)
    assert v_end > 0.99 * 5.0


def test_error_response_shape():
    resp = client.post("/api/dc", json={"netlist": {"elements": [
        {"name": "Q1", "type": "transistor", "nodes": ["a", "0"], "value": 1}]}})
    assert resp.status_code == 400
    body = resp.json()
    assert set(body) == {"code", "message"}
    assert body["code"] == "UNKNOWN_ELEMENT_TYPE"


def test_ac_endpoint_happy_path():
    resp = client.post("/api/ac", json={
        "netlist": RC,
        "f_start": 1.0, "f_stop": 1e6,
        "points_per_decade": 10,
        "output_node": "out",
        "input_source": "V1",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["input_source"] == "V1"
    assert body["output"] == "v(out)"
    points = body["points"]
    assert len(points) == 61  # 6 个十倍频程 × 10 点 + 1
    assert set(points[0]) == {"frequency", "transfer_real", "transfer_imag",
                              "magnitude", "magnitude_db", "phase_deg"}
    # 所有频点的所有数值都必须有限，绝不混入 NaN 或无穷
    for p in points:
        assert all(math.isfinite(v) for v in p.values())
    # 频率在对数轴上均匀分布
    freqs = [p["frequency"] for p in points]
    for a, b in zip(freqs, freqs[1:]):
        assert math.isclose(b / a, 10.0 ** 0.1, rel_tol=1e-9)
    # RC 低通（fc≈159Hz）：1Hz 处增益约 1、相位约 0；最后两个十倍频程衰减约 20dB
    assert math.isclose(points[0]["magnitude"], 1.0, rel_tol=1e-4)
    assert math.isclose(points[0]["phase_deg"], 0.0, abs_tol=0.5)
    assert math.isclose(points[-1]["magnitude_db"] - points[-11]["magnitude_db"],
                        -20.0, abs_tol=1e-3)
    # 复数传递函数与幅度/相位自洽
    for p in points:
        assert math.isclose(p["magnitude"],
                            math.hypot(p["transfer_real"], p["transfer_imag"]),
                            rel_tol=1e-12)


def test_ac_endpoint_with_num_points_and_branch_current():
    resp = client.post("/api/ac", json={
        "netlist": DIVIDER,
        "f_start": 10.0, "f_stop": 1e5,
        "num_points": 9,
        "output_branch": "V1",
        "input_source": "V1",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "i(V1)"
    assert len(body["points"]) == 9
    # 纯电阻网络：支路电流传递函数恒为 −1/(R1+R2)，不随频率变化
    for p in body["points"]:
        assert math.isclose(p["magnitude"], 1.0 / 3000.0, rel_tol=1e-12)
        assert math.isclose(abs(p["phase_deg"]), 180.0, abs_tol=1e-9)
