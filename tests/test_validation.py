"""各类非法网表 / 非法参数必须被干净地挡回：HTTP 400 + 机器可读错误码 + 人话说明。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post_dc(elements):
    return client.post("/api/dc", json={"netlist": {"elements": elements}})


def post_transient(elements, dt, tstop):
    return client.post("/api/transient",
                       json={"netlist": {"elements": elements}, "dt": dt, "tstop": tstop})


def check(resp, code):
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["code"] == code
    assert isinstance(body["message"], str) and body["message"]


def test_unknown_element_type():
    check(post_dc([{"name": "Q1", "type": "transistor", "nodes": ["a", "0"], "value": 1}]),
          "UNKNOWN_ELEMENT_TYPE")


def test_missing_node():
    # 只有一个节点
    check(post_dc([{"name": "R1", "type": "resistor", "nodes": ["a"], "value": 100}]),
          "MISSING_NODE")
    # 整个 nodes 字段缺失
    check(post_dc([{"name": "R1", "type": "resistor", "value": 100}]),
          "MISSING_NODE")


def test_no_ground():
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "b"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["a", "b"], "value": 100},
    ]), "NO_GROUND")


def test_ambiguous_ground():
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["a", "gnd"], "value": 100},
    ]), "AMBIGUOUS_GROUND")


def test_zero_resistance():
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": 0},
    ]), "ZERO_RESISTANCE")


def test_duplicate_element_names():
    check(post_dc([
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": 100},
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": 200},
    ]), "DUPLICATE_ELEMENT")


def test_voltage_source_loop():
    # 三个电压源围成三角形回路
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "b"], "value": 1},
        {"name": "V2", "type": "voltage_source", "nodes": ["b", "c"], "value": 2},
        {"name": "V3", "type": "voltage_source", "nodes": ["c", "a"], "value": 3},
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": 1000},
    ]), "VOLTAGE_SOURCE_LOOP")
    # 两个电压源并联也算回路
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "V2", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
    ]), "VOLTAGE_SOURCE_LOOP")


def test_floating_node_makes_dc_matrix_singular():
    # 节点 x 只接电容，直流下没有直流通路 → 悬浮节点
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "x"], "value": 1e-6},
        {"name": "C2", "type": "capacitor", "nodes": ["x", "0"], "value": 1e-6},
    ]), "MATRIX_SINGULAR")


def test_conflicting_capacitor_initial_conditions_singular():
    # 两个初始电压不同的电容并联，t=0 的 KVL 互相矛盾
    check(post_transient([
        {"name": "C1", "type": "capacitor", "nodes": ["x", "0"], "value": 1e-6, "initial": 1.0},
        {"name": "C2", "type": "capacitor", "nodes": ["x", "0"], "value": 1e-6, "initial": 2.0},
        {"name": "R1", "type": "resistor", "nodes": ["x", "0"], "value": 1000},
    ], dt=1e-6, tstop=1e-3), "MATRIX_SINGULAR")


def test_invalid_values():
    # 电容非正
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "C1", "type": "capacitor", "nodes": ["a", "0"], "value": 0},
    ]), "INVALID_VALUE")
    # 电感为负
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "L1", "type": "inductor", "nodes": ["a", "0"], "value": -1e-3},
    ]), "INVALID_VALUE")
    # 参数值不是有限数值（NaN；httpx 的 json= 不允许 NaN，故发原始请求体）
    resp = client.post(
        "/api/dc",
        content='{"netlist": {"elements": ['
                '{"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},'
                '{"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": NaN}]}}',
        headers={"content-type": "application/json"},
    )
    check(resp, "INVALID_VALUE")


def test_missing_value():
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"]},
    ]), "MISSING_VALUE")


def test_initial_condition_on_unsupported_element():
    check(post_dc([
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": 100, "initial": 1.0},
    ]), "UNSUPPORTED_INITIAL_CONDITION")


def test_empty_netlist():
    check(post_dc([]), "EMPTY_NETLIST")


def test_malformed_request():
    # elements 不是列表
    resp = client.post("/api/dc", json={"netlist": {"elements": "not-a-list"}})
    check(resp, "MALFORMED_REQUEST")
    # 元件缺少 name
    resp = client.post("/api/dc", json={"netlist": {"elements": [
        {"type": "resistor", "nodes": ["a", "0"], "value": 100}]}})
    check(resp, "MALFORMED_REQUEST")
    # 整个 netlist 缺失
    resp = client.post("/api/dc", json={})
    check(resp, "MALFORMED_REQUEST")


def test_transient_param_errors_via_api():
    elements = [
        {"name": "V1", "type": "voltage_source", "nodes": ["a", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["a", "0"], "value": 1000},
    ]
    check(post_transient(elements, dt=0.0, tstop=1.0), "INVALID_TIME_PARAMS")
    check(post_transient(elements, dt=-1e-3, tstop=1.0), "INVALID_TIME_PARAMS")
    check(post_transient(elements, dt=1e-3, tstop=1e-4), "INVALID_TIME_PARAMS")
    check(post_transient(elements, dt=1e-3, tstop=1e-3 * 20001), "TOO_MANY_STEPS")


# --- 交流频响 /api/ac 的参数与对象校验 ---------------------------------------


def post_ac(**overrides):
    body = {
        "netlist": {"elements": [
            {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5},
            {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000},
            {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": 1e-6},
        ]},
        "f_start": 1.0, "f_stop": 1e6, "num_points": 10,
        "output_node": "out", "input_source": "V1",
    }
    for key, value in overrides.items():
        if value is None:
            body.pop(key, None)  # 显式传 None 表示从请求里去掉这个字段
        else:
            body[key] = value
    return client.post("/api/ac", json=body)


def test_ac_invalid_frequency_range():
    check(post_ac(f_start=0.0), "INVALID_SWEEP_PARAMS")
    check(post_ac(f_start=-10.0), "INVALID_SWEEP_PARAMS")
    check(post_ac(f_stop=0.5), "INVALID_SWEEP_PARAMS")   # 终止不大于起始
    check(post_ac(f_start=1e6, f_stop=1e6), "INVALID_SWEEP_PARAMS")
    check(post_ac(f_start=None), "INVALID_SWEEP_PARAMS")
    check(post_ac(f_stop=None), "INVALID_SWEEP_PARAMS")


def test_ac_invalid_point_count():
    check(post_ac(num_points=1), "INVALID_SWEEP_PARAMS")   # 少于两点不成曲线
    check(post_ac(num_points=0), "INVALID_SWEEP_PARAMS")
    check(post_ac(num_points=10001), "TOO_MANY_SWEEP_POINTS")  # 超过硬上限
    check(post_ac(num_points=None, points_per_decade=0), "INVALID_SWEEP_PARAMS")
    check(post_ac(num_points=None, points_per_decade=2000), "TOO_MANY_SWEEP_POINTS")


def test_ac_point_spec_must_be_exactly_one():
    check(post_ac(points_per_decade=10), "INVALID_SWEEP_PARAMS")  # 两种都给
    check(post_ac(num_points=None), "INVALID_SWEEP_PARAMS")       # 都不给


def test_ac_output_object_not_found():
    check(post_ac(output_node="ghost"), "OUTPUT_NOT_FOUND")
    # 支路电流输出必须落在电压源上，电阻不行
    check(post_ac(output_node=None, output_branch="R1"), "OUTPUT_NOT_FOUND")
    check(post_ac(output_node=None, output_branch="V9"), "OUTPUT_NOT_FOUND")


def test_ac_input_source_not_found():
    check(post_ac(input_source="ghost"), "INPUT_SOURCE_NOT_FOUND")
    check(post_ac(input_source="R1"), "INPUT_SOURCE_NOT_FOUND")  # 电阻不是独立源


def test_ac_output_and_input_must_be_specified():
    check(post_ac(output_branch="V1"), "MALFORMED_REQUEST")  # 输出给了两个
    check(post_ac(output_node=None), "MALFORMED_REQUEST")    # 输出一个都没给
    check(post_ac(input_source=None), "MALFORMED_REQUEST")   # 参照源缺失


def test_ac_singular_frequency_point_reports_frequency():
    # 参照源之外置零后出现悬浮节点（iso 只接被置零的电流源）
    resp = post_ac(netlist={"elements": [
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000},
        {"name": "I1", "type": "current_source", "nodes": ["iso", "0"], "value": 1e-3},
    ]})
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "MATRIX_SINGULAR"
    assert "Hz" in body["message"]  # 必须说明是哪个频点出的问题
