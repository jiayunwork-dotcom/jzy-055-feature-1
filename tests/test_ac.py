"""交流频响：截止频率、−3 dB 点、−20 dB/十倍频斜率、高低通相位走向、纯电阻网络
频响平坦、频域与直流极限一致，以及各类非法参数 / 找不到的对象被干净挡回。"""

import math

import pytest
from fastapi.testclient import TestClient

from app.ac import MAX_SWEEP_POINTS, OutputSpec, make_log_sweep, run_ac
from app.dc import solve_dc
from app.errors import CircuitError, ErrorCode
from app.main import app
from app.netlist import parse_netlist

client = TestClient(app)

R, C = 1000.0, 1e-6
FC = 1.0 / (2.0 * math.pi * R * C)  # RC 转折频率 ≈ 159.155 Hz


def rc_lowpass():
    """一阶 RC 低通：输出取电容两端电压。"""
    return parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": C},
    ])


def rc_highpass():
    """R、C 位置对调的一阶高通：输出取电阻两端电压。"""
    return parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "C1", "type": "capacitor", "nodes": ["in", "out"], "value": C},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": R},
    ])


def divider():
    return parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
        {"name": "R2", "type": "resistor", "nodes": ["out", "0"], "value": 2000.0},
    ])


def run_voltage(circuit, freqs, node="out", source="V1"):
    return run_ac(circuit, freqs, OutputSpec(kind="voltage", node=node), source)


def nearest(points, f):
    return min(points, key=lambda p: abs(math.log10(p.frequency_hz / f)))


# ---------------------------------------------------------------- 低通：转折频率、−3 dB、斜率、相位

def test_lowpass_cutoff_frequency_matches_1_over_2pi_rc():
    freqs = make_log_sweep(FC / 100, FC * 100, num_points=401)  # 100 点/十倍频
    res = run_voltage(rc_lowpass(), freqs)
    target = 1.0 / math.sqrt(2.0)
    # 通带（低频端）幅度接近 1，第一个跌破 1/√2 的频点即测得的转折频率
    assert res.points[0].magnitude > 0.999
    f_measured = next(p.frequency_hz for p in res.points if p.magnitude <= target)
    # 网格分辨率 100 点/十倍频 → 量化误差不超过 10^0.01 ≈ 2.4%
    assert math.isclose(f_measured, FC, rel_tol=0.03)


def test_lowpass_minus_3db_at_cutoff():
    freqs = make_log_sweep(FC / 100, FC * 100, num_points=401)
    res = run_voltage(rc_lowpass(), freqs)
    p = nearest(res.points, FC)
    # 网格恰好命中 fc（端点之间的步进是 0.01 十倍频的整数倍）
    assert math.isclose(p.frequency_hz, FC, rel_tol=1e-9)
    # 该点幅度必须降到通带的约 0.707 倍，即约 −3 dB
    assert math.isclose(p.magnitude, 1.0 / math.sqrt(2.0), rel_tol=1e-3)
    assert math.isclose(p.magnitude_db, -3.0103, abs_tol=0.02)
    assert math.isclose(p.phase_deg, -45.0, abs_tol=0.5)


def test_lowpass_rolloff_20db_per_decade():
    # 远高于截止频率：每升高十倍频，幅度衰减必须逼近 −20 dB
    freqs = make_log_sweep(FC * 100, FC * 1000, num_points=101)
    res = run_voltage(rc_lowpass(), freqs)
    p1, p2 = res.points[0], res.points[-1]
    slope = (p2.magnitude_db - p1.magnitude_db) / math.log10(
        p2.frequency_hz / p1.frequency_hz
    )
    assert math.isclose(slope, -20.0, abs_tol=0.01)
    # 且整个阻带单调下降
    mags = [p.magnitude for p in res.points]
    assert all(b < a for a, b in zip(mags, mags[1:]))


def test_lowpass_phase_goes_from_0_to_minus_90():
    freqs = make_log_sweep(FC / 1000, FC * 1000, num_points=601)
    res = run_voltage(rc_lowpass(), freqs)
    assert abs(res.points[0].phase_deg) < 1.0          # 远低于 fc：≈ 0°
    assert abs(res.points[-1].phase_deg + 90.0) < 1.0  # 远高于 fc：≈ −90°
    phases = [p.phase_deg for p in res.points]
    assert all(p <= 0.0 for p in phases)               # 低通全程滞后（负相位）
    assert all(b <= a + 1e-9 for a, b in zip(phases, phases[1:]))  # 单调下降


# ---------------------------------------------------------------- 高通：与低通镜像

def test_highpass_blocks_low_and_passes_high():
    freqs = make_log_sweep(FC / 1000, FC * 1000, num_points=601)
    res = run_voltage(rc_highpass(), freqs)
    # 低频段被压制：0.001·fc 处幅度 ≈ f/fc = 1e-3
    assert res.points[0].magnitude < 1.1e-3
    assert res.points[0].magnitude_db < -59.0
    # 高频段趋于放行
    assert math.isclose(res.points[-1].magnitude, 1.0, rel_tol=1e-3)
    # 转折频率处同样是 −3 dB
    p = nearest(res.points, FC)
    assert math.isclose(p.magnitude, 1.0 / math.sqrt(2.0), rel_tol=1e-3)
    assert math.isclose(p.magnitude_db, -3.0103, abs_tol=0.02)


def test_highpass_phase_opposite_to_lowpass():
    freqs = make_log_sweep(FC / 1000, FC * 1000, num_points=601)
    lp = run_voltage(rc_lowpass(), freqs)
    hp = run_voltage(rc_highpass(), freqs)
    # 转折点处相位符号相反：低通 −45°，高通 +45°
    assert math.isclose(nearest(lp.points, FC).phase_deg, -45.0, abs_tol=0.5)
    assert math.isclose(nearest(hp.points, FC).phase_deg, +45.0, abs_tol=0.5)
    # 高通全程超前（正相位）：低频端趋 +90°，高频端回 0°，与低通走向相反
    assert all(p.phase_deg >= 0.0 for p in hp.points)
    assert hp.points[0].phase_deg > 89.0
    assert abs(hp.points[-1].phase_deg) < 1.0
    assert lp.points[-1].phase_deg < -89.0


# ---------------------------------------------------------------- 纯电阻网络：频响平坦

def test_resistive_divider_response_is_flat():
    # 没有储能元件：整个扫描区间内幅度恒定、相位恒为零
    freqs = make_log_sweep(1e-3, 1e9, num_points=121)
    res = run_voltage(divider(), freqs)
    expected_db = 20.0 * math.log10(2.0 / 3.0)
    for p in res.points:
        assert math.isclose(p.magnitude, 2.0 / 3.0, rel_tol=1e-12)
        assert math.isclose(p.magnitude_db, expected_db, rel_tol=1e-12)
        assert abs(p.phase_deg) < 1e-9
    # 传递函数与源的直流取值无关：V1=5V，但小信号传递函数是 2/3 而不是 10/3
    assert math.isclose(res.points[0].magnitude, 2.0 / 3.0, rel_tol=1e-12)


# ---------------------------------------------------------------- 与直流分析的一致性

def test_ac_low_frequency_limit_matches_dc_operating_point():
    """同一份 RC 网络：极低频的频域传递函数 == 电容当开路的直流电压比。"""
    circuit = rc_lowpass()
    dc = solve_dc(circuit)  # 电容开路 → out 被拉到电源电压
    dc_ratio = dc.node_voltages["out"] / dc.node_voltages["in"]
    assert math.isclose(dc_ratio, 1.0, rel_tol=1e-12)
    freqs = make_log_sweep(1e-3, 1e-2, num_points=10)  # ωRC ≤ 6.3e-5，深度通带
    res = run_voltage(circuit, freqs)
    for p in res.points:
        assert math.isclose(p.magnitude, dc_ratio, rel_tol=1e-6)
        assert abs(p.phase_deg) < 0.01


def test_zero_frequency_point_is_dc_limit():
    """f=0 按直流极限处理：电容开路、电感短路，不因除零而崩。"""
    circuit = rc_lowpass()
    dc = solve_dc(circuit)
    res = run_voltage(circuit, [0.0])
    p = res.points[0]
    assert p.frequency_hz == 0.0
    assert math.isclose(p.magnitude, dc.node_voltages["out"] / dc.node_voltages["in"],
                        rel_tol=1e-12)
    assert p.phase_deg == 0.0

    # 电感在 ω=0 是短路（阻抗为零），同样不能除零崩溃
    rl = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "L1", "type": "inductor", "nodes": ["out", "0"], "value": 1e-3},
    ])
    dc_rl = solve_dc(rl)
    assert dc_rl.node_voltages["out"] == 0.0  # 直流下电感把 out 短到地
    res0 = run_voltage(rl, [0.0])
    assert res0.points[0].magnitude == 0.0
    # 幅度为零时 dB 取有限的地板值，绝不报 −inf
    assert math.isfinite(res0.points[0].magnitude_db)
    assert res0.points[0].magnitude_db <= -6000.0


def test_ac_matches_dc_for_resistive_divider():
    circuit = divider()
    dc = solve_dc(circuit)
    dc_ratio = dc.node_voltages["out"] / dc.node_voltages["in"]
    res = run_voltage(circuit, make_log_sweep(1.0, 1e6, num_points=7))
    for p in res.points:
        assert math.isclose(p.magnitude, dc_ratio, rel_tol=1e-12)


# ---------------------------------------------------------------- 电感与支路电流输出

def test_rl_lowpass_cutoff():
    """电感经阻抗支路盖章：RL 低通截止频率 f_c = R/(2πL)。"""
    inductance = 1e-3
    fc = R / (2.0 * math.pi * inductance)  # ≈ 159.155 kHz
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "L1", "type": "inductor", "nodes": ["in", "out"], "value": inductance},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": R},
    ])
    freqs = make_log_sweep(fc / 100, fc * 100, num_points=401)
    res = run_voltage(circuit, freqs)
    p = nearest(res.points, fc)
    assert math.isclose(p.magnitude, 1.0 / math.sqrt(2.0), rel_tol=1e-3)
    assert math.isclose(res.points[0].magnitude, 1.0, rel_tol=1e-3)   # 低频放行
    assert res.points[-1].magnitude < 0.01                             # 高频压制


def test_branch_current_output_matches_hand_calculation():
    freqs = make_log_sweep(1.0, 1e6, num_points=13)
    # 分压网络 R1 电流：(1 − 2/3) / 1kΩ = 1/3 mA（单位激励下），全频段平坦
    res = run_ac(divider(), freqs, OutputSpec(kind="current", element="R1"), "V1")
    for p in res.points:
        assert math.isclose(p.magnitude, 1.0 / 3000.0, rel_tol=1e-12)
        assert abs(p.phase_deg) < 1e-9
    # 电压源支路电流：正方向为流入正极，供电故为负（与直流同一符号约定）
    res_v = run_ac(divider(), freqs, OutputSpec(kind="current", element="V1"), "V1")
    for p in res.points:
        assert math.isclose(p.transfer.real, -1.0 / 3000.0, rel_tol=1e-12)
        assert abs(abs(p.phase_deg) - 180.0) < 1e-9


def test_capacitor_and_inductor_current_outputs():
    # RC 低通在 fc 处：|i_C| = ωC·|v_out| = 1/(R·√2)
    freqs = make_log_sweep(FC / 100, FC * 100, num_points=401)
    res = run_ac(rc_lowpass(), freqs, OutputSpec(kind="current", element="C1"), "V1")
    p = nearest(res.points, FC)
    assert math.isclose(p.magnitude, 1.0 / (R * math.sqrt(2.0)), rel_tol=1e-3)

    # RL 低通深度通带：电感电流 ≈ 1/R（单位激励）
    rl = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "L1", "type": "inductor", "nodes": ["in", "out"], "value": 1e-3},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": R},
    ])
    fc = R / (2.0 * math.pi * 1e-3)
    res_l = run_ac(rl, make_log_sweep(fc / 1000, fc / 100, num_points=11),
                   OutputSpec(kind="current", element="L1"), "V1")
    for p in res_l.points:
        assert math.isclose(p.magnitude, 1.0 / R, rel_tol=1e-3)


def test_current_source_as_reference_input():
    """参照源也可以是电流源：跨阻传递函数 v_out/i_in = R，全频段平坦。"""
    circuit = parse_netlist([
        {"name": "I1", "type": "current_source", "nodes": ["0", "out"], "value": 2e-3},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": 1000.0},
    ])
    res = run_voltage(circuit, make_log_sweep(1.0, 1e6, num_points=7), source="I1")
    for p in res.points:
        assert math.isclose(p.magnitude, 1000.0, rel_tol=1e-12)
        assert abs(p.phase_deg) < 1e-9


def test_initial_conditions_are_ignored():
    """储能元件的初始条件不参与频域分析：带了不报错，结果与不带完全一致。"""
    freqs = make_log_sweep(1.0, 1e5, num_points=21)
    base = run_voltage(rc_lowpass(), freqs)
    with_v0 = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": C,
         "initial": 3.0},
    ])
    res_v0 = run_voltage(with_v0, freqs)
    assert [p.transfer for p in res_v0.points] == [p.transfer for p in base.points]

    rl_no_i0 = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "L1", "type": "inductor", "nodes": ["in", "out"], "value": 1e-3},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": R},
    ])
    rl_with_i0 = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "L1", "type": "inductor", "nodes": ["in", "out"], "value": 1e-3,
         "initial": 2e-3},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": R},
    ])
    r1 = run_voltage(rl_no_i0, freqs)
    r2 = run_voltage(rl_with_i0, freqs)
    assert [p.transfer for p in r1.points] == [p.transfer for p in r2.points]


# ---------------------------------------------------------------- 扫描频率轴

def test_log_sweep_is_log_uniform_with_exact_endpoints():
    freqs = make_log_sweep(1.0, 1e4, num_points=5)
    assert freqs[0] == 1.0 and freqs[-1] == 1e4
    ratios = [freqs[i + 1] / freqs[i] for i in range(len(freqs) - 1)]
    for r in ratios:
        assert math.isclose(r, 10.0, rel_tol=1e-12)  # 对数轴上均匀分布


def test_points_per_decade_equivalent_to_num_points():
    a = make_log_sweep(1.0, 1e4, points_per_decade=2)   # 4 个十倍频 × 2 → 9 点
    b = make_log_sweep(1.0, 1e4, num_points=9)
    assert len(a) == 9
    assert a == b


def test_sweep_param_errors():
    bad_ranges = [
        (0.0, 100.0), (-1.0, 100.0),          # 起始频率不为正
        (100.0, 0.0), (100.0, -5.0),          # 终止频率不为正
        (100.0, 100.0), (100.0, 1.0),         # 终止不大于起始
        (None, 100.0), (100.0, None),         # 缺失
        (float("nan"), 100.0), (100.0, float("inf")),
    ]
    for start, stop in bad_ranges:
        with pytest.raises(CircuitError) as excinfo:
            make_log_sweep(start, stop, num_points=10)
        assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS

    for n in (0, 1, -5, 2.5, "10", True, None):
        with pytest.raises(CircuitError) as excinfo:
            make_log_sweep(1.0, 100.0, num_points=n,
                           points_per_decade=None if n is not None else None)
        assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS

    for ppd in (0.0, -3.0, float("nan")):
        with pytest.raises(CircuitError) as excinfo:
            make_log_sweep(1.0, 100.0, points_per_decade=ppd)
        assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS

    # 两种取点方式必须且只能给一种
    with pytest.raises(CircuitError) as excinfo:
        make_log_sweep(1.0, 100.0)
    assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS
    with pytest.raises(CircuitError) as excinfo:
        make_log_sweep(1.0, 100.0, num_points=10, points_per_decade=5)
    assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS


def test_sweep_point_count_limit():
    # 上限本身可用
    freqs = make_log_sweep(1.0, 10.0, num_points=MAX_SWEEP_POINTS)
    assert len(freqs) == MAX_SWEEP_POINTS
    # 直接超上限
    with pytest.raises(CircuitError) as excinfo:
        make_log_sweep(1.0, 10.0, num_points=MAX_SWEEP_POINTS + 1)
    assert excinfo.value.code == ErrorCode.TOO_MANY_POINTS
    # 由每十倍频程点数推算后超上限
    with pytest.raises(CircuitError) as excinfo:
        make_log_sweep(1.0, 1e9, points_per_decade=2000.0)
    assert excinfo.value.code == ErrorCode.TOO_MANY_POINTS


# ---------------------------------------------------------------- 找不到的对象与奇异矩阵

def test_output_node_not_found():
    with pytest.raises(CircuitError) as excinfo:
        run_voltage(rc_lowpass(), [100.0], node="nope")
    assert excinfo.value.code == ErrorCode.OUTPUT_NOT_FOUND
    assert "nope" in excinfo.value.message


def test_output_element_not_found():
    with pytest.raises(CircuitError) as excinfo:
        run_ac(rc_lowpass(), [100.0], OutputSpec(kind="current", element="R99"), "V1")
    assert excinfo.value.code == ErrorCode.OUTPUT_NOT_FOUND
    assert "R99" in excinfo.value.message


def test_invalid_output_spec():
    with pytest.raises(CircuitError) as excinfo:
        run_ac(rc_lowpass(), [100.0], OutputSpec(kind="power", node="out"), "V1")
    assert excinfo.value.code == ErrorCode.INVALID_OUTPUT_SPEC
    # voltage 缺节点名 / current 缺元件名
    with pytest.raises(CircuitError) as excinfo:
        run_ac(rc_lowpass(), [100.0], OutputSpec(kind="voltage"), "V1")
    assert excinfo.value.code == ErrorCode.INVALID_OUTPUT_SPEC
    with pytest.raises(CircuitError) as excinfo:
        run_ac(rc_lowpass(), [100.0], OutputSpec(kind="current"), "V1")
    assert excinfo.value.code == ErrorCode.INVALID_OUTPUT_SPEC


def test_input_source_not_found():
    with pytest.raises(CircuitError) as excinfo:
        run_voltage(rc_lowpass(), [100.0], source="V9")
    assert excinfo.value.code == ErrorCode.INPUT_SOURCE_NOT_FOUND
    assert "V9" in excinfo.value.message


def test_floating_node_after_zeroing_sources_is_singular():
    # 节点 x 只经电流源 I1 连接；小信号分析中 I1 置零（开路）→ x 悬浮 → 矩阵奇异
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "I1", "type": "current_source", "nodes": ["in", "x"], "value": 0.5},
        {"name": "R1", "type": "resistor", "nodes": ["in", "0"], "value": 1000.0},
    ])
    with pytest.raises(CircuitError) as excinfo:
        run_voltage(circuit, [1234.0], node="in")
    assert excinfo.value.code == ErrorCode.MATRIX_SINGULAR
    assert "1234" in excinfo.value.message  # 必须说明是哪个频点出的问题


def test_capacitor_only_node_singular_at_zero_frequency():
    # 节点 x 只经电容连接：ω=0 时电容开路 → x 悬浮（与直流行为一致）；ω>0 则正常
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "C1", "type": "capacitor", "nodes": ["in", "x"], "value": 1e-6},
        {"name": "R1", "type": "resistor", "nodes": ["in", "0"], "value": 1000.0},
    ])
    with pytest.raises(CircuitError) as excinfo:
        run_voltage(circuit, [0.0], node="in")
    assert excinfo.value.code == ErrorCode.MATRIX_SINGULAR
    assert "0" in excinfo.value.message
    res = run_voltage(circuit, [100.0], node="x")
    assert math.isclose(res.points[0].magnitude, 1.0, rel_tol=1e-12)


def test_inductor_loop_singular_only_at_zero_frequency():
    # 纯电感回路：ω=0 时电感全是短路，KVL 约束冗余 → 奇异（同直流）；ω>0 正常
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "a"], "value": 1000.0},
        {"name": "L1", "type": "inductor", "nodes": ["a", "b"], "value": 1e-3},
        {"name": "L2", "type": "inductor", "nodes": ["b", "c"], "value": 2e-3},
        {"name": "L3", "type": "inductor", "nodes": ["c", "a"], "value": 3e-3},
    ])
    with pytest.raises(CircuitError) as excinfo:
        run_voltage(circuit, [0.0], node="a")
    assert excinfo.value.code == ErrorCode.MATRIX_SINGULAR
    res = run_voltage(circuit, [1000.0], node="a")
    assert math.isfinite(res.points[0].magnitude)


def test_sweep_across_lc_resonance_never_returns_nan_or_inf():
    # 串联 LC 到地、输出取中点：谐振点附近幅度巨大，但每个频点都必须有限
    inductance, capacitance = 1e-3, 1e-9
    f_res = 1.0 / (2.0 * math.pi * math.sqrt(inductance * capacitance))  # ≈ 159.155 kHz
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 1.0},
        {"name": "L1", "type": "inductor", "nodes": ["in", "out"], "value": inductance},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": capacitance},
    ])
    freqs = make_log_sweep(f_res / 10, f_res * 10, num_points=2001)
    res = run_voltage(circuit, freqs)
    peak = max(p.magnitude for p in res.points)
    assert peak > 100.0  # 确实扫过了谐振峰
    for p in res.points:
        assert math.isfinite(p.transfer.real) and math.isfinite(p.transfer.imag)
        assert math.isfinite(p.magnitude)
        assert math.isfinite(p.magnitude_db)
        assert math.isfinite(p.phase_deg)


# ---------------------------------------------------------------- HTTP 接口

RC_LOWPASS_BODY = {
    "netlist": {
        "elements": [
            {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 1.0},
            {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
            {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": 1e-6},
        ]
    },
    "sweep": {"start_hz": 1.0, "stop_hz": 1e5, "points_per_decade": 10},
    "output": {"kind": "voltage", "node": "out"},
    "input_source": "V1",
}


def post_ac(body):
    return client.post("/api/ac", json=body)


def test_ac_endpoint_happy_path():
    resp = post_ac(RC_LOWPASS_BODY)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["input_source"] == "V1"
    assert body["output"] == {"kind": "voltage", "node": "out", "element": None}
    points = body["points"]
    assert len(points) == 51  # 5 个十倍频 × 10 点 + 1
    assert points[0]["frequency_hz"] == 1.0
    assert points[-1]["frequency_hz"] == 1e5
    for p in points:
        # 每个频点：复数传递函数 + 线性幅度 + 分贝 + 相位，全部有限
        for key in ("frequency_hz", "real", "imag", "magnitude",
                    "magnitude_db", "phase_deg"):
            assert math.isfinite(p[key])
        # 幅度/相位与实部虚部自洽
        assert math.isclose(p["magnitude"], math.hypot(p["real"], p["imag"]),
                            rel_tol=1e-12)
        assert math.isclose(p["magnitude_db"], 20.0 * math.log10(p["magnitude"]),
                            rel_tol=1e-9)
        assert math.isclose(p["phase_deg"],
                            math.degrees(math.atan2(p["imag"], p["real"])),
                            abs_tol=1e-9)
    # 深度通带 ≈ 0 dB，阻带末端 ≈ −56 dB（1e5 Hz ≫ fc≈159 Hz）
    assert abs(points[0]["magnitude_db"]) < 0.01
    assert math.isclose(points[-1]["magnitude_db"], -55.97, abs_tol=0.1)


def test_ac_endpoint_with_num_points_and_current_output():
    body = {
        "netlist": {
            "elements": [
                {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
                {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
                {"name": "R2", "type": "resistor", "nodes": ["out", "0"], "value": 2000.0},
            ]
        },
        "sweep": {"start_hz": 10.0, "stop_hz": 1000.0, "num_points": 3},
        "output": {"kind": "current", "element": "R1"},
        "input_source": "V1",
    }
    resp = post_ac(body)
    assert resp.status_code == 200
    points = resp.json()["points"]
    assert len(points) == 3
    assert [p["frequency_hz"] for p in points] == [10.0, 100.0, 1000.0]
    for p in points:
        assert math.isclose(p["magnitude"], 1.0 / 3000.0, rel_tol=1e-12)


def test_ac_endpoint_rejects_bad_sweep_params():
    def check(sweep, code):
        body = {**RC_LOWPASS_BODY, "sweep": sweep}
        resp = post_ac(body)
        assert resp.status_code == 400, resp.text
        assert resp.json()["code"] == code
        assert resp.json()["message"]

    check({"start_hz": 0.0, "stop_hz": 100.0, "num_points": 10}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": -1.0, "stop_hz": 100.0, "num_points": 10}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": 100.0, "stop_hz": 100.0, "num_points": 10}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": 100.0, "stop_hz": 1.0, "num_points": 10}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": 1.0, "stop_hz": 100.0, "num_points": 1}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": 1.0, "stop_hz": 100.0}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": 1.0, "stop_hz": 100.0, "num_points": 10,
           "points_per_decade": 5}, "INVALID_SWEEP_PARAMS")
    check({"start_hz": 1.0, "stop_hz": 100.0, "num_points": MAX_SWEEP_POINTS + 1},
          "TOO_MANY_POINTS")


def test_ac_endpoint_rejects_missing_objects():
    base = RC_LOWPASS_BODY
    resp = post_ac({**base, "output": {"kind": "voltage", "node": "nope"}})
    assert resp.status_code == 400
    assert resp.json()["code"] == "OUTPUT_NOT_FOUND"
    assert "nope" in resp.json()["message"]

    resp = post_ac({**base, "output": {"kind": "current", "element": "R99"}})
    assert resp.status_code == 400
    assert resp.json()["code"] == "OUTPUT_NOT_FOUND"
    assert "R99" in resp.json()["message"]

    resp = post_ac({**base, "input_source": "V9"})
    assert resp.status_code == 400
    assert resp.json()["code"] == "INPUT_SOURCE_NOT_FOUND"
    assert "V9" in resp.json()["message"]

    resp = post_ac({**base, "output": {"kind": "power", "node": "out"}})
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_OUTPUT_SPEC"


def test_ac_endpoint_reports_singular_frequency():
    body = {
        "netlist": {
            "elements": [
                {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": 5.0},
                {"name": "I1", "type": "current_source", "nodes": ["in", "x"], "value": 0.5},
                {"name": "R1", "type": "resistor", "nodes": ["in", "0"], "value": 1000.0},
            ]
        },
        "sweep": {"start_hz": 100.0, "stop_hz": 1000.0, "num_points": 2},
        "output": {"kind": "voltage", "node": "in"},
        "input_source": "V1",
    }
    resp = post_ac(body)
    assert resp.status_code == 400
    assert resp.json()["code"] == "MATRIX_SINGULAR"
    assert "100" in resp.json()["message"]  # 报出具体频点


def test_ac_endpoint_malformed_request():
    resp = post_ac({"netlist": RC_LOWPASS_BODY["netlist"]})  # 缺 sweep / output / input_source
    assert resp.status_code == 400
    assert resp.json()["code"] == "MALFORMED_REQUEST"


def test_ac_example_is_hand_checkable():
    """随服务发布的 RC 低通算例：fc ≈ 159.155 Hz 处必须是 −3 dB。"""
    examples = client.get("/api/examples").json()
    assert "rc_lowpass" in examples
    ex = examples["rc_lowpass"]
    resp = post_ac({
        "netlist": ex["netlist"],
        "sweep": ex["suggested_sweep"],
        "output": ex["output"],
        "input_source": ex["input_source"],
    })
    assert resp.status_code == 200
    points = resp.json()["points"]
    p = min(points, key=lambda q: abs(math.log10(q["frequency_hz"] / FC)))
    assert math.isclose(p["magnitude_db"], -3.0103, abs_tol=0.35)  # 20 点/十倍频的网格量化
    assert math.isclose(p["magnitude"], 1.0 / math.sqrt(2.0), rel_tol=0.06)
