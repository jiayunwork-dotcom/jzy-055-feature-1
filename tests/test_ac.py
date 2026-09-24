"""交流频响：截止频率、−3dB 点、−20dB/十倍频斜率、高通/低通相位走向、
纯电阻网络频响平坦、频域与直流极限一致、支路电流输出、非法参数挡回。"""

import math

import pytest

from app.ac import MAX_SWEEP_POINTS, run_ac, solve_transfer_at
from app.dc import solve_dc
from app.errors import CircuitError, ErrorCode
from app.netlist import parse_netlist

V, R, C = 5.0, 1000.0, 1e-6
FC = 1.0 / (2.0 * math.pi * R * C)  # RC 转折频率 ≈ 159.155 Hz


def rc_lowpass():
    """一阶 RC 低通：输出取电容两端电压。"""
    return parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": C},
    ])


def rc_highpass():
    """一阶 RC 高通：R、C 位置对调，输出取电阻两端电压。"""
    return parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "C1", "type": "capacitor", "nodes": ["in", "out"], "value": C},
        {"name": "R1", "type": "resistor", "nodes": ["out", "0"], "value": R},
    ])


def divider():
    return parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
        {"name": "R2", "type": "resistor", "nodes": ["out", "0"], "value": 2000.0},
    ])


def sweep(circuit, **kwargs):
    kwargs.setdefault("input_source", "V1")
    kwargs.setdefault("output_node", "out")
    return run_ac(circuit, **kwargs)


# --- 一阶低通 ---------------------------------------------------------------


def test_lowpass_cutoff_frequency_and_minus_3db_point():
    """转折频率必须落在 1/(2πRC)，且该点幅度 ≈ 0.707（−3dB）、相位 −45°。"""
    # 网格中点精确落在 FC 上
    res = sweep(rc_lowpass(), f_start=FC / 10, f_stop=FC * 10, num_points=3)
    mid = res.points[1]
    assert math.isclose(mid.frequency, FC, rel_tol=1e-12)
    assert math.isclose(mid.magnitude, 1.0 / math.sqrt(2.0), abs_tol=1e-9)
    assert math.isclose(mid.magnitude_db, -3.010299956639812, abs_tol=1e-6)
    assert math.isclose(mid.phase_deg, -45.0, abs_tol=1e-6)

    # 加密扫描，按 −3dB 交叉点插值出的截止频率也必须逼近 1/(2πRC)
    dense = sweep(rc_lowpass(), f_start=FC / 10, f_stop=FC * 10, points_per_decade=1000)
    mags = [p.magnitude for p in dense.points]
    freqs = [p.frequency for p in dense.points]
    target = 1.0 / math.sqrt(2.0)
    i = next(k for k in range(len(mags) - 1) if mags[k] >= target >= mags[k + 1])
    t = (mags[i] - target) / (mags[i] - mags[i + 1])
    f_cross = freqs[i] * (freqs[i + 1] / freqs[i]) ** t
    assert math.isclose(f_cross, FC, rel_tol=1e-4)


def test_lowpass_rolloff_20db_per_decade():
    """远高于截止频率：每十倍频幅度衰减逼近 −20dB，相位趋向 −90°。"""
    res = sweep(rc_lowpass(), f_start=1e3 * FC, f_stop=1e4 * FC, num_points=2)
    slope = res.points[1].magnitude_db - res.points[0].magnitude_db
    assert math.isclose(slope, -20.0, abs_tol=1e-3)
    assert math.isclose(res.points[1].magnitude / res.points[0].magnitude, 0.1, rel_tol=1e-4)
    assert math.isclose(res.points[1].phase_deg, -90.0, abs_tol=0.01)


def test_lowpass_passband_flat_with_zero_phase():
    """远低于截止频率：增益 ≈ 1（0dB），相位 ≈ 0°。"""
    res = sweep(rc_lowpass(), f_start=FC / 1000, f_stop=FC / 100, num_points=2)
    assert math.isclose(res.points[0].magnitude, 1.0, rel_tol=1e-6)
    assert math.isclose(res.points[0].magnitude_db, 0.0, abs_tol=1e-4)
    assert abs(res.points[0].phase_deg) < 0.06


# --- 一阶高通 ---------------------------------------------------------------


def test_highpass_blocks_low_and_passes_high():
    """R、C 对调后：低频被压制、高频放行，转折频率仍是 1/(2πRC)。"""
    low, high = sweep(rc_highpass(), f_start=FC / 1000, f_stop=FC * 1000,
                      num_points=2).points
    assert low.magnitude == pytest.approx(1e-3, rel=0.1)   # 低频被压到约 f/fc
    assert low.magnitude_db < -50.0
    assert math.isclose(high.magnitude, 1.0, rel_tol=1e-6)  # 高频趋于放行

    # 转折点上幅度同样是 ≈0.707，但相位为 +45°（与低通相反）
    mid = sweep(rc_highpass(), f_start=FC / 10, f_stop=FC * 10, num_points=3).points[1]
    assert math.isclose(mid.magnitude, 1.0 / math.sqrt(2.0), abs_tol=1e-9)
    assert math.isclose(mid.phase_deg, 45.0, abs_tol=1e-6)


def test_highpass_phase_trend_opposite_to_lowpass():
    """同一网格上：低通相位恒 ≤0 且单调走向 −90°，高通恒 ≥0 且单调走向 0°。"""
    lp = sweep(rc_lowpass(), f_start=FC / 100, f_stop=FC * 100, points_per_decade=10)
    hp = sweep(rc_highpass(), f_start=FC / 100, f_stop=FC * 100, points_per_decade=10)
    lp_phases = [p.phase_deg for p in lp.points]
    hp_phases = [p.phase_deg for p in hp.points]
    assert all(p <= 0.0 for p in lp_phases)
    assert all(p >= 0.0 for p in hp_phases)
    # 单调走向相反：低通一路走向 −90°，高通从 +90° 一路走向 0°
    assert lp_phases == sorted(lp_phases, reverse=True)
    assert hp_phases == sorted(hp_phases, reverse=True)
    assert math.isclose(lp_phases[0], 0.0, abs_tol=0.6)
    assert math.isclose(lp_phases[-1], -90.0, abs_tol=0.6)
    assert math.isclose(hp_phases[0], 90.0, abs_tol=0.6)
    assert math.isclose(hp_phases[-1], 0.0, abs_tol=0.6)
    # 理论上 hp 相位 − lp 相位 ≡ 90°
    for p_lp, p_hp in zip(lp_phases, hp_phases):
        assert math.isclose(p_hp - p_lp, 90.0, abs_tol=1e-6)


# --- 纯电阻网络 -------------------------------------------------------------


def test_resistive_divider_response_is_flat():
    """没有储能元件：整个扫描区间内幅度恒定、相位恒为零。"""
    res = sweep(divider(), f_start=1.0, f_stop=1e6, num_points=50)
    expected_db = 20.0 * math.log10(2.0 / 3.0)
    for p in res.points:
        assert math.isclose(p.magnitude, 2.0 / 3.0, rel_tol=1e-12)
        assert math.isclose(p.magnitude_db, expected_db, rel_tol=1e-12)
        assert math.isclose(p.phase_deg, 0.0, abs_tol=1e-12)


# --- 与直流工作点的一致性 ----------------------------------------------------


def test_ac_low_frequency_limit_matches_dc_operating_point():
    """同一份 RC 网络：极低频的传递函数幅度 == 电容开路时直流算出的电压比。"""
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
        {"name": "R2", "type": "resistor", "nodes": ["out", "0"], "value": 2000.0},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": C},
    ])
    dc = solve_dc(circuit)
    dc_ratio = dc.node_voltages["out"] / dc.node_voltages["in"]
    assert math.isclose(dc_ratio, 2.0 / 3.0, rel_tol=1e-12)  # 电容开路 → 分压

    res = sweep(circuit, f_start=1e-3, f_stop=1e-2, num_points=2)
    for p in res.points:
        assert math.isclose(p.magnitude, dc_ratio, rel_tol=1e-8)
        assert math.isclose(p.phase_deg, 0.0, abs_tol=1e-2)


def test_zero_frequency_falls_back_to_dc_limit():
    """f=0 按直流极限处理：电容开路、电感短路，不因 1/(jωL) 除零而崩。"""
    # 电容开路：out 被拉到电源电压，传递函数 = 1
    h = solve_transfer_at(rc_lowpass(), 0.0, "V1", ("node", "out"))
    assert h == 1.0 + 0.0j

    # 电感短路：out 被拉到地，传递函数 = 0（与直流工作点一致）
    rl = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "L1", "type": "inductor", "nodes": ["out", "0"], "value": 1e-3},
    ])
    h = solve_transfer_at(rl, 0.0, "V1", ("node", "out"))
    assert h == 0.0j
    assert solve_dc(rl).node_voltages["out"] == 0.0


# --- 输出与参照源的各种取法 ---------------------------------------------------


def test_branch_current_output():
    """输出取电压源支路电流：分压网络下即 −1/(R1+R2) 西门子，恒定不随频率变。"""
    res = sweep(divider(), f_start=1.0, f_stop=1e6, num_points=10,
                output_node=None, output_branch="V1")
    assert res.output == "i(V1)"
    for p in res.points:
        assert math.isclose(p.magnitude, 1.0 / 3000.0, rel_tol=1e-12)
        assert math.isclose(abs(p.phase_deg), 180.0, abs_tol=1e-9)  # 电流流入正极


def test_current_source_as_input_reference():
    """参照源也可以是电流源：v(n1)/i(I1) 即电阻值本身。"""
    circuit = parse_netlist([
        {"name": "I1", "type": "current_source", "nodes": ["0", "n1"], "value": 0.001},
        {"name": "R1", "type": "resistor", "nodes": ["n1", "0"], "value": 1000.0},
    ])
    res = run_ac(circuit, f_start=1.0, f_stop=1e3, num_points=5,
                 output_node="n1", input_source="I1")
    for p in res.points:
        assert math.isclose(p.magnitude, 1000.0, rel_tol=1e-12)
        assert math.isclose(p.phase_deg, 0.0, abs_tol=1e-12)


def test_other_sources_are_zeroed():
    """非参照源在小信号意义下置零：第二个电压源短路后不影响传递函数。"""
    one_source = divider()
    two_sources = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "V2", "type": "voltage_source", "nodes": ["aux", "0"], "value": 3.0},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": 1000.0},
        {"name": "R2", "type": "resistor", "nodes": ["out", "0"], "value": 2000.0},
        {"name": "R3", "type": "resistor", "nodes": ["aux", "out"], "value": 4000.0},
    ])
    a = sweep(one_source, f_start=1.0, f_stop=1e3, num_points=5)
    b = sweep(two_sources, f_start=1.0, f_stop=1e3, num_points=5)
    # V2 置零（短路）后 aux 接地，R3 并在 R2 上：分压比 (R2‖R3)/(R1+R2‖R3)
    expected = (2000.0 * 4000.0 / 6000.0) / (1000.0 + 2000.0 * 4000.0 / 6000.0)
    for p in b.points:
        assert math.isclose(p.magnitude, expected, rel_tol=1e-12)
    assert not any(math.isclose(pa.magnitude, pb.magnitude, rel_tol=1e-9)
                   for pa, pb in zip(a.points, b.points))


def test_initial_conditions_are_ignored():
    """储能元件的初始条件不参与频域分析：带不带 initial 结果完全一致。"""
    with_ic = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "C1", "type": "capacitor", "nodes": ["out", "0"], "value": C,
         "initial": 2.0},
    ])
    a = sweep(rc_lowpass(), f_start=1.0, f_stop=1e6, points_per_decade=5)
    b = sweep(with_ic, f_start=1.0, f_stop=1e6, points_per_decade=5)
    assert [p.transfer for p in a.points] == [p.transfer for p in b.points]


# --- 频率网格 ---------------------------------------------------------------


def test_points_per_decade_grid_is_log_uniform():
    res = sweep(divider(), f_start=1.0, f_stop=1e3, points_per_decade=10)
    freqs = [p.frequency for p in res.points]
    assert len(freqs) == 31  # 3 个十倍频程 × 10 + 1
    assert freqs[0] == 1.0
    assert math.isclose(freqs[-1], 1e3, rel_tol=1e-12)
    for a, b in zip(freqs, freqs[1:]):
        assert math.isclose(b / a, 10.0 ** 0.1, rel_tol=1e-12)


def test_num_points_grid_hits_exact_decades():
    res = sweep(divider(), f_start=1.0, f_stop=1e4, num_points=5)
    freqs = [p.frequency for p in res.points]
    for got, want in zip(freqs, [1.0, 10.0, 100.0, 1000.0, 10000.0]):
        assert math.isclose(got, want, rel_tol=1e-12)


# --- 非法参数与找不到的对象 ---------------------------------------------------


def test_invalid_frequency_range_rejected():
    circuit = rc_lowpass()
    bad = [
        (0.0, 100.0), (-1.0, 100.0),          # 起始频率不为正
        (100.0, 0.0), (100.0, -5.0),          # 终止频率不为正
        (100.0, 100.0), (100.0, 50.0),        # 终止不大于起始
        (None, 100.0), (100.0, None),         # 缺失
        (float("nan"), 100.0), (1.0, float("inf")),  # 非有限
    ]
    for f_start, f_stop in bad:
        with pytest.raises(CircuitError) as excinfo:
            sweep(circuit, f_start=f_start, f_stop=f_stop, num_points=10)
        assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS


def test_invalid_point_count_rejected():
    circuit = rc_lowpass()
    for bad_points in (0, 1, -3, 2.5, True):  # 少于两点、非整数都非法
        with pytest.raises(CircuitError) as excinfo:
            sweep(circuit, f_start=1.0, f_stop=100.0, num_points=bad_points)
        assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS
    with pytest.raises(CircuitError) as excinfo:  # 每十倍频程点数必须 ≥1
        sweep(circuit, f_start=1.0, f_stop=100.0, points_per_decade=0)
    assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS


def test_point_spec_must_be_exactly_one():
    circuit = rc_lowpass()
    with pytest.raises(CircuitError) as excinfo:  # 两个都给
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=10, points_per_decade=10)
    assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS
    with pytest.raises(CircuitError) as excinfo:  # 都不给
        sweep(circuit, f_start=1.0, f_stop=100.0)
    assert excinfo.value.code == ErrorCode.INVALID_SWEEP_PARAMS


def test_too_many_sweep_points_rejected():
    circuit = rc_lowpass()
    with pytest.raises(CircuitError) as excinfo:
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=MAX_SWEEP_POINTS + 1)
    assert excinfo.value.code == ErrorCode.TOO_MANY_SWEEP_POINTS
    with pytest.raises(CircuitError) as excinfo:  # 按十倍频程算出来超上限也一样
        sweep(circuit, f_start=1.0, f_stop=1e12, points_per_decade=1000)
    assert excinfo.value.code == ErrorCode.TOO_MANY_SWEEP_POINTS


def test_output_object_not_found():
    circuit = rc_lowpass()
    with pytest.raises(CircuitError) as excinfo:  # 节点不存在
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2, output_node="ghost")
    assert excinfo.value.code == ErrorCode.OUTPUT_NOT_FOUND
    assert "ghost" in excinfo.value.message
    with pytest.raises(CircuitError) as excinfo:  # 支路不是电压源
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2,
              output_node=None, output_branch="R1")
    assert excinfo.value.code == ErrorCode.OUTPUT_NOT_FOUND
    assert "R1" in excinfo.value.message
    with pytest.raises(CircuitError) as excinfo:  # 支路名根本不存在
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2,
              output_node=None, output_branch="V9")
    assert excinfo.value.code == ErrorCode.OUTPUT_NOT_FOUND


def test_input_source_not_found():
    circuit = rc_lowpass()
    for bad in ("ghost", "R1", "C1"):  # 不存在 / 不是独立源
        with pytest.raises(CircuitError) as excinfo:
            sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2, input_source=bad)
        assert excinfo.value.code == ErrorCode.INPUT_SOURCE_NOT_FOUND
        assert bad in excinfo.value.message


def test_output_and_input_must_be_specified():
    circuit = rc_lowpass()
    with pytest.raises(CircuitError) as excinfo:  # 输出二选一，不能都给
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2, output_branch="V1")
    assert excinfo.value.code == ErrorCode.MALFORMED_REQUEST
    with pytest.raises(CircuitError) as excinfo:  # 都不给
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2, output_node=None)
    assert excinfo.value.code == ErrorCode.MALFORMED_REQUEST
    with pytest.raises(CircuitError) as excinfo:  # 参照源缺失
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=2, input_source=None)
    assert excinfo.value.code == ErrorCode.MALFORMED_REQUEST


def test_floating_node_after_zeroing_sources_is_singular():
    """参照源之外置零后出现悬浮节点：判错并说明是哪个频点。"""
    circuit = parse_netlist([
        {"name": "V1", "type": "voltage_source", "nodes": ["in", "0"], "value": V},
        {"name": "R1", "type": "resistor", "nodes": ["in", "out"], "value": R},
        {"name": "I1", "type": "current_source", "nodes": ["iso", "0"], "value": 1e-3},
    ])
    with pytest.raises(CircuitError) as excinfo:
        sweep(circuit, f_start=1.0, f_stop=100.0, num_points=3)
    assert excinfo.value.code == ErrorCode.MATRIX_SINGULAR
    assert "f=1 Hz" in excinfo.value.message  # 第一个频点就出事


def test_lc_resonance_on_grid_point_is_singular():
    """并联 LC 谐振点恰好落在扫描点上：该频点导纳抵消为零，必须判错。"""
    f0 = 10.0  # 网格 1→100 取 3 点，中点精确落在 10 Hz
    circuit = parse_netlist([
        {"name": "I1", "type": "current_source", "nodes": ["0", "a"], "value": 1e-3},
        {"name": "L1", "type": "inductor", "nodes": ["a", "0"],
         "value": 1.0 / (2.0 * math.pi * f0)},
        {"name": "C1", "type": "capacitor", "nodes": ["a", "0"],
         "value": 1.0 / (2.0 * math.pi * f0)},
    ])
    with pytest.raises(CircuitError) as excinfo:
        run_ac(circuit, f_start=1.0, f_stop=100.0, num_points=3,
               output_node="a", input_source="I1")
    assert excinfo.value.code == ErrorCode.MATRIX_SINGULAR
    assert "f=10 Hz" in excinfo.value.message
