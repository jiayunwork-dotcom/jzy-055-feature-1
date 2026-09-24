"""交流小信号频率响应分析（AC 频域扫描）。

物理定义：分析时独立源被视为给定角频率 ω 下的正弦相量；储能元件回到各自的
复阻抗——电容导纳 jωC、电感阻抗 jωL（即导纳 1/(jωL)）、电阻仍是实数电导。
整套修正节点方程在复数域组装求解（复用 mna.py 的同一套盖章骨架，
complex_mode=True），节点电位与支路电流都是复数相量。

传递函数：参照输入源设为单位激励（1∠0°），其余独立源在小信号意义下置零
（电压源短路、电流源开路，符合叠加与小信号线性化惯例），解一次复数网络，
所选输出相量即该频点的传递函数值。储能元件的初始条件不参与频域分析
（小信号分析假设已线性化、处于稳态正弦），网表里带了 initial 直接忽略。

频率网格在对数频率轴上均匀分布（频响最常用的取法）。f=0 按直流极限处理：
电容导纳为零（开路）、电感短路（以 0V 电压源建模），与 dc.py 口径一致，
不会因 1/(jωL) 除零而崩。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from .errors import CircuitError, ErrorCode
from .mna import MnaBuilder
from .netlist import Circuit

MIN_SWEEP_POINTS = 2        # 少于两点不成一条曲线
MAX_SWEEP_POINTS = 10_000   # 扫描点数硬上限，防止失控
MIN_MAGNITUDE = 1e-300      # 分贝换算的幅度下限，避免 log10(0) 产生 -inf

EXCITATION = 1.0 + 0.0j     # 参照源的单位激励相量（1∠0°）


@dataclass(frozen=True)
class AcPoint:
    """单个频点：复数传递函数及其幅度（线性 / 分贝）与相位（度）。"""

    frequency: float       # 赫兹
    transfer: complex      # 输出相量 / 参照源激励相量
    magnitude: float       # 线性幅度比
    magnitude_db: float    # 20·log10(magnitude)，下限 −6000 dB
    phase_deg: float       # 相位，度，(-180, 180]


@dataclass
class AcResult:
    input_source: str      # 参照输入源名
    output: str            # 输出描述，形如 "v(out)" / "i(V1)"
    points: list[AcPoint]  # 频率序列，按频率升序


def _is_int_like(x: object) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    return isinstance(x, float) and math.isfinite(x) and x.is_integer()


def _check_point_count(value: object, label: str, minimum: int) -> int:
    if not _is_int_like(value):
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS, f"{label} 必须是整数，得到 {value!r}")
    n = int(value)  # type: ignore[arg-type]
    if n < minimum:
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            f"{label} 至少为 {minimum}（少于 {minimum} 个扫描点不成一条曲线），得到 {n}")
    return n


def _frequency_grid(
    f_start: object,
    f_stop: object,
    points_per_decade: object,
    num_points: object,
) -> list[float]:
    """校验扫描参数并生成对数均匀频率网格（含两个端点）。"""
    for label, f in (("起始频率 f_start", f_start), ("终止频率 f_stop", f_stop)):
        if (
            not isinstance(f, (int, float)) or isinstance(f, bool)
            or not math.isfinite(f) or f <= 0
        ):
            raise CircuitError(
                ErrorCode.INVALID_SWEEP_PARAMS,
                f"{label}必须是正的有限数值，得到 {f!r}")
    assert isinstance(f_start, (int, float)) and isinstance(f_stop, (int, float))
    if f_stop <= f_start:
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            f"终止频率 f_stop（{f_stop}）必须大于起始频率 f_start（{f_start}）")

    if (points_per_decade is None) == (num_points is None):
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            "扫描点数必须且只能二选一：points_per_decade（每十倍频程点数）"
            "或 num_points（整段总点数）")

    if num_points is not None:
        n = _check_point_count(num_points, "num_points", MIN_SWEEP_POINTS)
    else:
        ppd = _check_point_count(points_per_decade, "points_per_decade", 1)
        decades = math.log10(f_stop / f_start)
        n = max(1, int(round(decades * ppd))) + 1  # 每十倍频程 ppd 个间隔
    if n > MAX_SWEEP_POINTS:
        raise CircuitError(
            ErrorCode.TOO_MANY_SWEEP_POINTS,
            f"需要 {n} 个扫描点，超过上限 {MAX_SWEEP_POINTS}；"
            "请缩小频率范围或减少点数")

    step = math.log10(f_stop / f_start) / (n - 1)
    return [f_start * 10.0 ** (i * step) for i in range(n)]


def _resolve_output(
    circuit: Circuit,
    output_node: object,
    output_branch: object,
) -> tuple[str, str]:
    """把输出指定解析为 ("node", 节点名) 或 ("branch", 电压源名)。"""
    if (output_node is None) == (output_branch is None):
        raise CircuitError(
            ErrorCode.MALFORMED_REQUEST,
            "输出必须且只能二选一：output_node（节点对地电压）"
            "或 output_branch（电压源支路电流）")
    if output_node is not None:
        if not isinstance(output_node, str) or not output_node:
            raise CircuitError(
                ErrorCode.MALFORMED_REQUEST, "output_node 必须是非空的节点名")
        known = (*circuit.nodes, circuit.ground)
        if output_node not in known:
            raise CircuitError(
                ErrorCode.OUTPUT_NOT_FOUND,
                f"输出节点 {output_node!r} 在网表中不存在；现有节点：{list(known)}")
        return ("node", output_node)
    if not isinstance(output_branch, str) or not output_branch:
        raise CircuitError(
            ErrorCode.MALFORMED_REQUEST, "output_branch 必须是非空的元件名")
    vs_names = [vs.name for vs in circuit.voltage_sources]
    if output_branch not in vs_names:
        raise CircuitError(
            ErrorCode.OUTPUT_NOT_FOUND,
            f"支路电流输出 {output_branch!r} 对不上任何电压源；"
            f"支路电流未知量只随电压源引入（现有电压源：{vs_names or '无'}）")
    return ("branch", output_branch)


def _resolve_input(circuit: Circuit, input_source: object) -> str:
    """校验参照输入源名，返回其规范名字。"""
    if input_source is None:
        raise CircuitError(
            ErrorCode.MALFORMED_REQUEST,
            "必须指定参照输入源 input_source（独立电压源或电流源的名字）")
    if not isinstance(input_source, str) or not input_source:
        raise CircuitError(
            ErrorCode.MALFORMED_REQUEST, "input_source 必须是非空的元件名")
    vs_names = {vs.name for vs in circuit.voltage_sources}
    cs_names = {cs.name for cs in circuit.current_sources}
    if input_source in vs_names or input_source in cs_names:
        return input_source
    raise CircuitError(
        ErrorCode.INPUT_SOURCE_NOT_FOUND,
        f"参照输入源 {input_source!r} 对不上网表中的任何独立源；"
        f"现有电压源：{sorted(vs_names) or '无'}，电流源：{sorted(cs_names) or '无'}")


def solve_transfer_at(
    circuit: Circuit,
    frequency: float,
    input_source: str,
    output: tuple[str, str],
) -> complex:
    """单个频点上的传递函数值：输出相量 / 参照源激励相量（单位激励）。

    其余独立源置零（电压源短路、电流源开路）。frequency 为 0 时按直流极限
    处理：电容开路、电感短路。矩阵奇异时抛出带频点信息的 MATRIX_SINGULAR。
    """
    omega = 2.0 * math.pi * frequency
    builder = MnaBuilder(circuit.nodes, circuit.ground, complex_mode=True)

    for r in circuit.resistors:
        builder.add_conductance(r.a, r.b, 1.0 / r.resistance)
    for c in circuit.capacitors:  # 电容导纳 jωC（ω=0 时自然为开路）
        builder.add_conductance(c.a, c.b, 1j * omega * c.capacitance)
    for ind in circuit.inductors:
        if omega == 0.0:
            # 直流极限：电感短路，与 dc.py 一致以 0V 电压源建模
            builder.add_voltage_source(ind.name, ind.a, ind.b, 0.0j)
        else:  # 电感阻抗 jωL → 导纳 1/(jωL)
            builder.add_conductance(ind.a, ind.b, 1.0 / (1j * omega * ind.inductance))
    for vs in circuit.voltage_sources:
        # 参照源单位激励，其余电压源置零（短路，仍保留支路电流未知量）
        e = EXCITATION if vs.name == input_source else 0.0j
        builder.add_voltage_source(vs.name, vs.a, vs.b, e)
    for cs in circuit.current_sources:
        if cs.name == input_source:
            builder.add_current_source(cs.a, cs.b, EXCITATION)
        # 其余电流源置零即开路：不盖章

    try:
        sol = builder.solve()
    except CircuitError as exc:
        raise CircuitError(
            exc.code,
            f"频点 f={frequency:.6g} Hz 处求解失败：{exc.message}",
        ) from exc

    kind, name = output
    out = sol.node_voltages[name] if kind == "node" else sol.branch_currents[name]
    return out / EXCITATION


def run_ac(
    circuit: Circuit,
    *,
    f_start: float | None = None,
    f_stop: float | None = None,
    points_per_decade: int | None = None,
    num_points: int | None = None,
    output_node: str | None = None,
    output_branch: str | None = None,
    input_source: str | None = None,
) -> AcResult:
    """在对数频率轴上扫描，给出每个频点输出相对参照源的传递函数。"""
    freqs = _frequency_grid(f_start, f_stop, points_per_decade, num_points)
    output = _resolve_output(circuit, output_node, output_branch)
    source = _resolve_input(circuit, input_source)

    points: list[AcPoint] = []
    for f in freqs:
        h = solve_transfer_at(circuit, f, source, output)
        mag = abs(h)
        mag_db = 20.0 * math.log10(max(mag, MIN_MAGNITUDE))
        phase = math.degrees(cmath.phase(h))  # 零相量的相位定义为 0°
        points.append(AcPoint(
            frequency=f, transfer=h, magnitude=mag,
            magnitude_db=mag_db, phase_deg=phase,
        ))

    label = f"v({output[1]})" if output[0] == "node" else f"i({output[1]})"
    return AcResult(input_source=source, output=label, points=points)
