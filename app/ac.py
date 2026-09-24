"""交流小信号频率响应分析（AC 频率扫描）。

物理定义：电路中的独立源视为正弦小信号激励（复数相量），储能元件回到各自的复
阻抗——电容导纳 jωC、电感阻抗 jωL、电阻仍为实数电导。整套修正节点方程在复数域
组装与求解：复用 mna.py 的 MNA 骨架（complex_mode=True），元件的符号约定、节点
与接地的处理、电压源/阻抗支路的扩维方式与直流、瞬态完全一致。

传递函数：参照输入源置单位激励 1∠0°，其余独立源在小信号意义下置零（电压源短路、
电流源开路），解一次复数网络，所选输出相量即该频点的传递函数值（激励为 1，相除
恒等）。独立源的直流取值（value）是工作点参数，不参与频响计算。

- ω=0 是直流极限：电容导纳为零（开路）、电感阻抗为零（以阻抗支路盖章，退化为
  0V 电压源即短路），不产生任何除零；
- 电容/电感的初始条件不参与频域分析（小信号稳态正弦假设），网表里带了也直接忽略；
- 每个频点独立求解，某频点矩阵奇异时以带 MATRIX_SINGULAR 错误码的 CircuitError
  报出具体频率，绝不让 NaN 或无穷大混进曲线。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .errors import CircuitError, ErrorCode
from .mna import MnaBuilder, MnaSolution
from .netlist import (
    Capacitor,
    Circuit,
    CurrentSource,
    Inductor,
    Resistor,
    VoltageSource,
)

MAX_SWEEP_POINTS = 10_000  # 频点硬上限，防止失控

_DB_FLOOR = 1e-300  # 幅度地板：log10(0) 无定义，dB 最低报 −6000 dB，保证结果有限

OUTPUT_KIND_VOLTAGE = "voltage"  # 节点对地电压
OUTPUT_KIND_CURRENT = "current"  # 支路电流（正方向 a→b；电压源为流入正极）


@dataclass(frozen=True)
class OutputSpec:
    """看哪里：某个节点对地的电压，或某条支路的电流。"""

    kind: str                # "voltage" | "current"
    node: str | None = None      # kind=voltage 时的节点名
    element: str | None = None   # kind=current 时的支路元件名


@dataclass(frozen=True)
class AcPoint:
    frequency_hz: float
    transfer: complex     # 复数传递函数 H = 输出相量 / 参照源激励相量
    magnitude: float      # |H|，线性倍数
    magnitude_db: float   # 20·log10(|H|)
    phase_deg: float      # 相位，度，(−180, 180]


@dataclass(frozen=True)
class AcResult:
    input_source: str
    output: OutputSpec
    points: tuple[AcPoint, ...]


def _is_finite_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def make_log_sweep(
    start_hz: float | None,
    stop_hz: float | None,
    *,
    num_points: int | None = None,
    points_per_decade: float | None = None,
) -> list[float]:
    """生成对数频率轴上均匀分布的扫描频点（含两个端点，至少 2 个）。

    取点密度二选一：num_points（整段总点数）或 points_per_decade（每十倍频程点数）。
    """
    if not _is_finite_number(start_hz) or start_hz <= 0:
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            f"起始频率 start_hz 必须是正的有限数值，得到 {start_hz!r}",
        )
    if not _is_finite_number(stop_hz) or stop_hz <= 0:
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            f"终止频率 stop_hz 必须是正的有限数值，得到 {stop_hz!r}",
        )
    if stop_hz <= start_hz:
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            f"终止频率必须大于起始频率，得到 start_hz={start_hz}、stop_hz={stop_hz}",
        )
    if (num_points is None) == (points_per_decade is None):
        raise CircuitError(
            ErrorCode.INVALID_SWEEP_PARAMS,
            "num_points（整段总点数）与 points_per_decade（每十倍频程点数）必须且只能提供一个",
        )

    if num_points is not None:
        if isinstance(num_points, bool) or not isinstance(num_points, int) or num_points < 2:
            raise CircuitError(
                ErrorCode.INVALID_SWEEP_PARAMS,
                f"扫描点数 num_points 必须是不小于 2 的整数，得到 {num_points!r}",
            )
        n = num_points
    else:
        if not _is_finite_number(points_per_decade) or points_per_decade <= 0:
            raise CircuitError(
                ErrorCode.INVALID_SWEEP_PARAMS,
                f"每十倍频程点数 points_per_decade 必须是正的有限数值，得到 {points_per_decade!r}",
            )
        decades = math.log10(stop_hz / start_hz)
        n = max(2, int(round(points_per_decade * decades)) + 1)

    if n > MAX_SWEEP_POINTS:
        raise CircuitError(
            ErrorCode.TOO_MANY_POINTS,
            f"需要 {n} 个频点，超过上限 {MAX_SWEEP_POINTS}；请缩小频率范围或降低取点密度",
        )

    log_lo = math.log10(start_hz)
    step = (math.log10(stop_hz) - log_lo) / (n - 1)
    freqs = [10.0 ** (log_lo + i * step) for i in range(n)]
    freqs[0] = float(start_hz)   # 端点取用户给定值，避免 log 往返的末位误差
    freqs[-1] = float(stop_hz)
    return freqs


def _validate_frequencies(frequencies: Iterable[float]) -> list[float]:
    freqs = list(frequencies)
    if not freqs:
        raise CircuitError(ErrorCode.INVALID_SWEEP_PARAMS, "频率序列至少需要一个频点")
    for f in freqs:
        if not _is_finite_number(f) or f < 0:
            raise CircuitError(
                ErrorCode.INVALID_SWEEP_PARAMS,
                f"频点 {f!r} 非法：频率必须是非负的有限数值（0 按直流极限处理）",
            )
    return [float(f) for f in freqs]


def _find_element(circuit: Circuit, name: str):
    for group in (
        circuit.resistors,
        circuit.capacitors,
        circuit.inductors,
        circuit.voltage_sources,
        circuit.current_sources,
    ):
        for elem in group:
            if elem.name == name:
                return elem
    return None


def _element_current(elem, sol: MnaSolution, omega: float, input_source: str) -> complex:
    """支路电流相量，正方向与 netlist.py 的约定一致（a→b；电压源为流入正极）。"""
    if isinstance(elem, Resistor):
        return (sol.voltage(elem.a) - sol.voltage(elem.b)) / elem.resistance
    if isinstance(elem, Capacitor):
        return 1j * omega * elem.capacitance * (sol.voltage(elem.a) - sol.voltage(elem.b))
    if isinstance(elem, (Inductor, VoltageSource)):
        return complex(sol.branch_currents[elem.name])  # 扩维支路电流未知量
    if isinstance(elem, CurrentSource):
        # 电流源的支路电流即源本身：参照源为单位激励，其余已置零
        return 1.0 + 0.0j if elem.name == input_source else 0.0 + 0.0j
    raise AssertionError(f"未知元件类型: {elem!r}")


def run_ac(
    circuit: Circuit,
    frequencies: Iterable[float],
    output: OutputSpec,
    input_source: str,
) -> AcResult:
    """逐频点求解复数 MNA，返回输出相对参照源的传递函数序列。

    频点允许取 0（直流极限）；矩阵奇异时抛 CircuitError(MATRIX_SINGULAR) 并注明频率。
    """
    freqs = _validate_frequencies(frequencies)

    source_names = {vs.name for vs in circuit.voltage_sources} | {
        cs.name for cs in circuit.current_sources
    }
    if not isinstance(input_source, str) or input_source not in source_names:
        raise CircuitError(
            ErrorCode.INPUT_SOURCE_NOT_FOUND,
            f"参照输入源 {input_source!r} 找不到：网表中的独立源有 "
            f"{sorted(source_names) if source_names else '（无）'}",
        )

    element = None
    if output.kind == OUTPUT_KIND_VOLTAGE:
        if not output.node:
            raise CircuitError(
                ErrorCode.INVALID_OUTPUT_SPEC,
                "输出类型为 voltage 时必须给出输出节点名 node",
            )
        if output.node not in (*circuit.nodes, circuit.ground):
            raise CircuitError(
                ErrorCode.OUTPUT_NOT_FOUND,
                f"输出节点 {output.node!r} 在网表中不存在；"
                f"可用节点：{[*circuit.nodes, circuit.ground]}",
            )
    elif output.kind == OUTPUT_KIND_CURRENT:
        if not output.element:
            raise CircuitError(
                ErrorCode.INVALID_OUTPUT_SPEC,
                "输出类型为 current 时必须给出支路元件名 element",
            )
        element = _find_element(circuit, output.element)
        if element is None:
            raise CircuitError(
                ErrorCode.OUTPUT_NOT_FOUND,
                f"输出支路元件 {output.element!r} 在网表中不存在",
            )
    else:
        raise CircuitError(
            ErrorCode.INVALID_OUTPUT_SPEC,
            f"输出类型 {output.kind!r} 无法识别；"
            f"支持 \"{OUTPUT_KIND_VOLTAGE}\"（节点电压）或 \"{OUTPUT_KIND_CURRENT}\"（支路电流）",
        )

    points: list[AcPoint] = []
    for f in freqs:
        omega = 2.0 * math.pi * f
        builder = MnaBuilder(circuit.nodes, circuit.ground, complex_mode=True)
        for r in circuit.resistors:
            builder.add_conductance(r.a, r.b, 1.0 / r.resistance)
        for c in circuit.capacitors:  # 导纳 jωC；ω=0 时为零，即开路（直流极限）
            builder.add_conductance(c.a, c.b, 1j * omega * c.capacitance)
        for ind in circuit.inductors:  # 阻抗 jωL；ω=0 时退化为 0V 电压源，即短路
            builder.add_impedance(ind.name, ind.a, ind.b, 1j * omega * ind.inductance)
        for vs in circuit.voltage_sources:  # 参照源单位激励，其余置零（短路）
            builder.add_voltage_source(
                vs.name, vs.a, vs.b, 1.0 if vs.name == input_source else 0.0
            )
        for cs in circuit.current_sources:  # 参照源单位激励，其余置零（开路）
            builder.add_current_source(
                cs.a, cs.b, 1.0 if cs.name == input_source else 0.0
            )
        try:
            sol = builder.solve()
        except CircuitError as exc:
            if exc.code == ErrorCode.MATRIX_SINGULAR:
                raise CircuitError(
                    ErrorCode.MATRIX_SINGULAR,
                    f"频点 f = {f:g} Hz 处复数系数矩阵奇异，无法求解：常见原因包括独立源"
                    "置零后出现悬浮节点、电感围成回路（ω=0 时）、或纯电抗回路在该频率谐振",
                ) from exc
            raise

        if output.kind == OUTPUT_KIND_VOLTAGE:
            out = complex(sol.voltage(output.node))
        else:
            out = _element_current(element, sol, omega, input_source)
        # 参照源激励相量恒为 1∠0°，输出相量即传递函数
        mag = abs(out)
        points.append(
            AcPoint(
                frequency_hz=f,
                transfer=out,
                magnitude=mag,
                magnitude_db=20.0 * math.log10(max(mag, _DB_FLOOR)),
                phase_deg=math.degrees(math.atan2(out.imag, out.real)),
            )
        )

    return AcResult(input_source=input_source, output=output, points=tuple(points))
