"""HTTP 请求/响应的数据结构。

这里只描述接口形状；领域校验（元件类型、取值、接地、时间参数等）
全部在 netlist.py / transient.py 中以带错误码的 CircuitError 完成。
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ElementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    type: str
    nodes: Optional[list[str]] = None
    value: Optional[float] = None
    initial: Optional[float] = None


class NetlistSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    elements: list[ElementSpec]


class DcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netlist: NetlistSpec


class TransientRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netlist: NetlistSpec
    dt: Optional[float] = None
    tstop: Optional[float] = None


class AcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netlist: NetlistSpec
    f_start: Optional[float] = None            # 起始频率，Hz（必须为正）
    f_stop: Optional[float] = None             # 终止频率，Hz（必须大于 f_start）
    points_per_decade: Optional[int] = None    # 每十倍频程点数（与 num_points 二选一）
    num_points: Optional[int] = None           # 整段总点数（与 points_per_decade 二选一）
    output_node: Optional[str] = None          # 输出：节点对地电压（与 output_branch 二选一）
    output_branch: Optional[str] = None        # 输出：电压源支路电流
    input_source: Optional[str] = None         # 参照输入源（独立电压源/电流源名）


class DcResponse(BaseModel):
    success: bool
    node_voltages: dict[str, float]
    voltage_source_currents: dict[str, float]
    inductor_currents: dict[str, float]


class TransientResponse(BaseModel):
    success: bool
    integration: str
    dt: float
    tstop: float
    steps: int
    time: list[float]
    node_voltages: dict[str, list[float]]
    voltage_source_currents: dict[str, list[float]]
    inductor_currents: dict[str, list[float]]


class AcPointResponse(BaseModel):
    frequency: float      # 赫兹
    transfer_real: float  # 复数传递函数实部
    transfer_imag: float  # 复数传递函数虚部
    magnitude: float      # 线性幅度比
    magnitude_db: float   # 20·log10(magnitude)
    phase_deg: float      # 相位，度


class AcResponse(BaseModel):
    success: bool
    input_source: str             # 参照输入源名
    output: str                   # 输出描述，形如 "v(out)" / "i(V1)"
    points: list[AcPointResponse]
