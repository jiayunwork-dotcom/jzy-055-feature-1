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


class SweepSpec(BaseModel):
    """频率扫描范围：num_points（整段总点数）与 points_per_decade（每十倍频程点数）二选一。"""

    model_config = ConfigDict(extra="forbid")

    start_hz: Optional[float] = None
    stop_hz: Optional[float] = None
    num_points: Optional[int] = None
    points_per_decade: Optional[float] = None


class OutputSelector(BaseModel):
    """看哪里：kind=voltage 时给节点名 node；kind=current 时给支路元件名 element。"""

    model_config = ConfigDict(extra="forbid")

    kind: Optional[str] = None
    node: Optional[str] = None
    element: Optional[str] = None


class AcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    netlist: NetlistSpec
    sweep: SweepSpec
    output: OutputSelector
    input_source: str = Field(min_length=1)


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
    frequency_hz: float
    real: float          # 传递函数实部
    imag: float          # 传递函数虚部
    magnitude: float     # 幅度，线性倍数
    magnitude_db: float  # 幅度，分贝（20·log10 幅度比）
    phase_deg: float     # 相位，度


class AcResponse(BaseModel):
    success: bool
    input_source: str
    output: OutputSelector
    points: list[AcPointResponse]
