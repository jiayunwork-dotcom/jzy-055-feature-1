"""HTTP 接口层：只做请求转发与结果封装。

解析、矩阵组装、时间步进、频率扫描等全部计算逻辑都在
netlist / mna / dc / transient / ac 模块中，本层不包含任何电学计算。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .ac import run_ac
from .dc import solve_dc
from .errors import CircuitError, ErrorCode
from .examples import load_examples
from .netlist import parse_netlist
from .schemas import (
    AcPointResponse,
    AcRequest,
    AcResponse,
    DcRequest,
    DcResponse,
    TransientRequest,
    TransientResponse,
)
from .transient import INTEGRATION_METHOD, run_transient

logger = logging.getLogger("circuit_sim")

app = FastAPI(
    title="电路仿真服务",
    version="1.1.0",
    summary="网表进、曲线出：直流工作点 + 后向欧拉瞬态 + 交流小信号频率响应",
)


@app.exception_handler(CircuitError)
async def circuit_error_handler(_: Request, exc: CircuitError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"code": exc.code, "message": exc.message})


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []))
    msg = f"请求格式不合法：{loc} {first.get('msg', '')}".strip()
    return JSONResponse(
        status_code=400,
        content={"code": ErrorCode.MALFORMED_REQUEST, "message": msg},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("未预期的错误: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"code": "INTERNAL_ERROR", "message": "服务内部错误"},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/examples")
async def examples() -> dict:
    """返回两个可手算核对的算例（分压网络 + RC 充电），可直接 POST 给计算接口。"""
    return load_examples()


@app.post("/api/dc", response_model=DcResponse)
async def dc_operating_point(req: DcRequest) -> DcResponse:
    """直流工作点：各节点电位、各电压源支路电流、各电感电流。"""
    circuit = parse_netlist([e.model_dump() for e in req.netlist.elements])
    result = solve_dc(circuit)
    return DcResponse(
        success=True,
        node_voltages=result.node_voltages,
        voltage_source_currents=result.voltage_source_currents,
        inductor_currents=result.inductor_currents,
    )


@app.post("/api/transient", response_model=TransientResponse)
async def transient(req: TransientRequest) -> TransientResponse:
    """瞬态分析：从 t=0 按定步长 dt 推进到 tstop，返回各节点电压时间序列。"""
    circuit = parse_netlist([e.model_dump() for e in req.netlist.elements])
    result = run_transient(circuit, req.dt, req.tstop)
    return TransientResponse(
        success=True,
        integration=INTEGRATION_METHOD,
        dt=result.dt,
        tstop=result.tstop,
        steps=len(result.time) - 1,
        time=result.time,
        node_voltages=result.node_voltages,
        voltage_source_currents=result.voltage_source_currents,
        inductor_currents=result.inductor_currents,
    )


@app.post("/api/ac", response_model=AcResponse)
async def ac_frequency_response(req: AcRequest) -> AcResponse:
    """交流小信号频率响应：在对数频率轴上扫描，给出输出相对参照源的传递函数。"""
    circuit = parse_netlist([e.model_dump() for e in req.netlist.elements])
    result = run_ac(
        circuit,
        f_start=req.f_start,
        f_stop=req.f_stop,
        points_per_decade=req.points_per_decade,
        num_points=req.num_points,
        output_node=req.output_node,
        output_branch=req.output_branch,
        input_source=req.input_source,
    )
    return AcResponse(
        success=True,
        input_source=result.input_source,
        output=result.output,
        points=[
            AcPointResponse(
                frequency=p.frequency,
                transfer_real=p.transfer.real,
                transfer_imag=p.transfer.imag,
                magnitude=p.magnitude,
                magnitude_db=p.magnitude_db,
                phase_deg=p.phase_deg,
            )
            for p in result.points
        ],
    )
