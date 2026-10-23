"""预测路由:批量单序列预测 + 多变量联合预测。"""

import math
import time

import numpy as np
from fastapi import APIRouter, HTTPException

from server import config
from server.api import schemas
from server.core.engine import TimesFMEngine

router = APIRouter(prefix="/api", tags=["forecast"])


# ---------------------------------------------------------------------- 工具


def _ensure_ready(engine: TimesFMEngine) -> None:
    if not engine.ready:
        detail = engine.load_error or "模型正在加载中,请稍后重试"
        raise HTTPException(status_code=503, detail=detail)


def _check_quantile_levels(
    levels: list[float] | None, model_quantiles: list[float]
) -> list[float] | None:
    """校验请求的分位数子集必须是模型支持的档位。"""
    if levels is None:
        return None
    for lv in levels:
        if not any(abs(lv - q) < 1e-6 for q in model_quantiles):
            raise HTTPException(
                status_code=400,
                detail=f"分位数 {lv} 不受支持,可用档位:{model_quantiles}",
            )
    return levels


def _fmt_quantile_key(q: float) -> str:
    return f"{q:g}"


def _sanitize(values) -> list[float | None]:
    """非有限值(NaN/Inf)转成 null,保证响应是合法 JSON。"""
    return [float(v) if math.isfinite(v) else None for v in values]


def _quantile_dict(
    quantiles: np.ndarray | None,
    model_quantiles: list[float],
    requested: list[float] | None,
) -> dict[str, list[float]] | None:
    """把 (horizon, n_quantiles) 数组转成 {分位水平: 曲线} 的 dict,可按子集过滤。"""
    if quantiles is None:
        return None
    arr = np.asarray(quantiles)
    out: dict[str, list[float]] = {}
    for i, q in enumerate(model_quantiles):
        if requested is not None and not any(abs(q - lv) < 1e-6 for lv in requested):
            continue
        out[_fmt_quantile_key(q)] = _sanitize(arr[:, i])
    return out


def _as_matrix(rows: list[list[float]] | None, name: str) -> np.ndarray | None:
    if rows is None:
        return None
    lengths = {len(r) for r in rows}
    if len(lengths) > 1:
        raise HTTPException(
            status_code=400, detail=f"{name} 各行长度必须一致,当前为 {sorted(lengths)}"
        )
    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------- 路由


@router.post("/forecast", response_model=schemas.ForecastResponse)
def forecast(req: schemas.ForecastRequest):
    """对一组相互独立的时间序列做零样本预测(内部批量推理)。"""
    engine = TimesFMEngine.instance()
    _ensure_ready(engine)

    if req.ts_ids is not None and len(req.ts_ids) != len(req.series):
        raise HTTPException(
            status_code=400,
            detail=f"ts_ids 数量({len(req.ts_ids)})与 series 数量({len(req.series)})不一致",
        )
    requested = _check_quantile_levels(req.quantile_levels, engine.quantiles)

    # 超长上下文截断,只保留最近的数据点(与模型内部行为一致,减小请求体)
    contexts = [
        np.asarray(s[-config.MAX_CONTEXT_LENGTH:], dtype=np.float32)
        for s in req.series
    ]
    ts_ids = req.ts_ids or [f"series_{i}" for i in range(len(req.series))]

    t0 = time.perf_counter()
    try:
        outputs = engine.predict_batch(
            contexts,
            req.horizon,
            ts_ids=ts_ids,
            return_quantiles=req.return_quantiles,
            make_positive=req.make_positive,
        )
    except Exception as exc:  # noqa: BLE001 - 推理异常统一转 500
        raise HTTPException(status_code=500, detail=f"推理失败:{type(exc).__name__}: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for out in outputs:
        results.append(
            schemas.SeriesForecast(
                ts_id=out.ts_id,
                point_forecast=_sanitize(np.asarray(out.forecast).ravel()),
                quantiles=_quantile_dict(
                    out.quantiles, engine.quantiles, requested
                ),
            )
        )

    return schemas.ForecastResponse(
        results=results,
        horizon=req.horizon,
        device=engine.device,
        elapsed_ms=round(elapsed_ms, 2),
    )


@router.post("/forecast/multivariate", response_model=schemas.MultivariateForecastResponse)
def forecast_multivariate(req: schemas.MultivariateForecastRequest):
    """多变量联合预测:目标变量 + 可选协变量一起送入模型(variate attention)。"""
    engine = TimesFMEngine.instance()
    _ensure_ready(engine)

    context_len = len(req.targets[0])
    n_channels = len(req.targets)
    if req.past_only_covariates:
        n_channels += len(req.past_only_covariates)
    if req.past_future_covariates:
        n_channels += len(req.past_future_covariates)
    if n_channels > config.MAX_VARIATES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"总通道数 {n_channels}(目标变量 + 协变量)超过模型上限 "
                f"{config.MAX_VARIATES},请减少变量或分批请求"
            ),
        )

    targets = np.asarray(req.targets, dtype=np.float32)
    past_only = _as_matrix(req.past_only_covariates, "past_only_covariates")
    past_future = _as_matrix(req.past_future_covariates, "past_future_covariates")

    if past_only is not None and past_only.shape[1] != context_len:
        raise HTTPException(
            status_code=400,
            detail=f"past_only_covariates 长度({past_only.shape[1]})须与 targets 上下文长度({context_len})一致",
        )
    if past_future is not None and past_future.shape[1] != context_len + req.horizon:
        raise HTTPException(
            status_code=400,
            detail=(
                f"past_future_covariates 长度({past_future.shape[1]})须为 "
                f"上下文长度 + horizon({context_len} + {req.horizon})"
            ),
        )

    t0 = time.perf_counter()
    try:
        out = engine.predict_multivariate(
            targets,
            req.horizon,
            past_only_covariates=past_only,
            past_future_covariates=past_future,
            return_quantiles=req.return_quantiles,
            make_positive=req.make_positive,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"推理失败:{type(exc).__name__}: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    forecast_arr = np.asarray(out.forecast)
    if forecast_arr.ndim == 1:  # 单变量时统一成 (1, horizon)
        forecast_arr = forecast_arr[None, :]
    quantiles_arr = None
    if out.quantiles is not None:
        quantiles_arr = np.asarray(out.quantiles)
        if quantiles_arr.ndim == 2:
            quantiles_arr = quantiles_arr[None, :, :]

    results = []
    for i in range(forecast_arr.shape[0]):
        results.append(
            schemas.SeriesForecast(
                ts_id=f"variate_{i}",
                point_forecast=_sanitize(forecast_arr[i]),
                quantiles=_quantile_dict(
                    None if quantiles_arr is None else quantiles_arr[i],
                    engine.quantiles,
                    None,
                ),
            )
        )

    return schemas.MultivariateForecastResponse(
        results=results,
        horizon=req.horizon,
        device=engine.device,
        elapsed_ms=round(elapsed_ms, 2),
    )
