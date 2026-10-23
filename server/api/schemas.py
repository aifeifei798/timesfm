"""API 请求/响应模型。"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ForecastRequest(BaseModel):
    """批量单序列预测请求:各序列相互独立,内部走 predict_batch。"""

    series: list[list[float]] = Field(
        ...,
        description="时间序列列表,每条为一维数值数组;传单条序列时也允许直接给一维数组",
        examples=[[[1.0, 2.2, 3.1, 4.5, 5.0], [10.0, 9.5, 9.8, 10.2, 11.0]]],
    )
    horizon: int = Field(24, ge=1, description="预测步长")
    ts_ids: Optional[list[str]] = Field(None, description="各序列的可选标识,缺省用 series_0/1/...")
    return_quantiles: bool = Field(True, description="是否返回分位数曲线")
    quantile_levels: Optional[list[float]] = Field(
        None,
        description="需要返回的分位数子集,如 [0.1, 0.5, 0.9];缺省返回全部 9 档(0.1~0.9)",
    )
    make_positive: bool = Field(False, description="上下文全为非负时,将预测裁剪为非负(适合销量等)")

    @field_validator("series", mode="before")
    @classmethod
    def _wrap_flat_series(cls, v):
        """允许直接传一条扁平序列(一维数字数组),自动包一层。"""
        if (
            isinstance(v, list)
            and v
            and all(isinstance(x, (int, float)) for x in v)
        ):
            return [v]
        return v

    @field_validator("series")
    @classmethod
    def _check_series(cls, v):
        if not v:
            raise ValueError("series 不能为空")
        return v


class SeriesForecast(BaseModel):
    """单条序列的预测结果。"""

    ts_id: str
    point_forecast: list[float] = Field(..., description="中位数(P50)点预测")
    quantiles: Optional[dict[str, list[float]]] = Field(
        None, description='分位数曲线,key 为分位水平字符串(如 "0.1"),value 为长度=horizon 的数组'
    )


class ForecastResponse(BaseModel):
    results: list[SeriesForecast]
    horizon: int
    device: str
    elapsed_ms: float


class MultivariateForecastRequest(BaseModel):
    """多变量联合预测请求:目标变量 + 可选协变量,利用 variate attention 建模变量间关联。"""

    targets: list[list[float]] = Field(
        ...,
        description="2D 矩阵:每行一个变量,行长度一致(共享上下文),最多 32 行",
        examples=[[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]],
    )
    horizon: int = Field(24, ge=1, description="预测步长")
    past_only_covariates: Optional[list[list[float]]] = Field(
        None, description="仅历史协变量,形状 (n, 上下文长度),与 targets 行长一致"
    )
    past_future_covariates: Optional[list[list[float]]] = Field(
        None, description="贯穿历史与未来的协变量,形状 (n, 上下文长度 + horizon),如节假日标记、日期特征"
    )
    return_quantiles: bool = Field(True, description="是否返回分位数曲线")
    make_positive: bool = Field(False, description="上下文全为非负时,将预测裁剪为非负")

    @field_validator("targets")
    @classmethod
    def _check_targets(cls, v):
        if not v:
            raise ValueError("targets 不能为空")
        lengths = {len(row) for row in v}
        if len(lengths) > 1:
            raise ValueError(f"targets 各行长度必须一致,当前为 {sorted(lengths)}")
        return v


class MultivariateForecastResponse(BaseModel):
    results: list[SeriesForecast] = Field(..., description="每个变量一条结果,ts_id 为 variate_0/1/...")
    horizon: int
    device: str
    elapsed_ms: float
