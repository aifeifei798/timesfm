"""TimesFM 3.0 推理引擎:单例持有模型,加锁串行推理。

加载流程(corpus-forge 模式):
- lifespan 里通过 asyncio.to_thread 在线程池中调用 load(),不阻塞事件循环
- 加载失败时记录 load_error,不抛异常,由 /api/health 暴露状态
"""

import json
import logging
import os
import threading
import time

# 国内镜像兜底(本地目录加载不走网络,仅保险起见,须在导入 timesfm 前设置)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np

from server import config

logger = logging.getLogger(__name__)


class TimesFMEngine:
    """TimesFM 3.0 推理引擎(单例)。"""

    _singleton_lock = threading.Lock()
    _instance = None

    def __init__(self) -> None:
        # 推理锁:模型非线程安全,所有预测串行执行
        self._infer_lock = threading.Lock()
        self.forecaster = None
        self.device: str | None = None
        self.quantiles: list[float] | None = None
        self.loaded = False
        self.load_error: str | None = None

    @classmethod
    def instance(cls) -> "TimesFMEngine":
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def ready(self) -> bool:
        return self.loaded and self.forecaster is not None

    # ------------------------------------------------------------------ 加载

    def load(self) -> None:
        """加载模型到指定设备(阻塞式,应放到线程中调用)。"""
        if self.ready:
            return
        t0 = time.time()
        try:
            from timesfm3 import ModelConfig, TimesFM3Forecaster

            device = config.DEVICE
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

            forecaster = TimesFM3Forecaster(
                ModelConfig(
                    checkpoint_path=config.MODEL_PATH,
                    device=device,
                    per_core_batch_size=config.PER_CORE_BATCH_SIZE,
                )
            )
            self.forecaster = forecaster
            self.device = device
            self.quantiles = self._resolve_quantiles(forecaster)
            self._warmup()
            self.loaded = True
            logger.info(
                "TimesFM 模型加载完成 path=%s device=%s 耗时 %.1fs",
                config.MODEL_PATH,
                device,
                time.time() - t0,
            )
        except Exception as exc:  # noqa: BLE001 - 加载失败要兜底暴露给 /api/health
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("TimesFM 模型加载失败")

    def _resolve_quantiles(self, forecaster) -> list[float]:
        """分位数列表:优先取 forecaster 同步出的配置,兜底读模型目录的 config.json。"""
        q = getattr(forecaster, "quantiles", None)
        if q:
            return [float(x) for x in q]
        cfg_path = os.path.join(config.MODEL_PATH, "config.json")
        try:
            with open(cfg_path, encoding="utf-8") as f:
                return [float(x) for x in json.load(f)["quantiles"]]
        except Exception:  # noqa: BLE001
            return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def _warmup(self) -> None:
        """跑一次最小预测,预热 CUDA kernel / 触发首次编译。"""
        dummy = np.zeros(64, dtype=np.float32)
        try:
            with self._infer_lock:
                self.forecaster.predict(dummy, horizon=8)
        except Exception:  # noqa: BLE001 - warmup 失败不影响服务
            logger.warning("warmup 预测失败(忽略)", exc_info=True)

    # ------------------------------------------------------------------ 推理

    def predict_batch(self, contexts: list[np.ndarray], horizon: int, **kwargs) -> list:
        """批量预测一组独立单变量序列,返回 ForecastOutput 列表。"""
        if not self.ready:
            raise RuntimeError("模型尚未加载")
        with self._infer_lock:
            return list(self.forecaster.predict_batch(contexts, horizon, **kwargs))

    def predict_multivariate(
        self,
        targets: np.ndarray,
        horizon: int,
        past_only_covariates: np.ndarray | None,
        past_future_covariates: np.ndarray | None,
        **kwargs,
    ):
        """多变量联合预测:targets 形状 (变量数, 上下文长度)。"""
        if not self.ready:
            raise RuntimeError("模型尚未加载")
        with self._infer_lock:
            return self.forecaster.predict(
                targets,
                horizon,
                past_only_covariates=past_only_covariates,
                past_future_covariates=past_future_covariates,
                **kwargs,
            )
