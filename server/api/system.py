"""系统路由:健康检查与模型信息。"""

import json
import os

from fastapi import APIRouter

from server import config
from server.core.engine import TimesFMEngine

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    """服务健康检查:模型是否就绪、加载错误信息。"""
    engine = TimesFMEngine.instance()
    if engine.ready:
        status = "ok"
    elif engine.load_error:
        status = "error"
    else:
        status = "loading"
    return {
        "status": status,
        "model_ready": engine.ready,
        "device": engine.device,
        "error": engine.load_error,
    }


@router.get("/model-info")
def model_info():
    """模型元信息:patch 长度、分位数档位、上下文/变量上限等。"""
    engine = TimesFMEngine.instance()
    forecaster = engine.forecaster
    info = {
        "model_path": config.MODEL_PATH,
        "model_ready": engine.ready,
        "device": engine.device,
        "quantiles": engine.quantiles,
        "max_context_length": config.MAX_CONTEXT_LENGTH,
        "max_variates": config.MAX_VARIATES,
        "per_core_batch_size": config.PER_CORE_BATCH_SIZE,
    }
    # patch 长度等架构参数以模型目录 config.json 为准
    try:
        with open(os.path.join(config.MODEL_PATH, "config.json"), encoding="utf-8") as f:
            model_cfg = json.load(f)
        info["input_patch_len"] = model_cfg.get("input_patch_len")
        info["output_patch_len"] = model_cfg.get("output_patch_len")
    except Exception:  # noqa: BLE001 - 读不到不影响主流程
        pass
    if forecaster is not None:
        # 从已加载的 forecaster 同步出的配置里取真实值,取不到就跳过
        for key in ("input_patch_length", "output_patch_length"):
            value = getattr(forecaster, key, None)
            if value is not None and key not in info:
                info[key] = value
    return info
