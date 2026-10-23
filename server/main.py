"""TimesFM 3.0 时间序列预测服务入口。

启动方式(项目根目录):
    .venv/bin/python -m server.main
    # 或
    .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server import config
from server.api import forecast as forecast_api
from server.api import system as system_api
from server.core.engine import TimesFMEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时在线程池中预加载模型(阻塞操作不卡事件循环),失败信息由 /api/health 暴露。"""
    engine = TimesFMEngine.instance()
    logger.info("开始加载 TimesFM 模型: %s", config.MODEL_PATH)
    asyncio.create_task(asyncio.to_thread(engine.load))
    yield
    # 关闭时无需特殊清理,torch 自行管理显存


app = FastAPI(
    title="TimesFM 3.0 时间序列预测服务",
    description=(
        "Google TimesFM 3.0 零样本时间序列预测 API:"
        "批量单序列预测、多变量联合预测、分位数区间输出。"
        "模型许可证为 TimesFM Non-Commercial v1.0,仅供测试/研究/评估使用。"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_api.router)
app.include_router(forecast_api.router)


@app.get("/")
def root():
    return {
        "service": "timesfm-forecast-api",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=config.HOST, port=config.PORT)
