"""服务配置:全部从环境变量读取,提供合理默认值。"""

import os

# 模型本地路径(必须是目录,内含 config.json + model.safetensors)
# 注意:不能直接指向裸 model.safetensors 文件,否则会用代码默认架构
# (use_rope_var=True)加载,与该 checkpoint 的 config.json 不一致导致权重错乱。
MODEL_PATH = os.environ.get(
    "TIMESFM_MODEL_PATH",
    "/home/feifei/develop/models/timesfm-3.0-pytorch",
)

# 推理设备:auto / cuda / cpu
DEVICE = os.environ.get("TIMESFM_DEVICE", "auto")

# 服务监听地址与端口
HOST = os.environ.get("TIMESFM_HOST", "0.0.0.0")
PORT = int(os.environ.get("TIMESFM_PORT", "8000"))

# 单次前向批大小(显存占用与吞吐的权衡)
PER_CORE_BATCH_SIZE = int(os.environ.get("TIMESFM_BATCH_SIZE", "8"))

# CORS 允许来源,逗号分隔
CORS_ORIGINS = os.environ.get(
    "TIMESFM_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")

# ---- 模型硬约束(来自 timesfm3 源码与 checkpoint config.json) ----
# 上下文最大长度,超出部分模型内部会截断保留最近的数据
MAX_CONTEXT_LENGTH = 15360
# 单次前向最大通道数(目标变量 + 协变量之和),Forecaster 不做自动分块
MAX_VARIATES = 32
