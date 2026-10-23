#!/usr/bin/env bash
# TimesFM 预测服务依赖安装脚本(可重复执行,已装过会跳过)
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON_VERSION="3.12"

# ---- 检查 uv,没有则回退到 python -m venv + pip ----
UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
fi

if [ -n "$UV_BIN" ]; then
    echo "==> 使用 uv: $UV_BIN"
    if [ ! -d "$VENV_DIR" ]; then
        echo "==> 创建虚拟环境 $VENV_DIR (Python $PYTHON_VERSION)"
        "$UV_BIN" venv "$VENV_DIR" --python "$PYTHON_VERSION"
    else
        echo "==> 虚拟环境已存在,跳过创建"
    fi
    echo "==> 安装依赖 (首次含 torch CUDA 版,约 3~5 分钟)..."
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" -r requirements.txt
else
    echo "==> 未找到 uv,回退到 python3 -m venv + pip"
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r requirements.txt
fi

# ---- 验证关键依赖可导入 + CUDA 可用性 ----
echo "==> 验证安装结果..."
"$VENV_DIR/bin/python" - <<'EOF'
import torch
print(f"torch {torch.__version__}  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
from timesfm3 import TimesFM3Forecaster  # noqa: F401
import fastapi, uvicorn  # noqa: F401
print("timesfm3 / fastapi / uvicorn 导入正常")
EOF

echo "==> 安装完成。启动服务: ./server.sh start"
