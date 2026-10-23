#!/usr/bin/env bash
# TimesFM 预测服务管理脚本: start | stop | restart | status | logs
set -euo pipefail

cd "$(dirname "$0")"

VENV_DIR=".venv"
PID_FILE=".server.pid"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/server.log"

# 端口/地址等配置通过环境变量透传给服务,例如:
#   TIMESFM_PORT=8001 ./server.sh start
HOST="${TIMESFM_HOST:-0.0.0.0}"
PORT="${TIMESFM_PORT:-8000}"

mkdir -p "$LOG_DIR"

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

wait_ready() {
    # 最多等 120 秒,直到 /api/health 返回 ok 或 error
    for _ in $(seq 1 60); do
        resp="$(curl -s -m 2 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null || true)"
        if echo "$resp" | grep -q '"status": *"ok"'; then
            echo "$resp"
            return 0
        fi
        if echo "$resp" | grep -q '"status": *"error"'; then
            echo "$resp"
            return 1
        fi
        sleep 2
    done
    echo "等待超时,请查看日志: $LOG_FILE"
    return 1
}

do_start() {
    if is_running; then
        echo "服务已在运行 (PID $(cat "$PID_FILE"))"
        return 0
    fi
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "虚拟环境不存在,请先执行: ./install.sh"
        exit 1
    fi
    echo "==> 启动服务 (http://${HOST}:${PORT}) ..."
    nohup "$VENV_DIR/bin/python" -m server.main >>"$LOG_FILE" 2>&1 &
    echo $! >"$PID_FILE"
    echo "==> PID $(cat "$PID_FILE"),日志: $LOG_FILE"
    echo "==> 等待模型加载..."
    if wait_ready; then
        echo "==> 服务就绪: http://127.0.0.1:${PORT}/docs"
    else
        echo "==> 服务启动异常,请查看日志: $LOG_FILE"
        exit 1
    fi
}

do_stop() {
    if ! is_running; then
        echo "服务未在运行"
        rm -f "$PID_FILE"
        return 0
    fi
    pid="$(cat "$PID_FILE")"
    echo "==> 停止服务 (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "==> 已停止"
}

case "${1:-start}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)
        if is_running; then
            echo "运行中 (PID $(cat "$PID_FILE"),端口 $PORT)"
            curl -s -m 2 "http://127.0.0.1:${PORT}/api/health" || echo "(health 接口无响应)"
            echo
        else
            echo "未运行"
            exit 1
        fi
        ;;
    logs)    tail -n "${2:-50}" -f "$LOG_FILE" ;;
    *)       echo "用法: $0 {start|stop|restart|status|logs [行数]}"; exit 1 ;;
esac
