"""TimesFM 预测服务调用示例(仅用标准库,无需额外依赖)。

先启动服务:
    .venv/bin/python -m server.main

再运行:
    .venv/bin/python examples/demo.py
"""

import json
import math
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def call_api(path: str, payload: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def show(title: str, resp: dict) -> None:
    print(f"\n===== {title} =====")
    print(json.dumps(resp, ensure_ascii=False, indent=2)[:1500])


def main() -> None:
    # 1. 健康检查
    show("健康检查 /api/health", call_api("/api/health"))
    show("模型信息 /api/model-info", call_api("/api/model-info"))

    # 2. 批量单序列预测:一条正弦波 + 一条带趋势的序列
    n = 200
    sine = [math.sin(i / 8.0) * 10 + 50 for i in range(n)]
    trend = [100 + i * 0.8 + math.sin(i / 5.0) * 5 for i in range(n)]
    show(
        "批量单序列预测 /api/forecast (horizon=24)",
        call_api(
            "/api/forecast",
            {
                "series": [sine, trend],
                "horizon": 24,
                "return_quantiles": True,
                "quantile_levels": [0.1, 0.5, 0.9],
            },
        ),
    )

    # 3. 多变量联合预测:两个相关变量(线性关系) + 一个未来协变量
    targets = [
        [100 + i for i in range(n)],            # 变量 0:线性上升
        [2 * (100 + i) + 10 for i in range(n)],  # 变量 1:约为变量 0 的 2 倍
    ]
    past_future_cov = [[float(i % 7 == 5) for i in range(n + 24)]]  # 未来 24 步已知,如周末标记
    show(
        "多变量联合预测 /api/forecast/multivariate (horizon=24)",
        call_api(
            "/api/forecast/multivariate",
            {
                "targets": targets,
                "horizon": 24,
                "past_future_covariates": past_future_cov,
                "return_quantiles": True,
            },
        ),
    )


if __name__ == "__main__":
    main()
