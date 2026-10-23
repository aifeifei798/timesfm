/* TimesFM 预测控制台前端逻辑 */

const PALETTE = ["#818cf8", "#34d399", "#fbbf24", "#f472b6", "#60a5fa", "#f87171", "#a78bfa", "#2dd4bf"];
const MAX_DISPLAY_POINTS = 300; // 图表最多展示的历史点数
let chart = null;
let lastMode = "univariate";
let lastResults = []; // 供 CSV 导出

// ---------------------------------------------------------------- 工具

const $ = (id) => document.getElementById(id);

function showToast(message, isError = true) {
  const toast = $("toast");
  toast.textContent = message;
  toast.className =
    "pointer-events-none fixed bottom-6 left-1/2 z-50 -translate-x-1/2 rounded-xl border px-4 py-2.5 text-sm shadow-lg " +
    (isError
      ? "border-rose-500/40 bg-rose-950/90 text-rose-200"
      : "border-emerald-500/40 bg-emerald-950/90 text-emerald-200");
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

/** 解析多行文本为数值数组:每行一条序列,行内逗号/空格/分号分隔 */
function parseLines(text) {
  const series = [];
  for (const line of text.split("\n")) {
    const nums = line
      .trim()
      .split(/[,;\s]+/)
      .filter((s) => s.length > 0)
      .map(Number);
    if (nums.length > 0 && nums.every((n) => Number.isFinite(n))) {
      series.push(nums);
    }
  }
  return series;
}

async function postJson(path, payload) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `请求失败 (${resp.status})`);
  return data;
}

// ---------------------------------------------------------------- Tab 切换

function switchTab(mode) {
  lastMode = mode;
  const uni = mode === "univariate";
  $("panel-univariate").classList.toggle("hidden", !uni);
  $("panel-multivariate").classList.toggle("hidden", uni);
  const activeCls = ["tab-active"];
  $("tab-univariate").classList.toggle(...activeCls, uni);
  $("tab-univariate").classList.toggle("text-slate-400", !uni);
  $("tab-multivariate").classList.toggle(...activeCls, !uni);
  $("tab-multivariate").classList.toggle("text-slate-400", uni);
}

// ---------------------------------------------------------------- 示例数据

function fillSample() {
  const n = 200;
  const sine = Array.from({ length: n }, (_, i) => +(Math.sin(i / 8) * 10 + 50).toFixed(2));
  const trend = Array.from({ length: n }, (_, i) => +(100 + i * 0.8 + Math.sin(i / 5) * 5).toFixed(2));
  if (lastMode === "univariate") {
    $("input-series").value = [sine.join(", "), trend.join(", ")].join("\n");
  } else {
    $("input-targets").value = [sine.join(", "), trend.map((v) => +(v * 2).toFixed(2)).join(", ")].join("\n");
    $("input-covariates").value = Array.from({ length: n + 24 }, (_, i) => (i % 7 === 5 ? 1 : 0)).join(",");
  }
}

// ---------------------------------------------------------------- 健康与模型信息

async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const dot = $("health-dot");
    const text = $("health-text");
    if (h.status === "ok") {
      dot.className = "h-2 w-2 rounded-full bg-emerald-400";
      text.textContent = `服务就绪 · ${h.device}`;
      text.className = "text-emerald-300";
    } else if (h.status === "error") {
      dot.className = "h-2 w-2 rounded-full bg-rose-500";
      text.textContent = "模型加载失败";
      text.className = "text-rose-300";
    } else {
      dot.className = "h-2 w-2 animate-pulse rounded-full bg-amber-400";
      text.textContent = "模型加载中...";
      text.className = "text-amber-300";
    }
  } catch {
    $("health-dot").className = "h-2 w-2 rounded-full bg-rose-500";
    $("health-text").textContent = "服务未连接";
    $("health-text").className = "text-rose-300";
  }
}

async function loadModelInfo() {
  try {
    const info = await (await fetch("/api/model-info")).json();
    const rows = [
      ["模型", "TimesFM 3.0"],
      ["设备", info.device || "—"],
      ["输入 patch", info.input_patch_len ?? "—"],
      ["输出 patch", info.output_patch_len ?? "—"],
      ["上下文上限", info.max_context_length?.toLocaleString() || "—"],
      ["最大变量数", info.max_variates ?? "—"],
    ];
    $("model-info").innerHTML = rows
      .map(([k, v]) => `<dt class="text-slate-500">${k}</dt><dd class="text-right font-mono text-slate-300">${v}</dd>`)
      .join("");
  } catch {
    /* 忽略,健康轮询会提示连接问题 */
  }
}

// ---------------------------------------------------------------- 预测

async function runForecast() {
  const btn = $("btn-forecast");
  const horizon = parseInt($("input-horizon").value, 10);
  const makePositive = $("input-positive").checked;

  let payload, histories;
  try {
    if (lastMode === "univariate") {
      const series = parseLines($("input-series").value);
      if (series.length === 0) throw new Error("请输入至少一条有效序列");
      histories = series;
      payload = { series, horizon, return_quantiles: true, make_positive: makePositive };
    } else {
      const targets = parseLines($("input-targets").value);
      if (targets.length === 0) throw new Error("请输入至少一个目标变量");
      const cov = parseLines($("input-covariates").value);
      payload = { targets, horizon, return_quantiles: true, make_positive: makePositive };
      if (cov.length > 0) payload.past_future_covariates = cov;
      histories = targets;
    }
  } catch (e) {
    showToast(e.message);
    return;
  }

  btn.disabled = true;
  btn.textContent = "预测中...";
  try {
    const path = lastMode === "univariate" ? "/api/forecast" : "/api/forecast/multivariate";
    const data = await postJson(path, payload);
    lastResults = data.results;
    renderStats(data, histories.length);
    renderChart(data.results, histories, data.horizon);
    renderTable(data.results);
    $("btn-csv").disabled = false;
    showToast(`预测完成,${data.results.length} 条结果,耗时 ${data.elapsed_ms}ms`, false);
  } catch (e) {
    showToast(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "开始预测";
  }
}

function renderStats(data, count) {
  $("stat-elapsed").textContent = `${data.elapsed_ms} ms`;
  $("stat-device").textContent = data.device;
  $("stat-count").textContent = count;
  $("stat-horizon").textContent = data.horizon;
}

// ---------------------------------------------------------------- 图表

function initChart() {
  chart = echarts.init($("chart"), null, { renderer: "canvas" });
  window.addEventListener("resize", () => chart.resize());
}

function renderChart(results, histories, horizon) {
  $("chart-empty").classList.add("hidden");
  const series = [];
  const legendNames = [];

  results.forEach((res, idx) => {
    const color = PALETTE[idx % PALETTE.length];
    const history = histories[idx] || [];
    const display = history.slice(-MAX_DISPLAY_POINTS);
    const offset = display.length; // 预测区从 x=offset 开始
    const point = res.point_forecast.map((v) => (v === null ? null : +v.toFixed(4)));

    // 历史曲线
    legendNames.push(`${res.ts_id} 历史`);
    series.push({
      name: `${res.ts_id} 历史`,
      type: "line",
      data: display,
      showSymbol: false,
      lineStyle: { width: 1.5, color },
      itemStyle: { color },
      z: 2,
    });

    // 预测曲线:第一个点桥接历史末尾,视觉上连贯
    legendNames.push(`${res.ts_id} 预测`);
    series.push({
      name: `${res.ts_id} 预测`,
      type: "line",
      data: [...Array(offset - 1).fill(null), display[display.length - 1], ...point],
      showSymbol: false,
      lineStyle: { width: 2.5, color, type: "dashed" },
      itemStyle: { color },
      z: 3,
    });

    // P10~P90 置信带
    const q = res.quantiles || {};
    if (q["0.1"] && q["0.9"]) {
      const lower = [...Array(offset).fill(null), display[display.length - 1], ...q["0.1"].map((v) => (v === null ? null : +v.toFixed(4)))];
      const upper = [...Array(offset).fill(null), display[display.length - 1], ...q["0.9"].map((v) => (v === null ? null : +v.toFixed(4)))];
      series.push({
        name: `${res.ts_id} 区间`,
        type: "line",
        data: upper,
        stack: `band-${idx}`,
        showSymbol: false,
        lineStyle: { opacity: 0 },
        areaStyle: { color, opacity: 0.12 },
        silent: true,
        z: 1,
      });
      series.push({
        name: `${res.ts_id} 区间基线`,
        type: "line",
        data: lower,
        stack: `band-${idx}`,
        showSymbol: false,
        lineStyle: { opacity: 0 },
        silent: true,
        z: 1,
      });
    }
  });

  const maxLen = Math.max(
    ...histories.map((h) => Math.min(h.length, MAX_DISPLAY_POINTS)),
  );
  const xLabels = [
    ...Array.from({ length: maxLen }, (_, i) => (i - maxLen + 1 === -1 ? "now" : `t${i - maxLen + 1}`)),
    ...Array.from({ length: horizon }, (_, i) => `+${i + 1}`),
  ];

  chart.setOption(
    {
      backgroundColor: "transparent",
      animationDuration: 400,
      legend: { data: legendNames, show: results.length <= 3, top: 0, textStyle: { color: "#94a3b8", fontSize: 11 } },
      grid: { left: 50, right: 20, top: 36, bottom: 40 },
      tooltip: {
        trigger: "axis",
        backgroundColor: "#0f172a",
        borderColor: "#334155",
        textStyle: { color: "#e2e8f0", fontSize: 12 },
      },
      xAxis: {
        type: "category",
        data: xLabels,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: {
          color: "#64748b",
          fontSize: 10,
          interval: (index) => {
            if (index >= maxLen) {
              const step = Math.ceil(horizon / 6); // 预测区最多约 6 个标签
              return (index - maxLen) % step === 0;
            }
            return index % Math.max(1, Math.floor(maxLen / 10)) === 0;
          },
        },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#1e293b" } },
        axisLabel: { color: "#64748b", fontSize: 10 },
      },
      series,
    },
    { notMerge: true },
  );
}

// ---------------------------------------------------------------- 数值表与 CSV

function renderTable(results) {
  const head = ["ts_id", "step", "point", "0.1", "0.3", "0.5", "0.7", "0.9"];
  $("table-head").innerHTML = `<tr>${head.map((h) => `<th class="px-2 py-1.5">${h}</th>`).join("")}</tr>`;
  const fmt = (v) => (v === null || v === undefined ? "—" : (+v).toFixed(3));
  const rows = [];
  results.forEach((res) => {
    res.point_forecast.forEach((v, i) => {
      const q = res.quantiles || {};
      rows.push(
        `<tr class="text-slate-400 hover:bg-slate-800/40">
          <td class="px-2 py-1 text-slate-300">${res.ts_id}</td>
          <td class="px-2 py-1">+${i + 1}</td>
          <td class="px-2 py-1 text-indigo-300">${fmt(v)}</td>
          <td class="px-2 py-1">${fmt(q["0.1"]?.[i])}</td>
          <td class="px-2 py-1">${fmt(q["0.3"]?.[i])}</td>
          <td class="px-2 py-1 text-slate-300">${fmt(q["0.5"]?.[i])}</td>
          <td class="px-2 py-1">${fmt(q["0.7"]?.[i])}</td>
          <td class="px-2 py-1">${fmt(q["0.9"]?.[i])}</td>
        </tr>`,
      );
    });
  });
  $("table-body").innerHTML = rows.join("");
}

function downloadCsv() {
  if (lastResults.length === 0) return;
  const qKeys = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"];
  const lines = ["ts_id,step,point_forecast," + qKeys.map((k) => `q${k}`).join(",")];
  lastResults.forEach((res) => {
    res.point_forecast.forEach((v, i) => {
      const q = res.quantiles || {};
      lines.push(
        [res.ts_id, i + 1, v ?? "", ...qKeys.map((k) => q[k]?.[i] ?? "")].join(","),
      );
    });
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "timesfm_forecast.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------- 初始化

window.addEventListener("DOMContentLoaded", () => {
  initChart();
  refreshHealth();
  loadModelInfo();
  setInterval(refreshHealth, 10000);
});
