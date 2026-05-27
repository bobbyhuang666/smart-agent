"""
TaskRouter API 服务器

提供 REST API 和 Web 仪表盘，用于：
- 任务路由 API（兼容 OpenAI 格式）
- 实时监控仪表盘
- 模型管理 API
- 批量任务处理

启动方式：
    python3 api_server.py                    # 默认端口 8930
    python3 api_server.py --port 9000        # 自定义端口
    python3 api_server.py --host 0.0.0.0     # 允许外部访问
"""

import os
import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# 添加 scripts 目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_router import (
    run_task, Task, estimate, classify_task, show_usage_stats,
    decompose_complex_task, CONFIG, cache,
    get_model_registry, run_batch,
)


# ─── API 处理器 ──────────────────────────────────────────────────

class TaskRouterAPI(BaseHTTPRequestHandler):
    """HTTP API 处理器"""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        routes = {
            "/": self._serve_dashboard,
            "/api/stats": self._api_stats,
            "/api/models": self._api_models,
            "/api/cache": self._api_cache_stats,
            "/api/health": self._api_health,
            "/api/history": self._api_history,
        }

        handler = routes.get(path)
        if handler:
            handler(params)
        elif path.startswith("/api/"):
            self._json_response({"error": "Not found"}, 404)
        else:
            self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_body()
        except Exception as e:
            self._json_response({"error": str(e)}, 400)
            return

        routes = {
            "/api/task": self._api_task,
            "/api/estimate": self._api_estimate,
            "/api/decompose": self._api_decompose,
            "/api/batch": self._api_batch,
            "/api/benchmark": self._api_benchmark,
        }

        handler = routes.get(path)
        if handler:
            handler(body)
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    # ─── API 端点 ─────────────────────────────────────────────

    def _api_task(self, body):
        """执行单个任务"""
        action = body.get("action", body.get("prompt", ""))
        text = body.get("text", body.get("input", ""))
        force = body.get("force", "")
        files = body.get("files", [])

        if not action:
            self._json_response({"error": "缺少 action 参数"}, 400)
            return

        task = Task(action=action, text=text, files=files)
        start = time.time()
        result = run_task(task, force_route=force)
        elapsed = int((time.time() - start) * 1000)

        self._json_response({
            "output": result.output,
            "route": result.route,
            "model": result.model_used,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "time_ms": elapsed,
            "cost_saved": result.cost_saved,
        })

    def _api_estimate(self, body):
        """估算任务路由"""
        action = body.get("action", body.get("prompt", ""))
        if not action:
            self._json_response({"error": "缺少 action 参数"}, 400)
            return
        result = estimate(action)
        self._json_response(result)

    def _api_decompose(self, body):
        """拆解复杂任务"""
        action = body.get("action", body.get("prompt", ""))
        text = body.get("text", "")
        if not action:
            self._json_response({"error": "缺少 action 参数"}, 400)
            return
        subtasks = decompose_complex_task(action, text)
        if not subtasks:
            # 尝试递归拆解
            from task_router import _recursive_decompose
            subtasks = _recursive_decompose(action, text) or []
        self._json_response({"task": action, "subtasks": subtasks})

    def _api_batch(self, body):
        """批量任务处理"""
        tasks = body.get("tasks", [])
        concurrency = body.get("concurrency", 1)
        if not tasks:
            self._json_response({"error": "缺少 tasks 数组"}, 400)
            return

        start = time.time()
        results = run_batch(tasks, concurrency=concurrency)
        elapsed = int((time.time() - start) * 1000)

        output = []
        for r in results:
            output.append({
                "action": r.action[:100],
                "output": r.output,
                "route": r.route,
                "model": r.model_used,
                "time_ms": r.time_ms,
                "cost_saved": r.cost_saved,
            })
        self._json_response({
            "results": output,
            "count": len(output),
            "total_time_ms": elapsed,
            "concurrency": concurrency,
        })

    def _api_stats(self, params=None):
        """使用统计"""
        stats_text = show_usage_stats()
        self._json_response({"stats": stats_text})

    def _api_models(self, params=None):
        """模型列表"""
        registry = get_model_registry()
        registry.discover()
        self._json_response(registry.get_status())

    def _api_cache_stats(self, params=None):
        """缓存统计"""
        stats = cache.stats()
        self._json_response(stats)

    def _api_health(self, params=None):
        """健康检查"""
        self._json_response({
            "status": "healthy",
            "version": "2.3",
            "timestamp": datetime.now().isoformat(),
            "local_model": CONFIG["local_model"],
            "cloud_configured": bool(CONFIG["cloud_api_key"]),
        })

    def _api_history(self, params=None):
        """最近任务历史"""
        log_file = os.path.join(CONFIG["cache_dir"], "usage.jsonl")
        limit = int(params.get("limit", ["20"])[0]) if params else 20
        entries = []
        if os.path.exists(log_file):
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        self._json_response({"history": entries, "count": len(entries)})

    def _api_benchmark(self, body):
        """运行基准测试"""
        model = body.get("model")
        registry = get_model_registry()
        registry.discover()
        results = registry.run_benchmark(model)
        self._json_response({"results": results})

    # ─── 仪表盘 ──────────────────────────────────────────────

    def _serve_dashboard(self, params=None):
        """主仪表盘页面"""
        html = DASHBOARD_HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_static(self, path):
        """静态文件服务"""
        self._json_response({"error": "Not found"}, 404)

    # ─── 工具方法 ─────────────────────────────────────────────

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        """静默日志（可配置）"""
        pass


# ─── 仪表盘 HTML ────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TaskRouter Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; }
.header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 20px 30px; border-bottom: 1px solid #475569; }
.header h1 { font-size: 24px; color: #f8fafc; }
.header p { color: #94a3b8; margin-top: 4px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-bottom: 20px; }
.card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
.card h3 { color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
.card .value { font-size: 32px; font-weight: bold; color: #f8fafc; margin-top: 8px; }
.card .sub { color: #64748b; font-size: 13px; margin-top: 4px; }
.green { color: #4ade80; }
.blue { color: #60a5fa; }
.yellow { color: #facc15; }
.red { color: #f87171; }
.test-section { background: #1e293b; border-radius: 12px; padding: 24px; border: 1px solid #334155; margin-bottom: 20px; }
.test-section h2 { margin-bottom: 16px; font-size: 18px; }
.input-group { display: flex; gap: 10px; margin-bottom: 12px; }
input, textarea, select { background: #0f172a; border: 1px solid #475569; color: #e2e8f0; padding: 10px 14px; border-radius: 8px; font-size: 14px; flex: 1; }
textarea { min-height: 80px; resize: vertical; }
button { background: #3b82f6; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 500; }
button:hover { background: #2563eb; }
button.secondary { background: #475569; }
button.secondary:hover { background: #64748b; }
.result { background: #0f172a; border-radius: 8px; padding: 16px; margin-top: 12px; white-space: pre-wrap; font-family: monospace; font-size: 13px; display: none; }
.route-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.route-local { background: #065f46; color: #6ee7b7; }
.route-cloud { background: #7c2d12; color: #fdba74; }
.route-cache { background: #1e3a5f; color: #93c5fd; }
.route-hybrid { background: #4c1d95; color: #c4b5fd; }
.model-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.model-table th, .model-table td { padding: 10px; text-align: left; border-bottom: 1px solid #334155; }
.model-table th { color: #94a3b8; font-size: 12px; text-transform: uppercase; }
.bar { display: inline-block; height: 16px; background: #3b82f6; border-radius: 3px; vertical-align: middle; }
</style>
</head>
<body>
<div class="header">
  <h1>TaskRouter Dashboard</h1>
  <p>智能任务路由系统 — 本地优先，成本最优</p>
</div>
<div class="container">
  <div class="grid" id="stats-grid">
    <div class="card"><h3>本地调用</h3><div class="value green" id="stat-local">-</div><div class="sub">免费执行</div></div>
    <div class="card"><h3>云端调用</h3><div class="value blue" id="stat-cloud">-</div><div class="sub">付费执行</div></div>
    <div class="card"><h3>缓存命中</h3><div class="value yellow" id="stat-cache">-</div><div class="sub">0 token 消耗</div></div>
    <div class="card"><h3>累计节约</h3><div class="value green" id="stat-saved">-</div><div class="sub">相比全云端</div></div>
  </div>

  <div class="test-section">
    <h2>测试任务路由</h2>
    <div class="input-group">
      <input type="text" id="task-action" placeholder="任务描述，如：翻译成中文、分类、提取关键词">
    </div>
    <div class="input-group">
      <textarea id="task-text" placeholder="待处理文本（可选）"></textarea>
    </div>
    <div class="input-group">
      <select id="task-force">
        <option value="">自动路由</option>
        <option value="local">强制本地</option>
        <option value="cloud">强制云端</option>
      </select>
      <button onclick="runTask()">执行任务</button>
      <button class="secondary" onclick="estimateTask()">预估路由</button>
    </div>
    <div class="result" id="task-result"></div>
  </div>

  <div class="test-section">
    <h2>模型管理</h2>
    <button onclick="loadModels()" style="margin-bottom: 12px">刷新模型列表</button>
    <table class="model-table" id="models-table">
      <thead><tr><th>模型</th><th>参数</th><th>延迟</th><th>速度</th><th>调用</th><th>成功率</th></tr></thead>
      <tbody id="models-body"></tbody>
    </table>
  </div>
</div>

<script>
const API = '';

async function api(method, path, body) {
  const opts = { method, headers: {'Content-Type': 'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(API + path, opts);
  return resp.json();
}

async function loadStats() {
  const data = await api('GET', '/api/stats');
  const stats = data.stats || '';
  const local = stats.match(/本地调用: (\\d+)/);
  const cloud = stats.match(/云端调用: (\\d+)/);
  const cache = stats.match(/缓存命中: (\\d+)/);
  const saved = stats.match(/累计节约成本: \\$([\\d.]+)/);
  if (local) document.getElementById('stat-local').textContent = local[1] + ' 次';
  if (cloud) document.getElementById('stat-cloud').textContent = cloud[1] + ' 次';
  if (cache) document.getElementById('stat-cache').textContent = cache[1] + ' 次';
  if (saved) document.getElementById('stat-saved').textContent = '$' + saved[1];
}

async function runTask() {
  const action = document.getElementById('task-action').value;
  const text = document.getElementById('task-text').value;
  const force = document.getElementById('task-force').value;
  if (!action) return alert('请输入任务描述');

  const result = document.getElementById('task-result');
  result.style.display = 'block';
  result.textContent = '执行中...';

  const data = await api('POST', '/api/task', { action, text, force });
  const routeClass = data.route?.includes('cache') ? 'route-cache' :
                     data.route?.includes('local') ? 'route-local' :
                     data.route?.includes('hybrid') ? 'route-hybrid' : 'route-cloud';

  result.innerHTML =
    `<span class="route-badge ${routeClass}">${data.route || 'unknown'}</span> ` +
    `模型: ${data.model || '-'} | 耗时: ${data.time_ms || 0}ms | ` +
    `输入: ${data.tokens_input || 0} tokens | 输出: ${data.tokens_output || 0} tokens` +
    (data.cost_saved > 0 ? ` | <span class="green">节约: $${data.cost_saved}</span>` : '') +
    `\\n\\n${data.output || data.error || '无输出'}`;

  loadStats();
}

async function estimateTask() {
  const action = document.getElementById('task-action').value;
  if (!action) return alert('请输入任务描述');

  const result = document.getElementById('task-result');
  result.style.display = 'block';
  result.textContent = '估算中...';

  const data = await api('POST', '/api/estimate', { action });
  const tag = data.will_save ? '本地 (免费)' : '云端 (付费)';
  const cls = data.will_save ? 'route-local' : 'route-cloud';
  result.innerHTML =
    `<span class="route-badge ${cls}">${tag}</span> 评分: ${data.score || '-'}\\n` +
    `原因: ${data.reason || '-'}\\n预估云端成本: ${data.estimated_cloud_cost || '-'}`;
}

async function loadModels() {
  const data = await api('GET', '/api/models');
  const tbody = document.getElementById('models-body');
  tbody.innerHTML = '';
  for (const [name, info] of Object.entries(data)) {
    const caps = Object.entries(info.capabilities || {})
      .filter(([k,v]) => v > 0)
      .map(([k,v]) => `${k}:${(v*100).toFixed(0)}%`)
      .join(', ');
    const latency = info.avg_latency_ms > 0 ? info.avg_latency_ms.toFixed(0) + 'ms' : '-';
    const speed = info.tokens_per_second > 0 ? info.tokens_per_second.toFixed(0) + 't/s' : '-';
    const rate = (info.success_rate * 100).toFixed(0) + '%';
    tbody.innerHTML += `<tr>
      <td>${name}</td>
      <td>${info.parameter_size || '-'}</td>
      <td>${latency}</td>
      <td>${speed}</td>
      <td>${info.total_calls}</td>
      <td>${rate}</td>
    </tr>`;
  }
}

// 初始加载
loadStats();
loadModels();
setInterval(loadStats, 10000);
</script>
</body>
</html>"""


# ─── 服务器启动 ──────────────────────────────────────────────────

def start_server(host="127.0.0.1", port=8930):
    """启动 API 服务器"""
    server = HTTPServer((host, port), TaskRouterAPI)
    print(f"TaskRouter API 服务器启动")
    print(f"  地址: http://{host}:{port}")
    print(f"  仪表盘: http://{host}:{port}/")
    print(f"  API 文档:")
    print(f"    POST /api/task      - 执行任务")
    print(f"    POST /api/estimate  - 预估路由")
    print(f"    POST /api/decompose - 拆解任务")
    print(f"    POST /api/batch     - 批量处理")
    print(f"    GET  /api/stats     - 使用统计")
    print(f"    GET  /api/models    - 模型列表")
    print(f"    GET  /api/health    - 健康检查")
    print(f"    GET  /api/history   - 任务历史")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TaskRouter API 服务器")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8930, help="监听端口")
    args = parser.parse_args()
    start_server(args.host, args.port)
