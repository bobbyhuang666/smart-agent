# TaskRouter

**简单任务走本地小模型不花钱，复杂任务走云端大模型才花钱，自动判断、越用越准。**

## 实测数据 (v4.5.1)

| 场景 | 全云端成本 | TaskRouter 成本 | 节省比例 |
|------|-----------|----------------|---------|
| 翻译任务 (5条) | $0.0068 | $0.0021 | **70%** |
| 分类任务 (5条) | $0.0068 | $0.0026 | **62%** |
| 代码生成 (5条) | $0.0068 | $0.0074 | -10%* |
| **总计 (15条)** | **$0.0203** | **$0.0120** | **41%** |

> *代码生成中 1 条复杂任务自动路由到云端（正确行为），其余 4 条走本地免费。路由准确率 100%。

```
Before: 所有任务 → 云端 API → $0.0203/15条
After:  简单任务 → 本地免费 + 复杂任务 → 云端 → $0.0120/15条
                                                          省 41%
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **A3M 多信号路由** | 动词强度 + 多步检测 + 领域复杂度，自动判断任务难度 |
| **五层路由决策** | 语义缓存 → 规则引擎 → 本地模型 → 递归拆解 → 云端 API |
| **蒸馏学习闭环** | 每次云端调用自动采集训练对，本地模型越用越强 |
| **输出质量验证** | 本地输出不合格自动回退云端，采集修正对 |
| **中文企业场景** | 合同、发票、会议纪要等 6 个专项模板 |
| **数据隐私保护** | PII 自动检测脱敏后再发送云端 |
| **企业级审计** | 完整操作日志 + 配额管理 + API 认证 |
| **自适应阈值** | 根据历史成功率动态调整路由策略 |
| **A3M 可学习权重** | 路由评分参数从反馈中自动学习优化 |

---

## 工作原理

```
用户任务 ─→ A3M 多信号评估 ─→ 五层路由决策
                              │
                              ├→ 语义缓存命中 → 0ms, 0 token
                              ├→ 规则引擎兜底 → 0ms, 100% 准确
                              ├→ 本地模型执行 → ~1-2s, 免费
                              ├→ 递归拆解混合 → 部分本地+部分云端
                              └→ 云端 API 调用 → 按量付费
                                    │
                                    ▼
                              蒸馏采集 → Judge 评判 → Few-Shot 注入
                                    │
                                    ▼
                              本地模型变强 → 更多任务走本地 → 更省钱
```

**关键：** 每次云端调用都在"训练"本地模型，系统越用越聪明。

---

## 快速开始

```bash
# 1. 安装
pip3 install requests aiohttp
ollama pull qwen-tool

# 2. 设置别名
alias sma="python3 /path/to/task-router/scripts/cli.py"

# 3. 执行任务
sma --task "翻译成中文" --text "Hello World"
# → 本地执行，免费，1-2秒

# 4. 查看累计节约
sma --stats

# 5. 运行基准测试
python3 scripts/benchmark.py

# 6. 启动 Web 仪表盘
python3 scripts/api_server.py --port 8930
# 访问 http://localhost:8930
```

---

## 实际使用 Demo

### 翻译任务 → 自动走本地（免费）

```bash
$ sma --task "翻译成中文" --text "Hello World, this is a test."

[LOCAL] qwen-tool:latest
耗时: 1200ms | 输入: 45 | 输出: 12 | 节约: $0.000420
--------------------------------------------------
你好世界，这是一个测试。
```

### 复杂代码生成 → 自动走云端（付费）

```bash
$ sma --task "设计一个分布式锁的实现，支持Redis和ZooKeeper两种后端，要求有自动续期和可重入功能"

[CLOUD] deepseek-chat
耗时: 3200ms | 输入: 128 | 输出: 512 | 节约: $0.000000
--------------------------------------------------
# Distributed Lock Implementation
import redis
import kazoo.client
...
```

### 查看路由预估

```bash
$ sma --estimate "分类这个产品属于哪个类别"

任务: 分类这个产品属于哪个类别
建议路由: 本地 (免费)
原因: 评分 1.2 ≤ 3.0 (动词: 分类(-0.15), 本地模式匹配 1 个)
预估云端成本: $0.00035
```

### 查看累计节约

```bash
$ sma --stats

TaskRouter 使用统计
  总任务数: 156
  本地执行: 142 (91%)
  云端执行: 14 (9%)
  累计节约: $0.0182
  节约比例: 82%
```

---

## 中文企业场景

内置 6 个中文企业专项模板，开箱即用：

| 场景 | 命令示例 | 路由 |
|------|---------|------|
| 合同条款提取 | `sma --task "合同条款提取" --text "..."` | 本地 |
| 发票解析 | `sma --task "发票信息提取" --text "..."` | 本地 |
| 会议纪要整理 | `sma --task "会议纪要整理" --text "..."` | 本地/云端 |
| 客户反馈分类 | `sma --task "客户反馈分类" --text "..."` | 本地 |
| 数据报表分析 | `sma --task "数据分析报告" --text "..."` | 云端 |
| 商品分类统计 | `sma --task "分类并统计" --text "..."` | 混合 |

---

## 数据隐私保护

自动检测并脱敏敏感信息后再发送到云端：

```python
from scripts.privacy import PrivacyFilter

pf = PrivacyFilter()
result = pf.anonymize("请联系 13812345678 或 test@example.com")
# → "请联系 [手机号]_0 或 [邮箱]_0"

original = pf.deanonymize(result.text)
# → "请联系 13812345678 或 test@example.com"
```

支持检测：手机号、身份证、邮箱、银行卡、IP 地址、护照号码。

---

## API 服务

```bash
python3 scripts/api_server.py --port 8930
```

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/task | 执行单个任务 |
| POST | /api/estimate | 预估路由 |
| POST | /api/decompose | 拆解复杂任务 |
| POST | /api/batch | 批量处理（支持并发） |
| POST | /v1/chat/completions | OpenAI 兼容 API |
| GET | /api/stats | 使用统计 |
| GET | /api/models | 模型列表 |
| GET | /api/audit | 审计日志 |
| GET | / | Web 仪表盘 |

---

## 学习闭环可视化

查看系统如何随时间进化：

```bash
python3 scripts/learning_viz.py           # 完整报告
python3 scripts/learning_viz.py --days 7  # 最近7天
python3 scripts/learning_viz.py --json    # JSON格式
```

---

## 模型管理

```bash
# 查看已安装模型及能力评分
sma --models

# 运行基准测试
sma --benchmark qwen-tool:latest

# 质量评估
python3 scripts/quality_eval.py --eval qwen-tool:latest
python3 scripts/quality_eval.py --ab model_a model_b
```

---

## 蒸馏系统

```bash
# 评判待处理的训练对
sma --distill

# 查看蒸馏健康状态
sma --distill-stats

# 导出可训练数据
sma --distill-export
```

蒸馏流程：云端响应 → Judge 评判 → SUPPORTED/CONTESTED → 提取 few-shot → 注入 prompt → 本地模型变强。

---

## 语义缓存

重复任务自动命中缓存，三级匹配：

1. **exact** — 精确匹配
2. **normalized** — 去空格、统一标点
3. **fuzzy** — trigram Jaccard 相似度（阈值 0.85）

不同任务类型使用不同 TTL：翻译/分类 7 天，提取 3 天，摘要 1 天。

---

## 技术架构

### A3M 多信号路由

```
复杂度评分 = 动词强度×3 + 多步惩罚 + 领域复杂度 + 文本长度 + 文件数量 + 本地模式奖励

动词强度: 设计+0.25, 分析+0.15, 分类-0.15, 提取-0.10 ...
多步检测: "并且"、"然后"、"先…再…"、序号等
领域检测: 金融/法律/医疗/算法 +0.5
```

### A3M 可学习权重

所有路由评分参数（动词乘数、多步权重、领域权重、文本阈值等）均可从执行反馈中自动学习：

```bash
# 查看当前权重和学习状态
sma --weights

# 重置为默认权重
sma --weights-reset
```

学习机制：本地成功 → 阈值升高（鼓励更多走本地）；本地失败 → 阈值降低（减少走本地）。阈值自动限制在 [1.0, 8.0] 范围内。

### 云端重试 + 熔断

```
失败 → 重试2次（指数退避）→ 连续3次失败 → 熔断120秒 → 期间保留本地输出
```

### 输出验证 + 降级

```
本地执行 → 验证质量 → 不通过 → 自动回退云端 → 采集修正对用于蒸馏
```

---

## 命令参考

| 命令 | 说明 |
|------|------|
| `sma --task "..." --text "..."` | 单任务执行 |
| `sma --task "..." --force local` | 强制本地 |
| `sma --decompose "大任务"` | 拆解任务 |
| `sma --estimate "..."` | 预估路由 |
| `sma --batch tasks.json` | 批量执行 |
| `sma --batch tasks.json --concurrency 3` | 并发批量 |
| `sma --stats` | 使用统计 |
| `sma --models` | 模型列表 |
| `sma --benchmark [model]` | 基准测试 |
| `sma --distill` | 蒸馏评判 |
| `sma --distill-stats` | 蒸馏状态 |
| `sma --weights` | 查看 A3M 可学习权重 |
| `sma --weights-reset` | 重置 A3M 权重为默认值 |
| `sma -i` | 交互模式 |

---

## 配置云端 API

```bash
# DeepSeek（推荐，性价比高）
export CLOUD_API_URL="https://api.deepseek.com"
export CLOUD_API_KEY="sk-xxxxxxxx"
export CLOUD_MODEL="deepseek-chat"

# Claude
export CLOUD_API_URL="https://api.anthropic.com"
export CLOUD_API_KEY="sk-ant-xxxxxxxx"
export CLOUD_MODEL="claude-sonnet-4-6"

# OpenAI
export CLOUD_API_URL="https://api.openai.com"
export CLOUD_API_KEY="sk-xxxxxxxx"
export CLOUD_MODEL="gpt-4o"
```

不配置时，跳过云端子任务，本地子任务正常执行。

---

## 测试 & 质量

- **110 个自动化测试**，覆盖核心路由、缓存、蒸馏、熔断器、API 认证、路由准确率
- **路由准确率基准**: 73 个标注用例，19 个测试方法，100% 通过
- **CI/CD**: GitHub Actions，Python 3.10-3.13 矩阵测试 + ruff 静态分析
- **覆盖率**: 45%（核心引擎 task_router.py 67%，路由模块 routing.py 90%）

```
scripts/task_router.py  379 stmts  125 miss  67%  ← 核心引擎
scripts/routing.py      126 stmts   12 miss  90%  ← 路由决策
scripts/cache.py         98 stmts   14 miss  86%  ← 语义缓存
scripts/distillation.py 139 stmts   27 miss  81%  ← 蒸馏系统
scripts/weights.py       89 stmts   19 miss  79%  ← A3M 可学习权重
TOTAL                  2501 stmts 1365 miss  45%
```

```bash
# 运行全部测试
python3 -m pytest tests/ -v

# 运行覆盖率
python3 -m pytest tests/ --cov=scripts --cov-report=term-missing

# 运行路由准确率基准
python3 -m pytest tests/test_routing_accuracy.py -v
```

---

## Docker 部署

```bash
docker build -t taskrouter .
docker run -p 8930:8930 -e TASKROUTER_API_KEY=your-secret-key taskrouter
```

---

## 开发者指南

```bash
# 安装 pre-commit hook（每次提交前自动运行 ruff 检查）
bash scripts/setup-hooks.sh

# 手动运行 ruff 检查
ruff check scripts/ --select E,F,W --ignore E501,E402,F541

# 自动修复 ruff 错误
ruff check scripts/ --select E,F,W --ignore E501,E402,F541 --fix
```

---

## 联系方式

- Email: huangweijiebobby@gmail.com

---

## License

MIT

---

---

# TaskRouter

**Self-evolving Enterprise AI Cost Optimization Engine** — Automatically routes tasks to the optimal model, and through distillation loops, continuously strengthens local models to save more over time.

> **Key Differentiator:** Not just routing, but learning routing. Every cloud call trains the local model, making the system smarter with use.

---

## Why TaskRouter?

| Feature | TaskRouter | Other Routing Tools |
|---------|-----------|---------------------|
| **Distillation Loop** | Cloud responses → continuous local model evolution | Static routing only |
| **Chinese Enterprise Scenarios** | Contracts, invoices, meeting minutes templates | English-focused |
| **Task Decomposition** | Compound tasks auto-split into subtasks | Single-layer classification |
| **Adaptive Thresholds** | Dynamic routing based on historical success rates | Fixed thresholds |
| **Learnable Weights** | A3M routing parameters auto-optimize from feedback | Static scoring |
| **Data Privacy** | PII auto-detection and anonymization before cloud | No privacy protection |
| **Enterprise Audit** | Complete operation logs + quota management | No audit capability |

---

## How It Works

```
User Task ─→ A3M Multi-signal Evaluation ─→ Five-layer Routing Decision
                                              │
                                              ├→ Semantic Cache Hit → 0ms, 0 tokens
                                              ├→ Rule Engine Fallback → 0ms, 100% accurate
                                              ├→ Local Model Execution → ~1-2s, free
                                              ├→ Recursive Decomposition → partial local + cloud
                                              └→ Cloud API Call → pay-per-use
                                                    │
                                                    ▼
                                              Distillation Collection → Judge → Few-Shot Injection
                                                    │
                                                    ▼
                                              Local Model Improves → More Tasks Local → More Savings
```

**Key:** Every cloud call "trains" the local model, making the system smarter over time.

---

## Quick Start

```bash
# 1. Install
pip3 install requests aiohttp
ollama pull qwen-tool

# 2. Set alias
alias sma="python3 /path/to/task-router/scripts/cli.py"

# 3. Execute tasks
sma --task "translate to Chinese" --text "Hello World"
# → Local execution, free, 1-2 seconds

# 4. View cumulative savings
sma --stats

# 5. Start Web dashboard
python3 scripts/api_server.py --port 8930
# Visit http://localhost:8930
```

---

## Chinese Enterprise Scenarios

6 built-in Chinese enterprise templates, ready to use:

| Scenario | Command Example | Route |
|----------|----------------|-------|
| Contract clause extraction | `sma --task "合同条款提取" --text "..."` | Local |
| Invoice parsing | `sma --task "发票信息提取" --text "..."` | Local |
| Meeting minutes | `sma --task "会议纪要整理" --text "..."` | Local/Cloud |
| Customer feedback classification | `sma --task "客户反馈分类" --text "..."` | Local |
| Data report analysis | `sma --task "数据分析报告" --text "..."` | Cloud |
| Product categorization | `sma --task "分类并统计" --text "..."` | Hybrid |

---

## Data Privacy Protection

Automatically detects and anonymizes sensitive information before sending to cloud:

```python
from scripts.privacy import PrivacyFilter

pf = PrivacyFilter()
result = pf.anonymize("Contact 13812345678 or test@example.com")
# → "Contact [Phone]_0 or [Email]_0"

original = pf.deanonymize(result.text)
# → "Contact 13812345678 or test@example.com"
```

Supports detection: phone numbers, ID cards, emails, bank cards, IP addresses, passport numbers.

---

## API Service

```bash
python3 scripts/api_server.py --port 8930
```

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/task | Execute single task |
| POST | /api/estimate | Estimate routing |
| POST | /api/decompose | Decompose complex task |
| POST | /api/batch | Batch processing (concurrent) |
| POST | /v1/chat/completions | OpenAI-compatible API |
| GET | /api/stats | Usage statistics |
| GET | /api/models | Model list |
| GET | /api/audit | Audit logs |
| GET | / | Web dashboard |

---

## Learning Loop Visualization

View how the system evolves over time:

```bash
python3 scripts/learning_viz.py           # Full report
python3 scripts/learning_viz.py --days 7  # Last 7 days
python3 scripts/learning_viz.py --json    # JSON format
```

---

## Model Management

```bash
# View installed models and capability scores
sma --models

# Run benchmark
sma --benchmark qwen-tool:latest

# Quality evaluation
python3 scripts/quality_eval.py --eval qwen-tool:latest
python3 scripts/quality_eval.py --ab model_a model_b
```

---

## Distillation System

```bash
# Judge pending training pairs
sma --distill

# View distillation health
sma --distill-stats

# Export trainable data
sma --distill-export
```

Distillation flow: Cloud response → Judge → SUPPORTED/CONTESTED → Extract few-shot → Inject into prompt → Local model improves.

---

## Semantic Cache

Duplicate tasks automatically hit cache with three-level matching:

1. **exact** — Exact match
2. **normalized** — Strip spaces, unify punctuation
3. **fuzzy** — Trigram Jaccard similarity (threshold 0.85)

Different task types use different TTLs: translation/classification 7 days, extraction 3 days, summarization 1 day.

---

## Technical Architecture

### A3M Multi-signal Routing

```
Complexity Score = Verb Intensity×3 + Multi-step Penalty + Domain Complexity + Text Length + File Count + Local Pattern Bonus

Verb Intensity: design+0.25, analyze+0.15, classify-0.15, extract-0.10 ...
Multi-step Detection: "and", "then", "first...then...", numbered lists
Domain Detection: finance/legal/medical/algorithm +0.5
```

### Cloud Retry + Circuit Breaker

```
Failure → Retry 2x (exponential backoff) → 3 consecutive failures → Circuit break 120s → Preserve local output during break
```

### Output Validation + Fallback

```
Local execution → Validate quality → Fail → Auto-fallback to cloud → Collect correction pair for distillation
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `sma --task "..." --text "..."` | Single task execution |
| `sma --task "..." --force local` | Force local |
| `sma --decompose "big task"` | Decompose task |
| `sma --estimate "..."` | Estimate routing |
| `sma --batch tasks.json` | Batch execution |
| `sma --batch tasks.json --concurrency 3` | Concurrent batch |
| `sma --stats` | Usage statistics |
| `sma --models` | Model list |
| `sma --benchmark [model]` | Benchmark |
| `sma --distill` | Distillation judge |
| `sma --distill-stats` | Distillation status |
| `sma --weights` | View A3M learnable weights |
| `sma --weights-reset` | Reset A3M weights to defaults |
| `sma -i` | Interactive mode |

---

## Configure Cloud API

```bash
# DeepSeek (recommended, cost-effective)
export CLOUD_API_URL="https://api.deepseek.com"
export CLOUD_API_KEY="sk-xxxxxxxx"
export CLOUD_MODEL="deepseek-chat"

# Claude
export CLOUD_API_URL="https://api.anthropic.com"
export CLOUD_API_KEY="sk-ant-xxxxxxxx"
export CLOUD_MODEL="claude-sonnet-4-6"

# OpenAI
export CLOUD_API_URL="https://api.openai.com"
export CLOUD_API_KEY="sk-xxxxxxxx"
export CLOUD_MODEL="gpt-4o"
```

When not configured, cloud subtasks are skipped, local subtasks execute normally.

---

## Contact

- Email: huangweijiebobby@gmail.com

---

## License

MIT
