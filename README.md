# TaskRouter

**企业 LLM 网关 — 安全审计 · 智能路由 · 自动优化**

统一管理所有 LLM 调用，简单任务走本地小模型（零成本、零延迟），复杂任务走云端大模型（按需付费）。三层决策融合确保路由准确，蒸馏闭环让系统越用越聪明。

## 核心价值

| 能力 | 说明 |
|------|------|
| **安全审计** | 完整操作日志 + 配额管理 + API 认证 + PII 自动脱敏 |
| **智能路由** | 三层决策融合（置信度级联 + Meta-Learner + 主动学习） |
| **自动优化** | 蒸馏闭环：云端响应 → 质量评估 → 本地模型持续进化 |
| **中文企业场景** | 合同、发票、会议纪要等 6 个专项模板，开箱即用 |

## 实测数据 (v5.0)

| 场景 | 全云端成本 | TaskRouter 成本 | 节省比例 |
|------|-----------|----------------|---------|
| 翻译任务 (5条) | $0.0068 | $0.0021 | **70%** |
| 分类任务 (5条) | $0.0068 | $0.0026 | **62%** |
| 代码生成 (5条) | $0.0068 | $0.0074 | -10%* |
| **总计 (15条)** | **$0.0203** | **$0.0120** | **41%** |

> *代码生成中 1 条复杂任务自动路由到云端（正确行为），其余 4 条走本地免费。路由准确率 100%。

---

## 工作原理

```
用户任务 ─→ 三层决策融合 ─→ 路由执行
                │
                ├→ 置信度级联：token 级 logprobs → 校准 → 阈值判断
                ├→ Meta-Learner：10 维特征 → 在线学习 → 全局决策
                └→ 主动学习：不确定性采样 → 优先收集高价值数据
                      │
                      ▼
                路由结果
                ├→ 语义缓存命中 → 0ms, 0 token
                ├→ 规则引擎兜底 → 0ms, 100% 准确
                ├→ 本地模型执行 → ~1-2s, 免费
                ├→ 递归拆解混合 → 部分本地+部分云端
                └→ 云端 API 调用 → 按量付费
                      │
                      ▼
                蒸馏闭环
                ├→ QualityEvaluator 5 维质量评估
                ├→ FailureClusterer 失败模式聚类
                └→ 闭环推进 hypothesis → supported/contested
```

---

## 三层决策融合

### 第一层：置信度门控级联

从 token 级 logprobs 提取置信度信号，通过 Pool Adjacent Violators (PAV) 算法校准，低于阈值自动升级到云端。

```
本地执行 → logprobs 提取 → 熵/置信度/边际 → PAV 校准 → 阈值判断
                                                        ├→ 置信度高 → 本地结果
                                                        └→ 置信度低 → 升级云端
```

### 第二层：Meta-Learner 全局决策

在线 Logistic Regression，10 维特征向量统一所有路由信号：

| 特征 | 说明 |
|------|------|
| complexity_score | A3M 复杂度评分 |
| confidence | 本地模型置信度 |
| entropy | token 熵 |
| margin | top-1 vs top-2 差距 |
| text_length | 输入文本长度 |
| file_count | 文件数量 |
| capability_success | 该能力的历史成功率 |
| strategy_cot/structured | 推理策略 |

### 第三层：主动学习

追踪每个任务类型的预测方差，对不确定的任务类型主动请求云端验证，优先收集高价值训练数据。

冷启动保护：样本数不足 5 时不触发验证，避免新任务类型的额外成本。

---

## 蒸馏闭环

### QualityEvaluator 5 维质量评估

| 维度 | 权重 | 说明 |
|------|------|------|
| 结构完整性 | 25% | 输出是否有组织（列表、分类、段落） |
| 内容相关性 | 25% | 输出是否与输入内容相关 |
| 失败信号 | 25% | 是否包含拒绝/错误信号 |
| 任务适配 | 15% | 输出格式是否匹配任务类型 |
| 一致性 | 10% | 与本地输出的重叠度 |

### 状态流转

```
云端响应 → hypothesis → QualityEvaluator 评估
                        ├→ score ≥ 0.9 → supported（可注入 few-shot）
                        ├→ score < 0.5 → contested（需人工审核）
                        └→ 0.5 ≤ score < 0.9 → hypothesis（继续观察）
```

### FailureClusterer 失败模式聚类

自动将相似失败归类：refusal（拒绝）、error（错误）、divergent（偏离）、empty_output（空输出）、quality_low（质量低）。

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

# 5. 查看三层决策状态
sma --cascade        # 置信度级联统计
sma --meta           # Meta-Learner 特征权重
sma --active         # 主动学习不确定性

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

蒸馏流程：云端响应 → 5 维质量评估 → 状态推进 → Few-Shot 注入 → 本地模型变强。

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
| `sma --cascade` | 置信度级联统计 |
| `sma --meta` | Meta-Learner 特征权重 |
| `sma --active` | 主动学习不确定性 |
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

- **276 个自动化测试**，覆盖三层决策融合、5 维质量评估、路由准确率、缓存、蒸馏、熔断器
- **路由准确率基准**: 73 个标注用例，19 个测试方法，100% 通过
- **CI/CD**: GitHub Actions，Python 3.10-3.13 矩阵测试 + ruff 静态分析

```bash
# 运行全部测试
python3 -m pytest tests/ -v

# 运行路由准确率基准
python3 -m pytest tests/test_routing_accuracy.py -v

# 运行三层决策融合测试
python3 -m pytest tests/test_decision_fusion.py -v
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
ruff check scripts/ --select E,F,W --ignore E501,E402

# 自动修复 ruff 错误
ruff check scripts/ --select E,F,W --ignore E501,E402 --fix
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

**Enterprise LLM Gateway — Security Audit · Intelligent Routing · Automatic Optimization**

Unified management of all LLM calls. Simple tasks go to local small models (zero cost, zero latency), complex tasks go to cloud large models (pay-per-use). Three-layer decision fusion ensures routing accuracy, and the distillation loop makes the system smarter over time.

## Core Value

| Capability | Description |
|------------|-------------|
| **Security Audit** | Complete operation logs + quota management + API authentication + PII auto-anonymization |
| **Intelligent Routing** | Three-layer decision fusion (confidence cascade + Meta-Learner + active learning) |
| **Automatic Optimization** | Distillation loop: cloud response → quality evaluation → local model evolution |
| **Chinese Enterprise Scenarios** | 6 built-in templates for contracts, invoices, meeting minutes, etc. |

## How It Works

```
User Task ─→ Three-Layer Decision Fusion ─→ Route Execution
                │
                ├→ Confidence Cascade: token-level logprobs → calibration → threshold
                ├→ Meta-Learner: 10-dim features → online learning → global decision
                └→ Active Learning: uncertainty sampling → prioritize high-value data
                      │
                      ▼
                Route Result
                ├→ Semantic Cache Hit → 0ms, 0 tokens
                ├→ Rule Engine Fallback → 0ms, 100% accurate
                ├→ Local Model Execution → ~1-2s, free
                ├→ Recursive Decomposition → partial local + cloud
                └→ Cloud API Call → pay-per-use
                      │
                      ▼
                Distillation Loop
                ├→ QualityEvaluator 5-dimension assessment
                ├→ FailureClusterer failure pattern clustering
                └→ State progression hypothesis → supported/contested
```

---

## Three-Layer Decision Fusion

### Layer 1: Confidence-Gated Cascade

Extract confidence signals from token-level logprobs, calibrate via Pool Adjacent Violators (PAV) algorithm, auto-escalate to cloud when below threshold.

### Layer 2: Meta-Learner Global Decision

Online Logistic Regression with 10-dimensional feature vector unifying all routing signals.

### Layer 3: Active Learning

Track prediction variance per task type, request cloud verification for uncertain types. Cold-start protection: no verification until 5+ samples.

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

# 5. View three-layer decision status
sma --cascade        # Confidence cascade stats
sma --meta           # Meta-Learner feature weights
sma --active         # Active learning uncertainty

# 6. Start Web dashboard
python3 scripts/api_server.py --port 8930
# Visit http://localhost:8930
```

---

## Contact

- Email: huangweijiebobby@gmail.com

---

## License

MIT
