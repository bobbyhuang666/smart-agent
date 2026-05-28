# TaskRouter

> **基于 Thompson Sampling 的自适应 LLM 路由引擎**
> 每次路由都在学习，越用越准

简单任务 → 本地 Ollama（免费）| 复杂任务 → 云端 API（按需付费）

**路由准确率 87.5% | 成本节省 41-70% | 435 个自动化测试**

---

## 为什么选 TaskRouter

| 特性 | TaskRouter | 传统路由 |
|------|------------|----------|
| **路由决策** | Thompson Sampling 贝叶斯决策，自动探索-利用平衡 | 固定规则/阈值 |
| **置信度校准** | Token 分位数 + 贝叶斯 Platt 缩放，解决长度偏差 | 简单平均/启发式 |
| **缓存策略** | 结果质量感知，高质量缓存更容易命中 | 固定阈值 |
| **推理策略** | Chain-of-Draft 节省 70% token，置信度驱动选择 | 固定 CoT/ few-shot |
| **学习能力** | 每次路由自动更新模型，冷启动有贝叶斯先验保护 | 静态或人工调参 |

---

## 基准测试 (v6.0)

### 路由准确率

| 指标 | 基线系统 | TaskRouter | 改进 |
|------|----------|------------|------|
| 路由准确率 | 37.5% | **87.5%** | +50.0% |
| Token 节省 | 0% | **5.3%** | +5.3% |
| 缓存命中率 | 23.3% | **28.2%** | +4.9% |

> 在边界场景（复杂度不能唯一决定路由）下，TQBC 的 logprobs 驱动方法显著优于纯复杂度阈值。

### 成本对比 (15 条任务)

| 场景 | 全云端成本 | TaskRouter 成本 | 节省比例 |
|------|-----------|----------------|---------|
| 翻译任务 (5条) | $0.0068 | $0.0021 | **70%** |
| 分类任务 (5条) | $0.0068 | $0.0026 | **62%** |
| 代码生成 (5条) | $0.0068 | $0.0074 | -10%* |
| **总计 (15条)** | **$0.0203** | **$0.0120** | **41%** |

> *代码生成中 1 条复杂任务自动路由到云端（正确行为），其余 4 条走本地免费。

---

## 核心创新：TQBC

**Token-Quantile Bayesian Cascade** — 解决 LLM 路由的三个关键问题：

1. **长度偏差**：使用 q25/q50/q75/q90 分位数替代简单平均，中位数对序列长度鲁棒
2. **探索-利用困境**：Thompson Sampling 贝叶斯线性回归，自动平衡探索与利用
3. **校准不足**：贝叶斯 Platt 缩放 + 分组校准，置信度与实际准确率匹配

```
┌─────────────────────────────────────────────────────────────┐
│                    TaskRouter v6.0 架构                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: 自适应 Prompt 压缩 (置信度驱动的 Token 预算优化)      │
│  Layer 4: Gatekeeper 置信度分离 (动态阈值调整)                  │
│  Layer 3: OATS 缓存质量感知 (结果反馈驱动的缓存优先级)          │
│  Layer 2: 自适应推理策略 (Chain-of-Draft + Token 分位数)       │
│  Layer 1: TQBC 路由决策 (Thompson Sampling + 贝叶斯校准)       │
│  Layer 0: Token 分位数特征提取 (解决长度偏差问题)               │
└─────────────────────────────────────────────────────────────┘
```

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

# 5. 启动 Web 仪表盘
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

---

## 功能特性

### 智能路由

- **三层决策融合**：置信度级联 + Meta-Learner + 主动学习投票
- **Token 分位数特征**：8 维特征向量解决长度偏差
- **Thompson Sampling**：贝叶斯决策，自动探索-利用平衡
- **冷启动保护**：贝叶斯先验编码初始不确定性

### 成本优化

- **语义缓存**：三级匹配（精确/归一化/模糊），重复任务 0 token 成本
- **自适应压缩**：置信度驱动的 token 预算优化（40%-100%）
- **Chain-of-Draft**：仅使用 7.6% 的推理 token 达到 CoT 准确率
- **批处理 API**：32 个请求批处理可降低 85% 单 token 成本

### 质量保障

- **QualityEvaluator 5 维评估**：结构/相关性/失败信号/任务适配/一致性
- **输出验证 + 云端降级**：本地输出质量差时自动回退云端
- **云端重试 + 熔断**：失败自动重试 2 次，连续 3 次失败触发 120 秒熔断

### 企业特性

- **安全审计**：完整操作日志 + 配额管理 + API 认证
- **多 API Key**：按团队管理，支持模型访问控制
- **用量告警**：80% 配额自动告警
- **PII 脱敏**：自动检测并脱敏敏感信息

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
| `sma --weights` | A3M 权重状态 |
| `sma --keys` | API Keys 管理 |
| `sma --quota-check` | 用量告警检查 |
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

- **435 个自动化测试**，覆盖 TQBC 路由、自适应压缩、缓存质量、推理策略等
- **路由准确率基准**: 边界场景 87.5% 准确率
- **CI/CD**: GitHub Actions，Python 3.10-3.13 矩阵测试 + ruff 静态分析

```bash
# 运行全部测试
python3 -m pytest tests/ -v

# 运行 TQBC 基准测试
python3 benchmark_tqbc.py

# 运行学习曲线测试
python3 benchmark_learning_curve.py
```

---

## Docker 部署

```bash
docker build -t taskrouter .
docker run -p 8930:8930 -e TASKROUTER_API_KEY=your-secret-key taskrouter
```

---

## 参考文献

| 论文 | 来源 | 核心贡献 |
|------|------|----------|
| Language Model Cascades | ICML 2024 | Token 分位数解决长度偏差 |
| PILOT | EMNLP 2025 | 上下文老虎机 LLM 路由 |
| OATS | vLLM-SR 2026 | 结果感知的工具选择 |
| Chain of Draft | - | 极简推理，7.6% token |
| Multicalibration | ICML 2024 | 分组校准 |
| Gatekeeper | arXiv 2502.19335 | 置信度分离训练 |
| RouteLLM | ICLR 2025 | 偏好数据路由 |
| FrugalGPT | - | 级联推理成本优化 |

---

## 联系方式

- Email: huangweijiebobby@gmail.com

---

## License

MIT
