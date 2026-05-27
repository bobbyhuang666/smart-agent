# TaskRouter

**自进化的企业级 AI 成本优化引擎** — 自动路由任务到最佳模型，通过蒸馏闭环让本地模型持续变强，越用越省钱。

> **核心差异化：** 不只是路由，而是会学习的路由。每次云端调用都在训练本地模型，系统随时间自我进化。

---

## 为什么选择 TaskRouter？

| 特性 | TaskRouter | 其他路由工具 |
|------|-----------|-------------|
| **蒸馏学习闭环** | 云端响应 → 本地模型持续进化 | 只做静态路由 |
| **中文企业场景** | 合同、发票、会议纪等专项模板 | 英文为主 |
| **任务拆解** | 复合任务自动拆分为子任务独立路由 | 单层分类路由 |
| **自适应阈值** | 根据历史成功率动态调整路由策略 | 固定阈值 |
| **数据隐私** | PII 自动检测脱敏后再发送云端 | 无隐私保护 |
| **企业级审计** | 完整操作日志 + 配额管理 | 无审计能力 |

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
pip3 install requests
ollama pull qwen-tool

# 2. 设置别名
alias sma="python3 /path/to/task-router/scripts/task_router.py"

# 3. 执行任务
sma --task "翻译成中文" --text "Hello World"
# → 本地执行，免费，1-2秒

# 4. 查看累计节约
sma --stats

# 5. 启动 Web 仪表盘
bash start_server.sh
# 访问 http://localhost:8930
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

## License

MIT
