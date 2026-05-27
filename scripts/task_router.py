#!/usr/bin/env python3
"""
TaskRouter — 任务路由系统
自动判断任务复杂度，路由到本地 Ollama 或云端 API，节约 token 成本。
"""

import sys
import os
import json
import time
import hashlib
import re
import random
from pathlib import Path
from dataclasses import dataclass, asdict, field

# 模型注册表（延迟加载）
_model_registry = None

def get_model_registry():
    global _model_registry
    if _model_registry is None:
        from model_registry import ModelRegistry
        _model_registry = ModelRegistry(
            cache_dir=CONFIG.get("cache_dir", os.path.expanduser("~/.cache/task_router"))
        )
        # 首次使用时自动发现
        if not _model_registry.models:
            _model_registry.discover()
    return _model_registry

# ─── 配置 ───────────────────────────────────────────────────────────

CONFIG = {
    "ollama_base": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    "local_model": os.environ.get("LOCAL_MODEL", "qwen-tool"),
    "local_max_tokens": 2048,
    "cloud_api_url": os.environ.get("CLOUD_API_URL", ""),
    "cloud_api_key": os.environ.get("CLOUD_API_KEY", ""),
    "cloud_model": os.environ.get("CLOUD_MODEL", ""),  # 通过环境变量设置，如 deepseek-chat / claude-sonnet-4-6
    "cost_per_1k_input": 0.003,
    "cost_per_1k_output": 0.015,
    "cache_dir": os.environ.get(
        "TASK_ROUTER_CACHE", str(Path.home() / ".cache" / "task_router")
    ),
    # 缓存 TTL（小时）：确定性任务缓存更久，非确定性任务缓存更短
    "cache_ttl_hours": {
        "translation": 168,      # 翻译：7天（确定性高）
        "classification": 168,   # 分类：7天
        "extraction": 72,        # 提取：3天
        "formatting": 168,       # 格式化：7天
        "summarization": 24,     # 摘要：1天（可能变化）
        "default": 48,           # 默认：2天
    },
}

# ─── 云端熔断器状态 ─────────────────────────────────────────────────
_circuit_breaker = {
    "failures": 0,           # 连续失败次数
    "last_failure": 0,       # 最后失败时间戳
    "open_until": 0,         # 熔断截止时间（> 当前时间则熔断中）
    "max_failures": 3,       # 触发熔断的连续失败次数
    "cooldown_seconds": 120, # 熔断冷却时间（秒）
}


# ─── 任务复杂度评估 ─────────────────────────────────────────────────

LOCAL_TASK_PATTERNS = [
    # 文件操作
    "分类", "归类", "整理", "排序", "重命名", "移动文件", "复制",
    # 文本处理
    "提取", "摘录", "格式化", "转换", "翻译", "概括",
    "替换", "补全", "填充", "合并", "拼接", "分割", "拆分",
    # 数据操作
    "排序", "过滤", "去重", "统计", "计数",
    "列出", "列举", "查询", "搜索",
    # 判断
    "属于哪", "是什么类型", "判定", "检查", "校验", "验证",
    # 标签
    "标记", "打标签",
    # 批量
    "批量", "所有文件", "全部", "遍历",
]

MANDATORY_LOCAL_PATTERNS = ["本地", "离线", "不上传"]
CLOUD_PATTERNS = [
    "代码", "编程", "debug", "调试", "重构", "架构",
    "bug", "错误", "异常", "项目", "仓库", "git", "复杂", "高级",
]

# ─── A3M 风格多信号路由 ──────────────────────────────────────────────
# 参考 A3M Router 论文：多信号启发式路由可达 99.5% 准确率
# 信号包括：领域检测、动词强度、查询结构、多步检测、专业度

# 动词/操作词强度映射（正数=复杂，负数=简单）
VERB_INTENSITY = {
    # 高复杂度动词 → +权重
    "设计": 0.25, "架构": 0.25, "规划": 0.20, "策略": 0.20,
    "分析": 0.15, "推理": 0.25, "对比": 0.15, "比较": 0.15,
    "评价": 0.15, "优化": 0.15, "重构": 0.25, "调试": 0.20,
    "生成": 0.10, "创建": 0.10, "实现": 0.15, "开发": 0.15,
    "编写": 0.10, "预测": 0.20, "推荐": 0.15, "建议": 0.10,
    # 中复杂度
    "解释": 0.10, "说明": 0.10, "描述": 0.05, "介绍": 0.05,
    # 低复杂度/简单动词 → 负权重
    "分类": -0.15, "归类": -0.15, "整理": -0.10, "列出": -0.15,
    "列举": -0.15, "提取": -0.10, "摘录": -0.10, "翻译": -0.05,
    "格式化": -0.15, "转换": -0.10, "概括": -0.05, "总结": -0.05,
    "排序": -0.15, "去重": -0.15, "过滤": -0.10, "统计": -0.05,
    "计数": -0.15, "检查": -0.05, "验证": -0.05, "打标签": -0.10,
    "标记": -0.10, "重命名": -0.10, "复制": -0.10, "移动": -0.05,
}

# 多步连接词 — 检测复合任务
MULTI_STEP_CONNECTORS = [
    "并且", "然后", "接着", "再", "再然后", "之后", "随后",
    "同时", "以及", "和", "与", "并",
    "先", "首先", "其次", "最后", "第一步", "第二步",
    "1.", "2.", "3.", "①", "②", "③",
]

# 高复杂度领域
HIGH_COMPLEXITY_DOMAINS = [
    "金融", "法律", "医疗", "医药", "投资", "税务", "合同",
    "机器学习", "深度学习", "算法", "神经网络", "量化",
    "安全", "加密", "密码", "网络协议", "编译",
]

# ─── 能力边界参考：Qwen2.5 1.5B 能做什么/不能做什么 ──────────────
# 这些不参与自动路由，而是给 agent 做人工判断参考
#
# ✅ 可以委托（准确率 ~80-95%）:
#   文本分类/标签     — 文件名归类、邮件分类、情感极性（正/负）
#   简单格式化        — CSV→JSON、换行转逗号、大小写转换
#   关键词提取        — 从短文本中抽人名、地名、日期
#   基础翻译          — 中英互译（短句，无专业术语）
#   简单提取          — 提取邮箱、电话、URL、数字
#   文件元数据整理    — 按扩展名/大小/日期分类
#   模板化回复        — "根据X总结三点"、固定格式输出
#   批量文本替换      — 查找替换、前缀后缀添加
#   简单排序/去重     — 列表去重、字母序排列
#
# ⚠️ 勉强可用（准确率 ~60-80%，需人工复核）:
#   多语言翻译（非中英）
#   长文本概括（>500字）
#   模糊分类（"这属于什么风格"）
#   格式不固定的提取
#
# ❌ 不要委托（准确率 < 50%）:
#   代码生成/调试      — 语法错误、逻辑 bug、API 调用
#   多步骤推理         — "如果A则B否则C，然后计算D"
#   数学计算           — 四则运算以外、公式推导
#   逻辑判断           — 涉及条件分支、因果推理
#   专业领域知识       — 法律、医疗、金融建议
#   数字精确提取/比较  — "哪个数字最大"、"总和是多少"
#   长上下文理解       — >1000字的文档问答
#   创意写作           — 故事、广告文案、诗歌
#   格式严格要求的输出 — 必须符合特定 JSON Schema


@dataclass
class Task:
    id: str = ""
    action: str = ""
    text: str = ""
    files: list = None
    output: str = ""
    model_used: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost_saved: float = 0.0
    time_ms: int = 0
    route: str = ""

    def __post_init__(self):
        if not self.id:
            raw = f"{self.action}{self.text[:100]}{time.time()}"
            self.id = hashlib.md5(raw.encode()).hexdigest()[:12]
        if self.files is None:
            self.files = []


def estimate_complexity(task: Task) -> dict:
    """
    A3M 风格多信号复杂度评估。
    结合动词强度、多步检测、领域检测、文本长度、文件数量等信号。
    """
    action_lower = task.action.lower()
    text_len = len(task.text or "")
    file_count = len(task.files)

    # ── 硬规则（优先级最高） ──
    for p in MANDATORY_LOCAL_PATTERNS:
        if p in action_lower:
            return {"route": "local", "reason": f"匹配强制本地模式: {p}", "score": 0}

    for p in CLOUD_PATTERNS:
        if p in action_lower:
            return {"route": "cloud", "reason": f"匹配需云端模式: {p}", "score": 10}

    # ── 信号 1：动词强度分析 ──
    verb_score = 0.0
    verb_signals = []
    for verb, weight in VERB_INTENSITY.items():
        if verb in action_lower:
            verb_score += weight
            verb_signals.append(f"{verb}({weight:+.2f})")

    # ── 信号 2：多步检测 ──
    multi_step_count = sum(1 for c in MULTI_STEP_CONNECTORS if c in action_lower)
    if multi_step_count >= 3:
        multi_step_penalty = 2.0
    elif multi_step_count >= 2:
        multi_step_penalty = 1.0
    elif multi_step_count >= 1:
        multi_step_penalty = 0.3
    else:
        multi_step_penalty = 0.0

    # ── 信号 3：领域复杂度 ──
    domain_score = 0.0
    for domain in HIGH_COMPLEXITY_DOMAINS:
        if domain in action_lower:
            domain_score += 0.5

    # ── 信号 4：文本长度（细粒度） ──
    if text_len > 2000:
        text_score = 3
    elif text_len > 1000:
        text_score = 2
    elif text_len > 500:
        text_score = 1
    else:
        text_score = 0
    # 极短文本几乎总是简单任务
    if 0 < text_len < 50:
        text_score = max(0, text_score - 1)

    # ── 信号 5：文件数量 ──
    if file_count > 20:
        file_score = 2
    elif file_count > 5:
        file_score = 1
    else:
        file_score = 0

    # ── 信号 6：本地模式匹配（LOCAL_TASK_PATTERNS 命中数） ──
    local_match = sum(1 for p in LOCAL_TASK_PATTERNS if p in action_lower)
    if local_match >= 3:
        local_pattern_bonus = -3
    elif local_match >= 2:
        local_pattern_bonus = -2
    elif local_match >= 1:
        local_pattern_bonus = -1
    else:
        local_pattern_bonus = 0

    # ── 信号 7：action 长度信号 ──
    action_len = len(action_lower)
    if action_len > 100:
        action_score = 0.5
    elif action_len < 10:
        action_score = -1
    else:
        action_score = 0

    # ── 综合评分（可解释的加权和） ──
    score = (
        verb_score * 3.0        # 动词强度 ×3 放大
        + multi_step_penalty    # 多步检测
        + domain_score          # 领域复杂度
        + text_score            # 文本长度
        + file_score            # 文件数量
        + local_pattern_bonus   # 本地模式匹配
        + action_score          # action 长度
    )

    # 构建详细原因
    signal_parts = []
    if verb_signals:
        signal_parts.append(f"动词强度: {''.join(verb_signals)}={verb_score:.2f}")
    if multi_step_count:
        signal_parts.append(f"多步检测: {multi_step_count}个连接词(penalty={multi_step_penalty:.1f})")
    if domain_score:
        signal_parts.append(f"领域复杂度: +{domain_score:.1f}")
    if text_score:
        signal_parts.append(f"文本长度: {text_len}字(score={text_score})")
    if file_score:
        signal_parts.append(f"文件数量: {file_count}个(score={file_score})")
    if local_pattern_bonus:
        signal_parts.append(f"本地模式: {local_match}个模式(bonus={local_pattern_bonus})")
    signal_detail = "; ".join(signal_parts) if signal_parts else "无显著信号"

    # 阈值判定（使用渐进式自适应阈值）
    task_type = detect_task_type(action_lower)
    capability = TASK_TO_CAPABILITY.get(task_type, "")
    effective_threshold = cap_tracker.get_adjusted_threshold(capability)

    route = "local" if score <= effective_threshold else "cloud"
    return {
        "route": route,
        "reason": f"复杂度评分={score:.2f} (<={effective_threshold:.1f} 走本地); {signal_detail}",
        "score": score,
    }


# ─── 提示词模板系统（针对 Qwen2.5 1.5B 优化） ────────────────────
#
# 核心原则：
# 1. 少样本(2-3例) >> 零样本
# 2. 输出格式越具体越好
# 3. 避免抽象指令，用示例说话
# 4. 短输入 + 简洁指令

# 系统级前缀，让模型进入"工具模式"而非"聊天模式"
TOOL_PREFIX = "你是一个精准的工具。只按要求输出，不说多余的话。\n\n"

PROMPT_TEMPLATES = {
    "general_classify": {
        "detect": ["分类", "归类", "分组", "产品类别", "按类别", "分为"],
        "template": TOOL_PREFIX + """将以下内容按类别分组，保持数据完整。每种类别一行，列出属于该类别的所有项：

示例：
iPhone15, 华为Mate60, 小米14, MacBookPro, 联想小新
→ 手机: iPhone15, 华为Mate60, 小米14
电脑: MacBookPro, 联想小新

苹果, 香蕉, 白菜, 萝卜, 牛肉, 鸡肉
→ 水果: 苹果, 香蕉
蔬菜: 白菜, 萝卜
肉类: 牛肉, 鸡肉

北京, 上海, 纽约, 伦敦, 东京, 巴黎
→ 中国城市: 北京, 上海
外国城市: 纽约, 伦敦, 东京, 巴黎

现在分类：
{text}
→""",
        "max_tokens": 256,
    },

    "file_classify": {
        "detect": ["扩展名", "按类型", "按格式"],
        "template": TOOL_PREFIX + """把以下每个文件按扩展名分到对应类别，**必须列出所有文件**，每个一行。

类别对应关系：
.pdf → 文档  .txt → 文档  .docx → 文档
.jpg → 图片  .png → 图片  .gif → 图片
.py → 代码  .js → 代码  .css → 代码  .html → 代码
.xlsx → 数据  .csv → 数据
其他 → 其他

示例：
report.pdf → 文档
photo.jpg → 图片
main.py → 代码
data.csv → 数据
logo.png → 图片

现在分类这些文件（不要漏掉任何一个）：
{text}""",
        "max_tokens": 512,
    },

    "sentiment": {
        "detect": ["情感", "情绪", "正面", "负面", "好评", "差评", "正负面"],
        "template": TOOL_PREFIX + """判断情感倾向，只输出"正面"或"负面"：

示例：
商品很好 → 正面
快递太慢 → 负面
服务不错 → 正面
质量很差 → 负面
发货速度快 → 正面
客服态度差 → 负面
功能很强大 → 正面
包装有破损 → 负面
性价比很高 → 正面
价格太贵了 → 负面
物流非常快 → 正面
用了就退货 → 负面

现在判断：
{text}
→""",
        "max_tokens": 32,
    },

    "translate_en2zh": {
        "detect": ["翻译成中文", "译成中文", "转成中文", "翻译为中文"],
        "template": TOOL_PREFIX + """将英文翻译成中文，只输出翻译结果：

示例：
Hello world → 你好世界
Good morning → 早上好
Thank you → 谢谢

现在翻译：
{text}
→""",
        "max_tokens": 256,
    },

    "translate_zh2en": {
        "detect": ["翻译成英文", "译成英文", "转成英文", "翻译为英文"],
        "template": TOOL_PREFIX + """将中文翻译成英文，只输出翻译结果：

示例：
你好世界 → Hello world
早上好 → Good morning
谢谢 → Thank you

现在翻译：
{text}
→""",
        "max_tokens": 256,
    },

    "extract_keywords": {
        "detect": ["关键词", "keyword", "关键字"],
        "template": TOOL_PREFIX + """从以下文本中提取最重要的3-5个关键词，用逗号分隔：

示例：
"今天天气真好，适合出去散步" → 天气,散步,户外
"这个手机续航长、拍照清晰、性价比高" → 手机,续航,拍照,性价比
"苹果发布新款MacBook Pro，搭载M4芯片，性能提升三倍" → 苹果,MacBook Pro,M4芯片,性能
"公司在上海举办年度开发者大会，吸引了全球5000名开发者参加" → 开发者大会,上海,全球开发者
"量子计算在药物研发领域的应用前景广阔，可大幅缩短新药上市时间" → 量子计算,药物研发,新药

现在提取：
{text}
→""",
        "max_tokens": 64,
    },

    "extract_info": {
        "detect": ["提取", "摘录", "找出", "抽取出"],
        "template": TOOL_PREFIX + """从文本中提取指定信息，每行一个：

示例：
文本：请联系张三(13800138000)或发邮件到 admin@test.com
提取：姓名、电话、邮箱
→ 张三
13800138000
admin@test.com

现在提取：
文本：{text}
提取：{target}
→""",
        "max_tokens": 128,
    },

    "summarize_short": {
        "detect": ["总结", "概括", "摘要", "要点"],
        "template": TOOL_PREFIX + """用1-3句话概括以下内容：

示例：
"苹果今天发布了新款iPhone，搭载A18芯片，起售价799美元。相机系统全面升级，新增AI拍照功能。"
→ 苹果发布新款iPhone，搭载A18芯片，起售价799美元，相机和AI功能升级。

"公司第一季度营收5000万元，同比增长20%。主要来自电商业务，其中日用品类增长最快达35%。"
→ 第一季度营收5000万同比增长20%，电商业务为主，日用品增长35%最突出。

"机器学习是人工智能的一个子领域，它使计算机能够从数据中学习而不需要明确编程。"
→ 机器学习是AI子领域，让计算机从数据中自主学习。

现在概括：
{text}
→""",
        "max_tokens": 256,
    },

    "format_json": {
        "detect": ["转成json", "格式化成json", "json格式"],
        "template": TOOL_PREFIX + """将以下内容转成JSON格式：

示例：
姓名:张三,年龄:28,城市:北京
→ {"name": "张三", "age": 28, "city": "北京"}

现在转换：
{text}
→""",
        "max_tokens": 256,
    },

    "dedup": {
        "detect": ["去重", "去重复", "重复"],
        "template": TOOL_PREFIX + """去重后输出，保留原始顺序。每行一个：

示例：
apple, banana, apple, orange, banana
→ apple
banana
orange

现在去重：
{text}
→""",
        "max_tokens": 256,
    },

    "sort_numbers": {
        "detect": ["排序", "排列", "按销量", "按数量", "从高到低", "从大到小"],
        "template": "",  # 不用模型，用规则
        "rule_based": True,
        "max_tokens": 8,
    },

    "sort_alpha": {
        "detect": ["按字母", "按名称", "按名字", "A到Z", "Z到A"],
        "template": TOOL_PREFIX + """按字母顺序排序，每行一个：

示例：
orange, apple, banana
→ apple
banana
orange

现在排序：
{text}
→""",
        "max_tokens": 256,
    },

    "clean_data": {
        "detect": ["清洗", "清理", "格式统一", "规范化"],
        "template": TOOL_PREFIX + """统一数据格式为标准CSV：每行一个记录，逗号分隔字段。

示例：
商品A,100;商品B,50;商品C,200
→ 商品A,100
商品B,50
商品C,200

现在格式化：
{text}
→""",
        "max_tokens": 256,
    },

    "rename_suggest": {
        "detect": ["重命名", "改名为"],
        "template": TOOL_PREFIX + """建议新文件名，保持扩展名不变。格式：原名 → 新名

示例：
IMG_001.jpg →  vacation_beach.jpg
Document1.docx → project_report.docx
Screenshot1.png → bug_screenshot.png

现在处理：
{text}
→""",
        "max_tokens": 512,
    },

    "tag": {
        "detect": ["打标签", "标记为"],
        "template": TOOL_PREFIX + """给以下内容打一个最合适的类别标签。可选：技术、生活、工作、学习、娱乐、其他
只输出一个类别名：

示例：
"Python列表推导式用法" → 技术
"周末去爬山攻略" → 生活
"季度工作总结" → 工作

现在打标签：
{text}
→""",
        "max_tokens": 16,
    },

    "qa_short": {
        "detect": ["是什么", "什么意思", "解释"],
        "template": TOOL_PREFIX + """用一句话回答：

示例：
问：API是什么意思？
答：应用程序编程接口，用于不同软件之间通信。

现在回答：
问：{text}
答：""",
        "max_tokens": 128,
    },
}

# 自动检测任务类型
SUBTASK_PATTERNS = {
    "sentiment": ["情感", "情绪", "正面", "负面"],
    "translate_en2zh": ["翻译成中文"],
    "extract_keywords": ["关键词"],
    "summarize_short": ["总结", "概括", "摘要"],
    "format_json": ["json"],
    "tag": ["打标签", "标签"],
}


# ─── 预处理/后处理 ────────────────────────────────────────────────

def preprocess_text(text: str, max_chars: int = 800) -> str:
    """输入清洗：去多余空行、截断防超长、统一编码"""
    if not text:
        return ""
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 压缩多余空行
    lines = [l for l in text.split("\n") if l.strip()]
    text = "\n".join(lines)
    # 如果文本是逗号分隔且没有换行（通常是文件列表），转成每行一个
    if "," in text and "\n" not in text:
        items = [x.strip() for x in text.split(",") if x.strip()]
        if len(items) > 3 and all(" " not in x or x.count(".") == 1 for x in items):
            text = "\n".join(items)
    # 截断
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [截断]"
    return text


def compress_prompt_tokens(prompt: str, task_type: str) -> str:
    """
    提示压缩：对长 prompt 进行 token 节约处理。
    - 去除冗余的模板说明（保留核心指令和示例）
    - 压缩长输入文本中的重复模式
    - 对简单任务类型使用更精简的指令
    """
    if not prompt or len(prompt) < 200:
        return prompt  # 短 prompt 不压缩

    # 简单任务类型的精简指令映射
    COMPACT_INSTRUCTIONS = {
        "sentiment": "判断情感，只输出正面/负面：\n{text}\n→",
        "translate_en2zh": "英译中：\n{text}\n→",
        "translate_zh2en": "中译英：\n{text}\n→",
        "tag": "提取标签：\n{text}\n→",
    }

    if task_type in COMPACT_INSTRUCTIONS:
        # 从原始 prompt 中提取 {text} 实际内容
        text_match = re.search(r'\{text\}', prompt)
        if text_match:
            # 尝试从 prompt 中提取实际文本
            pass  # 模板已 format，无法回退

    # 通用压缩：去除重复的空行和多余空格
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)
    prompt = re.sub(r' {2,}', ' ', prompt)

    # 如果 prompt 仍然很长，截断示例部分（保留最后的指令和输入）
    if len(prompt) > 1500:
        # 找到最后一个 "现在" 或 "→" 作为截断点
        for marker in ["现在", "→", "输入：", "内容："]:
            idx = prompt.rfind(marker)
            if idx > 200:  # 确保保留了一些上下文
                # 保留前100字符的指令 + 截断点后的内容
                first_line_end = prompt.index('\n') if '\n' in prompt else 100
                prompt = prompt[:first_line_end + 50] + "\n...\n" + prompt[idx:]
                return prompt
        # 没有找到标记 → 硬截断（保留首尾）
        prompt = prompt[:800] + "\n... [已压缩]\n" + prompt[-200:]

    return prompt


def postprocess_output(output: str, task_type: str = "") -> str:
    """输出清洗：去 markdown 包裹、去多余说明、格式化修复"""
    if not output:
        return ""
    out = output.strip()
    # 去 ``` 代码块
    if out.startswith("```"):
        out = out.split("\n", 1)[-1] if "\n" in out else out[3:]
        out = out.rsplit("```", 1)[0] if "```" in out else out
    # 去 markdown 段落标记
    out = out.strip("#* ")
    # 对于情感分析，只保留"正面"或"负面"
    if task_type == "sentiment" and out not in ("正面", "负面"):
        if "正面" in out or "积极" in out:
            out = "正面"
        elif "负面" in out or "消极" in out:
            out = "负面"
    return out.strip()


# ─── 任务拆解 ──────────────────────────────────────────────────────

def detect_task_type(action: str) -> str:
    """检测细粒度任务类型，返回模板 key"""
    action_lower = action.lower()
    for ttype, tpl in PROMPT_TEMPLATES.items():
        if any(d in action_lower for d in tpl.get("detect", [])):
            return ttype
    return ""


def decompose_complex_task(action: str, text: str) -> list[dict]:
    """
    将复杂任务拆解为多个简单子任务。
    例如 "分类并重命名所有文件" → [分类子任务, 重命名子任务]
    """
    action_lower = action.lower()
    subtasks = []

    # 复合任务拆解
    has_classify = any(w in action_lower for w in ["分类", "归类"])
    has_rename = any(w in action_lower for w in ["重命名", "改名"])
    has_sort = any(w in action_lower for w in ["排序", "排列"])
    has_dedup = any(w in action_lower for w in ["去重", "去重复"])
    has_count = any(w in action_lower for w in ["统计", "计数", "多少个"])

    if has_classify:
        # 检测文本中是否包含文件扩展名，决定用 file_classify 还是 general_classify
        ext_list = ["pdf", "jpg", "png", "py", "js", "css", "html", "csv", "xlsx", "docx", "txt", "gif"]
        has_extensions = False
        if text:
            text_lower = text.lower()
            has_extensions = any(f".{ext}" in text_lower for ext in ext_list)

        classify_type = "file_classify" if has_extensions else "general_classify"
        classify_action = "按文件类型分类" if has_extensions else "按类别分组"
        subtasks.append({
            "type": classify_type,
            "action": classify_action,
            "text": text,
        })
    if has_rename and text:
        subtasks.append({
            "type": "rename_suggest",
            "action": "建议新文件名",
            "text": text,
        })
    if has_sort and text:
        subtasks.append({
            "type": "sort_numbers",
            "action": "排序（按数字）",
            "text": text,
        })
    if has_dedup and text:
        subtasks.append({
            "type": "dedup",
            "action": "去重",
            "text": text,
        })
    if has_count and text:
        # 计数可以用规则处理，不需要模型
        items = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]
        subtasks.append({
            "type": "_rule_count",
            "action": "统计数量",
            "text": f"共 {len(items)} 项",
            "rule_result": f"共 {len(items)} 项",
        })

    return subtasks if subtasks else []


# ─── RLM 风格递归拆解 ──────────────────────────────────────────
# 参考 RLM (Recursive Language Models) 的 EXECUTE/RECURSE 模式：
#   当任务复杂度在"边界区间"时，不直接走云端，而是尝试拆解。
#   拆解后的子任务各自独立路由，部分可能适合本地。
#   递归终止条件：子任务已足够简单（EXECUTE）或无法再拆（RECURSE→cloud）。

RECURSE_CONNECTORS = [
    "并且", "然后", "接着", "再", "再然后", "之后",
    "同时", "以及", "并",
    "先", "首先", "其次", "最后",
    "1.", "2.", "3.",
]

# 边界区间：评分在此范围内尝试递归拆解
RECURSE_SCORE_MIN = 2.5
RECURSE_SCORE_MAX = 7.0


def _recursive_decompose(action: str, text: str, depth: int = 0) -> list[dict] | None:
    """
    递归拆解任务。返回子任务列表或 None（无法拆解）。
    每个子任务格式：{"action": str, "text": str, "route": str, "score": float}
    depth 防止无限递归（最大 3 层）。
    """
    if depth >= 3:
        return None

    action_lower = action.lower()

    # 1. 先检测已有 compound pattern（decompose_complex_task 已有的能力）
    known_subtasks = decompose_complex_task(action, text)
    if known_subtasks:
        # 注入路由信息
        for st in known_subtasks:
            if "route" not in st:
                t = Task(action=st.get("action", ""), text=text)
                dec = estimate_complexity(t)
                st["route"] = dec["route"]
                st["score"] = dec["score"]
                st["reason"] = dec["reason"]
        return known_subtasks

    # 2. 按连接词分割 action
    #   例如 "分析销售数据并生成报告" → ["分析销售数据", "生成报告"]
    parts = []
    remaining = action
    for conn in RECURSE_CONNECTORS:
        if conn in remaining:
            segments = remaining.split(conn, 1)
            if len(segments) == 2:
                # 前后都有内容才分割
                if segments[0].strip() and segments[1].strip():
                    parts = [segments[0].strip(), segments[1].strip()]
                    break

    if len(parts) < 2:
        # 尝试按逗号/顿号分割（仅当分割后每段都有动词）
        for sep in ["，", "、", ","]:
            if sep in remaining:
                segments = [s.strip() for s in remaining.split(sep) if s.strip()]
                # 需要每段 2+ 个字符（至少一个动词 + 宾语）
                valid_segments = [s for s in segments if len(s) >= 3]
                if len(valid_segments) >= 2:
                    parts = valid_segments
                    break

    if len(parts) < 2:
        return None

    # 3. 对每部分独立评估复杂度
    subtasks = []
    for part in parts:
        t = Task(action=part, text=text)
        dec = estimate_complexity(t)

        # 如果子任务仍然在边界区间，递归拆解
        if RECURSE_SCORE_MIN <= dec["score"] <= RECURSE_SCORE_MAX:
            deeper = _recursive_decompose(part, text, depth + 1)
            if deeper:
                subtasks.extend(deeper)
                continue

        subtasks.append({
            "type": "",
            "action": part,
            "text": text,
            "route": dec["route"],
            "score": dec["score"],
            "reason": dec["reason"],
        })

    return subtasks if len(subtasks) >= 2 else None


# ─── 渐进式置信度阈值（按能力自适应调整） ─────────────────────
# 参考 FrugalRoute: confidence threshold adapts per capability
# 基于历史执行数据动态调整各能力类型的路由阈值

CAPABILITY_THRESHOLD_FILE = "capability_thresholds.jsonl"

# 默认阈值（被 estimate_complexity 的 score <= 3 使用）
DEFAULT_CAPABILITY_THRESHOLD = 3.0

# 阈值调整参数
THRESHOLD_LOOKBACK = 20           # 取最近多少条记录
THRESHOLD_ADJUST_UP = 0.3         # 成功率低时上调（更保守）
THRESHOLD_ADJUST_DOWN = 0.2       # 成功率高时下调（更积极）
THRESHOLD_SUCCESS_MIN = 0.80      # 最低接受成功率
THRESHOLD_TARGET = 0.90           # 目标成功率


class CapabilityTracker:
    """
    按能力类型追踪本地执行成功率。
    数据持久化到 JSONL 文件。
    """

    def __init__(self):
        self.cache_dir = CONFIG["cache_dir"]
        self.data_file = os.path.join(self.cache_dir, CAPABILITY_THRESHOLD_FILE)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._threshold_cache = {}

    def _load_recent(self, capability: str, n: int = THRESHOLD_LOOKBACK) -> list[dict]:
        """加载某个能力最近 N 条记录"""
        records = []
        if not os.path.exists(self.data_file):
            return records
        with open(self.data_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("capability") == capability:
                        records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
        return records[-n:]

    def record(self, capability: str, success: bool, score: float = 0, task_type: str = ""):
        """记录一次执行结果"""
        entry = {
            "capability": capability,
            "success": success,
            "score": round(score, 2),
            "task_type": task_type,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.data_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # 清除缓存
        self._threshold_cache.pop(capability, None)

    def get_success_rate(self, capability: str) -> float:
        """获取某个能力的最近成功率"""
        records = self._load_recent(capability)
        if not records:
            return 1.0  # 无数据时乐观
        successes = sum(1 for r in records if r.get("success", False))
        return successes / len(records)

    def get_adjusted_threshold(self, capability: str) -> float:
        """获取调整后的阈值（默认 3.0，根据历史上调或下调）"""
        if not capability:
            return DEFAULT_CAPABILITY_THRESHOLD
        if capability in self._threshold_cache:
            return self._threshold_cache[capability]

        rate = self.get_success_rate(capability)
        threshold = DEFAULT_CAPABILITY_THRESHOLD

        if rate < THRESHOLD_SUCCESS_MIN:
            # 成功率低 → 上调阈值（更保守，更多走云端）
            steps = int((THRESHOLD_TARGET - rate) / 0.1)
            threshold += steps * THRESHOLD_ADJUST_UP
        elif rate > THRESHOLD_TARGET:
            # 成功率高 → 下调阈值（更积极，更多走本地）
            steps = int((rate - THRESHOLD_TARGET) / 0.05)
            threshold = max(1.0, threshold - steps * THRESHOLD_ADJUST_DOWN)

        threshold = round(max(1.0, min(8.0, threshold)), 1)
        self._threshold_cache[capability] = threshold
        return threshold

    def get_all_adjustments(self) -> dict:
        """返回所有能力的调整信息（用于展示）"""
        all_caps = set()
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        all_caps.add(rec.get("capability", ""))
                    except (json.JSONDecodeError, KeyError):
                        continue

        result = {}
        for cap in sorted(all_caps):
            if cap:
                rate = self.get_success_rate(cap)
                threshold = self.get_adjusted_threshold(cap)
                diff = round(threshold - DEFAULT_CAPABILITY_THRESHOLD, 1)
                result[cap] = {
                    "success_rate": round(rate, 2),
                    "threshold": threshold,
                    "adjustment": f"{'+' if diff > 0 else ''}{diff}",
                    "direction": "上调" if diff > 0 else "下调" if diff < 0 else "不变",
                }
        return result


# ─── 构建优化后的 Prompt ──────────────────────────────────────────

def rule_execute(task_type: str, text: str) -> str:
    """规则执行：对某些任务类型用 Python 规则替代模型，100% 准确"""
    items = [x.strip() for x in text.replace("\n", ",").split(",") if x.strip()]

    if task_type == "sort_numbers":
        def extract_num(s):
            # 提取冒号后的数字（支持 "商品:100件" "商品:100" "商品100件"）
            import re
            nums = re.findall(r'[：:]\s*(\d+(?:\.\d+)?)', s)
            if nums:
                return float(nums[0])
            nums = re.findall(r'(\d+(?:\.\d+)?)', s)
            if nums:
                return float(nums[-1])
            return 0
        items.sort(key=extract_num, reverse=True)
        return "\n".join(items)

    if task_type == "sort_alpha":
        items.sort()
        return "\n".join(items)

    if task_type == "dedup":
        seen = set()
        result = []
        for item in items:
            if item.lower() not in seen:
                seen.add(item.lower())
                result.append(item)
        return "\n".join(result)

    if task_type == "_count":
        return f"共 {len(items)} 项"

    return ""


def build_optimized_prompt(task_type: str, action: str, text: str, files: list) -> str:
    """根据任务类型选择最佳 prompt 模板"""
    if task_type in PROMPT_TEMPLATES:
        tpl = PROMPT_TEMPLATES[task_type]
        # 如果是规则执行，返回空
        if tpl.get("rule_based"):
            return ""
        content = text or action
        # 有目标提取特殊处理
        target = ""
        for kw in ["提取", "找出", "抽取出"]:
            if kw in action:
                # 尝试提取目标：提取 XXX
                parts = action.split(kw, 1)
                if len(parts) > 1:
                    target = parts[1].strip().split("从")[0].split("：")[0].strip()
                break
        prompt = tpl["template"].format(text=content, target=target)
        return prompt

    # 没有匹配模板时，自动构建带示例的泛化 prompt
    prompt = TOOL_PREFIX + action + "\n"
    if text:
        prompt += "\n内容：\n" + text + "\n"
    if files:
        prompt += "\n文件：\n" + "\n".join(f"  {f}" for f in files) + "\n"
    prompt += "\n只输出结果，不要解释。"
    return prompt


def get_max_tokens(task_type: str) -> int:
    """根据任务类型获取合适的 max_tokens"""
    if task_type in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[task_type].get("max_tokens", 256)
    return 512


# ─── 输出验证 + 云端降级 ─────────────────────────────────────────
# 参考 EdgeRouteAI 模式：本地输出质量差时自动升级到云端

FAILURE_SIGNALS = [
    "抱歉", "对不起", "无法", "不能", "不懂", "不明白", "不知道",
    "作为AI", "作为语言模型", "作为一个AI",
    "我无法", "我不能", "我不确定", "我不清楚",
    "超出", "不在我的", "没有足够信息",
    "error", "Error", "ERROR",
    "undefined", "null", "None",  # 模型吐出代码中的空值
]

# 各种任务类型的最小期望输出长度
MIN_OUTPUT_LENGTH = {
    "general_classify": 10,
    "file_classify": 20,
    "sentiment": 2,
    "translate_en2zh": 2,
    "translate_zh2en": 2,
    "extract_keywords": 3,
    "extract_info": 3,
    "summarize_short": 10,
    "format_json": 5,
    "sort_numbers": 3,
    "sort_alpha": 3,
    "dedup": 3,
    "clean_data": 5,
    "rename_suggest": 10,
    "tag": 2,
    "qa_short": 5,
}


def validate_local_output(output: str, task_type: str = "") -> dict:
    """
    验证本地模型输出质量。
    返回 {"valid": bool, "reason": str, "signals": list}
    """
    if not output or not output.strip():
        return {"valid": False, "reason": "空输出", "signals": ["empty"]}

    signals = []

    # 1. 最小长度检查
    min_len = MIN_OUTPUT_LENGTH.get(task_type, 3)
    if len(output.strip()) < min_len:
        signals.append(f"输出过短({len(output.strip())}<{min_len})")

    # 2. 失败信号词检测
    for signal in FAILURE_SIGNALS:
        if signal in output:
            signals.append(f"含失败信号词: {signal[:10]}")
            break

    # 3. 任务特定检查
    if task_type == "sentiment":
        if output not in ("正面", "负面"):
            signals.append(f"情感分析输出应为'正面'或'负面'，实际: {output[:10]}")
    elif task_type == "tag":
        valid_tags = {"技术", "生活", "工作", "学习", "娱乐", "其他"}
        if output not in valid_tags:
            signals.append(f"标签超出可选范围: {output[:10]}")
    elif task_type in ("translate_en2zh", "translate_zh2en"):
        # 翻译结果不应包含原文（复读）
        pass

    # 4. 输出质量问题（重复、填充、无意义）
    words = output.split()
    if len(words) >= 3:
        repeat_count = sum(1 for i in range(len(words)-2)
                          if words[i] == words[i+1] == words[i+2])
        if repeat_count > 0:
            signals.append(f"输出含重复内容({repeat_count}处)")

    filler_count = sum(output.count(w) for w in ["嗯", "这个", "那个", "就是", "然后", "其实"])
    if filler_count > 3:
        signals.append(f"填充词过多({filler_count}个)")

    is_valid = len(signals) == 0
    reason = "通过验证" if is_valid else "; ".join(signals[:3])
    return {"valid": is_valid, "reason": reason, "signals": signals}


# ─── 本地后端（增强版）───────────────────────────────────────────────

def call_ollama(prompt: str, model: str = None, max_tokens: int = None) -> dict:
    import requests

    model = model or CONFIG["local_model"]
    max_tokens = max_tokens or CONFIG["local_max_tokens"]
    start = time.time()
    try:
        resp = requests.post(
            f"{CONFIG['ollama_base']}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = int((time.time() - start) * 1000)
        result = {
            "text": data["response"].strip(),
            "tokens_input": data.get("prompt_eval_count", 0),
            "tokens_output": data.get("eval_count", 0),
            "time_ms": elapsed,
        }
        # 更新模型统计
        try:
            registry = get_model_registry()
            registry.update_after_call(
                model, success=True, latency_ms=elapsed,
                tokens_in=result["tokens_input"], tokens_out=result["tokens_output"]
            )
        except Exception:
            pass
        return result
    except Exception as e:
        # 记录失败
        try:
            registry = get_model_registry()
            registry.update_after_call(model, success=False, latency_ms=0)
        except Exception:
            pass
        raise


# ─── 云端后端 ───────────────────────────────────────────────────────

def call_cloud_api(prompt: str, text: str = "") -> dict:
    if not CONFIG["cloud_api_key"]:
        return {
            "text": "[云端未配置] 设置 CLOUD_API_KEY 环境变量启用云端路由",
            "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
        }

    # 熔断检查：连续失败超过阈值，暂时跳过云端
    now = time.time()
    if _circuit_breaker["open_until"] > now:
        remaining = int(_circuit_breaker["open_until"] - now)
        return {
            "text": f"[云端熔断中] 连续失败{_circuit_breaker['failures']}次，{remaining}秒后重试",
            "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
            "circuit_open": True,
        }

    import requests
    messages = [{"role": "user", "content": prompt}]
    if text:
        messages.append({"role": "user", "content": text})

    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            resp = requests.post(
                f"{CONFIG['cloud_api_url']}/v1/chat/completions",
                headers={"Authorization": f"Bearer {CONFIG['cloud_api_key']}"},
                json={
                    "model": CONFIG["cloud_model"],
                    "messages": messages,
                    "max_tokens": 4096,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = int((time.time() - start) * 1000)
            usage = data.get("usage", {})
            # 成功 → 重置熔断器
            _circuit_breaker["failures"] = 0
            _circuit_breaker["open_until"] = 0
            return {
                "text": data["choices"][0]["message"]["content"].strip(),
                "tokens_input": usage.get("prompt_tokens", 0),
                "tokens_output": usage.get("completion_tokens", 0),
                "time_ms": elapsed,
            }
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))  # 指数退避: 1s, 2s
                continue
            break

    # 所有重试失败 → 更新熔断器
    _circuit_breaker["failures"] += 1
    _circuit_breaker["last_failure"] = time.time()
    if _circuit_breaker["failures"] >= _circuit_breaker["max_failures"]:
        _circuit_breaker["open_until"] = time.time() + _circuit_breaker["cooldown_seconds"]

    return {
        "text": f"[云端调用失败] {last_error}",
        "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
        "error": str(last_error),
    }


# ─── 成本与日志 ────────────────────────────────────────────────────

def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1000 * CONFIG["cost_per_1k_input"]
        + output_tokens / 1000 * CONFIG["cost_per_1k_output"]
    )


def calc_savings(task: Task) -> float:
    if task.route != "local":
        return 0.0
    return round(calc_cost(task.tokens_input, task.tokens_output), 6)


def log_usage(task: Task):
    log_file = os.path.join(CONFIG["cache_dir"], "usage.jsonl")
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action": task.action[:100],
        "route": task.route,
        "model": task.model_used,
        "tokens_input": task.tokens_input,
        "tokens_output": task.tokens_output,
        "cost_saved": task.cost_saved,
        "time_ms": task.time_ms,
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def show_usage_stats() -> str:
    log_file = os.path.join(CONFIG["cache_dir"], "usage.jsonl")
    if not os.path.exists(log_file):
        return "暂无使用记录"

    total_local = total_cloud = total_cache = 0
    total_input = total_output = 0
    total_saved = 0.0
    daily_stats = {}  # date -> {local, cloud, cache, saved}

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            route = e.get("route", "")
            if route == "local":
                total_local += 1
            elif "cache" in route:
                total_cache += 1
            else:
                total_cloud += 1
            total_input += e.get("tokens_input", 0)
            total_output += e.get("tokens_output", 0)
            total_saved += e.get("cost_saved", 0)

            # 按天统计
            day = e.get("time", "")[:10]
            if day:
                if day not in daily_stats:
                    daily_stats[day] = {"local": 0, "cloud": 0, "cache": 0, "saved": 0.0}
                if route == "local":
                    daily_stats[day]["local"] += 1
                elif "cache" in route:
                    daily_stats[day]["cache"] += 1
                else:
                    daily_stats[day]["cloud"] += 1
                daily_stats[day]["saved"] += e.get("cost_saved", 0)

    # 语义缓存统计
    cache_stats = cache.stats()
    result = (
        f"TaskRouter 使用统计\n"
        f"{'='*40}\n"
        f"本地调用: {total_local} 次\n"
        f"云端调用: {total_cloud} 次\n"
        f"缓存命中: {total_cache} 次\n"
        f"总输入 tokens: {total_input:,}\n"
        f"总输出 tokens: {total_output:,}\n"
        f"累计节约成本: ${total_saved:.4f}\n"
    )
    if cache_stats["total"] > 0:
        result += (
            f"\n语义缓存:\n"
            f"  缓存条目: {cache_stats['total']}\n"
            f"  缓存节约: ${cache_stats['estimated_cost_saved']:.4f}\n"
        )

    # 每日统计（最近7天）
    if daily_stats:
        sorted_days = sorted(daily_stats.keys(), reverse=True)[:7]
        if sorted_days:
            result += f"\n每日统计 (最近{len(sorted_days)}天):\n"
            for day in sorted_days:
                d = daily_stats[day]
                total_day = d["local"] + d["cloud"] + d["cache"]
                result += (
                    f"  {day}: {total_day}次 "
                    f"(本地{d['local']}/云端{d['cloud']}/缓存{d['cache']}) "
                    f"省${d['saved']:.4f}\n"
                )

    # 熔断器状态
    if _circuit_breaker["failures"] > 0:
        now = time.time()
        if _circuit_breaker["open_until"] > now:
            remaining = int(_circuit_breaker["open_until"] - now)
            result += f"\n云端熔断器: 🔴 熔断中 (连续失败{_circuit_breaker['failures']}次, {remaining}秒后恢复)\n"
        else:
            result += f"\n云端熔断器: 🟡 已恢复 (上次连续失败{_circuit_breaker['failures']}次)\n"

    return result


# ═══════════════════════════════════════════════════════════════════
# 蒸馏系统 — 从云端响应中学习，持续提升本地模型准确率
# ═══════════════════════════════════════════════════════════════════
#
# 原理：每次请求转到云端（或本地失败后云端修正），
# 记录 (prompt, response, task_type) 作为训练对。
# Judge 评判质量后，优质样本被提取为 few-shot 示例，
# 注入到 PROMPT_TEMPLATES 中，使本地模型越来越强。
#
# 相比 FrugalRoute 简化处：
#   无 LoRA 微调 → 改为增强提示词模板
#   无 SQLite → 使用 JSONL (与 usage 一致)
#   无 TruthKeeper 完整溯源 → 简化版本号追踪

# 蒸馏对状态
PAIR_HYPOTHESIS = "hypothesis"   # 新采集，未验证
PAIR_SUPPORTED = "supported"     # 已验证，可用于训练
PAIR_CONTESTED = "contested"     # 质量低或矛盾
PAIR_OUTDATED = "outdated"       # 模型版本变了，数据过时

# Judge 置信度阈值
JUDGE_HIGH_THRESHOLD = 0.9
JUDGE_MODERATE_THRESHOLD = 0.5

# 能力类型 — 每种能力可独立配置 Judge 策略
CAPABILITY_TYPES = [
    "classification",    # 分类/打标签
    "translation",       # 翻译
    "extraction",        # 提取/关键词
    "summarization",     # 概括
    "formatting",        # 格式化
    "qa",                # 问答
    "reasoning",         # 推理（始终走 Judge）
]

# 默认跳过 Judge 的能力（性能优先）
SKIP_JUDGE_CAPABILITIES = {"formatting", "extraction"}

# 任务类型 → 能力映射
TASK_TO_CAPABILITY = {
    "general_classify": "classification",
    "file_classify": "classification",
    "sentiment": "classification",
    "tag": "classification",
    "translate_en2zh": "translation",
    "translate_zh2en": "translation",
    "extract_keywords": "extraction",
    "extract_info": "extraction",
    "summarize_short": "summarization",
    "format_json": "formatting",
    "clean_data": "formatting",
    "dedup": "formatting",
    "sort_numbers": "formatting",
    "sort_alpha": "formatting",
    "rename_suggest": "formatting",
    "qa_short": "qa",
}


@dataclass
class DistillationPair:
    """单个训练对"""
    prompt: str                          # 原始 prompt
    response: str                        # 模型输出
    task_type: str = ""                  # 任务类型 (模板 key)
    capability: str = ""                 # 能力分类
    route: str = "cloud"                 # 来源: cloud / local_fallback
    action: str = ""                     # 任务描述
    epistemic_state: str = PAIR_HYPOTHESIS  # 状态
    quality_score: float = 0.0           # Judge 评分 0-1
    judge_reason: str = ""               # Judge 评判理由
    model_used: str = ""                 # 产生该响应的模型
    model_version: str = ""              # 模型版本 (简化用时间戳)
    version_tag: str = ""                # 模型版本标签，用于级联失效
    time: str = ""                       # 采集时间
    pair_id: str = ""                    # 唯一 ID
    failure_type: str = ""               # 仅 correction pair: hallucination/format/reasoning/knowledge
    local_response: str = ""             # 仅 correction pair: 本地模型的错误输出

    def __post_init__(self):
        if not self.pair_id:
            raw = f"{self.prompt[:50]}{self.response[:50]}{time.time()}"
            self.pair_id = hashlib.md5(raw.encode()).hexdigest()[:12]
        if not self.time:
            self.time = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.version_tag:
            self.version_tag = f"{self.model_used or 'unknown'}@{self.time[:7] or 'v1'}"
        if not self.capability and self.task_type:
            self.capability = TASK_TO_CAPABILITY.get(self.task_type, "reasoning")


class DistillationStore:
    """蒸馏数据存储 — JSONL 文件 + 索引"""

    def __init__(self):
        self.cache_dir = CONFIG["cache_dir"]
        self.pairs_file = os.path.join(self.cache_dir, "distillation.jsonl")
        self.stats_file = os.path.join(self.cache_dir, "distillation_stats.json")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_all(self) -> list[dict]:
        """加载所有训练对"""
        pairs = []
        if not os.path.exists(self.pairs_file):
            return pairs
        with open(self.pairs_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pairs

    def _save_all(self, pairs: list[dict]):
        """覆写所有训练对"""
        with open(self.pairs_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def add_pair(self, pair: DistillationPair):
        """添加新训练对"""
        entry = asdict(pair)
        with open(self.pairs_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_pairs(self,
                  state: str = None,
                  capability: str = None,
                  task_type: str = None,
                  min_score: float = 0.0,
                  limit: int = 0) -> list[dict]:
        """按条件查询训练对"""
        pairs = self._load_all()
        if state:
            pairs = [p for p in pairs if p.get("epistemic_state") == state]
        if capability:
            pairs = [p for p in pairs if p.get("capability") == capability]
        if task_type:
            pairs = [p for p in pairs if p.get("task_type") == task_type]
        if min_score > 0:
            pairs = [p for p in pairs if p.get("quality_score", 0) >= min_score]
        if limit > 0:
            pairs = pairs[:limit]
        return pairs

    def update_pair_state(self, pair_id: str, new_state: str, score: float = None, reason: str = ""):
        """更新单个训练对的状态"""
        pairs = self._load_all()
        for p in pairs:
            if p.get("pair_id") == pair_id:
                p["epistemic_state"] = new_state
                if score is not None:
                    p["quality_score"] = score
                if reason:
                    p["judge_reason"] = reason
                break
        self._save_all(pairs)

    def cascade_invalidate(self, version_tag: str):
        """当模型版本变化时，使相关训练对失效"""
        pairs = self._load_all()
        count = 0
        for p in pairs:
            if p.get("version_tag", "").startswith(version_tag.split("@")[0]):
                if p["epistemic_state"] == PAIR_SUPPORTED:
                    p["epistemic_state"] = PAIR_OUTDATED
                    p["judge_reason"] = f"模型版本变更: {version_tag}"
                    count += 1
        self._save_all(pairs)
        return count

    def get_health(self) -> dict:
        """蒸馏系统健康状态"""
        pairs = self._load_all()
        by_state = {}
        by_capability = {}
        for p in pairs:
            s = p.get("epistemic_state", "unknown")
            by_state[s] = by_state.get(s, 0) + 1
            cap = p.get("capability", "unknown")
            if cap not in by_capability:
                by_capability[cap] = {"total": 0, "supported": 0}
            by_capability[cap]["total"] += 1
            if s == PAIR_SUPPORTED:
                by_capability[cap]["supported"] += 1

        total = len(pairs)
        supported = by_state.get(PAIR_SUPPORTED, 0)
        hypothesis = by_state.get(PAIR_HYPOTHESIS, 0)
        contested = by_state.get(PAIR_CONTESTED, 0)
        outdated = by_state.get(PAIR_OUTDATED, 0)

        return {
            "total_pairs": total,
            "supported": supported,
            "hypothesis": hypothesis,
            "contested": contested,
            "outdated": outdated,
            "health_score": round(supported / max(total, 1) * 100, 1),
            "by_capability": by_capability,
        }

    def get_eligible_pairs(self, capability: str = None,
                           min_score: float = 0.7,
                           max_age_days: int = 90) -> list[dict]:
        """获取符合训练条件的对（SUPPORTED + 质量分达标 + 时效内）"""
        cutoff = time.time() - max_age_days * 86400
        pairs = self._load_all()
        result = []
        for p in pairs:
            if p.get("epistemic_state") != PAIR_SUPPORTED:
                continue
            if p.get("quality_score", 0) < min_score:
                continue
            if capability and p.get("capability") != capability:
                continue
            # 检查时效
            t = p.get("time", "")
            if t:
                try:
                    t_ts = time.mktime(time.strptime(t, "%Y-%m-%dT%H:%M:%S"))
                    if t_ts < cutoff:
                        continue
                except (ValueError, OSError):
                    pass
            result.append(p)
        return result

    def export_jsonl(self, output_path: str = None) -> str:
        """导出可用的训练数据为 JSONL（instruction + output 格式）"""
        pairs = self.get_eligible_pairs()
        if not output_path:
            output_path = os.path.join(self.cache_dir, "distill_export.jsonl")
        with open(output_path, "w") as f:
            for p in pairs:
                f.write(json.dumps({
                    "instruction": p.get("prompt", ""),
                    "output": p.get("response", ""),
                    "task_type": p.get("task_type", ""),
                    "capability": p.get("capability", ""),
                }, ensure_ascii=False) + "\n")
        return output_path


# ─── 语义缓存（trigram Jaccard 相似度） ────────────────────────────
# 无需向量数据库，仅靠字符 trigram 重叠度检测相似 prompt
# 参考：Semantic Cache via N-gram Jaccard Similarity

class SemanticCache:
    """
    轻量语义缓存 — 基于字符 trigram Jaccard 相似度。
    为每个 action+text 缓存 local/cloud 的输出结果。
    命中时直接返回，完全节省 token。
    """

    CACHE_VERSION = "v1"

    def __init__(self, similarity_threshold: float = 0.85):
        self.cache_dir = CONFIG["cache_dir"]
        self.cache_file = os.path.join(self.cache_dir, "semantic_cache.jsonl")
        self.threshold = similarity_threshold
        os.makedirs(self.cache_dir, exist_ok=True)
        self._cache = self._load()

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化：去空格、统一标点、小写"""
        t = text.strip().lower()
        t = re.sub(r'\s+', '', t)          # 去所有空格
        t = t.replace('，', ',').replace('、', ',').replace('；', ';')  # 统一分隔符
        t = re.sub(r',+', ',', t)          # 压缩连续逗号
        t = t.strip(',').strip('.')        # 去首尾标点
        return t

    def _trigrams(self, text: str) -> set:
        """生成字符 trigram 集合（已归一化）"""
        normalized = self._normalize(text)
        if len(normalized) < 3:
            return {normalized}
        return set(normalized[i:i+3] for i in range(len(normalized) - 2))

    def _jaccard(self, a: set, b: set) -> float:
        """Jaccard 相似度"""
        union = len(a | b)
        if union == 0:
            return 1.0
        return len(a & b) / union

    def _make_key(self, action: str, text: str) -> str:
        """生成缓存键（结合 action 和 text 前 100 字符）"""
        raw = f"{action.strip().lower()}|{text.strip().lower()[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _normalized_key(self, action: str, text: str) -> str:
        """归一化后的备用键（去空格标点后）"""
        raw = f"{self._normalize(action)}|{self._normalize(text)[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _load(self) -> dict:
        """从磁盘加载缓存"""
        cache = {}
        if not os.path.exists(self.cache_file):
            return cache
        with open(self.cache_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    cache[entry["key"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
        return cache

    def _save(self):
        """持久化缓存到磁盘"""
        with open(self.cache_file, "w") as f:
            for entry in self._cache.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _is_expired(self, entry: dict) -> bool:
        """检查缓存条目是否已过期"""
        created = entry.get("created_ts", 0)
        ttl_hours = entry.get("ttl_hours", CONFIG["cache_ttl_hours"]["default"])
        if created <= 0:
            return False  # 旧条目无时间戳，不过期
        return (time.time() - created) > ttl_hours * 3600

    def get(self, action: str, text: str = "") -> dict | None:
        """
        查找缓存命中。先精确匹配 key，再归一化 key，最后模糊匹配 trigram 相似度。
        支持 TTL 过期检查。
        返回缓存条目或 None。
        """
        # 精确匹配
        exact_key = self._make_key(action, text)
        if exact_key in self._cache:
            entry = self._cache[exact_key]
            if self._is_expired(entry):
                del self._cache[exact_key]
                self._save()
            else:
                entry["match_type"] = "exact"
                return entry

        # 归一化匹配（去空格标点后）
        norm_key = self._normalized_key(action, text)
        if norm_key in self._cache:
            entry = self._cache[norm_key]
            if self._is_expired(entry):
                del self._cache[norm_key]
                self._save()
            else:
                entry["match_type"] = "normalized"
                return entry

        # 模糊匹配（trigram Jaccard）
        query = self._normalize(f"{action} {text[:100]}")
        if len(query) < 6:
            return None
        query_tri = self._trigrams(f"{action} {text[:100]}")

        best_score = 0
        best_entry = None
        best_key = None
        for key, entry in self._cache.items():
            if self._is_expired(entry):
                continue
            entry_query = self._normalize(entry.get("query", ""))
            if not entry_query:
                continue
            entry_tri = self._trigrams(entry.get("query", ""))
            score = self._jaccard(query_tri, entry_tri)
            if score > best_score:
                best_score = score
                best_entry = entry
                best_key = key

        if best_score >= self.threshold and best_entry:
            best_entry["match_type"] = f"fuzzy({best_score:.2f})"
            return best_entry

        return None

    def set(self, action: str, text: str, result: dict, task_type: str = "", route: str = ""):
        """存入缓存（含去重 + TTL）"""
        key = self._make_key(action, text)
        # 已存在则跳过
        if key in self._cache:
            return

        ttl_map = CONFIG["cache_ttl_hours"]
        ttl = ttl_map.get(task_type, ttl_map.get("default", 48))

        query = f"{action.strip().lower()} {text.strip().lower()[:100]}"
        self._cache[key] = {
            "key": key,
            "normalized_key": self._normalized_key(action, text),
            "query": query,
            "output": result.get("text", ""),
            "task_type": task_type,
            "route": route,
            "tokens_input": result.get("tokens_input", 0),
            "tokens_output": result.get("tokens_output", 0),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "created_ts": time.time(),
            "ttl_hours": ttl,
            "version": self.CACHE_VERSION,
        }
        # 限制缓存大小（最多 500 条）
        if len(self._cache) > 500:
            # 移除最旧的 100 条
            sorted_keys = sorted(self._cache.keys(),
                                key=lambda k: self._cache[k].get("time", ""))
            for old_key in sorted_keys[:100]:
                del self._cache[old_key]

        self._save()

    def cleanup_expired(self) -> int:
        """清理过期缓存条目，返回清理数量"""
        expired_keys = [k for k, v in self._cache.items() if self._is_expired(v)]
        for k in expired_keys:
            del self._cache[k]
        if expired_keys:
            self._save()
        return len(expired_keys)

    def stats(self) -> dict:
        """缓存统计"""
        # 先清理过期条目
        cleaned = self.cleanup_expired()
        total = len(self._cache)
        if total == 0:
            return {"total": 0, "estimated_tokens_saved": 0, "expired_cleaned": cleaned}
        total_input = sum(e.get("tokens_input", 0) for e in self._cache.values())
        total_output = sum(e.get("tokens_output", 0) for e in self._cache.values())
        total_saved = sum(
            calc_cost(e.get("tokens_input", 0), e.get("tokens_output", 0))
            for e in self._cache.values()
        )
        return {
            "total": total,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "estimated_cost_saved": round(total_saved, 6),
            "expired_cleaned": cleaned,
        }


# ─── Judge 裁判系统 ──────────────────────────────────────────────

class Judge:
    """训练对质量评判器

    两阶段评估：
      第一阶段: 启发式规则（始终可用，~0ms）
      第二阶段: LLM 裁判（仅在启发式结果模棱两可时触发）
    """

    def __init__(self):
        self.local_model = CONFIG["local_model"]

    def evaluate(self, pair: DistillationPair) -> tuple[float, str]:
        """返回 (score 0-1, reason)"""
        # 第一阶段：启发式评估
        score, reason = self._heuristic_eval(pair)
        pair.quality_score = score
        pair.judge_reason = reason

        # 如果能力类型跳过 Judge，直接返回
        if pair.capability in SKIP_JUDGE_CAPABILITIES:
            if score >= JUDGE_HIGH_THRESHOLD:
                pair.epistemic_state = PAIR_SUPPORTED
            else:
                pair.epistemic_state = PAIR_HYPOTHESIS
            return score, reason

        # 如果置信度在中等区间，启动第二阶段 LLM Judge
        if JUDGE_MODERATE_THRESHOLD <= score < JUDGE_HIGH_THRESHOLD:
            llm_score, llm_reason = self._llm_judge(pair)
            # 混合分数：40% 启发式 + 60% LLM
            score = round(score * 0.4 + llm_score * 0.6, 4)
            reason = f"heuristic({reason}) + llm({llm_reason})"
            pair.quality_score = score
            pair.judge_reason = reason

        # 确定状态
        if score >= JUDGE_HIGH_THRESHOLD:
            pair.epistemic_state = PAIR_SUPPORTED
        elif score < JUDGE_MODERATE_THRESHOLD:
            pair.epistemic_state = PAIR_CONTESTED
        else:
            pair.epistemic_state = PAIR_HYPOTHESIS

        return score, reason

    def _heuristic_eval(self, pair: DistillationPair) -> tuple[float, str]:
        """启发式评估：基于长度、格式、任务特定规则"""
        response = pair.response.strip()
        prompt = pair.prompt.strip()
        task_type = pair.task_type

        if not response:
            return 0.0, "空响应"

        score = 0.8  # 基础分
        penalties = []

        # 1. 长度检查
        resp_len = len(response)
        if resp_len < 3:
            penalties.append("响应过短")
            score -= 0.5
        elif resp_len > 4000:
            penalties.append("响应过长")
            score -= 0.2

        # 2. 响应中不应包含 prompt 本身（复读）
        if prompt[:50] in response:
            penalties.append("复读 prompt")
            score -= 0.3

        # 3. 任务特定检查
        if task_type == "sentiment":
            if response not in ("正面", "负面"):
                if "正面" in response or "积极" in response:
                    score -= 0.1
                elif "负面" in response or "消极" in response:
                    score -= 0.1
                else:
                    penalties.append("情感分析输出格式错误")
                    score -= 0.4
        elif task_type == "tag":
            valid_tags = {"技术", "生活", "工作", "学习", "娱乐", "其他"}
            if response not in valid_tags:
                penalties.append("标签超出可选范围")
                score -= 0.3
        elif task_type in ("translate_en2zh", "translate_zh2en"):
            if len(response) < 2:
                penalties.append("翻译结果过短")
                score -= 0.3
        elif task_type == "format_json":
            if not response.startswith("{"):
                penalties.append("JSON 格式可能不正确")
                score -= 0.2
        elif task_type == "extract_keywords":
            if "," not in response and "、" not in response:
                if len(response) > 10:
                    penalties.append("关键词可能不是逗号分隔")
                    score -= 0.2
        elif task_type in ("file_classify", "general_classify"):
            lines = response.split("\n")
            if len(lines) < 2:
                penalties.append("分类结果行数过少")
                score -= 0.2
            # 检查是否遗漏了输入项
            input_items = set()
            for sep in [",", "\n"]:
                for item in prompt.split(sep):
                    item = item.strip()
                    if item and len(item) > 2 and "." in item:
                        input_items.add(item.rsplit(".", 1)[0])
            if input_items:
                found_count = sum(1 for i in input_items if i in response)
                if found_count < len(input_items) * 0.5:
                    penalties.append("可能遗漏了部分文件")
                    score -= 0.3

        # 4. 通用质量信号
        if "抱歉" in response or "对不起" in response:
            penalties.append("模型在道歉")
            score -= 0.2
        if "作为AI" in response or "作为语言模型" in response:
            penalties.append("模型在自述身份")
            score -= 0.15
        if response.startswith("```"):
            penalties.append("含 markdown 包裹")
            score -= 0.1

        # 5. 高级质量检测
        # 重复词检测（如 "好的好的好的"）
        words = response.split()
        if len(words) >= 3:
            repeat_patterns = sum(1 for i in range(len(words)-2)
                                  if words[i] == words[i+1] == words[i+2])
            if repeat_patterns > 0:
                penalties.append(f"检测到重复({repeat_patterns}处)")
                score -= 0.15 * min(repeat_patterns, 3)

        # 无意义的填充词
        filler_words = ["嗯", "这个", "那个", "就是", "然后", "其实"]
        filler_count = sum(response.count(w) for w in filler_words)
        if filler_count > 3:
            penalties.append(f"填充词过多({filler_count}个)")
            score -= 0.1

        # 响应与输入长度严重不匹配（太长或太短）
        if len(prompt) > 50 and len(response) > 0:
            ratio = len(response) / len(prompt)
            if ratio > 10 and len(response) > 1000:
                penalties.append("响应远超输入长度")
                score -= 0.15
            elif ratio < 0.05 and len(prompt) > 100:
                penalties.append("响应过短")
                score -= 0.2

        # 扣分但保底
        score = max(0.0, min(1.0, score))
        reason = "; ".join(penalties) if penalties else "启发式规则通过"
        return score, reason

    def _llm_judge(self, pair: DistillationPair) -> tuple[float, str]:
        """第二阶段：用小模型做质量评判"""
        if not pair.response.strip():
            return 0.0, "LLM Judge: 空响应"

        judge_prompt = (
            "你是一个质量评估器。判断以下响应对输入来说是否准确、完整、格式正确。"
            "只输出一个数字 0-100 表示质量分，以及简短理由。\n\n"
            f"任务类型: {pair.task_type}\n"
            f"输入: {pair.prompt[:200]}\n"
            f"响应: {pair.response[:500]}\n\n"
            "评分:"
        )
        try:
            import requests
            resp = requests.post(
                f"{CONFIG['ollama_base']}/api/generate",
                json={
                    "model": CONFIG["local_model"],
                    "prompt": judge_prompt,
                    "stream": False,
                    "options": {"num_predict": 64},
                },
                timeout=30,
            )
            data = resp.json()
            text = data.get("response", "").strip()

            # 提取数字分数
            nums = re.findall(r'(\d{1,3})', text)
            if nums:
                score = min(100, max(0, int(nums[0]))) / 100.0
            else:
                score = 0.5  # 无法解析时取中间值

            reason = text[:100].replace("\n", " ")
            return score, f"LLM评分={score:.2f}: {reason}"
        except Exception as e:
            return 0.5, f"LLM Judge 失败: {e}"


# ─── 蒸馏主流程 ──────────────────────────────────────────────────

store = DistillationStore()
judge = Judge()
cache = SemanticCache()
cap_tracker = CapabilityTracker()


def collect_pair(task: Task, response_text: str = None,
                 is_correction: bool = False,
                 local_response: str = None,
                 failure_type: str = "") -> DistillationPair:
    """采集训练对（云端响应 / 本地失败云端修正）"""
    if not response_text:
        return None

    task_type = detect_task_type(task.action)
    capability = TASK_TO_CAPABILITY.get(task_type, "reasoning")
    model_used = task.model_used or CONFIG.get("cloud_model", "unknown")

    pair = DistillationPair(
        prompt=task.action[:500],
        response=response_text[:2000],
        task_type=task_type,
        capability=capability,
        route="cloud" if not is_correction else "local_fallback",
        action=task.action[:200],
        model_used=model_used,
        model_version=model_used,
    )

    if is_correction:
        pair.failure_type = failure_type
        pair.local_response = (local_response or "")[:1000]

    store.add_pair(pair)
    return pair


def run_distillation(dry_run: bool = False) -> dict:
    """执行蒸馏流程：评判 HYPOTHESIS 对 → 提取优质样本 → 报告结果"""
    pairs = store.get_pairs(state=PAIR_HYPOTHESIS)
    if not pairs:
        return {"status": "no_pairs", "message": "无待评判的训练对"}

    promoted = 0
    contested = 0
    kept_hypothesis = 0
    by_capability = {}

    for p in pairs:
        pair_obj = DistillationPair(**p)
        score, reason = judge.evaluate(pair_obj)

        if not dry_run:
            store.update_pair_state(
                pair_obj.pair_id,
                pair_obj.epistemic_state,
                score=score,
                reason=reason,
            )

        if pair_obj.epistemic_state == PAIR_SUPPORTED:
            promoted += 1
        elif pair_obj.epistemic_state == PAIR_CONTESTED:
            contested += 1
        else:
            kept_hypothesis += 1

        cap = pair_obj.capability
        if cap not in by_capability:
            by_capability[cap] = {"judged": 0, "promoted": 0}
        by_capability[cap]["judged"] += 1
        if pair_obj.epistemic_state == PAIR_SUPPORTED:
            by_capability[cap]["promoted"] += 1

    # 为每种能力提取最佳样本
    best_examples = {}
    if not dry_run and promoted > 0:
        best_examples = _extract_best_examples()

    health = store.get_health()
    return {
        "status": "completed",
        "judged": len(pairs),
        "promoted": promoted,
        "contested": contested,
        "kept_hypothesis": kept_hypothesis,
        "by_capability": by_capability,
        "best_examples": best_examples,
        "health": health,
        "dry_run": dry_run,
    }


def _extract_best_examples(top_n: int = 3) -> dict:
    """从 SUPPORTED 对中提取每种能力的最佳 few-shot 示例"""
    examples = {}
    for cap in CAPABILITY_TYPES:
        pairs = store.get_eligible_pairs(capability=cap, min_score=0.8)
        if pairs:
            # 按质量分排序取 top_n
            pairs.sort(key=lambda p: p.get("quality_score", 0), reverse=True)
            examples[cap] = [
                {
                    "prompt": p.get("prompt", "")[:100],
                    "response": p.get("response", "")[:200],
                    "score": p.get("quality_score", 0),
                    "task_type": p.get("task_type", ""),
                }
                for p in pairs[:top_n]
            ]
    return examples


def _get_dynamic_examples(task_type: str, max_examples: int = 2) -> list[dict]:
    """
    从蒸馏池中获取与当前任务类型相关的动态 few-shot 示例。
    优先选择多样性高的样本（不同 prompt 内容）。
    """
    capability = TASK_TO_CAPABILITY.get(task_type, "")
    if not capability:
        return []

    pairs = store.get_eligible_pairs(capability=capability, min_score=0.75)
    if not pairs:
        return []

    # 按质量分降序
    pairs.sort(key=lambda p: p.get("quality_score", 0), reverse=True)

    # 多样性选择：尽可能覆盖不同的原始 action
    selected = []
    seen_actions = set()
    for p in pairs:
        action = p.get("action", "")[:30]
        if action not in seen_actions or len(selected) < max_examples // 2:
            selected.append(p)
            seen_actions.add(action)
        if len(selected) >= max_examples:
            break

    # 如果多样性选择不够，补最高分的
    if len(selected) < max_examples:
        for p in pairs:
            if p not in selected:
                selected.append(p)
            if len(selected) >= max_examples:
                break

    return selected


def enrich_prompt_with_examples(base_prompt: str, task_type: str, user_text: str) -> str:
    """
    将动态 few-shot 示例注入到 prompt 末尾（在用户内容之前）。
    仅当有高质量蒸馏样本时追加，不影响已有模板示例。
    """
    examples = _get_dynamic_examples(task_type, max_examples=2)
    if not examples:
        return base_prompt

    # 构建额外的示例块
    extra_lines = ["", "# 额外参考示例（来自历史高质量响应）:"]
    for ex in examples:
        ex_prompt = ex.get("action", "") or ex.get("prompt", "")
        ex_response = ex.get("response", "")
        if ex_prompt and ex_response:
            extra_lines.append(f"# 输入: {ex_prompt[:80]}")
            extra_lines.append(f"# 输出: {ex_response[:120]}")
    extra_lines.append("")

    # 在 base_prompt 的末尾、用户内容之前注入
    # 找到 {text} 占位符，在它之前插入额外示例
    if "{text}" in base_prompt:
        return base_prompt.replace("{text}", "\n".join(extra_lines) + "\n{text}")
    return base_prompt + "\n".join(extra_lines)


def auto_collect_on_cloud(task: Task, result: dict):
    """云端调用后自动采集训练对"""
    try:
        pair = collect_pair(task, response_text=result.get("text", ""))
        if pair:
            # 对新采集的对做快速启发式评判
            score, reason = judge._heuristic_eval(pair)
            if score >= JUDGE_HIGH_THRESHOLD:
                store.update_pair_state(pair.pair_id, PAIR_SUPPORTED, score=score, reason=reason)
            elif score < JUDGE_MODERATE_THRESHOLD:
                store.update_pair_state(pair.pair_id, PAIR_CONTESTED, score=score, reason=reason)
            # 中等置信度保留 HYPOTHESIS，等 --distill 批量处理
    except Exception:
        pass  # 采集失败不影响主流程


def auto_collect_on_local_failure(task: Task, local_output: str, cloud_output: str, failure_type: str = "reasoning"):
    """本地失败→云端修正后采集 correction pair"""
    try:
        pair = collect_pair(
            task,
            response_text=cloud_output,
            is_correction=True,
            local_response=local_output,
            failure_type=failure_type,
        )
    except Exception:
        pass


def show_distill_stats(verbose: bool = False) -> str:
    """蒸馏系统状态仪表盘"""
    health = store.get_health()
    lines = [
        f"蒸馏系统健康状态",
        f"{'='*50}",
        f"总训练对: {health['total_pairs']}",
        f"  ✅ SUPPORTED:   {health['supported']} (可训练)",
        f"  ⏳ HYPOTHESIS:  {health['hypothesis']} (待评判)",
        f"  ❌ CONTESTED:   {health['contested']} (低质量)",
        f"  ⌛ OUTDATED:    {health['outdated']} (已过期)",
        f"健康度: {health['health_score']}%",
    ]

    if verbose and health.get("by_capability"):
        lines.append("")
        lines.append("按能力分布:")
        for cap, stats in sorted(health["by_capability"].items()):
            pct = round(stats["supported"] / max(stats["total"], 1) * 100, 1)
            lines.append(f"  {cap:20s}  {stats['total']:4d} 对  (SUPPORTED: {stats['supported']}, {pct}%)")

    # 显示最佳样本
    if verbose:
        examples = _extract_best_examples()
        if examples:
            lines.append("")
            lines.append("最佳 few-shot 候选:")
            for cap, exs in examples.items():
                if exs:
                    lines.append(f"  [{cap}] 最高分样本: {exs[0]['score']:.2f}")
                    lines.append(f"    prompt: {exs[0]['prompt']}")
                    lines.append(f"    response: {exs[0]['response']}")

    return "\n".join(lines)


def export_distillation(output_path: str = None) -> str:
    """导出可训练数据"""
    path = store.export_jsonl(output_path)
    pairs = store.get_eligible_pairs()
    return f"已导出 {len(pairs)} 条训练对到: {path}"


# ─── 在 run_task 中集成蒸馏采集 ─────────────────────────────────


# ─── 主逻辑 ────────────────────────────────────────────────────────

def run_task(task: Task, force_route: str = "") -> Task:
    if force_route:
        task.route = force_route
        task.model_used = (
            CONFIG["local_model"]
            if force_route == "local"
            else CONFIG["cloud_model"]
        )
    else:
        decision = estimate_complexity(task)
        task.route = decision["route"]
        task.model_used = (
            CONFIG["local_model"]
            if decision["route"] == "local"
            else CONFIG["cloud_model"]
        )

    # ── 语义缓存查找 ──
    cached = cache.get(task.action, task.text)
    if cached:
        task.output = cached["output"]
        task.route = f"cache({cached.get('match_type', 'hit')})"
        task.model_used = f"cache_{cached.get('route', 'unknown')}"
        task.tokens_input = 0
        task.tokens_output = 0
        task.time_ms = 0
        task.cost_saved = round(
            calc_cost(cached.get("tokens_input", 0), cached.get("tokens_output", 0)),
            6,
        )
        log_usage(task)
        return task

    if task.route == "local":
        # ── 本地任务：使用优化提示词系统 ──
        clean_text = preprocess_text(task.text or "")
        clean_action = preprocess_text(task.action, max_chars=200)

        # 1. 尝试拆解复合任务
        subtasks = decompose_complex_task(clean_action, clean_text)
        if not subtasks:
            # 没有已知模式 → 尝试通用递归拆解
            subtasks = _recursive_decompose(clean_action, clean_text)
        if subtasks:
            outputs = []
            for st in subtasks:
                if st.get("rule_result"):
                    outputs.append(st["rule_result"])
                    continue
                st_prompt = build_optimized_prompt(st["type"], st["action"], st["text"], task.files)
                st_prompt = enrich_prompt_with_examples(st_prompt, st["type"], st["text"])
                st_max_tokens = get_max_tokens(st["type"])
                result = call_ollama(st_prompt, max_tokens=st_max_tokens)
                out = postprocess_output(result["text"], st["type"])
                outputs.append(f"[{st['type']}]\n{out}")
                task.tokens_input += result["tokens_input"]
                task.tokens_output += result["tokens_output"]
                task.time_ms += result["time_ms"]
            task.output = "\n\n".join(outputs)
            task.model_used = CONFIG["local_model"]
            task.cost_saved = calc_savings(task)
            cache.set(
                task.action, task.text,
                {"text": task.output, "tokens_input": task.tokens_input,
                 "tokens_output": task.tokens_output},
                task_type="",
                route="local",
            )
            log_usage(task)
            return task

        # 2. 单任务：检测类型 + 优化 prompt
        task_type = detect_task_type(clean_action)

        # 智能模型选择：根据任务类型选择最佳本地模型
        selected_model = None
        try:
            registry = get_model_registry()
            if registry.models:
                capability = TASK_TO_CAPABILITY.get(task_type, "")
                if capability:
                    best = registry.select_best(capability, prefer_speed=True)
                    if best and best.name != CONFIG["local_model"]:
                        selected_model = best.name
        except Exception:
            pass

        prompt = build_optimized_prompt(task_type, clean_action, clean_text, task.files)
        # 动态 few-shot 增强
        if task_type:
            enhanced = enrich_prompt_with_examples(prompt, task_type, clean_text)
            if enhanced != prompt:
                prompt = enhanced
        # 提示压缩（减少 token 消耗）
        prompt = compress_prompt_tokens(prompt, task_type)
        max_tokens = get_max_tokens(task_type)
        result = call_ollama(prompt, model=selected_model, max_tokens=max_tokens)
        task.output = postprocess_output(result["text"], task_type)
        task.model_used = selected_model or CONFIG["local_model"]

        # ── 输出验证 + 云端降级 ──
        validation = validate_local_output(task.output, task_type)
        # 记录能力执行结果（用于自适应阈值）
        cap = TASK_TO_CAPABILITY.get(task_type, "")
        if cap:
            cap_tracker.record(cap, success=validation["valid"],
                               score=1.0 if validation["valid"] else 0.3,
                               task_type=task_type)
        if not validation["valid"] and CONFIG["cloud_api_key"]:
            # 本地输出质量差，自动回退到云端
            cloud_result = call_cloud_api(task.action, task.text)
            cloud_output = cloud_result["text"]
            # 检查是否熔断（熔断时保留本地输出，标记警告）
            if cloud_result.get("circuit_open"):
                task.output += f"\n\n[警告: 本地输出质量不佳，云端熔断中无法降级 - {validation['reason']}]"
                task.route = "local(degraded)"
            else:
                # 记录修正对用于蒸馏
                auto_collect_on_local_failure(
                    task, task.output, cloud_output,
                    failure_type="quality_fallback"
                )
                task.output = cloud_output
                task.route = "cloud_fallback"
                task.model_used = CONFIG["cloud_model"]
                result = cloud_result  # 用云端结果更新 token 统计
                task.cost_saved = 0  # 走云端，不节约成本
        elif not validation["valid"]:
            # 没有云端配置，在输出中标记警告
            task.output += f"\n\n[警告: 本地输出质量可能不佳 - {validation['reason']}]"
    else:
        # ── 云端任务：先检查是否能递归拆解 ──
        decision = estimate_complexity(task) if not force_route else {"score": 9, "route": force_route}
        if RECURSE_SCORE_MIN <= decision["score"] <= RECURSE_SCORE_MAX:
            subtasks = _recursive_decompose(task.action, task.text)
            if subtasks:
                # 有子任务 → 混合执行
                outputs = []
                for st in subtasks:
                    st_action = st.get("action", st.get("type", ""))
                    st_text = st.get("text", task.text)
                    st_task = Task(action=st_action, text=st_text)
                    st_route = st.get("route", "")
                    st_result = run_task(st_task, force_route=st_route if st_route else "")
                    outputs.append(f"[{st_action[:30]}]({st_result.route})\n{st_result.output}")
                    task.tokens_input += st_result.tokens_input
                    task.tokens_output += st_result.tokens_output
                    task.time_ms += st_result.time_ms
                    task.cost_saved += st_result.cost_saved
                task.output = "\n\n---\n\n".join(outputs)
                task.route = "hybrid(recurse)"
                task.model_used = "mixed"
                if task.output and len(task.output.strip()) >= 1:
                    cache.set(task.action, task.text,
                              {"text": task.output, "tokens_input": task.tokens_input,
                               "tokens_output": task.tokens_output})
                log_usage(task)
                return task

        # 无法拆解 → 直接调用云端
        result = call_cloud_api(task.action, task.text)
        task.output = result["text"]
        # 蒸馏：自动采集云端响应
        auto_collect_on_cloud(task, result)

    task.tokens_input = result["tokens_input"]
    task.tokens_output = result["tokens_output"]
    task.time_ms = result["time_ms"]
    task.cost_saved = calc_savings(task)
    # 存入语义缓存（仅缓存有效结果，最短1字符即可，如"正面"）
    if task.output and len(task.output.strip()) >= 1:
        original_route = force_route or estimate_complexity(task)["route"]
        cache.set(
            task.action, task.text,
            {"text": task.output, "tokens_input": task.tokens_input,
             "tokens_output": task.tokens_output},
            task_type=detect_task_type(task.action),
            route=original_route,
        )
    log_usage(task)
    return task


def run_batch(tasks_data: list[dict], concurrency: int = 1) -> list[Task]:
    """
    批量执行任务。

    参数:
        tasks_data: 任务列表，每个元素为 dict
        concurrency: 并发数 (1=串行, >1=并行)

    返回:
        Task 结果列表
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_one(t):
        task = Task(
            action=t.get("action", ""),
            text=t.get("text", ""),
            files=t.get("files", []),
        )
        try:
            task = run_task(task, force_route=t.get("force_route", ""))
        except Exception as e:
            task.output = f"[错误] {e}"
            task.route = "error"
        return task

    if concurrency <= 1:
        # 串行执行
        results = []
        for i, t in enumerate(tasks_data):
            task = _run_one(t)
            results.append(task)
            status = "✓" if task.route != "error" else "✗"
            print(
                f"  [{i+1}/{len(tasks_data)}] {status} {task.action[:50]} → {task.route} ({task.time_ms}ms)"
            )
        return results
    else:
        # 并行执行
        results = [None] * len(tasks_data)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(_run_one, t): i
                for i, t in enumerate(tasks_data)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    task = future.result()
                except Exception as e:
                    task = Task(action=tasks_data[idx].get("action", ""), output=f"[错误] {e}", route="error")
                results[idx] = task
                status = "✓" if task.route != "error" else "✗"
                print(
                    f"  [{idx+1}/{len(tasks_data)}] {status} {task.action[:50]} → {task.route} ({task.time_ms}ms)"
                )
        return results


def interactive_mode():
    print("TaskRouter 交互模式 (输入 'quit' 退出, 'stats' 看统计)")
    print(f"  本地模型: {CONFIG['local_model']}")
    print(f"  云端模型: {CONFIG['cloud_model'] or '(未配置)'}")
    print()
    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not cmd:
            continue
        if cmd == "quit":
            break
        if cmd == "stats":
            print(show_usage_stats())
            continue
        task = Task(action=cmd)
        try:
            task = run_task(task)
        except Exception as e:
            print(f"[错误] {e}")
            continue
        print(
            f"[路由: {task.route} | 模型: {task.model_used} | "
            f"耗时: {task.time_ms}ms | 节约: ${task.cost_saved:.6f}]"
        )
        print(task.output)
        print()


# ─── 细粒度分类 ────────────────────────────────────────────────────

def classify_task(task_description: str, text_content: str = "") -> dict:
    """
    对任务进行细粒度分析，给出是否适合同 Qwen2.5 1.5B 的详细判断。
    返回结构化分类结果供 agent 决策参考。
    """
    desc = task_description.lower()
    text = text_content.lower()
    text_len = len(text_content)

    # 分析维度
    reasons_local = []
    reasons_cloud = []
    warnings = []

    # ── 分析任务类型 ──
    is_file_task = any(kw in desc for kw in ["文件", "文件夹", "目录", "桌面", "扩展名", "重命名", "移动", "复制", "删除"])
    is_classify = any(kw in desc for kw in ["分类", "归类", "排序", "分组", "整理"])
    is_format = any(kw in desc for kw in ["格式化", "转换", "转成", "改成", "提取", "摘录"])
    is_translate = any(kw in desc for kw in ["翻译", "译成", "英文", "中文"])
    is_keyword = any(kw in desc for kw in ["关键词", "标签", "提取", "摘要"])
    is_simple_qa = any(kw in desc for kw in ["属于", "什么类型", "判断", "检查", "验证"])
    is_batch = any(kw in desc for kw in ["批量", "所有", "全部", "遍历", "每个"])
    is_code = any(kw in desc for kw in ["代码", "编程", "函数", "class", "def ", "bug", "调试", "重构", "git"])
    # "分析" 在很多分类场景只是做简单判断（情感、情绪等），不算推理
    has_simple_analysis = any(kw in desc for kw in ["情感", "情绪", "正负面", "好评", "差评"])
    is_reasoning = any(kw in desc for kw in ["推理", "对比", "评价优劣", "评价方案", "建议方案", "策略", "规划", "设计系统", "设计架构"])
    if "分析" in desc and not has_simple_analysis:
        is_reasoning = True
    is_math = any(kw in desc for kw in ["计算", "求和", "平均数", "统计", "公式", "方程"])
    is_creative = any(kw in desc for kw in ["写", "创作", "故事", "文案", "诗歌", "广告"])
    is_professional = any(kw in desc for kw in ["法律", "医疗", "金融", "投资", "税务", "合同"])

    # ── 内容复杂度 ──
    if text_len > 1000:
        warnings.append(f"文本过长({text_len}字)，小模型可能丢失细节")
    if text_len > 2000:
        reasons_cloud.append("远超小模型上下文窗口")

    # ── 逻辑判断 ──
    if "如果" in desc or "否则" in desc or "when" in desc:
        reasons_cloud.append("含条件分支逻辑")
    if text.count(",") > 20 or text.count("\n") > 30:
        warnings.append("数据量较大，建议分批处理")

    # ── 文件操作安全 ──
    if is_file_task and any(kw in desc for kw in ["删除", "覆盖", "修改"]):
        reasons_cloud.append("涉及破坏性文件操作")

    # ── 结论 ──
    local_votes = 0
    cloud_votes = 0

    if is_classify and not is_reasoning: local_votes += 2
    if is_format and not is_code: local_votes += 2
    if is_translate and text_len < 500: local_votes += 2
    elif is_translate: local_votes += 1; warnings.append("长文本翻译质量可能下降")
    if is_keyword and text_len < 1000: local_votes += 2
    if is_simple_qa: local_votes += 1
    if is_batch: local_votes += 1

    if is_code: cloud_votes += 3
    if is_reasoning: cloud_votes += 3
    if is_math: cloud_votes += 2
    if is_creative: cloud_votes += 2
    if is_professional: cloud_votes += 2
    if text_len > 2000: cloud_votes += 2

    verdict = "local" if local_votes > cloud_votes else "cloud"
    if local_votes == cloud_votes:
        verdict = "local" if text_len < 200 else "cloud"
        warnings.append("边界情况，按文本长度裁定")

    confidence = "high" if abs(local_votes - cloud_votes) >= 2 else "medium" if abs(local_votes - cloud_votes) == 1 else "low"

    # ── 子任务拆解潜力 ──
    compound_score = 0
    compound_hints = []
    if is_classify and is_batch:
        compound_score += 1; compound_hints.append("分类+批量 → 可拆解")
    if is_classify and ("重命名" in desc or "改名" in desc):
        compound_score += 1; compound_hints.append("分类+重命名 → 可拆解")
    if is_classify and ("统计" in desc or "计数" in desc):
        compound_score += 1; compound_hints.append("分类+统计 → 可拆解，计数用规则")
    if ("去重" in desc or "去重复" in desc) and ("排序" in desc):
        compound_score += 1; compound_hints.append("去重+排序 → 可分步执行")

    result = {
        "task": task_description[:100],
        "text_length": text_len,
        "verdict": verdict,
        "confidence": confidence,
        "local_score": local_votes,
        "cloud_score": cloud_votes,
        "local_reasons": reasons_local,
        "cloud_reasons": reasons_cloud,
        "warnings": warnings,
        "decomposition": compound_hints if compound_hints else None,
        "decision_tree": {
            "is_file_task": is_file_task,
            "is_classify": is_classify,
            "is_format": is_format,
            "is_translate": is_translate,
            "is_keyword": is_keyword,
            "is_simple_qa": is_simple_qa,
            "is_batch": is_batch,
            "is_code": is_code,
            "is_reasoning": is_reasoning,
            "is_math": is_math,
            "is_creative": is_creative,
            "is_professional": is_professional,
        },
    }
    return result


def estimate(task_description: str) -> dict:
    dummy = Task(action=task_description)
    decision = estimate_complexity(dummy)
    avg_len = len(task_description)
    est_input = max(50, avg_len // 2)
    est_output = max(50, avg_len)
    cloud_cost = calc_cost(est_input, est_output)
    return {
        "task": task_description[:100],
        "suggested_route": decision["route"],
        "reason": decision["reason"],
        "estimated_input_tokens": est_input,
        "estimated_output_tokens": est_output,
        "estimated_cloud_cost": f"${cloud_cost:.6f}",
        "will_save": decision["route"] == "local",
    }


# ─── 大任务拆解 + 逐子任务路由 ──────────────────────────────────
#
# 核心思想：一个大任务拆成多个子任务，每个子任务独立判断
# 走本地 Qwen 还是云端 API，实现 token 最大节约。

DECOMPOSE_TEMPLATES = {
    "电商数据分析": {
        "match": ["电商", "销售", "商品", "订单"],
        "subtasks": [
            "清洗数据（去空值、统一格式）",
            "按类别分类商品",
            "统计各品类销售额",
            "分析销售趋势和异常",
            "给出优化建议并生成报告",
        ],
        "routes": ["local", "local", "local", "cloud", "cloud"],
        "reasons": [
            "简单格式化",
            "文件分类",
            "统计可用规则",
            "需要推理",
            "需要推理+创作",
        ],
    },
    "数据处理流水线": {
        "match": ["数据", "处理", "清洗", "ETL"],
        "subtasks": [
            "数据格式统一和清洗",
            "去重和排序",
            "按字段分类",
            "统计分析",
            "数据可视化建议",
        ],
        "routes": ["local", "local", "local", "cloud", "cloud"],
        "reasons": [
            "格式化",
            "列表操作",
            "分类任务",
            "需要分析能力",
            "需要创意",
        ],
    },
    "文档处理": {
        "match": ["文档", "文章", "内容", "文本"],
        "subtasks": [
            "提取关键信息（日期、人名、数字）",
            "按主题分类",
            "概括主要内容",
            "分析核心观点",
            "生成摘要报告",
        ],
        "routes": ["local", "local", "local", "cloud", "cloud"],
        "reasons": [
            "信息提取",
            "文本分类",
            "简单概括",
            "需要推理",
            "需要创作",
        ],
    },
    "文件批量整理": {
        "match": ["文件", "桌面", "整理", "归类"],
        "subtasks": [
            "扫描并列出所有文件",
            "按扩展名分类",
            "建议目录结构",
            "生成整理脚本",
        ],
        "routes": ["local", "local", "local", "cloud"],
        "reasons": [
            "列表简单",
            "分类任务",
            "简单建议",
            "需要生成代码",
        ],
    },
    "内容创作": {
        "match": ["写", "创作", "文章", "文案", "博客", "邮件", "标题"],
        "subtasks": [
            "提取关键信息和要求",
            "列出大纲结构",
            "撰写初稿",
            "润色和优化",
        ],
        "routes": ["local", "local", "cloud", "cloud"],
        "reasons": [
            "信息提取",
            "结构化简单",
            "需要创意",
            "需要语言优化",
        ],
    },
    "数据提取处理": {
        "match": ["提取", "抓取", "收集", "采集", "爬取", "解析"],
        "subtasks": [
            "提取所有关键字段",
            "格式化为结构化数据",
            "去重和排序",
            "分析和总结",
        ],
        "routes": ["local", "local", "local", "cloud"],
        "reasons": [
            "提取任务",
            "格式化",
            "列表操作",
            "需要分析",
        ],
    },
    "多语言处理": {
        "match": ["翻译", "多语言", "国际化", "i18n", "本地化"],
        "subtasks": [
            "识别源语言",
            "逐段翻译",
            "术语一致性检查",
            "翻译质量评估",
        ],
        "routes": ["local", "local", "local", "cloud"],
        "reasons": [
            "语言识别",
            "逐句翻译",
            "关键词比对",
            "需要理解上下文",
        ],
    },
}


def decompose_task(task_description: str, text_content: str = "") -> dict:
    """
    将大任务拆解为子任务列表，每个子任务独立路由判断。
    返回结构化分解计划。
    """
    desc = task_description.lower()

    # 1. 尝试匹配预设模板
    for template_name, template in DECOMPOSE_TEMPLATES.items():
        if any(kw in desc for kw in template["match"]):
            subtasks = []
            for i, (sub_action, route, reason) in enumerate(
                zip(
                    template["subtasks"],
                    template["routes"],
                    template["reasons"],
                )
            ):
                subtasks.append({
                    "id": i + 1,
                    "action": sub_action,
                    "text": text_content,
                    "route": route,
                    "reason": reason,
                })
            return {
                "task": task_description[:200],
                "template": template_name,
                "total_subtasks": len(subtasks),
                "local_count": sum(1 for s in subtasks if s["route"] == "local"),
                "cloud_count": sum(1 for s in subtasks if s["route"] == "cloud"),
                "subtasks": subtasks,
            }

    # 2. 无匹配模板时，使用关键词判断逐段拆解
    subtasks = []
    seen = set()

    # 分析任务中出现的操作关键词
    operations = []
    for kw in [
        ("提取", "提取信息"),
        ("分类", "分类"),
        ("统计", "统计"),
        ("排序", "排序"),
        ("去重", "去重"),
        ("格式化", "格式化"),
        ("翻译", "翻译"),
        ("分析", "分析"),
        ("总结", "总结"),
        ("生成", "生成"),
        ("比较", "对比"),
        ("建议", "建议"),
    ]:
        if kw[0] in desc:
            operations.append(kw[1])

    if not operations:
        # 没有明确的操作步骤，返回整体判断
        cl = classify_task(task_description, text_content)
        return {
            "task": task_description[:200],
            "template": "none",
            "total_subtasks": 1,
            "local_count": 1 if cl["verdict"] == "local" else 0,
            "cloud_count": 1 if cl["verdict"] == "cloud" else 0,
            "subtasks": [{
                "id": 1,
                "action": task_description[:200],
                "text": text_content,
                "route": cl["verdict"],
                "reason": cl.get("cloud_reasons", ["整体判断"])[0] if cl["cloud_reasons"] else "适合本地",
            }],
        }

    # 逐操作拆解
    for i, op in enumerate(operations):
        if op in seen:
            continue
        seen.add(op)
        subtask_action = f"{op}（来自任务：{task_description[:50]}）"
        cl = classify_task(subtask_action, text_content)
        route = cl["verdict"]
        reasons = cl.get("cloud_reasons", [])
        reason = reasons[0] if reasons else "适合本地"
        subtasks.append({
            "id": i + 1,
            "action": op,
            "text": text_content,
            "route": route,
            "reason": reason,
        })

    if not subtasks:
        subtasks.append({
            "id": 1,
            "action": task_description[:200],
            "text": text_content,
            "route": "local",
            "reason": "默认",
        })

    return {
        "task": task_description[:200],
        "template": "auto_detected",
        "total_subtasks": len(subtasks),
        "local_count": sum(1 for s in subtasks if s["route"] == "local"),
        "cloud_count": sum(1 for s in subtasks if s["route"] == "cloud"),
        "subtasks": subtasks,
    }


def execute_plan(plan: dict) -> dict:
    """
    执行任务计划：对每个子任务按其 route 分配到对应模型执行。
    返回每个子任务的结果和汇总统计。
    """
    results = []
    total_input = 0
    total_output = 0
    total_time = 0
    total_saved = 0.0

    for i, step in enumerate(plan.get("subtasks", [])):
        step_num = i + 1
        total_steps = plan.get("total_subtasks", len(plan["subtasks"]))
        action = step["action"]
        text = step.get("text", "")
        route = step["route"]

        task = Task(action=action, text=text)
        task.route = route

        if route == "local":
            clean_text = preprocess_text(text)
            task_type = detect_task_type(action)
            # 先检查是否能规则执行
            rule_result = rule_execute(task_type, clean_text or action)
            if rule_result:
                output = rule_result
                result = {"text": output, "tokens_input": 0, "tokens_output": 0, "time_ms": 0}
            else:
                prompt = build_optimized_prompt(task_type, action, clean_text, [])
                max_tokens = get_max_tokens(task_type)
                if prompt:  # 非规则任务
                    model_result = call_ollama(prompt, max_tokens=max_tokens)
                    output = postprocess_output(model_result["text"], task_type)
                    result = model_result
                else:
                    output = ""
                    result = {"text": "", "tokens_input": 0, "tokens_output": 0, "time_ms": 0}
        else:
            # 云端：只做路由标记，不实际调用（没有配置key时返回提示）
            if CONFIG["cloud_api_key"]:
                result = call_cloud_api(action, text)
                output = result["text"]
            else:
                output = "[需要云端模型处理]"
                result = {"text": output, "tokens_input": 0, "tokens_output": 0, "time_ms": 0}

        task.output = output
        task.tokens_input = result["tokens_input"]
        task.tokens_output = result["tokens_output"]
        task.time_ms = result["time_ms"]
        task.cost_saved = calc_savings(task) if route == "local" else 0

        total_input += task.tokens_input
        total_output += task.tokens_output
        total_time += task.time_ms
        total_saved += task.cost_saved

        results.append({
            "id": step.get("id", step_num),
            "action": action,
            "route": route,
            "reason": step.get("reason", ""),
            "output": output,
            "tokens_input": task.tokens_input,
            "tokens_output": task.tokens_output,
            "time_ms": task.time_ms,
            "cost_saved": task.cost_saved,
        })

    return {
        "task": plan.get("task", ""),
        "total_steps": len(results),
        "local_steps": sum(1 for r in results if r["route"] == "local"),
        "cloud_steps": sum(1 for r in results if r["route"] == "cloud"),
        "total_tokens_input": total_input,
        "total_tokens_output": total_output,
        "total_time_ms": total_time,
        "total_cost_saved": round(total_saved, 6),
        "steps": results,
    }


# ─── CLI ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TaskRouter — 任务路由系统")
    parser.add_argument("--task", "-t", help="任务描述")
    parser.add_argument("--text", "-T", help="待处理的文本内容")
    parser.add_argument("--file", "-f", action="append", help="待处理的文件路径")
    parser.add_argument("--force", choices=["local", "cloud"], help="强制路由")
    parser.add_argument("--batch", "-b", help="批量任务 JSON 文件路径")
    parser.add_argument("--concurrency", type=int, default=1, help="批量任务并发数 (默认1=串行)")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--estimate", "-e", help="预估算任务路由")
    parser.add_argument("--classify", "-c", nargs="*", help="细粒度分类分析任务是否适 Qwen2.5 1.5B")
    parser.add_argument("--decompose", "-d", nargs="*", help="将大任务拆解为子任务，每步独立路由判断")
    parser.add_argument("--plan", "-p", help="执行任务计划文件（JSON），每步独立路由")
    parser.add_argument("--stats", action="store_true", help="查看使用统计")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    parser.add_argument("--distill", action="store_true", help="运行蒸馏：评判未处理的训练对")
    parser.add_argument("--distill-stats", action="store_true", help="查看蒸馏系统健康状态")
    parser.add_argument("--distill-export", nargs="?", const="auto", help="导出可训练数据为 JSONL")
    parser.add_argument("--thresholds", action="store_true", help="查看自适应阈值调整情况")
    parser.add_argument("--models", action="store_true", help="查看模型注册表")
    parser.add_argument("--benchmark", nargs="?", const="all", help="运行模型基准测试")
    args = parser.parse_args()

    if args.stats:
        print(show_usage_stats())
        return

    if args.models:
        registry = get_model_registry()
        registry.discover()
        print(registry.get_summary())
        return

    if args.benchmark:
        registry = get_model_registry()
        registry.discover()
        model = None if args.benchmark == "all" else args.benchmark
        results = registry.run_benchmark(model)
        for m, r in results.items():
            print(f"\n{m}:")
            for cap, score in r["capabilities"].items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                print(f"  {cap:15} {bar} {score:.0%}")
            print(f"  平均延迟: {r['avg_latency_ms']}ms")
        return

    if args.estimate:
        result = estimate(args.estimate)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            route_tag = "本地 (免费)" if result["will_save"] else "云端 (付费)"
            print(f"任务: {result['task']}")
            print(f"建议路由: {route_tag}")
            print(f"原因: {result['reason']}")
            print(f"预估云端成本: {result['estimated_cloud_cost']}")
        return

    if args.classify is not None:
        task_desc = " ".join(args.classify) if args.classify else ""
        text_content = args.text or ""
        if not task_desc:
            task_desc = input("请输入任务描述: ").strip()
            if not task_desc:
                print("任务描述不能为空")
                return
        result = classify_task(task_desc, text_content)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"任务: {result['task']}")
            if text_content:
                print(f"文本长度: {result['text_length']} 字")
            print(f"{'='*50}")

            # 决策树可视化
            dt = result["decision_tree"]
            tags = []
            for k, v in dt.items():
                if v:
                    tag = k.replace("is_", "").replace("_", " ")
                    tags.append(tag)
            if tags:
                print(f"检测到特征: {' | '.join(tags)}")
            print()

            verdict = result["verdict"]
            conf = result["confidence"]
            if verdict == "local":
                print(f"结论: ✅ 适合 Qwen2.5 1.5B (置信度: {conf})")
            else:
                print(f"结论: ❌ 不适合小模型 (置信度: {conf})")
            print(f"   本地得分: {result['local_score']} | 云端得分: {result['cloud_score']}")

            if result["cloud_reasons"]:
                print(f"\n需要云端模型的原因:")
                for r in result["cloud_reasons"]:
                    print(f"  · {r}")

            if result["warnings"]:
                print(f"\n注意事项:")
                for w in result["warnings"]:
                    print(f"  ⚠ {w}")
            print()
        return

    if args.decompose is not None:
        task_desc = " ".join(args.decompose) if args.decompose else ""
        text_content = args.text or ""
        if not task_desc:
            task_desc = input("请输入要拆解的大任务描述: ").strip()
            if not task_desc:
                print("任务描述不能为空")
                return
        plan = decompose_task(task_desc, text_content)
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            subtasks = plan.get("subtasks", [])
            print(f"大任务: {plan['task']}")
            print(f"拆解为 {plan['total_subtasks']} 个子任务")
            print(f"  {'='*40}")
            print(f"  🟢 本地处理: {plan['local_count']} 步")
            print(f"  🔵 云端处理: {plan['cloud_count']} 步")
            if plan['local_count'] > 0:
                pct = plan['local_count'] / plan['total_subtasks'] * 100
                print(f"  💰 预计节约: ~{pct:.0f}% token")
            print()
            for s in subtasks:
                icon = "🟢" if s["route"] == "local" else "🔵"
                print(f"  [{s['id']}] {icon} {s['action']}")
                print(f"       路由: {s['route'].upper()} — {s['reason']}")
        return

    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        print(f"执行任务计划: {plan.get('task', '未知')}")
        print(f"共 {plan.get('total_subtasks', len(plan['subtasks']))} 步")
        print()
        result = execute_plan(plan)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for step in result["steps"]:
                icon = "🟢" if step["route"] == "local" else "🔵"
                cost = step["cost_saved"]
                time_ms = step["time_ms"]
                print(f"  [{step['id']}] {icon} {step['action']}")
                print(f"       路由: {step['route'].upper()} | 耗时: {time_ms}ms | 节约: ${cost:.6f}")
                output_preview = step["output"][:100].replace("\n", " ")
                print(f"       结果: {output_preview}")
                print()
            print(f"{'='*50}")
            print(f"完成! 本地: {result['local_steps']} | 云端: {result['cloud_steps']} | "
                  f"总节约: ${result['total_cost_saved']:.4f} | "
                  f"总耗时: {result['total_time_ms']}ms")
        return

    if args.interactive:
        interactive_mode()
        return

    if args.batch:
        with open(args.batch) as f:
            tasks_data = json.load(f)
        results = run_batch(tasks_data, concurrency=args.concurrency)
        if args.json:
            print(json.dumps([asdict(t) for t in results], ensure_ascii=False, indent=2))
        else:
            local_count = sum(1 for t in results if t.route == "local")
            cloud_count = sum(1 for t in results if t.route == "cloud")
            total_saved = sum(t.cost_saved for t in results)
            print(
                f"\n完成! 本地: {local_count}, 云端: {cloud_count}, "
                f"总节约: ${total_saved:.4f}"
            )
        return

    if args.distill:
        # 检查是否需要先级联失效（模型版本变化）
        version_tag = f"{CONFIG['local_model']}@{time.strftime('%Y-%m')}"
        invalidated = store.cascade_invalidate(version_tag)
        if invalidated:
            print(f"  级联失效: {invalidated} 条过时训练对已标记为 OUTDATED\n")

        result = run_distillation(dry_run=args.json)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            status = result.get("status", "completed")
            if status == "no_pairs":
                print(f"无待评判的训练对")
            else:
                print(f"蒸馏完成!")
                print(f"  评判: {result.get('judged', 0)} 对")
                print(f"  ✅ 提升为 SUPPORTED: {result.get('promoted', 0)} 对")
                print(f"  ❌ CONTESTED: {result.get('contested', 0)} 对")
                print(f"  ⏳ 保持 HYPOTHESIS: {result.get('kept_hypothesis', 0)} 对")
                if result.get("health"):
                    h = result["health"]
                    print(f"\n蒸馏健康度: {h['health_score']}% ({h['supported']}/{h['total_pairs']})")
                if result.get("best_examples"):
                    total_examples = sum(len(v) for v in result["best_examples"].values())
                    print(f"\n已提取 {total_examples} 条候选 few-shot 示例")
            print(f"\n提示: 运行 sma --distill-stats 查看详细状态")
        return

    if args.distill_stats:
        print(show_distill_stats(verbose=True))
        return

    if args.distill_export:
        path = args.distill_export if args.distill_export != "auto" else None
        result = export_distillation(path)
        print(result)
        return

    if args.thresholds:
        adjs = cap_tracker.get_all_adjustments()
        if not adjs:
            print("暂无自适应阈值数据（执行一些本地任务后会自动生成）")
            return
        lines = ["自适应阈值调整情况", "="*50]
        for cap, info in adjs.items():
            lines.append(
                f"  {cap:20s} | 成功率: {info['success_rate']:.0%} | "
                f"阈值: {info['threshold']:.1f} ({info['direction']} {info['adjustment']})"
            )
        print("\n".join(lines))
        return

    if not args.task:
        parser.print_help()
        return

    task = Task(action=args.task, text=args.text or "", files=args.file or [])
    try:
        task = run_task(task, force_route=args.force)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

    if args.json:
        print(json.dumps(asdict(task), ensure_ascii=False, indent=2))
    else:
        route_icon = "[LOCAL]" if task.route == "local" else "[CLOUD]"
        print(f"{route_icon} {task.model_used}")
        print(f"耗时: {task.time_ms}ms | 输入: {task.tokens_input} | 输出: {task.tokens_output} | 节约: ${task.cost_saved:.6f}")
        print("-" * 50)
        print(task.output)


if __name__ == "__main__":
    main()
