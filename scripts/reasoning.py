"""
推理路径路由 — 根据任务复杂度选择最优推理策略

核心思路（vLLM Semantic Router + NeurIPS 2025）：
不同任务需要不同的推理深度，选择合适的推理策略可以减少 50% 的 token 消耗。

策略：
- direct: 简单任务 → 直接回答（最少 token）
- cot: 中等任务 → Chain-of-Thought（逐步推理）
- few_shot: 模式匹配 → 动态示例
- structured: 复杂输出 → 结构化格式
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from io_utils import read_jsonl, append_jsonl


# ─── 推理策略定义 ──────────────────────────────────────────────

STRATEGY_DIRECT = "direct"
STRATEGY_COT = "cot"
STRATEGY_FEW_SHOT = "few_shot"
STRATEGY_STRUCTURED = "structured"

# 策略对 token 消耗的影响（相对于 direct 的倍数）
STRATEGY_TOKEN_MULTIPLIER = {
    STRATEGY_DIRECT: 1.0,
    STRATEGY_COT: 1.8,
    STRATEGY_FEW_SHOT: 1.5,
    STRATEGY_STRUCTURED: 1.3,
}

# 策略适用的任务复杂度范围
STRATEGY_COMPLEXITY_RANGE = {
    STRATEGY_DIRECT: (0.0, 2.0),
    STRATEGY_COT: (2.0, 5.0),
    STRATEGY_FEW_SHOT: (1.5, 4.0),
    STRATEGY_STRUCTURED: (3.0, 8.0),
}

# CoT 触发关键词
COT_TRIGGERS = [
    "分析", "推理", "比较", "对比", "评价", "评估", "解释", "为什么",
    "原因", "影响", "优缺点", "利弊", "权衡", "选择", "决策",
    "推导", "论证", "证明", "假设", "如果",
]

# 结构化输出触发关键词
STRUCTURED_TRIGGERS = [
    "列出", "列举", "大纲", "结构", "框架", "步骤", "流程",
    "方案", "计划", "报告", "总结", "汇总", "清单",
]


# ─── 策略增强 Prompt ──────────────────────────────────────────

STRATEGY_ENHANCEMENTS = {
    STRATEGY_DIRECT: "",  # 不增强，使用原始 prompt

    STRATEGY_COT: "\n\n请一步一步思考：\n1. 先理解问题的关键点\n2. 逐步分析\n3. 给出最终答案\n",

    STRATEGY_FEW_SHOT: "",  # 由 distillation 系统动态注入

    STRATEGY_STRUCTURED: "\n\n请用以下格式输出：\n- 要点1\n- 要点2\n- ...\n- 结论\n",
}


# ─── 策略选择器 ──────────────────────────────────────────────

@dataclass
class StrategyDecision:
    """策略选择结果"""
    strategy: str
    reason: str
    complexity_score: float


def select_reasoning_strategy(
    action: str,
    text: str,
    complexity_score: float,
    task_type: str = "",
) -> StrategyDecision:
    """
    根据任务特征选择最优推理策略。

    参数:
        action: 任务描述
        text: 输入文本
        complexity_score: 复杂度评分（来自 estimate_complexity）
        task_type: 任务类型

    返回:
        StrategyDecision
    """
    action_lower = action.lower()

    # 1. 简单任务 → 直接回答
    if complexity_score < 2.0:
        return StrategyDecision(
            strategy=STRATEGY_DIRECT,
            reason=f"简单任务 (score={complexity_score:.1f})",
            complexity_score=complexity_score,
        )

    # 2. 检测是否需要 CoT
    cot_matches = [kw for kw in COT_TRIGGERS if kw in action_lower]
    if cot_matches and complexity_score >= 2.0:
        return StrategyDecision(
            strategy=STRATEGY_COT,
            reason=f"需要推理 (触发词: {','.join(cot_matches[:3])})",
            complexity_score=complexity_score,
        )

    # 3. 检测是否需要结构化输出
    struct_matches = [kw for kw in STRUCTURED_TRIGGERS if kw in action_lower]
    if struct_matches and complexity_score >= 3.0:
        return StrategyDecision(
            strategy=STRATEGY_STRUCTURED,
            reason=f"需要结构化输出 (触发词: {','.join(struct_matches[:3])})",
            complexity_score=complexity_score,
        )

    # 4. 中等复杂度 → CoT
    if 2.0 <= complexity_score < 5.0:
        return StrategyDecision(
            strategy=STRATEGY_COT,
            reason=f"中等复杂度 (score={complexity_score:.1f})",
            complexity_score=complexity_score,
        )

    # 5. 高复杂度 → 结构化
    return StrategyDecision(
        strategy=STRATEGY_STRUCTURED,
        reason=f"高复杂度 (score={complexity_score:.1f})",
        complexity_score=complexity_score,
    )


def enhance_prompt_with_strategy(prompt: str, strategy: str, examples: str = "") -> str:
    """
    用选定的策略增强 prompt。

    参数:
        prompt: 原始 prompt
        strategy: 推理策略
        examples: 动态示例（用于 few_shot）

    返回:
        增强后的 prompt
    """
    if strategy == STRATEGY_DIRECT:
        return prompt

    if strategy == STRATEGY_FEW_SHOT and examples:
        return prompt + "\n\n参考示例：\n" + examples

    enhancement = STRATEGY_ENHANCEMENTS.get(strategy, "")
    if enhancement:
        return prompt + enhancement

    return prompt


# ─── 策略性能追踪 ──────────────────────────────────────────────

class StrategyTracker:
    """追踪不同策略的性能表现"""

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "strategy_history.jsonl")
        self._history: list[dict] = []
        self._load()

    def _load(self) -> None:
        self._history = read_jsonl(self.data_file)

    def record(
        self,
        strategy: str,
        task_type: str,
        success: bool,
        tokens_used: int,
        latency_ms: int,
    ) -> None:
        """记录一次策略执行结果"""
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "strategy": strategy,
            "task_type": task_type,
            "success": success,
            "tokens": tokens_used,
            "latency_ms": latency_ms,
        }
        append_jsonl(self.data_file, entry)
        self._history.append(entry)

    def get_best_strategy(self, task_type: str) -> Optional[str]:
        """获取某任务类型的最佳策略"""
        type_entries = [e for e in self._history if e.get("task_type") == task_type]
        if len(type_entries) < 5:
            return None  # 数据不足

        # 按策略分组计算成功率
        strategy_stats: dict[str, dict] = {}
        for e in type_entries:
            s = e["strategy"]
            if s not in strategy_stats:
                strategy_stats[s] = {"success": 0, "total": 0, "tokens": 0}
            strategy_stats[s]["total"] += 1
            if e.get("success"):
                strategy_stats[s]["success"] += 1
            strategy_stats[s]["tokens"] += e.get("tokens", 0)

        # 选择成功率最高、token 消耗最低的策略
        best = None
        best_score = -1.0
        for s, stats in strategy_stats.items():
            if stats["total"] < 3:
                continue
            success_rate = stats["success"] / stats["total"]
            avg_tokens = stats["tokens"] / stats["total"]
            # 综合评分：成功率权重 70%，token 效率权重 30%
            token_score = max(0, 1.0 - avg_tokens / 2000)
            score = 0.7 * success_rate + 0.3 * token_score
            if score > best_score:
                best_score = score
                best = s

        return best

    def get_stats(self) -> dict:
        """获取策略统计"""
        if not self._history:
            return {"total": 0, "by_strategy": {}, "by_task_type": {}}

        by_strategy: dict[str, dict] = {}
        by_task_type: dict[str, dict] = {}

        for e in self._history:
            s = e.get("strategy", "unknown")
            t = e.get("task_type", "unknown")

            if s not in by_strategy:
                by_strategy[s] = {"total": 0, "success": 0, "tokens": 0}
            by_strategy[s]["total"] += 1
            if e.get("success"):
                by_strategy[s]["success"] += 1
            by_strategy[s]["tokens"] += e.get("tokens", 0)

            if t not in by_task_type:
                by_task_type[t] = {"total": 0, "success": 0}
            by_task_type[t]["total"] += 1
            if e.get("success"):
                by_task_type[t]["success"] += 1

        return {
            "total": len(self._history),
            "by_strategy": by_strategy,
            "by_task_type": by_task_type,
        }


# 全局实例（延迟初始化）
_tracker: Optional[StrategyTracker] = None


def get_strategy_tracker(cache_dir: Optional[str] = None) -> StrategyTracker:
    global _tracker
    if _tracker is None:
        if cache_dir is None:
            from config import get_config
            cache_dir = get_config().cache_dir
        _tracker = StrategyTracker(cache_dir)
    return _tracker
