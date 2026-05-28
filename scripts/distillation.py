"""
蒸馏系统 — 从云端响应中学习，持续提升本地模型准确率
"""

import os
import time
import threading
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

from config import get_config, TASK_TO_CAPABILITY
from io_utils import read_jsonl, append_jsonl, write_jsonl


# ─── 蒸馏对状态 ──────────────────────────────────────────────

PAIR_HYPOTHESIS = "hypothesis"
PAIR_SUPPORTED = "supported"
PAIR_CONTESTED = "contested"
PAIR_OUTDATED = "outdated"

JUDGE_HIGH_THRESHOLD = 0.9
JUDGE_MODERATE_THRESHOLD = 0.5

CAPABILITY_TYPES = [
    "classification", "translation", "extraction",
    "summarization", "formatting", "qa", "reasoning",
]

SKIP_JUDGE_CAPABILITIES = {"formatting", "extraction"}


@dataclass
class DistillationPair:
    """单个训练对"""
    prompt: str
    response: str
    task_type: str = ""
    capability: str = ""
    route: str = "cloud"
    action: str = ""
    epistemic_state: str = PAIR_HYPOTHESIS
    quality_score: float = 0.0
    judge_reason: str = ""
    model_used: str = ""
    model_version: str = ""
    version_tag: str = ""
    time: str = ""
    pair_id: str = ""
    failure_type: str = ""
    local_response: str = ""

    def __post_init__(self) -> None:
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
    """蒸馏数据存储（含 TTL 遗忘机制）"""

    DEFAULT_TTL_DAYS = 90       # 默认过期天数
    CONTESTED_TTL_DAYS = 14     # contested 状态更快过期
    OUTDATED_TTL_DAYS = 7       # outdated 状态最快过期
    MAX_PAIRS = 5000            # 最大条目数（FIFO 淘汰）

    def __init__(self, cache_dir: Optional[str] = None, ttl_days: int = 0):
        config = get_config()
        self.cache_dir = cache_dir or config.cache_dir
        self.pairs_file = os.path.join(self.cache_dir, "distillation.jsonl")
        self.stats_file = os.path.join(self.cache_dir, "distillation_stats.json")
        self.ttl_days = ttl_days or self.DEFAULT_TTL_DAYS
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_all(self) -> list[dict]:
        return read_jsonl(self.pairs_file)

    def add_pair(self, pair: DistillationPair) -> None:
        with self._lock:
            append_jsonl(self.pairs_file, asdict(pair))

    def _is_expired(self, pair: dict) -> bool:
        """检查蒸馏对是否过期"""
        pair_time = pair.get("time", "")
        if not pair_time:
            return True  # 无时间戳视为过期（保守清理）
        try:
            created = time.mktime(time.strptime(pair_time[:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, OverflowError):
            return True  # 时间戳格式错误视为过期
        state = pair.get("epistemic_state", PAIR_HYPOTHESIS)
        if state == PAIR_OUTDATED:
            ttl = self.OUTDATED_TTL_DAYS * 86400
        elif state == PAIR_CONTESTED:
            ttl = self.CONTESTED_TTL_DAYS * 86400
        else:
            ttl = self.ttl_days * 86400
        return (time.time() - created) > ttl

    def cleanup_expired(self) -> int:
        """清除过期条目，返回删除数量"""
        with self._lock:
            pairs = self._load_all()
            before = len(pairs)
            alive = [p for p in pairs if not self._is_expired(p)]
            if len(alive) < before:
                if len(alive) > self.MAX_PAIRS:
                    alive = alive[-self.MAX_PAIRS:]
                write_jsonl(self.pairs_file, alive)
            return before - len(alive)

    def get_pairs(self, state: Optional[str] = None, capability: Optional[str] = None,
                  task_type: Optional[str] = None, min_score: float = 0.0,
                  limit: int = 0) -> list[dict]:
        pairs = self._load_all()
        # 自动过滤过期条目
        pairs = [p for p in pairs if not self._is_expired(p)]
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

    def update_pair_state(self, pair_id: str, new_state: str,
                          score: Optional[float] = None, reason: str = "") -> None:
        with self._lock:
            pairs = self._load_all()
            for p in pairs:
                if p.get("pair_id") == pair_id:
                    p["epistemic_state"] = new_state
                    if score is not None:
                        p["quality_score"] = score
                    if reason:
                        p["judge_reason"] = reason
            write_jsonl(self.pairs_file, pairs)

    def get_supported_pairs(self, capability: str, limit: int = 5) -> list[dict]:
        """获取指定能力的已验证训练对（用于 few-shot 注入）"""
        return self.get_pairs(state=PAIR_SUPPORTED, capability=capability, min_score=0.7, limit=limit)

    def get_stats(self) -> dict:
        pairs = self._load_all()
        by_state: dict[str, int] = {}
        by_capability: dict[str, int] = {}
        expired = 0
        for p in pairs:
            if self._is_expired(p):
                expired += 1
            state = p.get("epistemic_state", "unknown")
            by_state[state] = by_state.get(state, 0) + 1
            cap = p.get("capability", "unknown")
            by_capability[cap] = by_capability.get(cap, 0) + 1

        return {
            "total": len(pairs),
            "expired": expired,
            "active": len(pairs) - expired,
            "by_state": by_state,
            "by_capability": by_capability,
            "supported": by_state.get(PAIR_SUPPORTED, 0),
            "ttl_days": self.ttl_days,
        }


def collect_distillation_pair(prompt: str, response: str, task_type: str,
                               route: str = "cloud", action: str = "",
                               model_used: str = "") -> DistillationPair:
    """采集一个蒸馏对"""
    pair = DistillationPair(
        prompt=prompt,
        response=response,
        task_type=task_type,
        route=route,
        action=action,
        model_used=model_used,
    )
    return pair


def get_dynamic_examples(capability: str, store: DistillationStore, limit: int = 3) -> list[dict]:
    """从蒸馏池中获取动态 few-shot 示例"""
    pairs = store.get_supported_pairs(capability, limit=limit * 2)
    if not pairs:
        return []

    # 多样性选择：不同的 action 文本优先
    seen_actions: set[str] = set()
    diverse: list[dict] = []
    for p in pairs:
        action = p.get("action", "")
        if action not in seen_actions:
            seen_actions.add(action)
            diverse.append(p)
        if len(diverse) >= limit:
            break
    return diverse


# ─── 质量评估器 ──────────────────────────────────────────────

# 失败信号词
FAILURE_SIGNALS = [
    "抱歉", "无法", "不能", "不懂", "作为AI", "我无法", "error", "Error",
    "失败", "错误", "异常", "不支持", "未找到", "不存在",
]

# 成功信号词（按能力分类）
SUCCESS_SIGNALS = {
    "classification": ["类别", "分类", "属于", "类型"],
    "translation": ["翻译", "译文"],
    "extraction": ["提取", "关键词", "摘录"],
    "summarization": ["总结", "概括", "摘要"],
    "formatting": ["格式", "格式化"],
}


class QualityEvaluator:
    """自动评估蒸馏对质量"""

    def evaluate(self, pair: dict) -> float:
        """
        评估蒸馏对质量，返回 [0, 1] 分数。

        评估维度：
        1. 输出长度合理性
        2. 失败信号检测
        3. 成功信号匹配
        4. 与本地输出的一致性（如有）
        """
        response = pair.get("response", "")
        capability = pair.get("capability", "")
        local_response = pair.get("local_response", "")

        if not response:
            return 0.0

        # 1. 长度分数
        length = len(response)
        if length < 5:
            length_score = 0.2
        elif length < 20:
            length_score = 0.5
        elif length < 500:
            length_score = 0.8
        else:
            length_score = 0.7

        # 2. 失败信号
        has_failure = any(s in response for s in FAILURE_SIGNALS)
        failure_penalty = 0.4 if has_failure else 0.0

        # 3. 成功信号
        success_bonus = 0.0
        if capability in SUCCESS_SIGNALS:
            if any(s in response for s in SUCCESS_SIGNALS[capability]):
                success_bonus = 0.15

        # 4. 一致性（本地 vs 云端）
        consistency_score = 0.0
        if local_response:
            # 简单的文本重叠度
            overlap = self._text_overlap(local_response, response)
            consistency_score = overlap * 0.2

        score = length_score - failure_penalty + success_bonus + consistency_score
        return max(0.0, min(1.0, round(score, 3)))

    def _text_overlap(self, text1: str, text2: str) -> float:
        """计算两段文本的重叠度（Jaccard）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0


# ─── 失败模式聚类 ──────────────────────────────────────────────

class FailureClusterer:
    """将相似的失败归类，用于针对性改进"""

    def __init__(self):
        self._clusters: dict[str, list[dict]] = {}

    def cluster_failures(self, pairs: list[dict]) -> dict[str, list[dict]]:
        """
        将失败的蒸馏对按失败类型聚类。

        返回: {failure_type: [pairs]}
        """
        clusters: dict[str, list[dict]] = {}

        for p in pairs:
            failure_type = self._classify_failure(p)
            if failure_type not in clusters:
                clusters[failure_type] = []
            clusters[failure_type].append(p)

        self._clusters = clusters
        return clusters

    def _classify_failure(self, pair: dict) -> str:
        """分类单个失败"""
        response = pair.get("response", "")
        local_response = pair.get("local_response", "")
        failure_type = pair.get("failure_type", "")

        if failure_type:
            return failure_type

        if not response or len(response) < 5:
            return "empty_output"

        if any(s in response for s in ["抱歉", "无法", "不能"]):
            return "refusal"

        if any(s in response for s in ["error", "Error", "失败", "错误"]):
            return "error"

        if local_response:
            overlap = len(set(response.split()) & set(local_response.split()))
            total = max(len(set(response.split())), 1)
            if overlap / total < 0.2:
                return "divergent"

        return "quality_low"

    def get_top_failures(self, n: int = 5) -> list[dict]:
        """获取最常见的失败模式"""
        result = []
        for ftype, pairs in sorted(
            self._clusters.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )[:n]:
            result.append({
                "failure_type": ftype,
                "count": len(pairs),
                "sample_action": pairs[0].get("action", "")[:50],
                "sample_capability": pairs[0].get("capability", ""),
            })
        return result


# ─── 闭环管理器 ──────────────────────────────────────────────

class ClosedLoopManager:
    """闭环蒸馏管理器：评估 → 聚类 → 推进状态"""

    def __init__(self, store: DistillationStore):
        self.store = store
        self.evaluator = QualityEvaluator()
        self.clusterer = FailureClusterer()

    def evaluate_pending(self) -> int:
        """评估所有 hypothesis 状态的蒸馏对，返回评估数量"""
        pairs = self.store.get_pairs(state=PAIR_HYPOTHESIS)
        evaluated = 0

        for p in pairs:
            score = self.evaluator.evaluate(p)

            if score >= JUDGE_HIGH_THRESHOLD:
                self.store.update_pair_state(
                    p["pair_id"], PAIR_SUPPORTED, score=score,
                    reason=f"auto_eval: score={score:.2f}",
                )
            elif score < JUDGE_MODERATE_THRESHOLD:
                self.store.update_pair_state(
                    p["pair_id"], PAIR_CONTESTED, score=score,
                    reason=f"auto_eval: score={score:.2f}",
                )
            else:
                self.store.update_pair_state(
                    p["pair_id"], PAIR_HYPOTHESIS, score=score,
                    reason=f"auto_eval: score={score:.2f} (pending review)",
                )
            evaluated += 1

        return evaluated

    def analyze_failures(self) -> dict:
        """分析失败模式"""
        # 获取所有低质量蒸馏对
        all_pairs = self.store.get_pairs()
        failed = [
            p for p in all_pairs
            if p.get("quality_score", 0) < JUDGE_MODERATE_THRESHOLD
            or p.get("epistemic_state") == PAIR_CONTESTED
        ]

        clusters = self.clusterer.cluster_failures(failed)
        top_failures = self.clusterer.get_top_failures()

        return {
            "total_failed": len(failed),
            "failure_types": len(clusters),
            "top_failures": top_failures,
        }

    def run_cycle(self) -> dict:
        """运行一个完整的闭环周期"""
        evaluated = self.evaluate_pending()
        failure_analysis = self.analyze_failures()
        cleaned = self.store.cleanup_expired()

        return {
            "evaluated": evaluated,
            "cleaned": cleaned,
            "failure_analysis": failure_analysis,
            "store_stats": self.store.get_stats(),
        }
