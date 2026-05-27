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
