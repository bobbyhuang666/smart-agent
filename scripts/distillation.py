"""
蒸馏系统 — 从云端响应中学习，持续提升本地模型准确率
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

from config import get_config, TASK_TO_CAPABILITY


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
    """蒸馏数据存储"""

    def __init__(self, cache_dir: Optional[str] = None):
        config = get_config()
        self.cache_dir = cache_dir or config.cache_dir
        self.pairs_file = os.path.join(self.cache_dir, "distillation.jsonl")
        self.stats_file = os.path.join(self.cache_dir, "distillation_stats.json")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_all(self) -> list[dict]:
        pairs: list[dict] = []
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

    def add_pair(self, pair: DistillationPair) -> None:
        entry = asdict(pair)
        with open(self.pairs_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_pairs(self, state: Optional[str] = None, capability: Optional[str] = None,
                  task_type: Optional[str] = None, min_score: float = 0.0,
                  limit: int = 0) -> list[dict]:
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

    def update_pair_state(self, pair_id: str, new_state: str,
                          score: Optional[float] = None, reason: str = "") -> None:
        pairs = self._load_all()
        for p in pairs:
            if p.get("pair_id") == pair_id:
                p["epistemic_state"] = new_state
                if score is not None:
                    p["quality_score"] = score
                if reason:
                    p["judge_reason"] = reason
        with open(self.pairs_file, "w") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    def get_supported_pairs(self, capability: str, limit: int = 5) -> list[dict]:
        """获取指定能力的已验证训练对（用于 few-shot 注入）"""
        return self.get_pairs(state=PAIR_SUPPORTED, capability=capability, min_score=0.7, limit=limit)

    def get_stats(self) -> dict:
        pairs = self._load_all()
        by_state: dict[str, int] = {}
        by_capability: dict[str, int] = {}
        for p in pairs:
            state = p.get("epistemic_state", "unknown")
            by_state[state] = by_state.get(state, 0) + 1
            cap = p.get("capability", "unknown")
            by_capability[cap] = by_capability.get(cap, 0) + 1

        return {
            "total": len(pairs),
            "by_state": by_state,
            "by_capability": by_capability,
            "supported": by_state.get(PAIR_SUPPORTED, 0),
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
