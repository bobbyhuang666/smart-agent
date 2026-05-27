"""
审计日志系统 — 企业级操作追踪

核心功能:
- 记录所有任务执行（谁、什么时候、做了什么、结果如何）
- 支持 JSON 格式输出（便于 ELK/Splunk 集成）
- 支持按时间/用户/操作类型查询
- 支持配额管理和速率限制
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional
from collections import defaultdict

from config import get_config
from io_utils import read_jsonl, append_jsonl


# ─── 审计事件 ──────────────────────────────────────────────────

@dataclass
class AuditEvent:
    """审计事件"""
    timestamp: str
    event_type: str          # task_execution / model_change / config_change / api_call
    user_id: str = "system"  # 用户标识（API 场景中来自 API Key）
    action: str = ""         # 具体操作
    details: dict = None     # 详细信息
    ip_address: str = ""     # 客户端 IP
    status: str = "success"  # success / failure / warning
    duration_ms: int = 0     # 执行耗时

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# ─── 配额管理 ──────────────────────────────────────────────────

@dataclass
class QuotaConfig:
    """配额配置"""
    user_id: str
    # 每日限额
    daily_task_limit: int = 1000       # 每日任务数
    daily_token_limit: int = 100000    # 每日 token 数
    daily_cost_limit: float = 10.0     # 每日成本上限（美元）
    # 速率限制
    requests_per_minute: int = 60      # 每分钟请求数
    requests_per_hour: int = 1000      # 每小时请求数


class QuotaManager:
    """配额管理器"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.quotas_file = os.path.join(self.cache_dir, "quotas.json")
        self.usage_file = os.path.join(self.cache_dir, "quota_usage.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.quotas: dict[str, QuotaConfig] = {}
        self._load_quotas()

    def _load_quotas(self):
        if not os.path.exists(self.quotas_file):
            return
        try:
            with open(self.quotas_file) as f:
                data = json.load(f)
            for uid, config in data.items():
                self.quotas[uid] = QuotaConfig(**config)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save_quotas(self):
        data = {uid: asdict(q) for uid, q in self.quotas.items()}
        with open(self.quotas_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def set_quota(self, user_id: str, **kwargs):
        """设置用户配额"""
        if user_id in self.quotas:
            for k, v in kwargs.items():
                if hasattr(self.quotas[user_id], k):
                    setattr(self.quotas[user_id], k, v)
        else:
            self.quotas[user_id] = QuotaConfig(user_id=user_id, **kwargs)
        self._save_quotas()

    def get_quota(self, user_id: str) -> QuotaConfig:
        """获取用户配额"""
        return self.quotas.get(user_id, QuotaConfig(user_id=user_id))

    def check_quota(self, user_id: str) -> dict:
        """
        检查用户配额状态。

        返回:
            {
                "allowed": bool,
                "reason": str,
                "daily_tasks_used": int,
                "daily_tasks_limit": int,
                "daily_tokens_used": int,
                "daily_tokens_limit": int,
            }
        """
        quota = self.get_quota(user_id)
        today = time.strftime("%Y-%m-%d")

        # 统计今日使用量
        daily_tasks = 0
        daily_tokens = 0
        daily_cost = 0.0

        if os.path.exists(self.usage_file):
            with open(self.usage_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("user_id") == user_id and entry.get("date") == today:
                            daily_tasks += 1
                            daily_tokens += entry.get("tokens", 0)
                            daily_cost += entry.get("cost", 0.0)
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 检查限制
        if daily_tasks >= quota.daily_task_limit:
            return {
                "allowed": False,
                "reason": f"每日任务数已达上限 ({quota.daily_task_limit})",
                "daily_tasks_used": daily_tasks,
                "daily_tasks_limit": quota.daily_task_limit,
                "daily_tokens_used": daily_tokens,
                "daily_tokens_limit": quota.daily_token_limit,
            }

        if daily_tokens >= quota.daily_token_limit:
            return {
                "allowed": False,
                "reason": f"每日 token 数已达上限 ({quota.daily_token_limit})",
                "daily_tasks_used": daily_tasks,
                "daily_tasks_limit": quota.daily_task_limit,
                "daily_tokens_used": daily_tokens,
                "daily_tokens_limit": quota.daily_token_limit,
            }

        if daily_cost >= quota.daily_cost_limit:
            return {
                "allowed": False,
                "reason": f"每日成本已达上限 (${quota.daily_cost_limit})",
                "daily_tasks_used": daily_tasks,
                "daily_tasks_limit": quota.daily_task_limit,
                "daily_tokens_used": daily_tokens,
                "daily_tokens_limit": quota.daily_token_limit,
            }

        return {
            "allowed": True,
            "reason": "配额充足",
            "daily_tasks_used": daily_tasks,
            "daily_tasks_limit": quota.daily_task_limit,
            "daily_tokens_used": daily_tokens,
            "daily_tokens_limit": quota.daily_token_limit,
        }

    def record_usage(self, user_id: str, tokens: int = 0, cost: float = 0.0,
                     action: str = ""):
        """记录使用量"""
        entry = {
            "user_id": user_id,
            "date": time.strftime("%Y-%m-%d"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tokens": tokens,
            "cost": cost,
            "action": action[:100],
        }
        with open(self.usage_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── 审计日志 ──────────────────────────────────────────────────

class AuditLogger:
    """
    审计日志记录器。

    使用方法:
        audit = AuditLogger()
        audit.log(AuditEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            event_type="task_execution",
            action="翻译成中文",
            details={"input": "Hello", "output": "你好", "route": "local"},
            status="success",
            duration_ms=500,
        ))
    """

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or get_config().cache_dir
        self.audit_file = os.path.join(self.cache_dir, "audit.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)

    def log(self, event: AuditEvent):
        """记录审计事件"""
        append_jsonl(self.audit_file, asdict(event))

    def query(self, event_type: str = None, user_id: str = None,
              start_time: str = None, end_time: str = None,
              limit: int = 100) -> list[dict]:
        """
        查询审计日志。

        参数:
            event_type: 事件类型过滤
            user_id: 用户过滤
            start_time: 开始时间 (ISO 格式)
            end_time: 结束时间 (ISO 格式)
            limit: 最大返回数

        返回:
            审计事件列表
        """
        all_entries = read_jsonl(self.audit_file)

        results = []
        for entry in all_entries:
            if event_type and entry.get("event_type") != event_type:
                continue
            if user_id and entry.get("user_id") != user_id:
                continue
            if start_time and entry.get("timestamp", "") < start_time:
                continue
            if end_time and entry.get("timestamp", "") > end_time:
                continue
            results.append(entry)

        return results[-limit:]

    def get_summary(self, days: int = 7) -> dict:
        """获取审计摘要"""
        start_time = time.strftime(
            "%Y-%m-%dT%H:%M:%S",
            time.localtime(time.time() - days * 86400)
        )
        events = self.query(start_time=start_time, limit=10000)

        summary = {
            "total_events": len(events),
            "by_type": defaultdict(int),
            "by_status": defaultdict(int),
            "by_user": defaultdict(int),
            "avg_duration_ms": 0,
        }

        total_duration = 0
        for e in events:
            summary["by_type"][e.get("event_type", "unknown")] += 1
            summary["by_status"][e.get("status", "unknown")] += 1
            summary["by_user"][e.get("user_id", "unknown")] += 1
            total_duration += e.get("duration_ms", 0)

        if events:
            summary["avg_duration_ms"] = round(total_duration / len(events))

        summary["by_type"] = dict(summary["by_type"])
        summary["by_status"] = dict(summary["by_status"])
        summary["by_user"] = dict(summary["by_user"])

        return summary


# ─── 单例实例 ──────────────────────────────────────────────────

_audit_logger = None
_quota_manager = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

def get_quota_manager() -> QuotaManager:
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager
