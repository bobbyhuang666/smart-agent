"""
配置管理 — 集中管理所有配置项，消除模块级全局状态
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouterConfig:
    """路由器配置"""
    # Ollama 设置
    ollama_base: str = ""
    local_model: str = "qwen-tool"
    local_max_tokens: int = 2048

    # 云端 API 设置
    cloud_api_url: str = ""
    cloud_api_key: str = ""
    cloud_model: str = ""

    # 成本计算
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015

    # 缓存
    cache_dir: str = ""
    cache_max_entries: int = 1000
    cache_fuzzy_threshold: float = 0.85

    # 缓存 TTL（小时）
    cache_ttl_hours: dict = field(default_factory=lambda: {
        "translation": 168,
        "classification": 168,
        "extraction": 72,
        "formatting": 168,
        "summarization": 24,
        "default": 48,
    })

    # 路由阈值
    base_threshold: float = 3.0
    recurse_score_min: float = 2.5
    recurse_score_max: float = 7.0
    max_recurse_depth: int = 3

    def __post_init__(self) -> None:
        if not self.ollama_base:
            self.ollama_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not self.cloud_api_url:
            self.cloud_api_url = os.environ.get("CLOUD_API_URL", "")
        if not self.cloud_api_key:
            self.cloud_api_key = os.environ.get("CLOUD_API_KEY", "")
        if not self.cloud_model:
            self.cloud_model = os.environ.get("CLOUD_MODEL", "")
        if not self.cache_dir:
            self.cache_dir = os.environ.get(
                "TASK_ROUTER_CACHE", str(Path.home() / ".cache" / "task_router")
            )

    @classmethod
    def from_env(cls) -> "RouterConfig":
        """从环境变量创建配置"""
        return cls()

    def get_cache_ttl(self, task_type: str) -> int:
        """获取指定任务类型的缓存 TTL"""
        capability = TASK_TO_CAPABILITY.get(task_type, "default")
        return self.cache_ttl_hours.get(capability, self.cache_ttl_hours.get("default", 48))


# ─── 能力映射 ──────────────────────────────────────────────────

TASK_TO_CAPABILITY: dict[str, str] = {
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


# ─── 全局配置实例（延迟初始化）──────────────────────────────────

_config: Optional[RouterConfig] = None


def get_config() -> RouterConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = RouterConfig.from_env()
    return _config


def set_config(config: RouterConfig) -> None:
    """设置全局配置（用于测试或自定义）"""
    global _config
    _config = config
