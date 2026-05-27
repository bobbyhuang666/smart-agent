"""
模型调用 — Ollama 本地 + 云端 API，含熔断器和重试
"""

import time
import threading
from typing import Any, Optional

from config import get_config

# 延迟加载的隐私过滤器单例（线程安全）
_privacy_filter_instance = None
_privacy_filter_lock = threading.Lock()


# ─── 熔断器（线程安全）─────────────────────────────────────────

class CircuitBreaker:
    """云端 API 熔断器"""

    def __init__(self, max_failures: int = 3, cooldown_seconds: int = 120):
        self._lock = threading.Lock()
        self.failures: int = 0
        self.last_failure: float = 0.0
        self.open_until: float = 0.0
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds

    def is_open(self) -> bool:
        with self._lock:
            return time.time() < self.open_until

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.open_until = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.max_failures:
                self.open_until = time.time() + self.cooldown_seconds

    def remaining_seconds(self) -> int:
        with self._lock:
            return max(0, int(self.open_until - time.time()))

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "failures": self.failures,
                "last_failure": self.last_failure,
                "open_until": self.open_until,
                "max_failures": self.max_failures,
                "cooldown_seconds": self.cooldown_seconds,
            }


# 全局熔断器实例
circuit_breaker = CircuitBreaker()


# ─── Ollama 调用 ──────────────────────────────────────────────

def call_ollama(prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> dict[str, Any]:
    """调用 Ollama 本地模型"""
    import requests
    config = get_config()
    model = model or config.local_model
    max_tokens = max_tokens or config.local_max_tokens
    start = time.time()

    try:
        resp = requests.post(
            f"{config.ollama_base}/api/generate",
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
            from model_registry import ModelRegistry
            registry = ModelRegistry(cache_dir=config.cache_dir)
            registry.update_after_call(
                model, success=True, latency_ms=elapsed,
                tokens_in=result["tokens_input"], tokens_out=result["tokens_output"]
            )
        except Exception:
            pass

        return result
    except Exception as e:
        try:
            from model_registry import ModelRegistry
            registry = ModelRegistry(cache_dir=config.cache_dir)
            registry.update_after_call(model, success=False, latency_ms=0)
        except Exception:
            pass
        raise


# ─── 云端 API 调用 ──────────────────────────────────────────────

def call_cloud_api(prompt: str, text: str = "") -> dict[str, Any]:
    """调用云端 API（含 PII 脱敏、重试、熔断）"""
    config = get_config()

    if not config.cloud_api_key:
        return {
            "text": "[云端未配置] 设置 CLOUD_API_KEY 环境变量启用云端路由",
            "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
        }

    # 熔断检查
    if circuit_breaker.is_open():
        remaining = circuit_breaker.remaining_seconds()
        return {
            "text": f"[云端熔断中] 连续失败{circuit_breaker.failures}次，{remaining}秒后重试",
            "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
            "circuit_open": True,
        }

    import requests

    # PII 脱敏
    pf = _get_privacy_filter()
    anon_result = None
    if pf:
        anon_result = pf.anonymize(prompt)
        prompt = anon_result.text
        if text:
            text_anon = pf.anonymize(text)
            text = text_anon.text
            anon_result.anonymized_count += text_anon.anonymized_count

    if text:
        full_content = f"{prompt}\n\n内容：\n{text}"
    else:
        full_content = prompt
    messages = [{"role": "user", "content": full_content}]

    max_retries = 2
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            resp = requests.post(
                f"{config.cloud_api_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.cloud_api_key}"},
                json={
                    "model": config.cloud_model,
                    "messages": messages,
                    "max_tokens": 4096,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = int((time.time() - start) * 1000)
            usage = data.get("usage", {})

            circuit_breaker.record_success()

            result_text = data["choices"][0]["message"]["content"].strip()
            if pf and anon_result and anon_result.anonymized_count > 0:
                result_text = pf.deanonymize(result_text)

            return {
                "text": result_text,
                "tokens_input": usage.get("prompt_tokens", 0),
                "tokens_output": usage.get("completion_tokens", 0),
                "time_ms": elapsed,
            }
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))
                continue
            break

    circuit_breaker.record_failure()

    return {
        "text": f"[云端调用失败] {last_error}",
        "tokens_input": 0, "tokens_output": 0, "time_ms": 0,
        "error": str(last_error),
    }


def _get_privacy_filter() -> Any:
    """延迟加载隐私过滤器（线程安全单例）"""
    global _privacy_filter_instance
    if _privacy_filter_instance is not None:
        return _privacy_filter_instance
    with _privacy_filter_lock:
        # 双重检查锁定
        if _privacy_filter_instance is not None:
            return _privacy_filter_instance
        try:
            from privacy import PrivacyFilter
            _privacy_filter_instance = PrivacyFilter()
        except ImportError:
            pass
    return _privacy_filter_instance
