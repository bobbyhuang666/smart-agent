#!/usr/bin/env python3
"""
TaskRouter — 自进化的企业级 AI 成本优化引擎

自动路由任务到最佳模型，通过蒸馏闭环让本地模型持续变强。
模块化架构：config / routing / cache / models / prompts / validation / distillation / rules
"""

import sys
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

# ─── 模块导入 ──────────────────────────────────────────────────

from config import get_config, set_config, RouterConfig, TASK_TO_CAPABILITY
from routing import (
    Task, estimate_complexity, detect_task_type,
    decompose_complex_task, _recursive_decompose,
    LOCAL_TASK_PATTERNS, VERB_INTENSITY, MULTI_STEP_CONNECTORS,
    HIGH_COMPLEXITY_DOMAINS, MANDATORY_LOCAL_PATTERNS, CLOUD_PATTERNS,
)
from cache import SemanticCache
from models import call_ollama, call_cloud_api, circuit_breaker
from prompts import (
    PROMPT_TEMPLATES, build_optimized_prompt, get_max_tokens,
    compress_prompt_tokens, TOOL_PREFIX,
)
from rules import rule_execute
from validation import validate_local_output
from distillation import (
    DistillationStore, DistillationPair, collect_distillation_pair,
    get_dynamic_examples, PAIR_SUPPORTED, PAIR_CONTESTED,
    JUDGE_HIGH_THRESHOLD, JUDGE_MODERATE_THRESHOLD, SKIP_JUDGE_CAPABILITIES,
)

# ─── 全局实例（延迟初始化）──────────────────────────────────────

CONFIG = get_config()
cache = SemanticCache(
    cache_dir=CONFIG.cache_dir,
    max_entries=CONFIG.cache_max_entries,
    fuzzy_threshold=CONFIG.cache_fuzzy_threshold,
)
store = DistillationStore(cache_dir=CONFIG.cache_dir)

# 模型注册表（延迟加载）
_model_registry = None
_privacy_filter = None
_cap_tracker = None
_log_lock = threading.Lock()

RECURSE_SCORE_MIN = CONFIG.recurse_score_min
RECURSE_SCORE_MAX = CONFIG.recurse_score_max
DEFAULT_CAPABILITY_THRESHOLD = CONFIG.base_threshold


def get_model_registry() -> Any:
    global _model_registry
    if _model_registry is None:
        from model_registry import ModelRegistry
        _model_registry = ModelRegistry(cache_dir=CONFIG.cache_dir)
        if not _model_registry.models:
            _model_registry.discover()
    return _model_registry


def get_privacy_filter() -> Any:
    global _privacy_filter
    if _privacy_filter is None:
        try:
            from privacy import PrivacyFilter
            _privacy_filter = PrivacyFilter()
        except ImportError:
            pass
    return _privacy_filter


class CapabilityTracker:
    """自适应阈值追踪器"""

    LOOKBACK = 20

    def __init__(self, cache_dir: str):
        self.data_file = os.path.join(cache_dir, "capability_tracker.jsonl")
        os.makedirs(cache_dir, exist_ok=True)

    def record(self, capability: str, success: bool, score: float = 0.0, task_type: str = "") -> None:
        entry = {
            "capability": capability, "success": success, "score": score,
            "task_type": task_type, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.data_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_success_rate(self, capability: str) -> float:
        records = self._load_recent(capability)
        if not records:
            return 0.8  # 默认
        return sum(1 for r in records if r.get("success")) / len(records)

    def get_adjusted_threshold(self, capability: str) -> float:
        rate = self.get_success_rate(capability)
        if rate >= 0.9:
            return DEFAULT_CAPABILITY_THRESHOLD - 0.5
        elif rate >= 0.7:
            return DEFAULT_CAPABILITY_THRESHOLD
        elif rate >= 0.5:
            return DEFAULT_CAPABILITY_THRESHOLD + 0.5
        else:
            return DEFAULT_CAPABILITY_THRESHOLD + 1.0

    def get_all_adjustments(self) -> dict:
        all_caps: set[str] = set()
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
                    "success_rate": round(rate, 2), "threshold": threshold,
                    "adjustment": f"{'+' if diff > 0 else ''}{diff}",
                    "direction": "上调" if diff > 0 else "下调" if diff < 0 else "不变",
                }
        return result

    def _load_recent(self, capability: str, n: int = LOOKBACK) -> list[dict]:
        records: list[dict] = []
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


cap_tracker = CapabilityTracker(CONFIG.cache_dir)


# ─── 预处理/后处理 ──────────────────────────────────────────────

def preprocess_text(text: str, max_chars: int = 800) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "," in text and "\n" not in text:
        items = [x.strip() for x in text.split(",") if x.strip()]
        if len(items) > 3:
            text = "\n".join(items)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (截断，共 {len(text)} 字符)"
    return text


def postprocess_output(output: str, task_type: str = "") -> str:
    if not output:
        return ""
    output = output.strip()
    # 移除模型可能添加的前缀
    for prefix in ["输出：", "结果：", "答案：", "Output:", "Result:", "Answer:"]:
        if output.startswith(prefix):
            output = output[len(prefix):].strip()
    # 情感分析特殊处理
    if task_type == "sentiment":
        if "正面" in output and "负面" not in output:
            return "正面"
        elif "负面" in output and "正面" not in output:
            return "负面"
    return output


def enrich_prompt_with_examples(prompt: str, task_type: str, text: str) -> str:
    """从蒸馏池注入动态 few-shot 示例"""
    capability = TASK_TO_CAPABILITY.get(task_type, "")
    if not capability:
        return prompt
    examples = get_dynamic_examples(capability, store, limit=2)
    if not examples:
        return prompt
    example_block = "\n".join(f"示例: {e.get('prompt', '')[:50]} → {e.get('response', '')[:50]}" for e in examples)
    return f"{example_block}\n\n{prompt}"


# ─── 使用日志 ──────────────────────────────────────────────────

def log_usage(task: Task) -> None:
    log_file = os.path.join(CONFIG.cache_dir, "usage.jsonl")
    os.makedirs(CONFIG.cache_dir, exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "action": task.action[:100],
        "route": task.route,
        "model": task.model_used,
        "tokens_input": task.tokens_input,
        "tokens_output": task.tokens_output,
        "cost_saved": task.cost_saved,
        "time_ms": task.time_ms,
    }
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with _log_lock:
        with open(log_file, "a") as f:
            f.write(line)


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000 * CONFIG.cost_per_1k_input
            + output_tokens / 1000 * CONFIG.cost_per_1k_output)


def calc_savings(task: Task) -> float:
    if task.route != "local":
        return 0.0
    return round(calc_cost(task.tokens_input, task.tokens_output), 6)


# ─── 蒸馏辅助 ──────────────────────────────────────────────────

def auto_collect_on_cloud(task: Task, result: dict) -> None:
    if not result.get("text"):
        return
    pair = collect_distillation_pair(
        prompt=task.action, response=result["text"],
        task_type=detect_task_type(task.action, PROMPT_TEMPLATES),
        route="cloud", action=task.action, model_used=task.model_used,
    )
    store.add_pair(pair)


def auto_collect_on_local_failure(task: Task, local_output: str, cloud_output: str, failure_type: str = "") -> None:
    pair = collect_distillation_pair(
        prompt=task.action, response=cloud_output,
        task_type=detect_task_type(task.action, PROMPT_TEMPLATES),
        route="local_fallback", action=task.action, model_used=task.model_used,
    )
    pair.local_response = local_output
    pair.failure_type = failure_type
    store.add_pair(pair)


# ─── 核心执行 ──────────────────────────────────────────────────

def run_task(task: Task, force_route: str = "") -> Task:
    """执行单个任务（核心入口）"""
    if force_route:
        task.route = force_route
        task.model_used = CONFIG.local_model if force_route == "local" else CONFIG.cloud_model
    else:
        decision = estimate_complexity(task, base_threshold=CONFIG.base_threshold)
        task.route = decision["route"]
        task.model_used = CONFIG.local_model if decision["route"] == "local" else CONFIG.cloud_model

    # 语义缓存查找
    cached = cache.get(task.action, task.text)
    if cached:
        task.output = cached.get("result", {}).get("text", cached.get("output", ""))
        task.route = f"cache({cached.get('match_type', 'hit')})"
        task.model_used = "cache"
        task.tokens_input = 0
        task.tokens_output = 0
        task.time_ms = 0
        task.cost_saved = round(calc_cost(
            cached.get("result", {}).get("tokens_input", 0),
            cached.get("result", {}).get("tokens_output", 0)), 6)
        log_usage(task)
        return task

    if task.route == "local":
        clean_text = preprocess_text(task.text or "")
        clean_action = preprocess_text(task.action, max_chars=200)

        # 尝试拆解复合任务
        subtasks = decompose_complex_task(clean_action, clean_text)
        if not subtasks:
            subtasks = _recursive_decompose(clean_action, clean_text)
        if subtasks:
            outputs: list[str] = []
            for st in subtasks:
                st_type = st.get("type", "")
                st_action = st.get("action", "")
                st_text = st.get("text", clean_text)
                rule_result = rule_execute(st_type, st_text)
                if rule_result:
                    outputs.append(rule_result)
                    continue
                st_prompt = build_optimized_prompt(st_type, st_action, st_text, task.files)
                st_prompt = enrich_prompt_with_examples(st_prompt, st_type, st_text)
                result = call_ollama(st_prompt, max_tokens=get_max_tokens(st_type))
                out = postprocess_output(result["text"], st_type)
                outputs.append(f"[{st_type}]\n{out}")
                task.tokens_input += result["tokens_input"]
                task.tokens_output += result["tokens_output"]
                task.time_ms += result["time_ms"]
            task.output = "\n\n".join(outputs)
            task.model_used = CONFIG.local_model
            task.cost_saved = calc_savings(task)
            cache.set(task.action, task.text,
                      {"text": task.output, "tokens_input": task.tokens_input, "tokens_output": task.tokens_output})
            log_usage(task)
            return task

        # 单任务
        task_type = detect_task_type(clean_action, PROMPT_TEMPLATES)

        # 规则执行
        rule_result = rule_execute(task_type, clean_text or clean_action)
        if rule_result:
            task.output = rule_result
            task.model_used = "rule_engine"
            task.route = "local(rule)"
            cache.set(task.action, task.text, {"text": rule_result, "tokens_input": 0, "tokens_output": 0})
            log_usage(task)
            return task

        # 智能模型选择
        selected_model: Optional[str] = None
        try:
            registry = get_model_registry()
            if registry.models:
                capability = TASK_TO_CAPABILITY.get(task_type, "")
                if capability:
                    best = registry.select_best(capability, prefer_speed=True)
                    if best and best.name != CONFIG.local_model:
                        selected_model = best.name
        except Exception:
            pass

        prompt = build_optimized_prompt(task_type, clean_action, clean_text, task.files)
        prompt = enrich_prompt_with_examples(prompt, task_type, clean_text)
        prompt = compress_prompt_tokens(prompt, task_type)
        max_tokens = get_max_tokens(task_type)
        result = call_ollama(prompt, model=selected_model, max_tokens=max_tokens)
        task.output = postprocess_output(result["text"], task_type)
        task.model_used = selected_model or CONFIG.local_model

        # 输出验证 + 云端降级
        validation = validate_local_output(task.output, task_type)
        cap = TASK_TO_CAPABILITY.get(task_type, "")
        if cap:
            cap_tracker.record(cap, success=validation["valid"],
                               score=1.0 if validation["valid"] else 0.3, task_type=task_type)

        if not validation["valid"] and CONFIG.cloud_api_key:
            cloud_result = call_cloud_api(task.action, task.text)
            if cloud_result.get("circuit_open"):
                task.output += f"\n\n[警告: 本地输出质量不佳，云端熔断中 - {validation['reason']}]"
                task.route = "local(degraded)"
            else:
                auto_collect_on_local_failure(task, task.output, cloud_result["text"], "quality_fallback")
                task.output = cloud_result["text"]
                task.route = "cloud_fallback"
                task.model_used = CONFIG.cloud_model
                result = cloud_result
                task.cost_saved = 0
        elif not validation["valid"]:
            task.output += f"\n\n[警告: 本地输出质量可能不佳 - {validation['reason']}]"
    else:
        # 云端任务：尝试递归拆解
        if not force_route:
            decision = estimate_complexity(task, base_threshold=CONFIG.base_threshold)
        else:
            decision = {"score": 9, "route": force_route}
        if RECURSE_SCORE_MIN <= decision["score"] <= RECURSE_SCORE_MAX:
            subtasks = _recursive_decompose(task.action, task.text)
            if subtasks:
                outputs = []
                for st in subtasks:
                    st_action = st.get("action", st.get("type", ""))
                    st_text = st.get("text", task.text)
                    st_task = Task(action=st_action, text=st_text)
                    st_result = run_task(st_task)
                    outputs.append(f"[{st_action[:30]}]({st_result.route})\n{st_result.output}")
                    task.tokens_input += st_result.tokens_input
                    task.tokens_output += st_result.tokens_output
                    task.time_ms += st_result.time_ms
                    task.cost_saved += st_result.cost_saved
                task.output = "\n\n---\n\n".join(outputs)
                task.route = "hybrid(recurse)"
                task.model_used = "mixed"
                if task.output and task.output.strip():
                    cache.set(task.action, task.text,
                              {"text": task.output, "tokens_input": task.tokens_input, "tokens_output": task.tokens_output})
                log_usage(task)
                return task

        result = call_cloud_api(task.action, task.text)
        task.output = result["text"]
        auto_collect_on_cloud(task, result)

    task.tokens_input = result.get("tokens_input", 0)
    task.tokens_output = result.get("tokens_output", 0)
    task.time_ms = result.get("time_ms", 0)
    task.cost_saved = calc_savings(task)

    if task.output and len(task.output.strip()) >= 1:
        cache.set(task.action, task.text,
                  {"text": task.output, "tokens_input": task.tokens_input, "tokens_output": task.tokens_output},
                  ttl_hours=CONFIG.get_cache_ttl(detect_task_type(task.action, PROMPT_TEMPLATES)))
    log_usage(task)

    # 审计日志
    try:
        from audit import get_audit_logger, AuditEvent
        audit = get_audit_logger()
        audit.log(AuditEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            event_type="task_execution",
            action=task.action[:100],
            details={"route": task.route, "model": task.model_used,
                     "tokens_input": task.tokens_input, "tokens_output": task.tokens_output,
                     "cost_saved": task.cost_saved, "output_length": len(task.output or "")},
            status="success" if task.route != "error" else "failure",
            duration_ms=task.time_ms,
        ))
    except Exception:
        pass

    return task


# ─── 批量执行 ──────────────────────────────────────────────────

def run_batch(tasks_data: list[dict], concurrency: int = 1) -> list[Task]:
    """批量执行任务"""
    tasks = [Task(action=t.get("action", t.get("prompt", "")),
                  text=t.get("text", t.get("input", "")),
                  files=t.get("files", []))
             for t in tasks_data]

    if concurrency <= 1:
        return [run_task(t) for t in tasks]

    results: list[Optional[Task]] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {executor.submit(run_task, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = Task(action=tasks[idx].action, output=f"[错误] {e}", route="error")
    return [r for r in results if r is not None]


# ─── 预估和分类 ──────────────────────────────────────────────────

def estimate(task_description: str) -> dict:
    dummy = Task(action=task_description)
    decision = estimate_complexity(dummy, base_threshold=CONFIG.base_threshold)
    est_input = max(50, len(task_description) // 2)
    est_output = max(50, len(task_description))
    return {
        "task": task_description[:100],
        "suggested_route": decision["route"],
        "reason": decision["reason"],
        "score": decision["score"],
        "estimated_cloud_cost": f"${calc_cost(est_input, est_output):.6f}",
        "will_save": decision["route"] == "local",
    }


def classify_task(task_desc: str, text: str = "") -> dict:
    """细粒度任务分类"""
    task = Task(action=task_desc, text=text)
    decision = estimate_complexity(task, base_threshold=CONFIG.base_threshold)
    task_type = detect_task_type(task_desc, PROMPT_TEMPLATES)

    return {
        "task": task_desc[:100],
        "task_type": task_type,
        "verdict": decision["route"],
        "score": decision["score"],
        "reason": decision["reason"],
        "confidence": "high" if abs(decision["score"] - CONFIG.base_threshold) > 2 else "medium",
        "text_length": len(text),
    }


# ─── 统计 ──────────────────────────────────────────────────────

def show_usage_stats() -> str:
    log_file = os.path.join(CONFIG.cache_dir, "usage.jsonl")
    if not os.path.exists(log_file):
        return "暂无使用记录"

    total_local = total_cloud = total_cache = 0
    total_input = total_output = 0
    total_saved = 0.0
    daily_stats: dict[str, dict] = {}

    with open(log_file) as f:
        for line in f:
            try:
                e = json.loads(line.strip())
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

            day = e.get("date", e.get("time", "")[:10])
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

    cache_stats = cache.stats()
    result = (
        f"TaskRouter 使用统计\n{'='*40}\n"
        f"本地调用: {total_local} 次\n云端调用: {total_cloud} 次\n缓存命中: {total_cache} 次\n"
        f"总输入 tokens: {total_input:,}\n总输出 tokens: {total_output:,}\n累计节约成本: ${total_saved:.4f}\n"
    )
    if cache_stats["total"] > 0:
        result += f"\n语义缓存:\n  缓存条目: {cache_stats['total']}\n  缓存节约: ${cache_stats['estimated_cost_saved']:.4f}\n"

    if daily_stats:
        sorted_days = sorted(daily_stats.keys(), reverse=True)[:7]
        if sorted_days:
            result += f"\n每日统计 (最近{len(sorted_days)}天):\n"
            for day in sorted_days:
                d = daily_stats[day]
                total_day = d["local"] + d["cloud"] + d["cache"]
                result += f"  {day}: {total_day}次 (本地{d['local']}/云端{d['cloud']}/缓存{d['cache']}) 省${d['saved']:.4f}\n"

    if circuit_breaker.failures > 0:
        if circuit_breaker.is_open():
            result += f"\n云端熔断器: 🔴 熔断中 (连续失败{circuit_breaker.failures}次, {circuit_breaker.remaining_seconds()}秒后恢复)\n"
        else:
            result += f"\n云端熔断器: 🟡 已恢复 (上次连续失败{circuit_breaker.failures}次)\n"

    return result


# ─── 任务计划执行 ──────────────────────────────────────────────

DECOMPOSE_TEMPLATES: dict[str, dict] = {
    "电商数据分析": {
        "match": ["电商", "销售", "商品", "订单"],
        "subtasks": ["清洗数据（去空值、统一格式）", "按类别分类商品", "统计各品类销售额", "分析销售趋势和异常", "给出优化建议并生成报告"],
        "routes": ["local", "local", "local", "cloud", "cloud"],
    },
    "文件批量整理": {
        "match": ["文件", "桌面", "整理", "归类"],
        "subtasks": ["扫描并列出所有文件", "按扩展名分类", "建议目录结构", "生成整理脚本"],
        "routes": ["local", "local", "local", "cloud"],
    },
}


def decompose_task(task_description: str, text_content: str = "") -> dict:
    """将大任务拆解为子任务列表"""
    desc = task_description.lower()
    for template_name, template in DECOMPOSE_TEMPLATES.items():
        if any(kw in desc for kw in template["match"]):
            subtasks = [{"id": i + 1, "action": sa, "text": text_content, "route": r}
                        for i, (sa, r) in enumerate(zip(template["subtasks"], template["routes"]))]
            return {"task": task_description[:200], "template": template_name,
                    "total_subtasks": len(subtasks),
                    "local_count": sum(1 for s in subtasks if s["route"] == "local"),
                    "cloud_count": sum(1 for s in subtasks if s["route"] == "cloud"),
                    "subtasks": subtasks}

    # 自动检测
    cl = classify_task(task_description, text_content)
    return {"task": task_description[:200], "template": "auto", "total_subtasks": 1,
            "local_count": 1 if cl["verdict"] == "local" else 0,
            "cloud_count": 1 if cl["verdict"] == "cloud" else 0,
            "subtasks": [{"id": 1, "action": task_description[:200], "text": text_content,
                          "route": cl["verdict"], "reason": cl["reason"]}]}


def execute_plan(plan: dict) -> dict:
    """执行任务计划"""
    results: list[dict] = []
    total_input = total_output = total_time = 0
    total_saved = 0.0

    for i, step in enumerate(plan.get("subtasks", [])):
        task = Task(action=step["action"], text=step.get("text", ""))
        task = run_task(task, force_route=step.get("route", ""))
        total_input += task.tokens_input
        total_output += task.tokens_output
        total_time += task.time_ms
        total_saved += task.cost_saved
        results.append({"id": step.get("id", i + 1), "action": step["action"],
                        "route": task.route, "output": task.output,
                        "tokens_input": task.tokens_input, "tokens_output": task.tokens_output,
                        "time_ms": task.time_ms, "cost_saved": task.cost_saved})

    return {"task": plan.get("task", ""), "total_steps": len(results),
            "local_steps": sum(1 for r in results if r["route"] == "local"),
            "cloud_steps": sum(1 for r in results if "cloud" in r["route"]),
            "total_tokens_input": total_input, "total_tokens_output": total_output,
            "total_time_ms": total_time, "total_cost_saved": round(total_saved, 6), "steps": results}


# ─── CLI ──────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TaskRouter — 自进化 AI 成本优化引擎")
    parser.add_argument("--task", "-t", help="任务描述")
    parser.add_argument("--text", "-T", help="待处理的文本内容")
    parser.add_argument("--file", "-f", action="append", help="待处理的文件路径")
    parser.add_argument("--force", choices=["local", "cloud"], help="强制路由")
    parser.add_argument("--batch", "-b", help="批量任务 JSON 文件路径")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--estimate", "-e", help="预估路由")
    parser.add_argument("--decompose", "-d", nargs="*", help="拆解大任务")
    parser.add_argument("--plan", "-p", help="执行任务计划文件")
    parser.add_argument("--stats", action="store_true", help="使用统计")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--distill", action="store_true", help="运行蒸馏")
    parser.add_argument("--distill-stats", action="store_true", help="蒸馏状态")
    parser.add_argument("--thresholds", action="store_true", help="自适应阈值")
    parser.add_argument("--models", action="store_true", help="模型列表")
    parser.add_argument("--benchmark", nargs="?", const="all", help="基准测试")
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
            print(f"任务: {result['task']}\n建议路由: {route_tag}\n原因: {result['reason']}\n预估云端成本: {result['estimated_cloud_cost']}")
        return

    if args.decompose is not None:
        task_desc = " ".join(args.decompose) if args.decompose else input("请输入要拆解的大任务: ").strip()
        if not task_desc:
            print("任务描述不能为空")
            return
        plan = decompose_task(task_desc, args.text or "")
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            print(f"大任务: {plan['task']}\n拆解为 {plan['total_subtasks']} 个子任务")
            print(f"  🟢 本地: {plan['local_count']} | 🔵 云端: {plan['cloud_count']}")
            for s in plan["subtasks"]:
                icon = "🟢" if s["route"] == "local" else "🔵"
                print(f"  [{s['id']}] {icon} {s['action']}")
        return

    if args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        result = execute_plan(plan)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for step in result["steps"]:
                icon = "🟢" if step["route"] == "local" else "🔵"
                print(f"  [{step['id']}] {icon} {step['action']} | {step['route']} | {step['time_ms']}ms")
            print(f"\n完成! 本地: {result['local_steps']} | 云端: {result['cloud_steps']} | 节约: ${result['total_cost_saved']:.4f}")
        return

    if args.thresholds:
        adjs = cap_tracker.get_all_adjustments()
        if not adjs:
            print("暂无自适应阈值数据")
            return
        for cap, info in adjs.items():
            print(f"  {cap:20s} | 成功率: {info['success_rate']:.0%} | 阈值: {info['threshold']:.1f} ({info['direction']})")
        return

    if args.distill_stats:
        stats = store.get_stats()
        print(f"蒸馏统计: 总计 {stats['total']} | SUPPORTED {stats['supported']}")
        for cap, count in stats.get("by_capability", {}).items():
            print(f"  {cap}: {count}")
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
        from dataclasses import asdict
        print(json.dumps(asdict(task), ensure_ascii=False, indent=2))
    else:
        route_icon = "[LOCAL]" if task.route == "local" else "[CLOUD]"
        print(f"{route_icon} {task.model_used}")
        print(f"耗时: {task.time_ms}ms | 输入: {task.tokens_input} | 输出: {task.tokens_output} | 节约: ${task.cost_saved:.6f}")
        print("-" * 50)
        print(task.output)


if __name__ == "__main__":
    main()
