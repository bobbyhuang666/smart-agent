#!/usr/bin/env python3
"""
TaskRouter CLI — 命令行接口

从 task_router 核心引擎分离，负责参数解析和用户交互。
"""

import sys
import json

from task_router import (
    Task, run_task, estimate, show_usage_stats,
    decompose_task, execute_plan, get_model_registry, store,
    cap_tracker,
)


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
    parser.add_argument("--distill-cleanup", action="store_true", help="清除过期蒸馏条目")
    parser.add_argument("--thresholds", action="store_true", help="自适应阈值")
    parser.add_argument("--models", action="store_true", help="模型列表")
    parser.add_argument("--benchmark", nargs="?", const="all", help="基准测试")
    parser.add_argument("--weights", action="store_true", help="查看 A3M 权重状态")
    parser.add_argument("--weights-reset", action="store_true", help="重置 A3M 权重")
    parser.add_argument("--cascade", action="store_true", help="置信度级联统计")
    args = parser.parse_args()

    if args.stats:
        print(show_usage_stats())
        return

    if args.weights:
        from weights import get_weight_tracker
        wt = get_weight_tracker()
        stats = wt.get_stats()
        w = wt.get_weights()
        print("A3M 可学习权重状态")
        print(f"  学习数据: {stats['total']} 条记录")
        print(f"  本地成功率: {stats.get('local_success_rate', 0):.0%}")
        print(f"  当前阈值: {stats['current_threshold']:.3f} (默认: 3.0)")
        print(f"  学习率: {stats['learning_rate']}")
        print("\n权重参数:")
        for k, v in w.to_dict().items():
            print(f"  {k}: {v}")
        return

    if args.weights_reset:
        from weights import get_weight_tracker
        wt = get_weight_tracker()
        wt.reset()
        print("A3M 权重已重置为默认值")
        return

    if args.cascade:
        from task_router import get_cascade
        cascade = get_cascade()
        stats = cascade.get_stats()
        print("置信度门控级联统计")
        print(f"  总任务数: {stats['total']}")
        print(f"  升级到云端: {stats['escalated']}")
        print(f"  本地保留: {stats['local_kept']}")
        print(f"  升级率: {stats['escalation_rate']:.0%}")
        print(f"  本地准确率: {stats['local_accuracy']:.0%}")
        cal = stats.get('calibration', {})
        print(f"  校准状态: {'已校准' if cal.get('is_calibrated') else '未校准（需 ≥20 个样本）'}")
        print(f"  校准样本数: {cal.get('total_samples', 0)}")
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

    if args.distill_cleanup:
        removed = store.cleanup_expired()
        print(f"已清除 {removed} 条过期蒸馏条目")
        return

    if args.distill_stats:
        stats = store.get_stats()
        print(f"蒸馏统计: 总计 {stats['total']} | 活跃 {stats['active']} | 过期 {stats['expired']} | TTL {stats['ttl_days']}天")
        print(f"  SUPPORTED: {stats['supported']}")
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
