"""
A3M 路由准确率基准测试 — 50+ 标注用例

验证 A3M 多信号路由在各类任务上的准确率。
每个用例标注了期望路由 (local/cloud) 和任务类型。
"""

import pytest
from routing import Task, estimate_complexity, detect_task_type
from prompts import PROMPT_TEMPLATES


# ─── 测试用例（action, text, expected_route, task_type）─────────

LOCAL_CASES = [
    # 翻译任务
    ("翻译成中文", "Hello World", "translation"),
    ("翻译这段英文", "The quick brown fox jumps over the lazy dog", "translation"),
    ("将以下内容翻译为中文", "Machine learning is transforming industries", "translation"),
    ("帮我翻译", "Please confirm receipt of this email", "translation"),
    ("翻译", "Good morning, how are you today?", "translation"),

    # 分类任务
    ("分类", "iPhone 15 Pro Max 256GB", "general_classify"),
    ("判断情感", "这个产品太好用了，强烈推荐！", "sentiment"),
    ("分类以下文本", "今日A股三大指数集体收涨", "general_classify"),
    ("情感分析", "服务态度极差，等了一个小时", "sentiment"),
    ("判断类别", "患者主诉头痛三天", "general_classify"),

    # 提取任务
    ("提取关键词", "人工智能正在改变制造业的生产方式", "extraction"),
    ("提取所有邮箱", "联系 alice@example.com 或 bob@test.org", "extraction"),
    ("提取日期", "会议定于2024年3月15日下午2点", "extraction"),
    ("提取手机号", "请联系 13812345678 或 13987654321", "extraction"),
    ("摘录重点", "本季度营收增长15%，净利润增长8%", "extraction"),

    # 排序/统计
    ("排序", "5, 3, 8, 1, 9, 2, 7", "sort_numbers"),
    ("统计数量", "苹果,香蕉,橙子,苹果,香蕉,苹果", "_count"),
    ("去重", "apple, banana, apple, cherry, banana", "dedup"),
    ("过滤", "1,2,3,4,5,6,7,8,9,10 中大于5的", "general_classify"),
    ("计数", "文件列表: a.txt, b.py, c.js, d.py", "_count"),

    # 格式化
    ("格式化为表格", "姓名:张三 年龄:25 部门:研发", "general_classify"),
    ("转换为JSON", "name=张三,age=25,dept=研发", "general_classify"),
    ("整理格式", "第一行内容 第二行内容 第三行内容", "general_classify"),
    ("重命名", "IMG_20240315_143022.jpg", "rename_suggest"),
    ("概括", "本文讨论了人工智能在医疗领域的应用前景", "general_classify"),

    # 简单查询
    ("列出", "所有Python文件", "general_classify"),
    ("检查", "这段代码有没有语法错误", "general_classify"),
    ("验证", "这个邮箱格式是否正确", "general_classify"),
    ("是什么类型", "这属于哪个分类", "general_classify"),
    ("属于哪", "这个文件应该放在哪个目录", "general_classify"),

    # 文件操作
    ("整理桌面文件", "桌面上有20个文件需要整理", "general_classify"),
    ("合并", "将以下CSV文件合并", "general_classify"),
    ("拆分", "将这个大文件按行拆分", "general_classify"),
    ("替换", "将所有 'old' 替换为 'new'", "general_classify"),
    ("补全", "补全这个句子：今天天气", "general_classify"),
]

CLOUD_CASES = [
    # 代码生成（复杂）
    ("写一个分布式锁的实现，支持Redis和ZooKeeper两种后端，要求有自动续期和可重入功能", "", "code_generation"),
    ("设计一个高并发消息队列，支持持久化、消费者组、死信队列，用Go实现核心组件", "", "code_generation"),
    ("实现一个Raft共识算法，包含Leader选举、日志复制、安全性保证", "", "code_generation"),
    ("编写一个分布式SQL优化器，能分析查询计划、自动选择索引、生成改写建议", "", "code_generation"),
    ("设计一个分布式事务框架，支持TCC和Saga模式，要有完整的异常处理和补偿机制", "", "code_generation"),

    # 复杂分析（含架构/系统设计关键词）
    ("分析这个系统的架构设计，指出潜在的单点故障和性能瓶颈，给出优化方案和优先级排序", "", "general_classify"),
    ("设计一个微服务架构的服务注册发现系统，支持健康检查和故障转移", "", "general_classify"),
    ("编写一个分布式爬虫框架，支持代理池和反爬虫策略", "", "general_classify"),
    ("分析这段代码的时间复杂度和空间复杂度，找出可以优化的地方并给出具体方案", "", "general_classify"),
    ("评估这个系统的架构设计是否合理，给出重构建议和具体的实施方案", "", "general_classify"),
]

# 边界用例（容易误判的）
EDGE_CASES = [
    # 简单代码任务 → 应该走本地
    ("写一个Hello World", "Python", "local"),
    ("写一个斐波那契函数", "", "local"),
    ("写一个简单的排序", "冒泡排序", "local"),

    # 复合任务 → 应该走本地（拆解后本地执行）
    ("翻译并提取关键词", "AI is changing the world", "local"),
    ("分类并统计", "apple,banana,apple,cherry", "local"),

    # 短任务 → 应该走本地
    ("翻译", "Hi", "local"),
    ("分类", "好", "local"),
    ("提取", "13812345678", "local"),
]


class TestRoutingAccuracy:
    """A3M 路由准确率测试"""

    def _check_route(self, action: str, text: str, expected: str) -> bool:
        """检查路由是否符合预期"""
        task = Task(action=action, text=text)
        decision = estimate_complexity(task)
        return decision["route"] == expected

    def test_local_translation(self):
        """翻译任务应走本地"""
        for action, text, _ in LOCAL_CASES[:5]:
            assert self._check_route(action, text, "local"), f"翻译任务应走本地: {action}"

    def test_local_classification(self):
        """分类任务应走本地"""
        for action, text, _ in LOCAL_CASES[5:10]:
            assert self._check_route(action, text, "local"), f"分类任务应走本地: {action}"

    def test_local_extraction(self):
        """提取任务应走本地"""
        for action, text, _ in LOCAL_CASES[10:15]:
            assert self._check_route(action, text, "local"), f"提取任务应走本地: {action}"

    def test_local_sort_count(self):
        """排序/统计任务应走本地"""
        for action, text, _ in LOCAL_CASES[15:20]:
            assert self._check_route(action, text, "local"), f"排序/统计任务应走本地: {action}"

    def test_local_format(self):
        """格式化任务应走本地"""
        for action, text, _ in LOCAL_CASES[20:25]:
            assert self._check_route(action, text, "local"), f"格式化任务应走本地: {action}"

    def test_local_query(self):
        """查询任务应走本地"""
        for action, text, _ in LOCAL_CASES[25:30]:
            assert self._check_route(action, text, "local"), f"查询任务应走本地: {action}"

    def test_local_file_ops(self):
        """文件操作应走本地"""
        for action, text, _ in LOCAL_CASES[30:]:
            assert self._check_route(action, text, "local"), f"文件操作应走本地: {action}"

    def test_cloud_complex_code(self):
        """复杂代码生成应走云端"""
        for action, text, _ in CLOUD_CASES[:5]:
            assert self._check_route(action, text, "cloud"), f"复杂代码应走云端: {action[:30]}"

    def test_cloud_complex_analysis(self):
        """复杂分析应走云端"""
        for action, text, _ in CLOUD_CASES[5:]:
            assert self._check_route(action, text, "cloud"), f"复杂分析应走云端: {action[:30]}"

    def test_edge_simple_code_local(self):
        """简单代码任务应走本地"""
        for action, text, expected in EDGE_CASES[:3]:
            assert self._check_route(action, text, expected), f"简单代码应走本地: {action}"

    def test_edge_compound_local(self):
        """复合任务应走本地"""
        for action, text, expected in EDGE_CASES[3:5]:
            assert self._check_route(action, text, expected), f"复合任务应走本地: {action}"

    def test_edge_short_tasks_local(self):
        """短任务应走本地"""
        for action, text, expected in EDGE_CASES[5:]:
            assert self._check_route(action, text, expected), f"短任务应走本地: {action}"

    def test_overall_accuracy(self):
        """总体准确率测试"""
        all_cases = (
            [(a, t, "local") for a, t, _ in LOCAL_CASES] +
            [(a, t, "cloud") for a, t, _ in CLOUD_CASES] +
            [(a, t, e) for a, t, e in EDGE_CASES]
        )

        correct = 0
        total = len(all_cases)
        errors = []

        for action, text, expected in all_cases:
            if self._check_route(action, text, expected):
                correct += 1
            else:
                task = Task(action=action, text=text)
                decision = estimate_complexity(task)
                errors.append(f"  期望: {expected}, 实际: {decision['route']}, "
                            f"评分: {decision['score']:.1f}, 原因: {decision['reason'][:50]}")

        accuracy = correct / total * 100
        print(f"\n路由准确率: {correct}/{total} = {accuracy:.1f}%")
        if errors:
            print(f"错误用例 ({len(errors)}):")
            for e in errors[:10]:
                print(e)

        assert accuracy >= 80, f"准确率 {accuracy:.1f}% 低于 80% 阈值"

    def test_task_type_detection(self):
        """任务类型检测准确率"""
        type_cases = [
            ("翻译成中文", "Hello", "translate_en2zh"),
            ("分类这个产品", "iPhone 15", "general_classify"),
            ("判断情感", "太棒了", "sentiment"),
            ("提取关键词", "人工智能改变世界", "extract_keywords"),
            ("排序", "5,3,8,1", "sort_numbers"),
            ("统计数量", "苹果,香蕉,苹果", ""),
            ("去重", "a,b,a,c", "dedup"),
        ]

        for action, text, expected_type in type_cases:
            detected = detect_task_type(action, PROMPT_TEMPLATES)
            assert detected == expected_type, f"类型检测: {action} → {detected}, 期望 {expected_type}"
