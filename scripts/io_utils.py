"""
I/O 工具 — 共享的 JSONL 读写和文件操作
"""

import os
import json
from typing import Optional


def read_jsonl(path: str) -> list[dict]:
    """读取 JSONL 文件，跳过损坏行"""
    if not os.path.exists(path):
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def append_jsonl(path: str, entry: dict) -> None:
    """追加单条记录到 JSONL 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_jsonl(path: str, entries: list[dict]) -> None:
    """全量重写 JSONL 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
