#!/usr/bin/env python3
"""调用本地 Ollama 模型的简单封装，供 Claude 委托任务使用。"""

import sys
import json
import requests
import subprocess

OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"
MAX_TOKENS = 2048


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """调用 Ollama 模型，返回文本结果"""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {
            "num_predict": MAX_TOKENS,
        }},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "请提供 prompt 参数"}))
        sys.exit(1)

    prompt = sys.argv[1]

    try:
        result = call_ollama(prompt)
        print(result)
    except requests.exceptions.ConnectionError:
        print(json.dumps({"error": "无法连接到 Ollama，请确保已运行: ollama serve"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
