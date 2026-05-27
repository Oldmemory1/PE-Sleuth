# api_client.py
# -*- coding: utf-8 -*-
"""
DeepSeek V4 Pro API 客户端（OpenAI 兼容协议）。

替代原 model_server.py 中的 LLMInterface / LoRALLMInterface，
将本地模型推理替换为远程 HTTP API 调用。

用法:
    from api_client import APIClient, APIConfig

    config = APIConfig()
    client = APIClient(config)
    reply = client.chat("Hello, classify this code...")
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from openai import OpenAI, APIError, RateLimitError

# =========================
# Configuration
# =========================


class APIConfig:
    """
    从 settings.json 读取 API 连接参数和生成参数。

    对应的 settings.json 格式 (OpenAI 兼容):
    {
        "model_name": "deepseek-chat",
        "api_key": "sk-xxxxxxxx",
        "base_url": "https://api.deepseek.com",
        "max_new_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_retries": 5,
        "retry_backoff_sec": 1.0
    }
    """

    def __init__(self, settings_path: Optional[Path] = None):
        if settings_path is None:
            settings_path = Path(__file__).resolve().parent / "settings.json"

        with open(settings_path, "r", encoding="utf-8") as f:
            settings: Dict[str, Any] = json.load(f)

        # ── API 连接 ──
        self.MODEL_NAME: str = settings.get("model_name", "deepseek-chat")
        self.API_KEY: str = settings.get("api_key", "")
        self.BASE_URL: str = settings.get("base_url", "https://api.deepseek.com")

        # ── 生成参数 ──
        self.MAX_NEW_TOKENS: int = settings.get("max_new_tokens", 512)
        self.TEMPERATURE: float = settings.get("temperature", 0.5)
        self.TOP_P: float = settings.get("top_p", 0.9)

        # ── 重试策略 ──
        self.MAX_RETRIES: int = settings.get("max_retries", 5)
        self.RETRY_BACKOFF_SEC: float = settings.get("retry_backoff_sec", 1.0)

        # ── 速率控制（每次调用的间隔秒数，避免触发限流） ──
        self.REQUEST_DELAY_SEC: float = settings.get("request_delay_sec", 0.3)


# =========================
# API Client
# =========================


class APIClient:
    """
    OpenAI 兼容的 API 客户端。

    替代:
        - model_server.LLMInterface (基础模型)
        - model_server.LoRALLMInterface (LoRA 微调模型)

    提供:
        - chat(): 发送对话请求
        - count_tokens(): token 计数估算
    """

    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()

        if not self.config.API_KEY or self.config.API_KEY == "your-api-key-here":
            raise ValueError(
                "API key 未配置。请在 settings.json 中设置 'api_key' 字段"
            )

        self.client = OpenAI(
            api_key=self.config.API_KEY,
            base_url=self.config.BASE_URL,
            timeout=120.0,
        )

    # ─── 核心方法 ───

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """
        发送 chat completion 请求。

        参数:
            prompt:        用户消息内容
            system_prompt:  system role 消息（用于需要强指令约束的场景）

        返回:
            str: 模型回复文本（去除首尾空白）

        自动处理:
            - 429 限流 → 指数退避重试
            - 5xx 服务端错误 → 重试
            - 网络超时/异常 → 重试
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Optional[Exception] = None

        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    max_tokens=self.config.MAX_NEW_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""

            except RateLimitError:
                wait = self.config.RETRY_BACKOFF_SEC * (2 ** attempt)
                logging.warning(
                    f"API 限流 (429)，{wait:.1f}s 后重试 "
                    f"(attempt {attempt + 1}/{self.config.MAX_RETRIES})"
                )
                time.sleep(wait)
                last_error = None

            except APIError as e:
                if e.status_code is not None and 500 <= e.status_code < 600:
                    wait = self.config.RETRY_BACKOFF_SEC * (2 ** attempt)
                    logging.warning(
                        f"服务端错误 ({e.status_code})，{wait:.1f}s 后重试"
                    )
                    time.sleep(wait)
                    last_error = e
                else:
                    raise RuntimeError(
                        f"API 调用失败 (status={e.status_code}): {e}"
                    ) from e

            except Exception as e:
                last_error = e
                if attempt < self.config.MAX_RETRIES - 1:
                    wait = self.config.RETRY_BACKOFF_SEC * (2 ** attempt)
                    logging.warning(
                        f"请求异常: {e}, {wait:.1f}s 后重试 "
                        f"(attempt {attempt + 1}/{self.config.MAX_RETRIES})"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"API 调用失败（{self.config.MAX_RETRIES} 次重试后仍不成功）: {last_error}"
        )

    # ─── Token 计数 ───

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        使用 tiktoken 估算 token 数。

        编码: cl100k_base（OpenAI GPT-4 / DeepSeek 兼容的 BPE 编码器）
        Fallback: 字符数 / 3.5（粗略估算）

        返回:
            int: 估算的 token 数量
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, int(len(text) / 3.5))
