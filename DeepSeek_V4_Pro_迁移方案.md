# PE-Sleuth 迁移至 DeepSeek V4 Pro API 方案

> **目标：** 复现论文，不微调（skip LoRA），用 DeepSeek V4 Pro 远程 API 替代本地 Qwen3-14B。

---

## 目录

1. [改动总览](#1-改动总览)
2. [需新增：api_client.py](#2-需新增-api_clientpy-约150行)
3. [需改动：classify_from_c_code.py](#3-需改动-classify_from_c_codepy-约10处)
4. [需改动：batch_decompile_ida.py](#4-需改动-batch_decompile_idapy-不改动)
5. [需删除/可跳过](#5-需删除可跳过)
6. [风险点与缓解方案](#6-风险点与缓解方案)
7. [推荐实施顺序](#7-推荐实施顺序)

---

## 1. 改动总览

```
现有架构                              目标架构
─────────                           ─────────
model_server.py                     api_client.py (新建 ~150 行)
  ├─ ModelConfig                      ├─ APIConfig
  ├─ 4-bit / FlashAttention / YaRN   ├─ (删除 — API 无本地模型)
  ├─ _load_tokenizer                 ├─ count_tokens() (tiktoken 估算)
  ├─ _load_base_model_only           ├─ (删除)
  ├─ _decode_chat_response           ├─ (保留但重写)
  ├─ LLMInterface + unload()         ├─ APIClient (纯 HTTP)
  └─ LoRALLMInterface + unload()     └─ (删除 — 不做微调)

classify_from_c_code.py  改动 ~10 处   classify_from_c_code.py
  ─ 导入、main() 模型管理、TokenCounter
  ─ 消融实验逻辑不变

batch_decompile_ida.py                batch_decompile_ida.py
  ─ 不改动 (仍在 Windows 主机跑 IDA)
```

**改动量：** 1 个新文件 + 1 个文件部分修改。Steps 1A~1B~2~3（纯静态分析步骤）完全不用动。

---

## 2. 需新增：`api_client.py`（~150 行）

### 2.1 `APIConfig`（替代原 `ModelConfig`）

```python
import os

class APIConfig:
    # ── API 连接 ──
    API_KEY: str       = os.getenv("DEEPSEEK_API_KEY", "")
    BASE_URL: str      = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    MODEL_NAME: str    = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ── 生成参数 ──
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float  = 0.5
    TOP_P: float        = 0.9

    # ── 重试策略 ──
    MAX_RETRIES: int       = 5
    RETRY_BACKOFF_SEC: float = 1.0

    # ── 速率控制 ──
    REQUEST_DELAY_SEC: float = 0.3  # 调用间隔避免触发限制
```

**被删除的原 `ModelConfig` 属性：**

| 删除 | 原因 |
|------|------|
| `BASE_MODEL_PATH` | 无本地模型 |
| `LORA_WEIGHTS_PATH` | 不做微调 |
| `QUANTIZATION_BITS` | API 不量化 |
| `USE_FLASH_ATTENTION` | API 侧重推理优化 |
| `ENABLE_THINKING` | Qwen 特有标记 |
| `DEVICE` | 无 GPU 概念 |
| `REPO_ROOT` | 无本地路径 |

---

### 2.2 `APIClient`（替代原 `LLMInterface` + `LoRALLMInterface` + `_GenerationMixin`）

```python
import time
import logging
from typing import List, Dict
from openai import OpenAI, APIError, RateLimitError

class APIClient:
    """
    DeepSeek API 客户端（OpenAI 兼容协议）。
    替代原 LLMInterface / LoRALLMInterface。
    """

    def __init__(self, config: APIConfig):
        if not config.API_KEY:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
        self.config = config
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            timeout=120.0,
        )

    # ─── 核心方法：替代原 _GenerationMixin.generate() ───

    def chat(self, prompt: str,
             system_prompt: str = "") -> str:
        """
        发送 chat completion 请求到 DeepSeek。

        参数:
            prompt:       用户消息内容
            system_prompt: 可选的 system role 消息（用于分类等需要强指令的场景）

        返回:
            模型回复文本

        自动处理:
            - 429 限流 → 指数退避重试
            - 5xx 服务端错误 → 重试
            - 超时 → 重试
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=messages,
                    max_tokens=self.config.MAX_NEW_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    top_p=self.config.TOP_P,
                )
                return response.choices[0].message.content or ""

            except RateLimitError:
                wait = self.config.RETRY_BACKOFF_SEC * (2 ** attempt)
                logging.warning(f"API 限流，{wait:.1f}s 后重试 (attempt {attempt+1})")
                time.sleep(wait)

            except APIError as e:
                last_error = e
                if e.status_code and e.status_code >= 500:
                    wait = self.config.RETRY_BACKOFF_SEC * (2 ** attempt)
                    logging.warning(f"服务端错误 {e.status_code}，{wait:.1f}s 后重试")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API 调用失败: {e}") from e

            except Exception as e:
                last_error = e
                wait = self.config.RETRY_BACKOFF_SEC * (2 ** attempt)
                logging.warning(f"请求异常: {e}，{wait:.1f}s 后重试")
                time.sleep(wait)

        raise RuntimeError(f"API 调用失败（重试 {self.config.MAX_RETRIES} 次后）: {last_error}")

    # ─── Token 计数：替代原 _load_tokenizer ───

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        使用 tiktoken 估算 token 数（cl100k_base 编码）。
        Fallback: 字符数 / 3.5（如果 tiktoken 未安装）。
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, int(len(text) / 3.5))
```

**接口映射（新旧对照）：**

| 原调用 | 新调用 | 备注 |
|--------|--------|------|
| `LLMInterface(ModelConfig())` | `APIClient(APIConfig())` | 构造函数 |
| `llm.generate(prompt)` | `client.chat(prompt)` | 对话生成 |
| `llm.unload()` | *(删除)* | API 无需释放显存 |
| `llm.get_tokenizer()` | `APIClient.count_tokens(text)` | 变为静态方法 |
| `LoRALLMInterface(ModelConfig())` | *(删除)* | 不做微调 |

---

## 3. 需改动：`classify_from_c_code.py`（~10 处）

### 3.1 导入行 [classify_from_c_code.py:21](classify_from_c_code.py#L21)

```diff
- from model_server import LLMInterface, LoRALLMInterface, ModelConfig
+ from api_client import APIClient, APIConfig
```

---

### 3.2 `ExperimentConfig` — 删除一个字段

```diff
class ExperimentConfig:
    ...
-   USE_LORA_FOR_CLASSIFICATION: bool = True       # ← 删除
-   RUN_ABLATION_E_META: bool = True                # ← 保留不变
-   RUN_ABLATION_E_SUMMARY: bool = True             # ← 保留不变
```

> `CHUNK_TOKEN_SIZE` 和 `MAX_METADATA_BYTES` 暂时保持原值（30K / 4KB）。DeepSeek V4 支持 128K 上下文，但初次运行保守不变，后续可酌情调大。

---

### 3.3 `TokenCounter` 类 [classify_from_c_code.py:193-210](classify_from_c_code.py#L193-L210)

```diff
class TokenCounter:
-   def __init__(self, llm: Optional[LLMInterface]):
+   def __init__(self, client: Optional[APIClient]):
-       self.llm = llm
+       self.client = client
-       self.tok = None
-       if llm is not None:
-           try:
-               self.tok = llm.get_tokenizer()
-           except Exception as e:
-               logging.warning(...)

    def count(self, text: str) -> int:
-       if self.tok:
-           try:
-               ids = self.tok.encode(text, add_special_tokens=False)
-               return len(ids)
-           except Exception:
-               pass
-       return max(1, int(len(text) / 3.8))
+       if self.client:
+           return self.client.count_tokens(text)
+       return max(1, int(len(text) / 3.8))
```

---

### 3.4 `ask_llm()` 函数 [classify_from_c_code.py:215-222](classify_from_c_code.py#L215-L222)

```diff
- def ask_llm(llm: LLMInterface, prompt: str, max_retries: int = 3, backoff: float = 0.7) -> str:
+ def ask_llm(client: APIClient, prompt: str,
+              system_prompt: str = "",
+              max_retries: int = 3, backoff: float = 0.7) -> str:
      last = None
      for i in range(max_retries):
          try:
-             return llm.generate(prompt=prompt)
+             return client.chat(prompt=prompt, system_prompt=system_prompt)
          except Exception as e:
              last = e
              time.sleep(backoff * (2 ** i))
      raise RuntimeError(f"LLM generation failed after retries: {last}")
```

> **说明：** 新增 `system_prompt` 参数。原有的 `ask_llm` 调用如果不传该参数行为不变（兼容旧调用）。

---

### 3.5 `main()` 函数 — **大幅简化** [classify_from_c_code.py:1065-1332](classify_from_c_code.py#L1065-L1332)

**原代码（~80 行，涉及多次模型加载/卸载）：**

```python
def main():
    ...
    base_llm = None
    tc = TokenCounter(None)

    try:
        base_llm = LLMInterface(ModelConfig())     # ① 本地加载 14B 模型（~28GB VRAM）
        tc = TokenCounter(base_llm)
    except Exception as e:
        ...

    # Step 4-6 用 base_llm
    ...

    # Step 7:
    if use_lora:
        base_llm.unload()                           # ② 卸载
        base_llm = None
        clf_llm = LoRALLMInterface(ModelConfig())   # ③ 重新加载 + 应用 LoRA
    else:
        clf_llm = base_llm
    # ... 分类 ...
    if use_lora:
        clf_llm.unload()                            # ④ 卸载 LoRA

    # Step 8:
    if base_llm is None:
        base_llm = LLMInterface(ModelConfig())      # ⑤ 再次加载基础模型
    # ... 理由生成 ...
```

**新代码（~15 行，无任何模型管理逻辑）：**

```python
def main():
    ...
    # 一个 API 客户端贯穿全部 LLM 步骤
    client = APIClient(APIConfig())
    tc = TokenCounter(client)

    # Step 4-5-7-8 全部用同一个 client（无加载/卸载/切换）
    ...
```

**具体改动点：**

```diff
  def main():
      setup_logging()
      ...
      files = [p for p in input_dir.rglob("*.c") if p.is_file()]
      ...

-     base_llm = None
-     tc = TokenCounter(None)
-     try:
-         base_llm = LLMInterface(ModelConfig())
-         tc = TokenCounter(base_llm)
-     except Exception as e:
-         logging.exception(...)
-         return
+     client = APIClient(APIConfig())
+     tc = TokenCounter(client)

      # ── Steps 1A, 1B, 2, 3 ── 不变

      # ── Step 4 ──
-     if base_llm is None:
-         logging.error(...)
-         return
      for p in tqdm(files, desc="Step 4: chunk summaries"):
          ...
-         block_summaries = summarize_blocks(base_llm, chunks, ...)
+         block_summaries = summarize_blocks(client, chunks, ...)

      # ── Step 5 ──
-     if base_llm is None:
-         logging.error(...)
-         return
      for p in tqdm(files, desc="Step 5: program summary"):
          ...
-         program_summary = summarize_program(base_llm, block_summaries, ...)
+         program_summary = summarize_program(client, block_summaries, ...)

      # ── Step 6 ── 不变（无 LLM 调用）

      # ── Step 7 ── 大幅简化
-     use_lora = ExperimentConfig.USE_LORA_FOR_CLASSIFICATION
-     if use_lora:
-         base_llm.unload()  # 卸载
-         clf_llm = LoRALLMInterface(ModelConfig())  # 加载 LoRA
-     else:
-         clf_llm = base_llm

      for p in tqdm(files, desc="Step 7: classification"):
          ...
-         out_full = ask_llm(clf_llm, prompt_full)
+         out_full = ask_llm(client, prompt_full)
          ...  # 消融实验同样改为 client

-     if use_lora:
-         clf_llm.unload()

      # ── Step 8 ──
-     if base_llm is None:
-         base_llm = LLMInterface(ModelConfig())  # 重载
      for p in tqdm(files, desc="Step 8: rationale"):
          ...
-         rationale = ask_llm(base_llm, prompt).strip()
+         rationale = ask_llm(client, prompt).strip()
```

---

### 3.6 `summarize_blocks()` 和 `summarize_program()` 类型注解 [classify_from_c_code.py:981-1015](classify_from_c_code.py#L981-L1015)

```diff
- def summarize_blocks(llm: LLMInterface, chunks: ...) -> ...:
+ def summarize_blocks(client: APIClient, chunks: ...) -> ...:

- def summarize_program(llm: LLMInterface, block_summaries: ...) -> ...:
+ def summarize_program(client: APIClient, block_summaries: ...) -> ...:
```

内部调用也相应改为 `ask_llm(client, ...)`。

---

## 4. 需改动：`batch_decompile_ida.py` — **不改动**

反编译步骤仍在 Windows 主机上运行 IDA Pro，与 LLM 无关。

---

## 5. 需删除 / 可跳过

### 5.1 整个文件：`model_server.py`

被 `api_client.py` 完全替代。所有本地模型加载/量化/卸载逻辑不再需要。

### 5.2 目录（不再需要）

```
LLMs/Base_Model/        ← 不再下载 Qwen3-14B（省 ~28GB 磁盘）
LLMs/LoRA_Weight/       ← 不做微调
```

### 5.3 Python 依赖变更

| 删除（不再需要） | 新增 |
|------------------|------|
| `bitsandbytes` (量化) | `openai>=1.0.0` |
| `flash-attn` (FlashAttention) | `tiktoken>=0.5.0` |
| `peft` (LoRA 适配器) | `python-dotenv>=1.0.0` |
| `accelerate` (设备映射) | |

`torch` 和 `transformers` 的使用范围缩小：`transformers` 在 `classify_from_c_code.py` 中**没有被直接使用**（仅 `model_server.py` 使用了），所以 `transformers` 也可以删除。

实际上检查一下 classify_from_c_code.py 的导入——它只从 model_server 导入了 `LLMInterface, LoRALLMInterface, ModelConfig`，没有直接 import transformers。所以：

```
可删除的依赖:
  - torch (如仅为此项目，无其他用途则可删)
  - transformers (仅 model_server.py 使用)
  - bitsandbytes
  - flash-attn / flash-attention
  - peft
  - accelerate
```

---

## 6. 风险点与缓解方案

### 6.1 Token 计数精度偏差

**风险：** 原方案用 Qwen 分词器精确计数，改为 tiktoken cl100k_base 估算，偏差可能达 ±15%。

**影响范围：**
- Step 2 智能分块：token 预算控制精度下降
- Step 1A 元数据压缩：`compress_metadata_to_budget()` 依赖 `TokenCounter`

**缓解：**
- 分块预算 `CHUNK_TOKEN_SIZE` 保留安全余量（如设为 25K 而非 30K）
- 在 `APIClient.chat()` 中增加 pre-flight 检查：发请求前估算 prompt 的 token 数，超过 128K 则拒绝 + warn
- 全量运行后抽查几个样本的 LLM 输入日志，确认未超出上下文

### 6.2 API 速率限制（Rate Limiting）

**风险：** DeepSeek API 存在 RPM（每分钟请求数）和 TPM（每分钟 token 数）上限。每个样本的 LLM 调用量：

| 步骤 | 每个样本调用次数 |
|------|-----------------|
| Step 4 | N = 块数（通常 1~10） |
| Step 5 | 1 |
| Step 7 | 3（Full + E-Meta + E-Summary） |
| Step 8 | 1 |
| **合计** | **6~15 次/样本** |

数据集 2000+ 样本意味着 **12000~30000 次 API 调用**。

**缓解：**
- `APIClient.chat()` 已内置 429 → 指数退避重试
- 调用间增加 `time.sleep(REQUEST_DELAY_SEC)`（配置项，默认 0.3s）
- **首次运行用小样本（10~20 个）先验证速率和成本**

### 6.3 API 调用成本

**估算（单样本）：**
- 输入 tokens：~40K~200K（取决于函数数量和复杂性）
- 输出 tokens：~2K~5K（摘要 + 分类 + 理由）

**缓解：**
- 先跑 10 个样本，统计实际 token 消耗
- 再决定是否全量运行或抽样运行

### 6.4 DeepSeek 思考模式（Reasoning Tokens）

**风险：** DeepSeek V4 Pro 的 `deepseek-reasoner` 变体可能会在输出中产生思考过程（类似 DeepSeek-R1 的 `
` 标记），这与 Qwen 的 `<think>` 标记类似但格式不同。

**影响：** `_decode_chat_response`（原在 model_server.py）中仅处理了 Qwen 的 think 标记。若使用 DeepSeek reasoner，需要类似处理。

**缓解：**
- 推荐使用 `deepseek-chat`（非 reasoner 模型），输出更干净，不产生额外思考 token
- 如果确实使用 reasoner 模型，需在 `parse_classification_output()` 中增加 robustness：先提取 `[RANSOMWARE]` / `[BENIGN]` 标签，再解析理由

### 6.5 Prompt 模板微调（可选优化）

**当前情况：** Prompt 模板为 Qwen3-14B 设计，单一 `user` role 消息。

**建议改进（可选）：** 对 Step 7 分类这一步，拆分 system + user 两层消息以获得更强的指令遵循：

```python
PROMPT_CLASSIFICATION_SYSTEM = (
    "You are a malware analyst. Your task is binary classification. "
    "Output format: EXACTLY one label [RANSOMWARE] or [BENIGN] as the FIRST text."
)

PROMPT_CLASSIFICATION_USER = (
    "HARD RULE: If any of these indicators is present — vssadmin, delete shadows, "
    ".onion, decrypt, ransom, CryptAcquireContext — OR program_summary.text contains "
    "[RANSOMWARE], output [RANSOMWARE]. Otherwise decide holistically.\n\n"
    "--- IR (JSON) ---\n{ir_json_block}"
)
```

API 调用时传入：

```python
out_full = ask_llm(
    client,
    prompt=PROMPT_CLASSIFICATION_USER.format(ir_json_block=...),
    system_prompt=PROMPT_CLASSIFICATION_SYSTEM,
)
```

> 这是**可选优化**，不修改也能正常工作。DeepSeek V4 Pro 的指令遵循能力对原始 prompt 格式也能正确处理。

---

## 7. 推荐实施顺序

```
Phase 1: 新建 api_client.py
├── 1. 定义 APIConfig
├── 2. 实现 APIClient.chat()（含重试 + 限流处理）
├── 3. 实现 APIClient.count_tokens()
└── 4. 写一个简单的验证脚本：发一条 "Reply with just: [BENIGN]"
    确认 API 连通性，检查输出格式

Phase 2: 修改 classify_from_c_code.py
├── 5. 改导入行
├── 6. 改 TokenCounter
├── 7. 改 ask_llm() 函数签名
├── 8. 改 main() — 删除所有模型加载/卸载逻辑，替换为 APIClient
├── 9. 删除 ExperimentConfig.USE_LORA_FOR_CLASSIFICATION
└── 10. 改 summarize_blocks / summarize_program 的类型注解

Phase 3: 小规模验证（1~2 个样本）
├── 11. 准备 1~2 个 .c 文件放入 Classify/Classify_Input_C_Code/
├── 12. 运行 python classify_from_c_code.py
├── 13. 检查每个步骤的输出 JSON 是否正确
├── 14. 检查 LLM 输入日志（设置 SAVE_LLM_INPUTS=True）是否超上下文
└── 15. 确认分类结果和理由格式符合预期

Phase 4: 中等规模验证（10~20 个样本）
├── 16. 放入更多 .c 文件
├── 17. 监控 API 速率限制触发频率
├── 18. 统计每个样本的平均 token 消耗和 API 调用次数
└── 19. 估算全量运行的成本和时间

Phase 5: （可选）Prompt 优化
├── 20. 将 Step 7 的 prompt 拆分为 system + user 双层
└── 21. A/B 对比：原始 prompt vs 优化后 prompt 的分类一致性

Phase 6: 全量运行
├── 22. 运行全部样本
└── 23. 汇总分类结果、计算 ACC / Precision / Recall / F1
```

---

## 附录：关键文件改动对照表

| 文件 | 改动性质 | 行数变化 | 关键内容 |
|------|----------|----------|----------|
| `api_client.py` | **新建** | ~150 行 | APIConfig + APIClient |
| `classify_from_c_code.py` | 修改 ~10 处 | 约 -40 行 | 导入/TokenCounter/ask_llm/main() |
| `model_server.py` | **废弃**（可删除或保留备份） | -274 行 | 被 api_client.py 替代 |
| `batch_decompile_ida.py` | 不变 | 0 | IDA 反编译不涉及 LLM |
| `requirements_api.txt` | 新建 | ~5 行 | openai, tiktoken, python-dotenv |
