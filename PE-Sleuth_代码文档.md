# PE-Sleuth 代码文档

> 基于论文 *PE-Sleuth: Program-level Semantics and Static Feature Fusion for Interpretable Ransomware Detection with LLMs* 的代码实现。

---

## 目录

1. [项目总览](#1-项目总览)
2. [model_server.py — LLM 模型服务层（原始论文）](#2-model_serverpy--llm-模型服务层原始论文)
3. [batch_decompile_ida.py — PE 批量反编译驱动](#3-batch_decompile_idapy--pe-批量反编译驱动)
4. [classify_from_c_code.py — 端到端分类流水线](#4-classify_from_c_codepy--端到端分类流水线)
   - [Step 1A: 静态特征提取（元数据解析）](#step-1a-静态特征提取元数据解析)
   - [Step 1B: 函数调用图构建](#step-1b-函数调用图构建)
   - [Step 2: 智能分块](#step-2-智能分块)
   - [Step 3: 可视化](#step-3-可视化)
   - [Step 4: 块级语义摘要](#step-4-块级语义摘要)
   - [Step 5: 程序级语义摘要](#step-5-程序级语义摘要)
   - [Step 6: 中间表示构建](#step-6-中间表示构建)
   - [Step 7: 分类](#step-7-分类)
   - [Step 8: 分类理由生成](#step-8-分类理由生成)
5. [数据流图](#5-数据流图)
6. [api_client.py — DeepSeek API 客户端](#6-api_clientpy--deepseek-api-客户端)
7. [pipeline.py — 单文件端到端流水线](#7-pipelinepy--单文件端到端流水线)
8. [settings.json — API 配置](#8-settingsjson--api-配置)
9. [Taskfile.yml — 任务编排](#9-taskfileyml--任务编排)

---

## 1. 项目总览

PE-Sleuth 是一个基于大语言模型 (LLM) 的 Windows PE 勒索软件检测框架。其核心流程为：

```
PE 二进制 → IDA Pro 反编译为 C 源码 → 静态特征提取 + 调用图感知分块 →
块级语义分析 → 程序级语义聚合 → 语义+特征融合 IR → LLM 二分类 + 理由生成
```

**三个核心 Python 模块：**

| 文件 | 行数 | 角色 |
|------|------|------|
| `model_server.py` | 274 | *(原始论文)* 本地 LLM 加载、量化、推理 |
| `api_client.py` | ~160 | *(API 适配)* 远程 DeepSeek API 客户端 |
| `batch_decompile_ida.py` | ~370 | 调用 IDA Pro 批量反编译 PE 文件 |
| `classify_from_c_code.py` | ~1290 | 完整 8 步检测流水线（已适配 API） |
| `pipeline.py` | ~280 | 单文件端到端分析流水线 |
| `Taskfile.yml` | ~78 | 测试/反编译/分类/清理任务编排 |
| `settings.json` | 10 | API 连接与模型参数配置 |

---

## 2. model_server.py — LLM 模型服务层

提供 Qwen3-14B 模型的加载、量化、推理和卸载功能，支持基础模型和 LoRA 微调模型两种模式。

### 2.1 类 `ModelConfig`

**作用：** 集中管理模型路径和运行时参数。

```python
class ModelConfig:
```

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `BASE_MODEL_PATH` | `str` | `"LLMs/Base_Model/Qwen3-14B"` | 基础模型路径（相对于仓库根目录） |
| `LORA_WEIGHTS_PATH` | `str` | `"LLMs/LoRA_Weight/PE-Sleuth-Qwen3-14B-LoRA"` | LoRA 适配器权重路径 |
| `CTX_LENGTH` | `int` | `32768` (32K) | 最大上下文长度（tokens） |
| `MAX_NEW_TOKENS` | `int` | `512` | 每次生成的最大新 token 数 |
| `QUANTIZATION_BITS` | `Optional[int]` | `4` | 量化位数：4 / 8 / None |
| `USE_FLASH_ATTENTION` | `bool` | `True` | 是否启用 FlashAttention-2 |
| `ENABLE_THINKING` | `bool` | `False` | 是否启用 Qwen 思考模式 |
| `DEVICE` | `str` | `"cuda"` 或 `"cpu"` | 自动检测 CUDA 可用性选择设备 |

---

### 2.2 内部工具函数

#### `_get_quantization_config(bits)`

**作用：** 根据指定位数创建 BitsAndBytes 量化配置。

| 参数 | 类型 | 说明 |
|------|------|------|
| `bits` | `Optional[int]` | 量化位数，支持 `4` / `8` / `None` |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 量化配置 | `Optional[BitsAndBytesConfig]` | 4-bit 使用 nf4 + 双重量化；8-bit 使用标准配置；None 返回 None |

---

#### `_load_tokenizer(cfg)`

**作用：** 加载 Qwen3-14B 分词器并配置 YaRN 支持的最大长度。

| 参数 | 类型 | 说明 |
|------|------|------|
| `cfg` | `ModelConfig` | 模型配置对象 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| tokenizer | `AutoTokenizer` | Qwen3 tokenizer 实例 |

---

#### `_load_base_model_only(cfg)`

**作用：** 加载 Qwen3-14B 基础模型（含 YaRN 扩展、量化和 FlashAttention）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `cfg` | `ModelConfig` | 模型配置对象 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| model | `AutoModelForCausalLM` | 已加载并设为 eval 模式的模型实例 |

**细节：**
- 启用 `use_yarn` 支持 32K 上下文
- 自动选择 `bfloat16` 或 `float16` 数据类型
- 使用 `device_map="auto"` 自动分配设备

---

#### `_decode_chat_response(full_text, prompt_text)`

**作用：** 从生成的完整文本中提取助手回复。

| 参数 | 类型 | 说明 |
|------|------|------|
| `full_text` | `str` | 模型生成的完整文本 |
| `prompt_text` | `str` | 输入的提示文本（用于去除前缀） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 提取的回复 | `str` | 去除提示前缀和 Qwen think 标记后的纯回复文本 |

---

### 2.3 混入类 `_GenerationMixin`

**作用：** 提供 `generate()` 和 `get_tokenizer()` 共用方法，子类需定义 `self.model`、`self.tokenizer`、`self.config`。

#### `generate(prompt)`

**作用：** 使用 chat template 格式化用户输入并生成回复。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str` | 用户输入的提示文本（单轮对话） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| response | `str` | 模型生成的文本回复 |

**生成参数：**
- `temperature=0.5`、`top_p=0.9`
- `do_sample=True`, `use_cache=True`
- 自动处理 pad_token_id (fallback 到 eos_token_id)

#### `get_tokenizer()`

**作用：** 返回已加载的分词器实例。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| tokenizer | `AutoTokenizer` 或 `None` | 当前加载的分词器，未加载时返回 None |

---

### 2.4 类 `LLMInterface(_GenerationMixin)`

**作用：** 基础模型（无 LoRA）接口，封装模型加载与显存卸载。

#### `__init__(config)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `ModelConfig` | 模型配置对象 |

初始化后自动调用 `_load_model()` 加载模型和分词器。

#### `_load_model()`

**作用：** 加载基础模型和分词器（幂等：已加载则跳过）。

#### `unload()`

**作用：** 卸载模型和分词器以释放 GPU 显存。安全地可多次调用。

**细节：** 删除模型/分词器引用 → `gc.collect()` → `torch.cuda.empty_cache()`

---

### 2.5 类 `LoRALLMInterface(_GenerationMixin)`

**作用：** LoRA 微调模型接口，在基础模型上应用 PE-Sleuth 的 LoRA 适配器。

#### `__init__(config)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `ModelConfig` | 需设置 `LORA_WEIGHTS_PATH`，否则抛出 `ValueError` |

**加载流程：** 加载基础模型 → 通过 PEFT `PeftModel.from_pretrained()` 应用 LoRA 适配器 → 设为 eval 模式

#### `_load_model()`

**作用：** 加载带 LoRA 适配器的模型（幂等：已加载则跳过）。

#### `unload()`

**作用：** 卸载 LoRA 模型（含底层基础模型）和分词器以释放 GPU 显存。

---

## 3. batch_decompile_ida.py — PE 批量反编译驱动

**平台要求：** Windows + IDA Pro 9+（含 Hex-Rays 反编译器）

调用 IDA Pro 的 `idat.exe` 批量将 PE 文件反编译为 C 源码。

### 3.1 配置常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `TARGET_FOLDER` | `Decompile/Decompile_Input_Raw_PE/` | 输入 PE 文件目录 |
| `OUTPUT_FOLDER` | `Decompile/Decompile_Output_C_Code/` | 输出 .c 文件目录 |
| `FAIL_LOG` | `Decompile/decompile_failure_log.txt` | 失败日志路径 |
| `TIMEOUT` | `600` (秒) | 单个文件反编译超时 |
| `MIN_C_OUTPUT_BYTES` | `10 * 1024` (10 KB) | 输出最小合理大小 |
| `MAX_C_OUTPUT_BYTES` | `3 * 1024 * 1024` (3 MB) | 输出最大合理大小 |
| `IDA_ENV` | `"IDA_PATH"` | IDA 路径环境变量名 |

---

### 3.2 函数列表

#### `locate_idat()`

**作用：** 仅通过 `IDA_PATH` 环境变量查找 `idat.exe`。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 路径 | `Optional[str]` | 找到的 `idat.exe` 完整路径，未找到返回 None |

**逻辑：** 接受 `IDA_PATH` 指向 `idat.exe` 文件本身，或指向 IDA 安装目录（自动拼接 `idat.exe`）。

---

#### `resolve_ida_path()`

**作用：** 定位并验证 `idat.exe`。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| 路径 | `str` | 经验证的 `idat.exe` 路径 |

**行为：** 非 Windows 系统直接报错退出；找不到 `idat.exe` 或文件名不正确时报错退出。

---

#### `load_failed_set()`

**作用：** 读取失败日志，返回已失败的文件名集合（用于跳过重复处理）。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| failed | `set` | 之前处理失败的文件名集合 |

---

#### `append_failure(failed, filename, reason)`

**作用：** 将失败记录追加到失败日志（同一文件只记录一次）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `failed` | `set` | 当前失败集合（会被修改） |
| `filename` | `str` | 失败的文件名 |
| `reason` | `Optional[str]` | 失败原因描述 |

---

#### `kill_process_tree(proc)`

**作用：** 终止超时的 IDA 进程树（Windows）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `proc` | `subprocess.Popen` | 需要终止的进程对象 |

---

#### `output_size_is_abnormal(path)`

**作用：** 检查反编译输出文件大小是否合理（[10KB, 3MB]）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `path` | `str` | 输出 .c 文件路径 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `(is_abnormal, size_bytes)` | `Tuple[bool, int]` | 是否异常 + 文件大小（字节） |

---

#### `main()`

**作用：** 主循环——遍历输入 PE 文件，对每个文件执行：

1. 检查是否已在失败日志中 → 跳过
2. 检查输出 .c 文件是否已存在 → 跳过
3. 构造 IDA 命令：`idat.exe -Ohexrays:-new:-nosave:<output_name>:ALL -A <input_file>`
4. 以 `CREATE_NEW_PROCESS_GROUP` 标志启动子进程
5. 等待最多 600 秒
6. 输出大小超范围 → 删除文件 + 记录失败
7. 清理 IDA 临时数据库文件（`.i64`, `.id0`, `.id1`, `.id2`, `.nam`, `.til`）
8. 打印摘要统计

---

## 4. classify_from_c_code.py — 端到端分类流水线

### 4.1 类 `ExperimentConfig`

**作用：** 流水线所有可配置参数的集中管理。

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `INPUT_DIR` | `Path` | `Classify/Classify_Input_C_Code/` | 输入 C 源码目录 |
| `OUTPUT_DIR` | `Path` | `Classify/Classify_Output_All/` | 所有输出根目录 |
| `RUN_STEP_1A_METADATA_EXTRACTION` | `bool` | `True` | 是否执行步骤 1A |
| `RUN_STEP_1B_CALL_GRAPH_GENERATION` | `bool` | `True` | 是否执行步骤 1B |
| `RUN_STEP_2_SMART_CHUNKING` | `bool` | `True` | 是否执行步骤 2 |
| `RUN_STEP_3_VISUALIZATIONS` | `bool` | `True` | 是否执行步骤 3 |
| `RUN_STEP_4_CHUNK_SUMMARIZATION` | `bool` | `True` | 是否执行步骤 4 |
| `RUN_STEP_5_PROGRAM_SUMMARIZATION` | `bool` | `True` | 是否执行步骤 5 |
| `RUN_STEP_6_IR_CONSTRUCTION` | `bool` | `True` | 是否执行步骤 6 |
| `RUN_STEP_7_CLASSIFICATION` | `bool` | `True` | 是否执行步骤 7 |
| `RUN_STEP_8_CLASSIFICATION_RATIONALE` | `bool` | `True` | 是否执行步骤 8 |
| `USE_LORA_FOR_CLASSIFICATION` | `bool` | `True` | Step 7 是否使用 LoRA 模型 |
| `RUN_ABLATION_E_META` | `bool` | `True` | 是否执行消融实验 E-Meta（仅程序语义） |
| `RUN_ABLATION_E_SUMMARY` | `bool` | `True` | 是否执行消融实验 E-Summary（仅静态特征） |
| `OVERWRITE_EXISTING` | `bool` | `False` | 是否覆盖已有输出（断点续跑） |
| `SAVE_LLM_INPUTS` | `bool` | `False` | 是否保存发送给 LLM 的完整输入便于审计 |
| `CHUNK_TOKEN_SIZE` | `int` | `30 * 1024` (30K) | 每个块的 token 上限 |
| `MAX_METADATA_BYTES` | `int` | `4096` (4KB) | 元数据 JSON 的字节上限 |
| `MAX_TOP_STRINGS` | `int` | `120` | 保留字符串最大数量 |
| `MAX_STRING_LEN` | `int` | `200` | 单条字符串最大长度（截断） |
| `MAX_GLOBAL_DECLS` | `int` | `80` | 全局声明最大数量 |
| `MAX_API_IMPORTS` | `int` | `120` | API 导入最大数量 |

**LLM 提示模板（Prompt Templates）：**

| 模板变量 | 用途 |
|----------|------|
| `PROMPT_CHUNK_SUMMARY` | Step 4：让 LLM 分析代码片段的关键行为 |
| `PROMPT_PROGRAM_SUMMARY` | Step 5：聚合所有块摘要为程序级语义 |
| `PROMPT_CLASSIFICATION` | Step 7：基于融合 IR 进行二分类判断 |
| `PROMPT_CLASSIFICATION_RATIONALE` | Step 8：生成分析师可审计的分类理由 |

---

### 4.2 通用工具函数

#### `out_dir(name) → Path`

**作用：** 在输出根目录下创建子目录并返回路径（幂等）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 子目录名称（如 `"1a_parsed_metadata"`） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| path | `Path` | 已创建的子目录路径 |

#### `read_text(p) → str`

读取文件为 UTF-8 字符串（忽略编码错误）。

#### `read_json(p) → Dict[str, Any]`

读取 JSON 文件并解析为字典。

#### `write_json(p, obj)`

将对象序列化为 JSON 写入文件（`ensure_ascii=False`, `indent=2`）。

#### `sha1_text(s) → str`

计算字符串的 SHA-1 哈希值（用于函数体去重）。

#### `natural_key(name) → List`

文件名自然排序键函数，按数字部分数值排序而非字典序。

#### `out_paths(sample_path) → Dict[str, Path]`

**作用：** 根据输入文件路径生成所有步骤的输出文件路径映射。

| 参数 | 类型 | 说明 |
|------|------|------|
| `sample_path` | `Path` | 输入 .c 文件路径 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| paths | `Dict[str, Path]` | 键名涵盖 metadata / callgraph / chunks / viz_html / chunk_summaries / program_summary / ir / classification / rationale |

---

### 4.3 类 `TokenCounter`

**作用：** 封装 token 计数逻辑，优先使用真实分词器，fallback 为字符/3.8 的估算。

#### `__init__(llm)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `llm` | `Optional[LLMInterface]` | LLM 接口实例（用于获取分词器） |

#### `count(text) → int`

| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 待计数的文本 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| tokens | `int` | token 数量（真实或估算） |

---

### 4.4 LLM 辅助函数

#### `ask_llm(llm, prompt, max_retries=3, backoff=0.7) → str`

**作用：** 带重试的 LLM 调用包装器（指数退避）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `llm` | `LLMInterface` | LLM 接口实例 |
| `prompt` | `str` | 发送给 LLM 的提示文本 |
| `max_retries` | `int` | 最大重试次数（默认 3） |
| `backoff` | `float` | 退避基础系数（默认 0.7s） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| response | `str` | LLM 生成的回复文本 |

---

### Step 1A: 静态特征提取（元数据解析）

从反编译后的 C 源码中提取四类静态特征：字符串、API 导入、全局声明、可疑指标。

#### 正则表达式常量

| 变量 | 用途 |
|------|------|
| `STRING_RE` | 匹配 C 代码中的宽字符 (`L"..."`) 和普通 (`"..."`) 字符串字面量 |
| `EXTERN_LINE_RE` | 匹配 IDA 生成的 `// extern ...` 注释行（外部 API 导入） |
| `GLOBAL_DECL_RE` | 匹配全局变量声明（含 Windows 类型如 `DWORD`, `HANDLE` 等） |

---

#### `_shorten(s, max_len) → str`

**作用：** 在字符串过长时从中间截断并插入省略标记 `…`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `s` | `str` | 原始字符串 |
| `max_len` | `int` | 最大允许长度 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| shortened | `str` | 截断后的字符串 |

---

#### `extract_strings(code) → List[str]`

**作用：** 从 C 代码中提取所有字符串字面量（去重后返回）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | 反编译后的 C 源码全文 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| strings | `List[str]` | 去重后的字符串列表（过滤掉长度 < 3 的） |

---

#### `rank_strings(strings) → List[str]`

**作用：** 对提取到的字符串进行可疑度评分排序，保留 Top-N。

| 参数 | 类型 | 说明 |
|------|------|------|
| `strings` | `List[str]` | 已提取的字符串列表 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| ranked | `List[str]` | 按可疑度降序排列的前 `MAX_TOP_STRINGS` 条字符串 |

**评分规则：**
- 长度贡献：`min(len(s) / 20, 3.0)`
- 包含路径分隔符 `/` 或 `\`：+1.0
- 包含文件扩展名（如 `.exe`）：+0.8
- 匹配高危关键词（`vssadmin`, `delete shadows`, `.onion`, `ransom`, `CryptAcquireContext`, `RegSetValue` 等）：+2.5

---

#### `extract_extern_imports(code) → List[str]`

**作用：** 从 IDA 的 `// extern` 注释行提取外部 API 导入。

| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | C 源码全文 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| imports | `List[str]` | 排序后的 API 导入列表（最多 `MAX_API_IMPORTS` 条） |

---

#### `compress_metadata_to_budget(meta, tc, max_tokens=1000, max_bytes=4096) → Dict[str, Any]`

**作用：** 将元数据压缩到严格的 token 和字节预算内。

| 参数 | 类型 | 说明 |
|------|------|------|
| `meta` | `Dict[str, Any]` | 原始元数据字典 |
| `tc` | `TokenCounter` | Token 计数器 |
| `max_tokens` | `int` | token 上限（默认 1000） |
| `max_bytes` | `int` | 字节上限（默认 4096） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| compressed | `Dict[str, Any]` | 压缩后的元数据（JSON 序列化后 ≤ max_bytes） |

**压缩策略（按优先级依次执行）：**
1. 截断 `global_declarations` 到 `MAX_GLOBAL_DECLS` 并缩短
2. 精简 `functions` 仅保留名称和 token 数，限制 50 个
3. 缩减 `api_imports` 到 60 条
4. 缩减 `top_strings` 到 80 条
5. 重组为最简结构，进一步限制各类别数量
6. 迭代缩减各类列表（每次 -10 条）
7. 缩短单条字符串的最大长度
8. 最终极简 fallback（仅保留 20 条字符串 + 15 条 API）

---

#### `step_1a_parse_metadata(sample_path, code, tc) → Dict[str, Any]`

**作用：** 步骤 1A 的主函数——从 C 源码提取并组装标准化的静态特征。

| 参数 | 类型 | 说明 |
|------|------|------|
| `sample_path` | `Path` | 输入 .c 文件路径 |
| `code` | `str` | C 源码文本 |
| `tc` | `TokenCounter` | Token 计数器 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| metadata | `Dict[str, Any]` | 包含以下字段的元数据字典 |

**返回字典结构：**

```json
{
  "file_size_bytes": 12345,
  "top_strings": ["..."],
  "api_imports": ["CreateFileW", "..."],
  "global_declarations": ["DWORD var;", "..."],
  "suspicious_indicators": {
    "vssadmin": false,
    "delete shadows": false,
    ".onion": false,
    "decrypt": true,
    "ransom": false,
    "CryptAcquireContext": true,
    ...
  }
}
```

**处理流程：**
1. `extract_strings()` → `rank_strings()` 获取排序后的 Top 字符串
2. `extract_extern_imports()` 获取 API 导入列表
3. 正则匹配 `GLOBAL_DECL_RE` 获取全局声明
4. 正则搜索 14 种可疑关键词，生成布尔型可疑指标
5. 通过 `compress_metadata_to_budget()` 压缩到预算内
6. 移除 `source_path` 防止标签泄露

---

### Step 1B: 函数调用图构建

#### `strip_strings_and_comments(code) → str`

**作用：** 移除 C 代码中的字符串字面量和注释（防止其中的函数名干扰调用图构建）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | 原始 C 代码 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| cleaned | `str` | 清除了字符串和注释的代码 |

---

#### `parse_functions_ida_style(code, tc) → List[Dict[str, Any]]`

**作用：** 解析 IDA Hex-Rays 导出的 C 代码，按 `//---- (地址) ----` 分割符提取每个函数的信息。

| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | IDA 导出的 C 源码 |
| `tc` | `TokenCounter` | Token 计数器 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| functions | `List[Dict[str, Any]]` | 函数列表 |

**每个函数的字典结构：**

```json
{
  "id": 0,
  "display_name": "main",
  "unique": "main#L42",
  "line_start": 42,
  "line_end": 120,
  "code": "int main(int argc, char **argv) { ... }",
  "tokens": 512,
  "content_sha1": "a1b2c3d4..."
}
```

**函数名提取逻辑：** `_extract_func_name_from_signature(sig_line)` 从函数签名行提取函数名，去除返回类型、指针符号 `*&`，并移除 IDA 的 `@<eax>` 等寄存器尾缀。

---

#### `build_call_graph(code, tc) → Dict[str, Any]`

**作用：** 构建完整函数调用图（Function Call Graph, FCG）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `code` | `str` | C 源码全文 |
| `tc` | `TokenCounter` | Token 计数器 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| graph | `Dict[str, Any]` | 调用图 JSON 对象 |

**返回字典结构：**

```json
{
  "graph_type": "call_graph",
  "node_count": 42,
  "edge_count": 87,
  "nodes": [ { "id": 0, "display_name": "main", "code": "...", ... }, ... ],
  "edges": [ { "src": 0, "dst": 3 }, ... ]
}
```

**构建逻辑：**
1. `parse_functions_ida_style()` 获取函数列表
2. 将函数名按长度降序排列（避免短名称误匹配）
3. 用正则 `(?<![A-Za-z0-9_:])(name)\s*\(` 在每个函数体中匹配对其他函数的调用
4. 去除自环边（函数调用自身）
5. 边排重并排序

---

### Step 2: 智能分块

#### `split_long_function(name, code, tc, max_tokens) → List[Dict[str, Any]]`

**作用：** 将超大函数按语法边界拆分（优先花括号边界，其次滑动窗口）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 函数名 |
| `code` | `str` | 函数完整代码 |
| `tc` | `TokenCounter` | Token 计数器 |
| `max_tokens` | `int` | 每块 token 上限 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| parts | `List[Dict[str, Any]]` | 拆分后的函数片段列表 |

**拆分策略：**
1. 函数总 token ≤ 上限 → 直接返回原函数
2. 按花括号深度切分（depth = 0 时切分）
3. 若某块仍超大，按 `max_tokens * 3.8` 字符窗口滑动切分
4. 每块命名为 `funcName#part1`, `funcName#part2` 或 `funcName#part1.1` 等

---

#### `connected_components(n, edges) → List[List[int]]`

**作用：** 使用 BFS 计算无向图的连通分量。

| 参数 | 类型 | 说明 |
|------|------|------|
| `n` | `int` | 节点总数 |
| `edges` | `List[Dict[str, int]]` | 边列表（每条边含 `src`, `dst`） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| components | `List[List[int]]` | 连通分量列表，每个分量为节点 ID 排序列表 |

---

#### `step_2_smart_chunk(graph, tc, max_tokens) → Dict[str, Any]`

**作用：** 步骤 2 的主函数——基于调用图的智能分块。

| 参数 | 类型 | 说明 |
|------|------|------|
| `graph` | `Dict[str, Any]` | Step 1B 输出的调用图 |
| `tc` | `TokenCounter` | Token 计数器 |
| `max_tokens` | `int` | 每块 token 上限 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| chunks | `Dict[str, Any]` | 分块结果 |

**分块算法：**
1. 计算图的连通分量（不同连通分量天然分割到不同块）
2. 在每个连通分量内，按优先级排序函数（`main`/`WinMain`/`wmain`/`start` 优先级最高）
3. 贪心填充当前块：token 不超限 → 加入；超限 → flush 当前块，开新块
4. SHA-1 哈希全局去重（跨块消除重复函数体）
5. **相邻小块贪心合并**：若两相邻块合并后 token ≤ `max_tokens * 0.98`，则合并，减少 LLM 调用次数

**输出结构：**

```json
{
  "sample_blocks": [
    {
      "block_id": "block_1",
      "function_count": 5,
      "block_tokens": 25000,
      "functions": [ { "name": "main", "code": "...", "tokens": 512, "sha1": "..." } ],
      "code": "/* --- BLOCK SEPARATOR --- */\n\n..."
    }
  ],
  "total_blocks": 3,
  "max_block_tokens": 28000
}
```

---

### Step 3: 可视化

#### `render_visualization_html(sample_id, graph, chunks) → str`

**作用：** 生成交互式 D3.js 调用图可视化 HTML 页面。

| 参数 | 类型 | 说明 |
|------|------|------|
| `sample_id` | `str` | 样本标识符（用于标题） |
| `graph` | `Dict[str, Any]` | Step 1B 输出的调用图 |
| `chunks` | `Dict[str, Any]` | Step 2 输出的分块结果 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| html | `str` | 完整 HTML 文档字符串 |

**可视化特性：**
- **节点** = 函数（圆圈），大小与函数 token 数成正比
- **边** = 调用关系（连线）
- **颜色** = 分块归属（相同颜色的函数属于同一块）
- **交互**：
  - 拖拽节点 + 力导向布局
  - 缩放/平移画布
  - 悬停显示函数名、token 数、所属块信息
  - "Freeze Layout" 复选框冻结力模拟，支持整体拖动连通分量
  - "Show function names" 复选框控制标签显示，含智能去重叠算法（贪心 + 重要性排序）
- **标签去重**：按 token 数（重要性）排序，贪心保留无重叠标签，其余自动变暗

---

### Step 4: 块级语义摘要

#### `summarize_blocks(llm, chunks, save_inputs_to=None) → List[Dict[str, Any]]`

**作用：** 对每个代码块调用 LLM 进行行为分析，生成块级语义摘要。

| 参数 | 类型 | 说明 |
|------|------|------|
| `llm` | `LLMInterface` | LLM 接口实例 |
| `chunks` | `Dict[str, Any]` | Step 2 输出的分块结果 |
| `save_inputs_to` | `Optional[Path]` | 若提供，将发送给 LLM 的 prompt 保存到该路径 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| summaries | `List[Dict[str, Any]]` | 块摘要列表 |

**每项结构：**

```json
{
  "block_id": "block_1",
  "block_tokens": 25000,
  "summary": "The code exhibits the following key behaviors: ..."
}
```

**Prompt 策略：** 使用中性提示，仅要求提取行为（"code behavior extractor"），避免引入良性/恶意倾向。

---

### Step 5: 程序级语义摘要

#### `summarize_program(llm, block_summaries, save_input_to=None) → str`

**作用：** 合并所有块级摘要，调用 LLM 提炼为程序级语义。

| 参数 | 类型 | 说明 |
|------|------|------|
| `llm` | `LLMInterface` | LLM 接口实例 |
| `block_summaries` | `List[Dict[str, Any]]` | Step 4 输出的块摘要列表 |
| `save_input_to` | `Optional[Path]` | 若提供，保存 LLM prompt 到该路径 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| summary | `str` | 程序级语义文本（必须以 `[RANSOMWARE]` 或 `[BENIGN]` 开头） |

**Prompt 策略：**
- 赋予 LLM "资深恶意软件分析师"角色
- 明确指示：少数强信号（文件加密、卷影副本删除、勒索信）比大量无害操作更关键
- 要求输出 `[RANSOMWARE]` 或 `[BENIGN]` 标签 + 理由

---

### Step 6: 中间表示构建

#### `build_final_ir(global_meta, program_summary_text) → Dict[str, Any]`

**作用：** 将静态特征（global_metadata）与程序级语义（program_summary）融合为统一中间表示。

| 参数 | 类型 | 说明 |
|------|------|------|
| `global_meta` | `Dict[str, Any]` | Step 1A 输出的静态特征元数据 |
| `program_summary_text` | `str` | Step 5 输出的程序级语义文本 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| ir | `Dict[str, Any]` | 融合后的中间表示 |

**输出结构：**

```json
{
  "global_metadata": { "top_strings": [...], "api_imports": [...], "suspicious_indicators": {...}, ... },
  "program_summary": { "text": "[RANSOMWARE] This program exhibits ..." }
}
```

---

### Step 7: 分类

#### `sanitize_ir_for_llm(ir_obj) → Dict[str, Any]`

**作用：** 过滤 IR 对象，仅保留 `global_metadata` 和 `program_summary` 两个字段（移除其他可能干扰的字段）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ir_obj` | `Dict[str, Any]` | 完整 IR 对象 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| safe_ir | `Dict[str, Any]` | 仅含 `global_metadata` 和 `program_summary` 的纯净 IR |

---

#### `build_classification_prompt(ir_obj) → str`

**作用：** 构建分类步骤的 LLM prompt。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ir_obj` | `Dict[str, Any]` | IR 对象 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| prompt | `str` | 完整的分类 prompt |

**Prompt 包含硬规则（Hard Rule）：** 如果 `program_summary.text` 包含 `[RANSOMWARE]` 标签，或者元数据中存在 `vssadmin`、`delete shadows`、`.onion`、`decrypt`、`ransom`、`CryptAcquireContext` 任一指标 → 强制输出 `[RANSOMWARE]`。

---

#### `parse_classification_output(out) → Tuple[str, str]`

**作用：** 解析 LLM 的分类输出，提取标签和理由。

| 参数 | 类型 | 说明 |
|------|------|------|
| `out` | `str` | LLM 原始输出文本 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `(label, justification)` | `Tuple[str, str]` | 分类标签（`RANSOMWARE` / `BENIGN` / `UNKNOWN`）+ 理由文本 |

---

### Step 8: 分类理由生成

#### `build_rationale_prompt(ir_obj, label) → str`

**作用：** 构建理由生成的 LLM prompt。

| 参数 | 类型 | 说明 |
|------|------|------|
| `ir_obj` | `Dict[str, Any]` | IR 对象 |
| `label` | `str` | 已确定的分类标签（`RANSOMWARE` 或 `BENIGN`） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| prompt | `str` | 理由生成 prompt |

**Prompt 要求：** 简短段落 + 3~6 条要点，聚焦具体证据（字符串、API、行为）及其与勒索/良性模式的关联。

---

### 4.5 `main()` — 流水线编排器

**作用：** 按顺序执行全部分类流水线步骤，管理模型加载/卸载。

**执行流程：**

```
Step 1A: 静态特征提取        → 1a_parsed_metadata/*.json
Step 1B: 调用图构建          → 1b_call_graphs/*.json
Step 2:  智能分块            → 2_smart_chunks/*.json
Step 3:  可视化              → 3_visualizations/*.html
         [加载基础模型]
Step 4:  块级语义摘要        → 4_chunk_summaries/*.json
Step 5:  程序级语义摘要      → 5_program_summaries/*.json
Step 6:  中间表示 (IR) 构建   → 6_final_ir/*.json
         [卸载基础模型 → 加载 LoRA 模型]
Step 7:  分类 (全量+消融)     → 7_classification_results/*.json
         [卸载 LoRA → 重载基础模型]
Step 8:  分类理由生成         → 8_classification_rationale/*.json
```

**关键设计要点：**

1. **断点续跑**：每一步检查输出文件是否已存在，若存在且 `OVERWRITE_EXISTING=False` 则跳过
2. **显存管理**：基础模型和 LoRA 模型在不同步骤间加载/卸载，避免同时占用显存
3. **消融实验**：Step 7 自动生成三种输入变体：
   - `Full`：完整 IR（静态特征 + 程序语义）
   - `E-Meta`：仅程序语义（去除 static features）
   - `E-Summary`：仅静态特征（去除 program summary）
4. **自包含**：文件名自然排序，支持原地追加处理新文件

---

## 5. 数据流图

```
PE 二进制文件
    │
    ▼
[batch_decompile_ida.py]
    │  IDA Pro Hex-Rays (-Ohexrays)
    ▼
  .c 源码
    │
    ▼
[classify_from_c_code.py]
    │
    ├── Step 1A ──────────────────────► metadata JSON
    │   (extract_strings, rank_strings,      │
    │    extract_extern_imports,              │
    │    compress_metadata_to_budget)         │
    │                                         │
    ├── Step 1B ──────────────────────► callgraph JSON
    │   (parse_functions_ida_style,           │
    │    build_call_graph)                    │
    │                                         │
    ├── Step 2 ───────────────────────► chunks JSON
    │   (connected_components,                │
    │    step_2_smart_chunk)                  │
    │                                         │
    ├── Step 3 ───────────────────────► visualization HTML
    │   (render_visualization_html)           │
    │                                         │
    ├── Step 4 ── [基础模型] ─────────► chunk_summaries JSON
    │   (summarize_blocks)                    │
    │                                         │
    ├── Step 5 ── [基础模型] ─────────► program_summary JSON
    │   (summarize_program)                   │
    │                                         │
    ├── Step 6 ───────────────────────► IR JSON
    │   (build_final_ir)                      │
    │                                         │
    ├── Step 7 ── [LoRA模型] ─────────► classification JSON
    │   (build_classification_prompt,         │
    │    parse_classification_output)         │
    │                                         │
    └── Step 8 ── [基础模型] ─────────► rationale JSON
        (build_rationale_prompt)
```

### 输出目录结构

```
Classify/Classify_Output_All/
├── 1a_parsed_metadata/         # 静态特征（字符串, API, 全局声明, 可疑指标）
├── 1b_call_graphs/             # 函数调用图
├── 2_smart_chunks/             # 调用图感知的代码块
├── 3_visualizations/           # 交互式 D3.js HTML 可视化
├── 4_chunk_summaries/          # 块级行为语义摘要
├── 5_program_summaries/        # 程序级语义摘要
├── 6_final_ir/                 # 融合中间表示（静态特征 + 语义）
├── 7_classification_results/   # 分类预测 + 消融实验结果
├── 8_classification_rationale/ # 分析师可审计的分类理由
└── llm_input/                  # （可选）LLM 原始输入记录
```

---

## 6. api_client.py — DeepSeek API 客户端

替代 `model_server.py` 中的本地 LLM 推理，改为通过 HTTP 调用远程 DeepSeek V4 Pro API（OpenAI 兼容协议）。

### 6.1 类 `APIConfig`

**作用：** 从 `settings.json` 加载 API 连接参数和生成参数。

```python
class APIConfig:
    def __init__(self, settings_path: Optional[Path] = None):
```

若未传入 `settings_path`，默认读取项目根目录下的 `settings.json`。

| 属性 | 类型 | 默认值 | 来源 | 说明 |
|------|------|--------|------|------|
| `MODEL_NAME` | `str` | `"deepseek-chat"` | settings.json | 模型名称 |
| `API_KEY` | `str` | `""` | settings.json | API 密钥 |
| `BASE_URL` | `str` | `"https://api.deepseek.com"` | settings.json | API 端点 |
| `MAX_NEW_TOKENS` | `int` | `512` | settings.json | 单次生成最大 token 数 |
| `TEMPERATURE` | `float` | `0.5` | settings.json | 采样温度 |
| `TOP_P` | `float` | `0.9` | settings.json | 核采样参数 |
| `MAX_RETRIES` | `int` | `5` | settings.json | 最大重试次数 |
| `RETRY_BACKOFF_SEC` | `float` | `1.0` | settings.json | 指数退避基础秒数 |
| `REQUEST_DELAY_SEC` | `float` | `0.3` | settings.json | 调用间隔（避免限流） |

---

### 6.2 类 `APIClient`

**作用：** OpenAI 兼容协议的 HTTP 客户端，替代原 `LLMInterface` 和 `LoRALLMInterface`。

```python
class APIClient:
    def __init__(self, config: Optional[APIConfig] = None):
    def chat(self, prompt: str, system_prompt: str = "") -> str:
    @staticmethod
    def count_tokens(text: str) -> int:
```

#### `__init__(config)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `config` | `Optional[APIConfig]` | 若为 None，自动从 settings.json 加载 |

初始化 OpenAI 兼容客户端，若 `API_KEY` 未配置则抛出 `ValueError`。

#### `chat(prompt, system_prompt="") → str`

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | `str` | 用户消息内容 |
| `system_prompt` | `str` | 可选的 system role 消息 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| reply | `str` | 模型回复文本（去除首尾空白） |

**自动处理：**
- `429` 限流 → 指数退避重试（最多 `MAX_RETRIES` 次）
- `5xx` 服务端错误 → 重试
- 网络超时/异常 → 重试

#### `count_tokens(text) → int`

**静态方法。** 使用 `tiktoken` (`cl100k_base` 编码) 估算 token 数。Fallback：字符数 / 3.5。

| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | `str` | 待计数的文本 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| tokens | `int` | 估算的 token 数量 |

---

## 7. pipeline.py — 单文件端到端流水线

将反编译 + 8 步分类流水线合并为一步操作，输入一个 PE 文件，输出一个完整的分析报告目录。

### 7.1 CLI 入口

```bash
python pipeline.py --input sample.exe --output ./reports
python pipeline.py -i toolkit.exe -o ./reports --ida-path "D:\\IDA Pro"
```

| 参数 | 必需 | 说明 |
|------|------|------|
| `-i, --input` | 是 | PE 二进制文件路径 |
| `-o, --output` | 是 | 报告输出根目录（其下自动创建以二进制文件名命名的子目录） |
| `--ida-path` | 否 | IDA Pro 安装目录（默认 `$env:IDA_PATH` 或 `D:\Program Files\IDA Professional 9.2`） |

### 7.2 函数列表

#### `clean_workspace()`

**作用：** 清空所有中间 I/O 目录（Decompile_Input / Decompile_Output / Classify_Input / Classify_Output_All），为单文件运行准备干净的运行环境。

#### `run_decompile(ida_path) → bool`

| 参数 | 类型 | 说明 |
|------|------|------|
| `ida_path` | `str` | IDA Pro 安装目录 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| success | `bool` | 反编译是否成功 |

通过 `subprocess` 调用 `batch_decompile_ida.py`，自动设置 `IDA_PATH` 和 `PYTHONIOENCODING` 环境变量。超时限制 600 秒。

#### `run_classify() → bool`

| 返回值 | 类型 | 说明 |
|--------|------|------|
| success | `bool` | 分类是否成功 |

通过 `subprocess` 调用 `classify_from_c_code.py`，超时限制 3600 秒。

#### `collect_results(binary_name, results_dir)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `binary_name` | `str` | 二进制文件名（不含扩展名） |
| `results_dir` | `Path` | 报告输出目录 |

将 `Classify_Output_All/` 中匹配 `binary_name` 的输出复制到 `results_dir` 的对应子目录中，同时复制反编译的 `.c` 源文件。

#### `generate_summary(binary_name, results_dir)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `binary_name` | `str` | 二进制文件名 |
| `results_dir` | `Path` | 报告输出目录 |

**生成 `summary.md` 综合报告，包含 7 个部分：**

1. **Classification Result** — 分类结论表格（Full / E-Meta / E-Summary）
2. **Analyst Rationale** — LLM 生成的分析师理由
3. **Program-Level Semantics** — 程序级语义摘要
4. **Static Features** — 可疑指标 / API 导入 / 字符串 / 全局声明
5. **Call Graph** — 节点数和边数统计
6. **Smart Chunking** — 块数及最大块 token 数
7. **Output Artifacts** — 所有输出文件清单

#### `run(input_pe, output_dir, ida_path) → bool`

**作用：** 主流水线函数，按顺序执行全部 8 步。

| 参数 | 类型 | 说明 |
|------|------|------|
| `input_pe` | `str` | PE 文件路径 |
| `output_dir` | `str` | 报告输出根目录 |
| `ida_path` | `str` | IDA Pro 路径 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| success | `bool` | 全流程是否成功完成 |

**执行流程：** Clean workspace → Copy PE → Decompile → Copy .c → Classify (8 Steps) → Collect results → Generate summary.md

### 7.3 输出目录结构

```
{output_dir}/{binary_name}/
├── summary.md                    # 综合分析报告
├── {binary_name}.c               # 反编译 C 源码
├── 1a_parsed_metadata/           # 静态特征 (字符串/API/可疑指标)
├── 1b_call_graphs/               # 函数调用图
├── 2_smart_chunks/               # 调用图感知分块
├── 3_visualizations/             # 交互式 D3.js 调用图
├── 4_chunk_summaries/            # 块级行为语义
├── 5_program_summaries/          # 程序级语义
├── 6_final_ir/                   # 融合中间表示
├── 7_classification_results/     # 分类预测 (含消融实验)
└── 8_classification_rationale/   # 分类理由
```

---

## 8. settings.json — 统一配置

项目的单一配置文件，集中管理 API 连接参数、模型生成参数和 IDA Pro 路径。

```json
{
    "model_name": "deepseek-v4-pro",
    "api_key": "sk-xxxxxxxx",
    "base_url": "https://api.deepseek.com",
    "max_new_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.9,
    "max_retries": 5,
    "retry_backoff_sec": 1.0,
    "ida_path": "D:\\Program Files\\IDA Professional 9.2"
}
```

### API 配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_name` | `str` | DeepSeek 模型名称（`deepseek-chat` / `deepseek-v4-pro` / `deepseek-reasoner`） |
| `api_key` | `str` | API 密钥 |
| `base_url` | `str` | API 端点地址 |
| `max_new_tokens` | `int` | 单次生成最大 token 数 |
| `temperature` | `float` | 采样温度（0~2，越低越确定性） |
| `top_p` | `float` | 核采样概率阈值 |
| `max_retries` | `int` | API 调用最大重试次数 |
| `retry_backoff_sec` | `float` | 指数退避基础等待秒数 |

### IDA 配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `ida_path` | `str` | IDA Pro 安装目录（含 `idat.exe`） |

**读取优先级（`pipeline.py`）：**
1. `--ida-path` CLI 参数（显式覆盖）
2. `settings.json` 的 `ida_path` 字段
3. `$env:IDA_PATH` 环境变量
4. 以上均未配置时报错退出

> `.gitignore` 已配置忽略 `settings.json`，防止 API 密钥被提交到版本控制。

---

## 9. Taskfile.yml — 任务编排

使用 [Task](https://taskfile.dev) 工具管理项目的常用操作。所有任务共享以下全局配置：

| 变量 | 值 | 说明 |
|------|-----|------|
| `PYTHON` | `./venv/Scripts/python.exe` | Python 解释器路径 |
| `TEST_PE` | `Binaries\tests\toolkit.exe` | 测试用 PE 文件 |
| `TEST_NAME` | `toolkit` | 测试样本名称 |
| `REPORT_DIR` | `reports` | pipeline.py 报告输出目录 |

| 环境变量 | 值 | 说明 |
|----------|-----|------|
| `PYTHONIOENCODING` | `utf-8` | 强制 UTF-8 输出（避免 GBK 编码错误） |

> `IDA_PATH` 不再硬编码在 Taskfile 中，而是从 `settings.json` 的 `ida_path` 字段动态读取。直接调用 `batch_decompile_ida.py` 的任务（`test`、`decompile`）通过 PowerShell 读取 `settings.json` 后设置环境变量；`pipeline-test` 则由 `pipeline.py` 内部从 `settings.json` 读取。

### 任务列表

#### `task test`

**作用：** 端到端测试（批量模式），分 4 步手动编排。

**流程：**
1. 将 `toolkit.exe` 拷贝到 `Decompile/Decompile_Input_Raw_PE/`
2. 运行 `batch_decompile_ida.py`（IDA Pro 反编译）
3. 将 `.c` 输出拷贝到 `Classify/Classify_Input_C_Code/`
4. 运行 `classify_from_c_code.py`（8 步分类流水线）

**结果输出：** `Classify/Classify_Output_All/` 下的各子目录。

**适用场景：** 需要分步观察中间产物的调试场景。

---

#### `task pipeline-test`

**作用：** 单文件全自动分析（推荐），通过 `pipeline.py` 一键完成。

**流程：** 直接调用 `pipeline.py --input toolkit.exe --output reports/`

**结果输出：** `reports/toolkit/` 目录，含 `summary.md` 综合报告及全部制品。

**适用场景：** 日常使用的单文件分析和报告生成。

---

#### `task decompile`

**作用：** 仅反编译——对 `Decompile/Decompile_Input_Raw_PE/` 中的所有 PE 文件运行 IDA Pro 反编译。

**适用场景：** 单独批量反编译，不与分类步骤耦合。

---

#### `task classify`

**作用：** 仅分类——对 `Classify/Classify_Input_C_Code/` 中的所有 `.c` 文件运行 8 步分类流水线。

**适用场景：** 已有反编译好的 `.c` 文件时，跳过反编译直接分类。

---

#### `task clean`

**作用：** 清理测试产物。删除：

- `Decompile/Decompile_Input_Raw_PE/toolkit.exe`
- `Decompile/Decompile_Output_C_Code/toolkit.c`
- `Classify/Classify_Input_C_Code/toolkit.c`
- `Classify/Classify_Output_All/` 下的所有子目录
- `reports/toolkit/` 下的完整报告目录

**适用场景：** 重置到干净的初始状态后重新测试。

### 推荐工作流

```powershell
# 一键分析 + 生成报告（推荐）
task pipeline-test

# 查看报告
start reports/toolkit/summary.md

# 清理
task clean
```
