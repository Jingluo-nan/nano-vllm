---
name: code-analyzer
description: 为 nano-vllm 项目生成架构总览报告——模块职责、数据流、关键控制流。当用户要求"分析项目""讲讲架构""onboarding""这个项目是怎么工作的"等场景时调用。直接在对话中输出，不落盘。
---

# code-analyzer

为 nano-vllm 项目生成结构化的**架构总览**报告，目标读者是首次接触本项目的工程师。

## 何时调用

- 用户说"分析项目""分析这个 codebase""讲一下整体架构""给我做个 onboarding""这项目是怎么跑起来的"
- 不要用于：单文件代码审查、性能调优建议、bug 定位、写新代码——那些有更合适的工具或直接做。

## 输出协议

**直接在对话中回复**，不要创建 `analysis.md` 或任何文件。除非用户后续明确要求落盘。

回复必须严格按以下 6 节顺序，使用 markdown 二级标题。每节有篇幅上限，**超出请删减而不是溢出**。

### 1. 一句话定位（≤ 2 行）
告诉读者这是什么东西、和已知的什么对标。例：
> nano-vllm 是 vLLM 离线推理引擎的从零重写，~1200 行 Python，API 与 vLLM 同名（`LLM` / `SamplingParams` / `generate`），目前仅支持 Qwen3 模型族。

### 2. 分层结构（一张表）
用 markdown 表格，三列：**层 | 主要文件 | 职责一句话**。覆盖：
- 入口层（`nanovllm/__init__.py`, `llm.py`）
- 引擎层（`engine/llm_engine.py`, `engine/scheduler.py`, `engine/block_manager.py`, `engine/sequence.py`, `engine/model_runner.py`）
- 模型层（`models/qwen3.py`）
- 算子层（`layers/*.py`）
- 工具层（`utils/context.py`, `utils/loader.py`）

### 3. 一次 generate 调用的数据流（编号步骤，≤ 10 步）
从 `LLM.generate(prompts, sampling_params)` 一路追到 token 返回。必须点名以下关键事件，按时间顺序：
1. tokenize → `Sequence` 创建 → 入 `scheduler.waiting`
2. `step()` 循环：`Scheduler.schedule()` 决定 prefill / decode
3. `BlockManager` 分配 / 命中前缀缓存
4. `ModelRunner.run` → `prepare_prefill` 或 `prepare_decode` → 写入全局 `Context`
5. 模型前向（eager 或 CUDA graph replay）
6. `Attention.forward` 读 `Context`，调 flash-attn 三条路径之一，Triton kernel 写 KV cache
7. `Sampler` 出 token（仅 rank 0）
8. `Scheduler.postprocess` → `hash_blocks` 注册前缀缓存
9. 完成的 seq 释放 block，循环到 `is_finished`

### 4. 三个非显然的设计决策（每条 2-4 行）
**必须**包含以下这三条（这是新人最容易踩坑的地方）：

- **一个 step 要么全 prefill 要么全 decode**：`Scheduler.schedule` 优先排 prefill，且只有第一个 seq 允许 chunked prefill（`scheduler.py` 中 `if remaining < num_tokens and scheduled_seqs` 的守卫）。
- **KV cache 大小是动态算的**：`ModelRunner` 先做 warmup 前向测峰值显存，再用剩余显存按 `gpu_memory_utilization` 切 block。所以 `num_kvcache_blocks` 在构造时是 `-1`。
- **调度器和 attention 通过模块级全局 `Context` 通信**：不是通过函数参数。`prepare_prefill/decode` 写 `set_context(...)`，`Attention.forward` 读 `get_context()`。CUDA graph 重放时也读这个全局。

如果代码层面发现还有其他高价值的非显然点（如前缀缓存的 xxhash 链式哈希、TP 共享内存协议、`Sequence.__getstate__` 精简 pickle），可加 1-2 条，但总数 ≤ 5。

### 5. Tensor Parallelism 路径（≤ 6 行）
回答这几个问题：
- worker 进程怎么起的？（`mp.Process` + `spawn` context, in `LLMEngine.__init__`）
- rank 0 和 rank≥1 怎么通信？（`SharedMemory` name=`"nanovllm"` + 每 worker 一个 `mp.Event`，pickle `(method_name, args)`）
- NCCL 怎么初始化？（`tcp://localhost:2333`）
- 权重怎么分片？（`layers/linear.py` 的 `ColumnParallel/RowParallel/QKVParallel/MergedColumnParallel`，`packed_modules_mapping` 在 `Qwen3ForCausalLM` 上）

### 6. 改动指引（3-5 条 bullet）
针对常见改动方向给出"先看哪里"的指引，例如：
- 加新模型 → `models/qwen3.py` + 定义 `packed_modules_mapping` + 在 `ModelRunner.__init__` 替换实例化
- 改调度策略 → `engine/scheduler.py`（注意 prefill / decode 互斥规则）
- 改 KV cache 布局 → `engine/block_manager.py` + `layers/attention.py` 的 `store_kvcache` Triton kernel + `ModelRunner.allocate_kv_cache`
- 加采样策略 → `layers/sampler.py` + `sampling_params.py`（注意当前显式禁止 greedy）

## 执行流程

1. **读 `AGENTS.md`**（如果存在）——里面已有大部分架构信息，作为基础。
2. **快速校验关键文件存在且未大幅变动**：用 Read 抽查 `engine/llm_engine.py`、`engine/scheduler.py`、`engine/model_runner.py`、`layers/attention.py` 的关键函数签名。AGENTS.md 可能过时——以代码为准。
3. **生成报告**，严格按上述 6 节结构输出。
4. 报告末尾**不要**加"如有问题继续问我"之类的客套话。

## 风格约束

- 文件引用一律用 `path:line` 格式（如 `engine/scheduler.py:42`），方便用户跳转。
- 不要罗列每个文件——`ls` 就能做。
- 不要解释"什么是 vLLM""什么是 KV cache"——读者已经知道。
- 不要复读 README——README 自己会读。聚焦在**读多个文件才能拼出来的图景**。
- 全程中文。
