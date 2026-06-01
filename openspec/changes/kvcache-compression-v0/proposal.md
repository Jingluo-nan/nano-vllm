## Why

Decode 阶段 KV-Cache 随每生成一个 token 单调增长。高并发长生成下物理 block 很快耗尽：`BlockManager.can_append` 没有空闲 block 时，`Scheduler` 抢占（preempt）running 队尾序列、释放其 block，被抢占的序列下次必须从头 prefill（recompute）。抢占越多 → 同时存活序列越少 → 有效 batch size 越小 → decode 吞吐越低。

v0 引入基于 **StreamingLLM** 的 KV-Cache 压缩：周期性把每条序列的 KV 裁剪为"开头 sink 块 + 最近 recent 块"，物理 gather 紧凑后回收空出的 block。**不需要注意力打分**（保留集确定性可算），目的是先把最难的"物理整理 + 元数据/位置一致性"管线跑通，为后续 v1（注意力 Top-K）复用。

## What Changes

> v0 是分阶段计划的第一步。v1 把保留策略换成注意力 Top-K（管线复用），v2 加触发调优与 A/B 数据。本 change 只覆盖 v0。

- **新增压缩开关**：`Config` 增加 `enable_kv_compression`（默认 `False`）、`kv_sink_blocks`、`kv_recent_blocks`。关闭时行为与原实现完全一致。
- **解耦 `len(seq)` 与"cache 内 KV 数"**：`Sequence` 新增 `num_kv` 字段。压缩后 cache 里的 KV 少于序列 token 数，`context_lens`/`slot_mapping`/`may_append` 等不能再从 `len(seq)` 推导。
- **新增物理 gather/compact**：按确定性 keep_indices 把幸存 KV 在 paged cache 中搬紧（copy，不重旋转），对**所有层共用同一份 block_table 与同一份 slot 置换**，回收尾部整块还给 `BlockManager`。
- **新增 StreamingLLM 保留集**：`streaming_keep_indices(num_kv, sink_blocks, recent_blocks, block_size)`，块对齐返回保留下标。
- **前缀缓存隔离**：被 gather 改写内容的 block 从 `hash_to_block_id` 注销、置 `hash=-1`；v0 对已压缩序列跳过后续前缀缓存注册（简化，代价是这些序列失去前缀复用）。
- **位置语义保持**：新 token 的 RoPE 位置仍用真实生成步（`len(seq)-1`），保留 KV 不重排、不重旋转——因总长远小于 RoPE 训练范围，copy 语义下自动正确。

## Capabilities

### New Capabilities
- `kvcache-compression`: 基于 StreamingLLM 的 decode 阶段 KV-Cache 物理压缩——确定性 keep-set、跨层统一的 gather/block 回收、`num_kv` 元数据解耦、前缀缓存隔离与 RoPE 位置保持。

### Modified Capabilities
<!-- 当前分支 openspec/specs/ 为空，无既有 spec 的需求变更。 -->

## Impact

- `nanovllm/config.py` — 压缩配置项。
- `nanovllm/engine/sequence.py` — `num_kv` 字段、基于 num_kv 的尾块填充、`__getstate__/__setstate__` 同步（TP 热路径）。
- `nanovllm/engine/block_manager.py` — `evict(seq, keep_indices)`、`may_append`/`can_append` 改用 num_kv、hash 注销、已压缩序列跳过 `hash_blocks`。
- `nanovllm/engine/model_runner.py` — 跨层 gather kernel、`prepare_decode` 的 `context_lens`/`slot_mapping` 改用 num_kv。
- `nanovllm/engine/scheduler.py` 或 `llm_engine.py` — 压缩触发点 + `streaming_keep_indices` helper。
- **不动**：`attention.py` 签名、CUDA graph 捕获逻辑、rotary 代码（gather 在 model forward 之外、copy 语义、位置已烘焙）。
- **依赖**：不新增第三方依赖。
- **风险/约束**：丢中段 KV 有损，需 PPL/质量对照；gather 必须遵守 `kvcache_block_size` 256 对齐；slot 越界是最大实现风险。
