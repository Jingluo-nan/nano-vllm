# v0 实施步骤（StreamingLLM + 物理 gather 管线）

> 定义：StreamingLLM 确定性 keep-set + 物理 gather + 所有层共用 block_table + copy-gather（不重旋转）。
> 黄金法则：先有度量基线，再动压缩；gather 先 CPU 对拍再上 Triton；默认关闭、可回退。

## 1. 基线度量（独立、零行为变化，先合）

- [x] 1.1 `scheduler.py` 的 `preempt` 累加 `num_preemptions`（`__init__` 初始化、`generate` 起始清零）。
- [x] 1.2 `generate` 累计每 decode step 的 batch（`-num_tokens`）求平均 decode batch。
- [x] 1.3 `generate` 累计 decode token 数与耗时算吞吐；存 `self.metrics` 并在 verbose 时打印 `{preemptions, avg_decode_batch, decode_tok_s}`。
- [ ] 1.4 跑默认配置拿 baseline 三项数字并记录（需 CUDA 机器；本地 Mac 无法实跑）。**验证**：可复现的 baseline。

## 2. 配置开关（默认关闭）

- [x] 2.1 `config.py` 增加 `enable_kv_compression=False`、`kv_sink_blocks=1`、`kv_recent_blocks=3`，`__post_init__` 在开启时断言 sink/recent>0。透传链经 `llm_engine` kwargs 过滤已通。

## 3. num_kv 解耦（F2 / 设计 D4，压缩前提）

- [x] 3.1 `sequence.py` 新增 `num_dropped_kv` 字段 + `num_kv` 属性(=num_tokens-num_dropped_kv，decode 自动+1，关闭时恒等 num_tokens)。✅ 单测通过（`scratch/test_task3_1_num_kv.py`，4/4 PASS，2026-06-02，CPU 零依赖）。
- [x] 3.2 `sequence.py` 增加 `num_kv_blocks`/`last_kv_block_num_tokens` 属性(基于 num_kv)。✅ 跨块边界单测通过（`scratch/test_task3_2_num_kv_blocks.py`，6/6 PASS，2026-06-02，CPU 零依赖）。
- [x] 3.3 `sequence.py` 的 `__getstate__/__setstate__` 同步 `num_dropped_kv`。✅ round-trip 单测通过（`scratch/test_task3_3_seq_getstate.py`，3/3 PASS，2026-06-01，CPU 零依赖）。

## 4. StreamingLLM 保留集（设计 D7）

- [x] 4.1 `kv_compression.py` 实现 `streaming_keep_indices(...)`，块对齐返回 sink+recent 下标（[0,num_kv) 有序去重，覆盖全部块时返回全部）。✅ 边界单测通过（`scratch/test_task4_1_streaming_keep_indices.py`，6/6 PASS，2026-06-01，CPU 零依赖）。
- [x] 4.2 `kv_compression.py` 实现 `should_compress(num_kv_blocks, sink_blocks, recent_blocks)` 触发谓词。实际接入 decode 循环在第 8 步。✅ 触发单测通过（`scratch/test_task4_2_should_compress.py`，5/5 PASS，2026-06-01，CPU 零依赖）。

## 5. 物理 gather 与 block 回收（设计 D1/D2/D3，最难、给 v1 复用）

- [x] 5.1 `block_manager.evict(seq, keep_indices)`：紧凑块数 `ceil(len/bs)`（通用含零散）、还尾部整块、`del block_table[new:]`、设 `num_dropped_kv=num_tokens-num_keep-1`（pending token 时序约定）。✅ free_block_ids 增量验证通过（`scratch/test_task5_1_block_manager_evict.py`，5/5 PASS，2026-06-01，CPU 零依赖）。TODO(step7) hash 注销留待。
- [x] 5.2 `model_runner.compact_kv(block_table, keep_indices)`：token 级通用置换，kv_cache 展平按 slot `index_select().clone()→index_copy_`，所有层一并搬、copy 不重旋转。须在 evict 前调用。✅ 逐位一致验证通过（`scratch/test_task5_2_compact_kv.py`，3/3 PASS，2026-06-01，CUDA）。
- [x] 5.3 CPU 参考对拍，**用两组 keep_indices**：①块对齐（streaming 风格）②零散（手构造散落下标，提前压通用搬迁路径，不等 v1）。给定 q/k/v 与 keep_indices，手算"只在保留 KV 上的注意力"与 gather 后走 flash-attn 比对。**验证**✅：两组 `torch.allclose` 均通过（`scratch/test_task5_3_gather_attention.py`，2/2 PASS，max_abs_diff 7.6e-5 / 2.7e-4，2026-06-01，CUDA + flash-attn）。

## 6. prepare_decode 改用 num_kv（设计 D4/D6）

- [x] 6.1 `model_runner.py` 的 `prepare_decode`：`context_lens ← seq.num_kv`。**验证**：压缩后 decode 不越界。✅ 索引算术对拍通过（`scratch/test_task6_prepare_decode_num_kv.py`，6/6）。
- [x] 6.2 新 token `slot_mapping ← block_table[-1]*block_size + (num_kv-1)%block_size`（用 `last_kv_block_num_tokens-1` 等价表达，贴合原 `last_block_num_tokens-1` 风格）；`positions` 保持 `len(seq)-1`。**验证**：新 token KV 写入紧凑布局的正确 slot。✅ 复刻公式对拍：零回归(num_dropped_kv==0 与旧公式逐位一致) + 压缩落点 + evict 后起新块均通过。真 `prepare_decode` 走 `.cuda()`，端到端实跑挂起 GPU 环境。
- [x] 6.3 `block_manager.py` 的 `may_append`/`can_append` 判断改用 `num_kv % block_size`。**验证**：压缩后继续 decode 能正确按需开新块。✅ 真 BlockManager+Sequence 实测（开块触发用 num_kv、零回归、can_append）通过。

## 7. 前缀缓存隔离（设计 D5）

- [x] 7.1 `evict` 改写块前将其从 `hash_to_block_id` 注销、置 `hash=-1`；断言只压 `ref_count==1` 的块，否则跳过该序列。**验证**：压缩后改写块不在 `hash_to_block_id` 的 value 中。✅ 用 identity 前缀(keep_indices[j]==j)区分未搬的 sink 块与被改写块：只注销+断言被改写块，sink 块 hash 保留。CPU 单测通过（`scratch/test_task7_1_evict_prefix_isolation.py`，4/4：改写块注销+sink保留、共享块断言拦截、全 identity 不误伤、无关 hash 零回归）+ 独立逻辑对拍 4 场景。ref_count 断言为防御性不变量，真正"跳过该序列"的门控在 step8 触发处（须在 compact_kv 之前判，否则数据已改写）。
- [x] 7.2 已压缩序列跳过后续 `hash_blocks` 注册（加标志位）。**验证**：压缩序列不再写入 `hash_to_block_id`。✅ `Sequence.kv_compressed`（默认 False、仅 rank0 调度侧用、不入 __getstate__），`evict` 末尾置 True，`hash_blocks` 开头 `if seq.kv_compressed: return`。CPU 单测通过（`scratch/test_task7_2_compressed_skip_hash.py`，4/4：默认 False、压缩序列 no-op、未压缩零回归、evict 置位后 hash_blocks 不再写入）。

## 8. 触发接入 + 端到端

- [x] 8.1 在 decode 循环接入：开关开启时，对触发条件成立的序列算 `streaming_keep_indices` → `evict` → gather。接口固定 `evict(seq, keep_indices)`（v1 仅换 keep_indices 来源）。**验证**：开启压缩端到端跑通长生成不报错、不崩、不非法内存。✅ 代码接入：`LLMEngine.step` 在 decode + 开关开启时调 `maybe_compress_kv(seqs)`；管线 = `should_compress`(physical_kv=num_kv-1) → `streaming_keep_indices` → `block_manager.can_evict`(gather 前 ref==1 门控) → `model_runner.call("compact_kv", block_table, keep_indices)`(所有 rank 各搬分片) → `block_manager.evict`。新增 `BlockManager.can_evict`/`_rewritten_block_ids`。CPU 单测通过（`scratch/test_task8_1_compress_pipeline.py`，6/6：can_evict 门控、触发决策真子集、evict 块账、短序列不触发）+ 独立逻辑对拍。⏸ **端到端实跑（开启压缩长生成不崩/不非法内存）挂起 GPU + flash-attn 环境**。

## 9. v0 验收（全过才进 v1）

- [ ] 9.1 **block 确实释放**：开/关压缩对比，开启时 `free_block_ids` 在压缩点回升、峰值 `used_block_ids` 更低。**验证**：计数证明回收。
- [ ] 9.2 **PPL 没崩**：相同种子开/关压缩算 perplexity，仅小幅上升不爆炸。**验证**：PPL 报告在阈值内。
- [ ] 9.3 **收益对照**（依赖第 1 步度量）：preemption↓、平均 decode batch↑。**验证**：与 baseline 可对照的数字。
- [ ] 9.4 `openspec validate kvcache-compression-v0` 通过，归档前任务全勾选。**验证**：validate 通过。