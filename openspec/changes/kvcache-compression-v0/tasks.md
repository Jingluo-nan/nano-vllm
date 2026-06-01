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

- [x] 3.1 `sequence.py` 新增 `num_dropped_kv` 字段 + `num_kv` 属性(=num_tokens-num_dropped_kv，decode 自动+1，关闭时恒等 num_tokens)。⏸ 单测挂起。
- [x] 3.2 `sequence.py` 增加 `num_kv_blocks`/`last_kv_block_num_tokens` 属性(基于 num_kv)。⏸ 跨块边界单测挂起。
- [x] 3.3 `sequence.py` 的 `__getstate__/__setstate__` 同步 `num_dropped_kv`。⏸ round-trip 单测挂起。

## 4. StreamingLLM 保留集（设计 D7）

- [x] 4.1 `kv_compression.py` 实现 `streaming_keep_indices(...)`，块对齐返回 sink+recent 下标（[0,num_kv) 有序去重，覆盖全部块时返回全部）。⏸ 边界单测挂起。
- [x] 4.2 `kv_compression.py` 实现 `should_compress(num_kv_blocks, sink_blocks, recent_blocks)` 触发谓词。实际接入 decode 循环在第 8 步。⏸ 触发单测挂起。

## 5. 物理 gather 与 block 回收（设计 D1/D2/D3，最难、给 v1 复用）

- [x] 5.1 `block_manager.evict(seq, keep_indices)`：紧凑块数 `ceil(len/bs)`（通用含零散）、还尾部整块、`del block_table[new:]`、设 `num_dropped_kv=num_tokens-num_keep-1`（pending token 时序约定）。⏸ free_block_ids 增量验证挂起。TODO(step7) hash 注销留待。
- [x] 5.2 `model_runner.compact_kv(block_table, keep_indices)`：token 级通用置换，kv_cache 展平按 slot `index_select().clone()→index_copy_`，所有层一并搬、copy 不重旋转。须在 evict 前调用。⏸ 逐位一致验证挂起。
- [ ] 5.3 CPU 参考对拍，**用两组 keep_indices**：①块对齐（streaming 风格）②零散（手构造散落下标，提前压通用搬迁路径，不等 v1）。给定 q/k/v 与 keep_indices，手算"只在保留 KV 上的注意力"与 gather 后走 flash-attn 比对。**验证**⏸：两组 `torch.allclose` 均通过。

## 6. prepare_decode 改用 num_kv（设计 D4/D6）

- [ ] 6.1 `model_runner.py` 的 `prepare_decode`：`context_lens ← seq.num_kv`。**验证**：压缩后 decode 不越界。
- [ ] 6.2 新 token `slot_mapping ← block_table[-1]*block_size + (num_kv-1)%block_size`；`positions` 保持 `len(seq)-1`。**验证**：新 token KV 写入紧凑布局的正确 slot。
- [ ] 6.3 `block_manager.py` 的 `may_append`/`can_append` 判断改用 `num_kv % block_size`。**验证**：压缩后继续 decode 能正确按需开新块。

## 7. 前缀缓存隔离（设计 D5）

- [ ] 7.1 `evict` 改写块前将其从 `hash_to_block_id` 注销、置 `hash=-1`；断言只压 `ref_count==1` 的块，否则跳过该序列。**验证**：压缩后改写块不在 `hash_to_block_id` 的 value 中。
- [ ] 7.2 已压缩序列跳过后续 `hash_blocks` 注册（加标志位）。**验证**：压缩序列不再写入 `hash_to_block_id`。

## 8. 触发接入 + 端到端

- [ ] 8.1 在 decode 循环接入：开关开启时，对触发条件成立的序列算 `streaming_keep_indices` → `evict` → gather。接口固定 `evict(seq, keep_indices)`（v1 仅换 keep_indices 来源）。**验证**：开启压缩端到端跑通长生成不报错、不崩、不非法内存。

## 9. v0 验收（全过才进 v1）

- [ ] 9.1 **block 确实释放**：开/关压缩对比，开启时 `free_block_ids` 在压缩点回升、峰值 `used_block_ids` 更低。**验证**：计数证明回收。
- [ ] 9.2 **PPL 没崩**：相同种子开/关压缩算 perplexity，仅小幅上升不爆炸。**验证**：PPL 报告在阈值内。
- [ ] 9.3 **收益对照**（依赖第 1 步度量）：preemption↓、平均 decode batch↑。**验证**：与 baseline 可对照的数字。
- [ ] 9.4 `openspec validate kvcache-compression-v0` 通过，归档前任务全勾选。**验证**：validate 通过。