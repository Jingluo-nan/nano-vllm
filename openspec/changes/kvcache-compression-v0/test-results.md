# 单测结果记录（kvcache-compression-v0）

> 留痕用：记录各任务挂起单测的编写、运行结果与环境，供归档与回溯。
> 约定：每条注明 任务号 / 测试文件 / 结果 / 日期 / 环境 / 备注。

## 命名规则

测试文件统一放在 `scratch/`，命名 `test_task<步骤>_<子步骤>_<被测对象简述>.py`：

- `<步骤>_<子步骤>` 对应 `tasks.md` 的任务号（如 5.1 → `task5_1`，点用下划线）；
- `<被测对象简述>` 用被测函数/方法名或其语义（snake_case，如 `compact_kv`、`block_manager_evict`）；
- 一个任务号对应一个测试文件，文件内每个 `test_*` 函数是一个用例；
- 文件 docstring 首行写「task X.Y 单测：<被测对象> <验证点>」，并列出约定与运行命令。

示例：

| 任务 | 文件名 |
|------|--------|
| 3.3 | `test_task3_3_seq_getstate.py` |
| 4.1 | `test_task4_1_streaming_keep_indices.py` |
| 5.2 | `test_task5_2_compact_kv.py` |

## 汇总

| 任务 | 测试文件 | 结果 | 日期 | 环境 |
|------|----------|------|------|------|
| 1.4 默认配置 baseline 三项度量 | `scratch/test_task1_4_baseline.py` | ✅ 记录（preempt=0/batch=8/tok_s≈260） | 2026-06-02 | CUDA + flash-attn + Qwen3-0.6B 权重 |
| 3.3 `__getstate__/__setstate__` round-trip 同步 `num_dropped_kv` | `scratch/test_task3_3_seq_getstate.py` | ✅ 3/3 PASS | 2026-06-01 | CPU，零依赖（无需 pytest/CUDA） |
| 3.1 `num_kv` 属性 | `scratch/test_task3_1_num_kv.py` | ✅ 4/4 PASS | 2026-06-02 | CPU，零依赖 |
| 3.2 `num_kv_blocks`/`last_kv_block_num_tokens` 跨块边界 | `scratch/test_task3_2_num_kv_blocks.py` | ✅ 6/6 PASS | 2026-06-02 | CPU，零依赖 |
| 4.1 `streaming_keep_indices` 边界 | `scratch/test_task4_1_streaming_keep_indices.py` | ✅ 6/6 PASS | 2026-06-01 | CPU，零依赖 |
| 4.2 `should_compress` 触发 | `scratch/test_task4_2_should_compress.py` | ✅ 5/5 PASS | 2026-06-01 | CPU，零依赖 |
| 5.1 `evict` free_block_ids 增量 | `scratch/test_task5_1_block_manager_evict.py` | ✅ 5/5 PASS | 2026-06-01 | CPU，零依赖（纯 Python 记账，无需 CUDA） |
| 5.2 `compact_kv` 逐位一致 | `scratch/test_task5_2_compact_kv.py` | ✅ 3/3 PASS | 2026-06-01 | CUDA（轻量桩 + 哨兵编码） |
| 5.3 CPU 参考对拍（两组 keep_indices） | `scratch/test_task5_3_gather_attention.py` | ✅ 2/2 PASS | 2026-06-01 | CUDA + flash-attn |
| 6.1/6.2/6.3 prepare_decode/may_append 改用 num_kv（索引算术） | `scratch/test_task6_prepare_decode_num_kv.py` | ✅ 6/6 PASS（修正 2 处用例计算错误后） | 2026-06-02 | CPU，零依赖 |
| 8.1 端到端实跑（开启压缩长生成） | `scratch/test_task8_1_e2e_gpu.py` | ✅ PASS（不崩 + 压缩触发 + used_peak 下降） | 2026-06-02 | CUDA + flash-attn + Qwen3-0.6B 权重 |
| 9.1 block 确实释放（开/关对照 + 压缩点回升） | `scratch/test_task9_1_block_reclaim.py` | ✅ PASS（used_peak 6→4、free 回升、无泄漏） | 2026-06-02 | CUDA + flash-attn + Qwen3-0.6B 权重 |
| 9.2 PPL 没崩（teacher forcing 开/关对照） | `scratch/test_task9_2_perplexity.py` | ✅ PASS（全程 +4.65%，未爆炸） | 2026-06-02 | CUDA + flash-attn + Qwen3-0.6B 权重 |
| 9.3 收益对照（preemption↓ / decode batch↑） | `scratch/test_task9_3_benefit.py` | ✅ PASS（preempt 4→0、batch 12.93→16） | 2026-06-02 | CUDA + flash-attn + Qwen3-0.6B 权重 |
| 9.4 openspec validate --strict | （CLI）`@fission-ai/openspec` | ✅ "Change is valid"（exit=0） | 2026-06-03 | npx @fission-ai/openspec@latest |

## 明细

### 1.4 — 默认配置 baseline 三项度量（2026-06-02）

- 测试文件：`scratch/test_task1_4_baseline.py`
- 运行方式：`python scratch/test_task1_4_baseline.py`
- 环境：CUDA + flash-attn + Qwen3-0.6B（RTX 4050）；enforce_eager=True；默认配置（压缩关闭）
- 工作负载：8 序列 × 512 token，gpu_util=0.9，seed=0（常规运行，不刻意制造 KV 压力）
- 结果（`LLMEngine.metrics`）：
  - preemptions = 0
  - avg_decode_batch = 8.00
  - decode_tok_s ≈ 260（多次 258~263，计时波动）
- 说明：preemptions / avg_decode_batch 确定可复现；decode_tok_s 受计时影响有小波动。此为 task 9.3 收益对照的参照基线（9.3 用更高并发+长生成单独制造压力场景，数值不直接可比）。

### 3.1 — `num_kv` 属性（2026-06-02）

- 测试文件：`scratch/test_task3_1_num_kv.py`
- 运行方式：`python scratch/test_task3_1_num_kv.py`
- 环境：CPU，零依赖
- 结果：**4/4 PASS**
  - `test_num_kv_closed_equals_num_tokens` — PASS（num_dropped_kv==0 时 num_kv≡num_tokens，1..49 全覆盖）
  - `test_num_kv_subtracts_dropped` — PASS（num_kv = num_tokens - num_dropped_kv）
  - `test_num_kv_auto_increments_on_decode` — PASS（append_token 后 num_kv 自动 +1，num_dropped_kv 不变）
  - `test_num_kv_closed_auto_increments` — PASS（关闭压缩 decode 后仍恒等 num_tokens）
- 覆盖约定：num_kv 定义、decode 自增、零回归。

### 3.2 — `num_kv_blocks` / `last_kv_block_num_tokens` 跨块边界（2026-06-02）

- 测试文件：`scratch/test_task3_2_num_kv_blocks.py`
- 运行方式：`python scratch/test_task3_2_num_kv_blocks.py`
- 环境：CPU，零依赖（顶部 `Sequence.block_size = BS=4` 对齐类属性以测小块边界）
- 结果：**6/6 PASS**
  - `test_exact_block_boundary_last_block_full` — PASS（整除时末块满 == BS）
  - `test_half_full_last_block` — PASS（非整除，末块余数 = num_kv % BS）
  - `test_single_block` — PASS
  - `test_based_on_num_kv_not_num_tokens` — PASS（压缩后基于 num_kv 算块，对照旧 num_blocks 基于 num_tokens）
  - `test_zero_regression_equals_num_blocks` — PASS（num_dropped_kv==0 时等于 num_blocks/last_block_num_tokens）
  - `test_last_block_num_tokens_range_invariant` — PASS（不变量 1≤last≤BS）
- 覆盖约定：ceil 分块、末块 token 数公式、基于 num_kv 而非 num_tokens、零回归、范围不变量。

### 3.3 — `__getstate__/__setstate__` round-trip（2026-06-01）

- 测试文件：`scratch/test_task3_3_seq_getstate.py`
- 运行方式：`python scratch/test_task3_3_seq_getstate.py`
- 环境：CPU，零依赖（不需要 pytest / CUDA）
- 结果：**3/3 PASS**
  - `test_prefill_roundtrip_preserves_dropped_kv` — PASS（prefill 带完整 token_ids，`num_dropped_kv` 存活）
  - `test_decode_roundtrip_preserves_dropped_kv` — PASS（decode 只带 last_token、丢弃 token_ids，`num_dropped_kv` 存活）
  - `test_default_dropped_kv_is_zero_after_roundtrip` — PASS（压缩关闭路径 `num_dropped_kv==0`，`num_kv==num_tokens`，零回归）
- 覆盖约定：`num_dropped_kv` 被编入序列化元组并跨 pickle 往返保持；prefill/decode 两条分支的 token_ids 行为不被破坏。

### 4.1 — `streaming_keep_indices` 块对齐保留集 + 边界（2026-06-01）

- 测试文件：`scratch/test_task4_1_streaming_keep_indices.py`
- 运行方式：`python scratch/test_task4_1_streaming_keep_indices.py`
- 环境：CPU，零依赖（不需要 pytest / CUDA）
- 结果：**6/6 PASS**
  - `test_drops_middle_blocks_full_blocks` — PASS（满块场景丢中段，保留 sink+recent）
  - `test_last_block_half_full` — PASS（末块半满，recent 段含半满末块正确收尾到 num_kv）
  - `test_covers_all_blocks_returns_all` — PASS（sink+recent==num_blocks 时返回全部）
  - `test_sink_plus_recent_exceeds_blocks_returns_all` — PASS（sink+recent 超过块数仍返回全部、不越界不重叠）
  - `test_just_one_block_droppable` — PASS（仅一块可丢的临界场景）
  - `test_no_overlap_between_sink_and_recent` — PASS（多块场景 sink/recent 不重叠、中段连续丢弃）
- 覆盖约定：返回下标落在 [0, num_kv)、有序、去重；块对齐保留 sink+recent；覆盖全部块时退化为全保留。

### 4.2 — `should_compress` 触发谓词（2026-06-01）

- 测试文件：`scratch/test_task4_2_should_compress.py`
- 运行方式：`python scratch/test_task4_2_should_compress.py`
- 环境：CPU，零依赖（不需要 pytest / CUDA）
- 结果：**5/5 PASS**
  - `test_below_threshold_no_compress` — PASS（块数不足不触发）
  - `test_equal_threshold_no_compress` — PASS（恰好等于 sink+recent 不触发）
  - `test_just_above_threshold_compress` — PASS（刚超过一块即触发）
  - `test_well_above_threshold_compress` — PASS（远超阈值触发）
  - `test_zero_blocks_no_compress` — PASS（0 块不触发）
- 覆盖约定：num_kv_blocks > sink+recent 才触发；临界点（等于不触发、+1 触发）锁定。

### 5.1 — `BlockManager.evict` 尾块回收 + free_block_ids 增量（2026-06-01）

- 测试文件：`scratch/test_task5_1_block_manager_evict.py`
- 运行方式：`python scratch/test_task5_1_block_manager_evict.py`
- 环境：CPU，零依赖（BlockManager 是纯 Python 记账，无需 CUDA）
- 结果：**5/5 PASS**
  - `test_basic_tail_reclaim_and_free_increment` — PASS（还尾 3 块，free 增量精确 == 被还块数，且为原尾部块）
  - `test_half_full_last_block_block_count_ceil` — PASS（num_keep=9 → ceil=3 块）
  - `test_shared_block_not_reclaimed` — PASS（ref_count=2 的共享块只 -1 不归还，符合当前行为）
  - `test_no_reclaim_when_keep_fills_all_blocks` — PASS（new==len 时不删块、free 不变）
  - `test_num_dropped_kv_formula_various` — PASS（多组 num_dropped_kv == num_tokens-num_keep-1）
- 覆盖约定：紧凑块数 ceil(num_keep/bs)；尾块按 ref_count 归还到 free_block_ids；block_table 截断；num_dropped_kv 公式。
- 备注：step7 的 hash 注销 / 仅压 ref_count==1 尚未实现，本轮不在测试范围。

### 5.2 — `ModelRunner.compact_kv` 物理 gather 逐位一致（2026-06-01）

- 测试文件：`scratch/test_task5_2_compact_kv.py`
- 运行方式：`python scratch/test_task5_2_compact_kv.py`
- 环境：CUDA（轻量桩 _Dummy 持 block_size/kv_cache，把 compact_kv 当未绑定方法调；哨兵编码 val(k,layer,slot) 反查每 slot 内容）
- 结果：**3/3 PASS**
  - `test_block_aligned_gather` — PASS（非恒等 block_table[3,1,4,2,0]，保留块0+末块，逐位匹配）
  - `test_scattered_gather` — PASS（散落下标 [0,2,5,7,9,13,18]，token 级通用置换）
  - `test_overlap_safe` — PASS（前部槽内乱序，old/new 大量重叠，.clone() 保证不自我污染）
- 覆盖约定：new 位置 j 内容 == 旧 keep_indices[j] 位置；断言遍历 k×layer×j → 同时验证无串层/串 k-v 半；new_slots 紧凑到 block_table 前部物理块。
- 未覆盖：copy 不重旋转的 RoPE 正确性 —— 留给 5.3 的 flash-attn 端到端对拍。

### 5.3 — gather 后 flash-attn 注意力 vs CPU 手算对拍（2026-06-01）

- 测试文件：`scratch/test_task5_3_gather_attention.py`
- 运行方式：`python scratch/test_task5_3_gather_attention.py`
- 环境：CUDA + flash-attn（BS=256 满足 flash 分页路径约束；fp16 KV，参考用同批值上采样 fp32 手算）
- 结果：**2/2 PASS**
  - `test_block_aligned` — PASS（num_kv=600 跨 3 块，非恒等 block_table，保留块0+末块丢中段，max_abs_diff=7.6e-5）
  - `test_scattered` — PASS（零散跨块下标，token 级通用 gather，max_abs_diff=2.7e-4）
- 容差：rtol=1e-2, atol=2e-2（fp16 flash vs fp32 参考的合理量级）。
- 覆盖约定：compact_kv gather → 紧凑 block_table + cache_seqlens=len(keep) 走 flash_attn_with_kvcache，结果与「只在 K[keep]/V[keep] 上手算的注意力」逐元素一致；注意力对 key 排列不变，故 gather 顺序不影响。
- 未覆盖：GQA（num_heads>num_kv_heads，本测试为 MHA）；pending last_token 的拼接（本测试 cache_seqlens=len(keep)，隔离 gather 正确性）。

### 6.1/6.2/6.3 — prepare_decode / may_append 改用 num_kv（2026-06-02 重跑修正）

- 测试文件：`scratch/test_task6_prepare_decode_num_kv.py`
- 运行方式：`python scratch/test_task6_prepare_decode_num_kv.py`
- 环境：CPU，零依赖（6.1/6.3 用真 BlockManager+Sequence；6.2 复刻 prepare_decode 逐条公式对拍，因真函数走 .cuda()）
- 结果：**6/6 PASS**（重跑时发现并修正 2 处用例自身的计算错误，详见下）
  - `test_6_2_zero_regression_when_no_compression` — PASS（num_dropped_kv==0 时新旧公式逐位一致）
  - `test_6_2_slot_lands_in_compacted_layout` — PASS（压缩落点）
  - `test_6_2_just_after_evict_pending_token_starts_new_block` — PASS（evict 后起新块）
  - `test_6_3_may_append_uses_num_kv` — PASS（开块判据用 num_kv）
  - `test_6_3_zero_regression_may_append` — PASS（零回归）
  - `test_6_3_can_append_uses_num_kv` — PASS
- **本次修正（之前 tasks.md 误标 6/6，实际执行早死在 6.2 未跑到 6.3）**：
  1. 6.2：`seq.last_kv_block_num_tokens` 用 `Sequence.block_size`(类属性=256)，与 slot 公式传的 BS=4 打架 → 测试模块顶部加 `Sequence.block_size = BS` 对齐。
  2. 6.3：两个用例误用 `num_kv=9` 当作 `9%4 != 1`（实际 `9%4==1` 会触发开块）→ 改用 `num_kv=10`（`10%4=2`）。
- 仍挂起：真 `prepare_decode` 走 `.cuda()` 的端到端实跑（需加载完整模型构建 ModelRunner），归入 step 8.1 的 GPU 端到端范畴（见下，已于 2026-06-02 跑通）。

### 8.1 — 端到端实跑（开启 KV 压缩长生成）（2026-06-02）

- 测试文件：`scratch/test_task8_1_e2e_gpu.py`
- 运行方式：`python scratch/test_task8_1_e2e_gpu.py`
- 环境：CUDA + flash-attn + Qwen3-0.6B 权重（RTX 4050 Laptop 6.4GB）；enforce_eager=True 隔离 CUDA graph
- 配置：block_size=256, sink=1, recent=3；单序列 ignore_eos 生成 1400 token（>1024 才跨过触发阈值）
- 设计：每个配置在**独立子进程**跑（init_process_group 与 KV 显存预算不可在同进程内重建两次）
- 结果：**PASS**
  - 关闭压缩（baseline）：1400 token，used_peak=6 块，evict=0
  - 开启压缩：1400 token，used_peak=**4** 块，evict=**2** 次，free_low 略升
  - 断言：生成满 max_tokens ✓；evict 触发>0 ✓；used_peak(on)≤used_peak(off)（6→4）✓
- 意义：① 8.1「不崩/不非法内存」达成；② 同时为 9.1「block 确实释放」提供初步证据（used_peak 6→4，压到 sink1+recent3=4 块）。
- 未覆盖：多序列并发 + 抢占场景；CUDA graph（enforce_eager=False）路径；PPL 质量（task 9.2）。

### 9.1 — block 确实释放（开/关对照 + 压缩点 free 回升）（2026-06-02）

- 测试文件：`scratch/test_task9_1_block_reclaim.py`
- 运行方式：`python scratch/test_task9_1_block_reclaim.py`
- 环境：CUDA + flash-attn + Qwen3-0.6B（RTX 4050）；enforce_eager=True；每配置独立子进程
- 配置：block_size=256, sink=1, recent=3；单序列 ignore_eos 生成 1400 token
- 结果：**PASS**
  | 指标 | 关闭 | 开启 |
  |------|------|------|
  | used_block 峰值 | 6 | 4 |
  | free 起始/结束 | 98/98 | 98/98 |
  | evict 次数 | 0 | 2 |
  - 压缩点 free 回升：evict#1 93→94(+1)、evict#2 93→94(+1)
- 断言：used_peak(on)<used_peak(off)（6→4）✓；每个压缩点 reclaimed>0 ✓；序列结束 free 复原无泄漏（98/98）✓
- 结论：block 回收实时发生（不是延迟）、峰值被钉在 sink+recent=4 块、生命周期无泄漏。
- 未覆盖：多序列并发 + 抢占下的回收/回升（v0 单序列已足够证明回收机制；并发留待 9.3 收益对照或后续）。

### 9.2 — PPL 没崩（teacher forcing 开/关对照）（2026-06-02）

- 测试文件：`scratch/test_task9_2_perplexity.py`
- 运行方式：`python scratch/test_task9_2_perplexity.py`
- 环境：CUDA + flash-attn + Qwen3-0.6B（RTX 4050）；enforce_eager=True；每配置独立子进程
- 方法：① 关闭压缩+种子0 生成参考序列 ref（prompt 21 + 续写 1400 = 1421 token）；② 对同一 ref，开/关压缩各重跑，monkeypatch `model_runner.sampler` 强制逐位返回 ref 的 token（teacher forcing，走同一路径），记录 `log_softmax(原始 logits)[token]`；③ PPL=exp(-平均 logprob)。唯一变量=压缩。
- 结果：**PASS**
  | | 关闭 | 开启 |
  |---|---|---|
  | PPL（全程） | 1.6697 | 1.7473（+4.65%） |
  | PPL（>1024 压缩生效段） | 1.7177 | 2.0346 |
- 断言：PPL 有限 ✓；全程相对上升 <50%（实际 +4.65%）✓
- 解读：退化集中在压缩生效后段（1.72→2.03，仍很低），符合"丢中段 KV 后越靠后越依赖被丢上下文"的直觉；StreamingLLM sink+recent 足以维持流畅度。与肉眼 essay 连贯的定性结论一致。
- 注意：PPL 用模型真实分布（不做 temperature 缩放）；teacher forcing 使开/关走完全相同 token 序列，差异纯归因于压缩。

### 9.3 — 收益对照（preemption↓ / avg decode batch↑）（2026-06-02）

- 测试文件：`scratch/test_task9_3_benefit.py`
- 运行方式：`python scratch/test_task9_3_benefit.py`
- 环境：CUDA + flash-attn + Qwen3-0.6B（RTX 4050）；enforce_eager=True；每配置独立子进程
- 工作负载：16 序列并发 × 2000 token，gpu_util=0.9（KV cache 98 块）。长生成让"压缩稳态"主导：关闭压缩 16×⌈2000/256⌉=128≫98 全程抢占；开启压缩钉在 sink1+recent3=4 块 →16×4=64<98 爬坡后稳住。
- 结果：**PASS**
  | 指标 | 关闭 | 开启 |
  |------|------|------|
  | preemptions | 4 | **0** |
  | avg_decode_batch | 12.93 | **16.00** |
  | decode_tok_s | 407.3 | 474.5（+16.5%） |
  | used_block 峰值 | 98/98（打满） | 64/98 |
- 断言：preemptions(on)≤off（4→0）✓；avg_decode_batch(on)≥off（12.93→16）✓
- 调参记录：先用 20/22 序列×1200 token，preemption 持平（3→3）——因抢占集中在压缩阈值前的"爬坡期"（前 1024 token 开/关相同）。改长生成（2000 token）使压缩稳态主导后，preemption 才显著下降（4→0）。
- 指标来源：`LLMEngine.metrics`（task 1 基线度量：preemptions / avg_decode_batch / decode_tok_s）。

### 9.4 — openspec validate（2026-06-03）

- 命令：`npx --yes @fission-ai/openspec@latest validate kvcache-compression-v0 --strict`
- CLI：`@fission-ai/openspec` v1.4.1（npm 上 `openspec` 是 0.0.0 空壳包，真正的工具在 `@fission-ai/openspec`；bin 名为 `openspec`）
- 结果：**`Change 'kvcache-compression-v0' is valid`，exit=0**（node 18 引擎警告不影响校验）
- 前置：tasks.md 全部任务已勾选（22→23，9.4 自身随此条勾上）。
- 备注：需联网（npx 下载包）；本机权限规则见 `.claude/settings.local.json`。
