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
| 3.3 `__getstate__/__setstate__` round-trip 同步 `num_dropped_kv` | `scratch/test_task3_3_seq_getstate.py` | ✅ 3/3 PASS | 2026-06-01 | CPU，零依赖（无需 pytest/CUDA） |
| 3.1 `num_kv` 属性 | — | ⏸ 挂起 | — | — |
| 3.2 `num_kv_blocks`/`last_kv_block_num_tokens` 跨块边界 | — | ⏸ 挂起 | — | — |
| 4.1 `streaming_keep_indices` 边界 | `scratch/test_task4_1_streaming_keep_indices.py` | ✅ 6/6 PASS | 2026-06-01 | CPU，零依赖 |
| 4.2 `should_compress` 触发 | `scratch/test_task4_2_should_compress.py` | ✅ 5/5 PASS | 2026-06-01 | CPU，零依赖 |
| 5.1 `evict` free_block_ids 增量 | `scratch/test_task5_1_block_manager_evict.py` | ✅ 5/5 PASS | 2026-06-01 | CPU，零依赖（纯 Python 记账，无需 CUDA） |
| 5.2 `compact_kv` 逐位一致 | `scratch/test_task5_2_compact_kv.py` | ✅ 3/3 PASS | 2026-06-01 | CUDA（轻量桩 + 哨兵编码） |
| 5.3 CPU 参考对拍（两组 keep_indices） | `scratch/test_task5_3_gather_attention.py` | ✅ 2/2 PASS | 2026-06-01 | CUDA + flash-attn |

## 明细

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
