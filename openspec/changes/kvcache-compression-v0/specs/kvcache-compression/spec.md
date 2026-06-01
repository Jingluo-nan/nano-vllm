## ADDED Requirements

### Requirement: 压缩功能可配置且默认关闭

引擎 SHALL 通过 `Config` 提供 `enable_kv_compression`（默认 `False`）、`kv_sink_blocks`、`kv_recent_blocks`。当 `enable_kv_compression=False` 时，调度、KV 写入、注意力路径 MUST 与未引入本功能前完全一致。

#### Scenario: 默认关闭时零回归

- **WHEN** 用户不设置任何压缩参数运行 `generate`
- **THEN** 输出 token 序列与原实现一致，`num_kv` 始终等于序列已缓存 token 数，不触发任何 gather

#### Scenario: 显式开启

- **WHEN** 用户设置 `enable_kv_compression=True` 并提供 sink/recent 块数
- **THEN** 引擎在 decode 阶段对超出 (sink+recent) 块的序列执行 StreamingLLM 压缩

### Requirement: StreamingLLM 确定性保留集

引擎 SHALL 提供 `streaming_keep_indices`，按块对齐返回"开头 `kv_sink_blocks` 块 + 最近 `kv_recent_blocks` 块"对应的保留 token 下标（有序）。保留集 MUST 仅由序列长度与配置决定，不依赖任何注意力分数。

#### Scenario: 块对齐保留

- **WHEN** 一条序列物理占用块数超过 `kv_sink_blocks + kv_recent_blocks`
- **THEN** `streaming_keep_indices` 返回开头 sink 块与最近 recent 块的全部 token 下标，中间块的下标被排除

#### Scenario: 未超限不压缩

- **WHEN** 序列物理占用块数不超过 `kv_sink_blocks + kv_recent_blocks`
- **THEN** 不触发压缩，按常规 decode 执行

### Requirement: 跨层统一的物理 gather 与 block 回收

压缩 SHALL 按 keep_indices 把幸存 KV 在 paged cache 中搬紧到前部连续 slot，对该序列**所有层使用同一份 block_table 与同一份 slot 置换**，并将空出的尾部整块归还 `BlockManager` 的空闲池。搬运 MUST 以字节拷贝语义进行（不重新施加 RoPE），并遵守 `kvcache_block_size` 256 对齐。

#### Scenario: 回收空闲 block

- **WHEN** 压缩后某序列保留 KV 占用块数少于压缩前
- **THEN** 多出的物理块归还 `free_block_ids`，`used_block_ids` 相应减少，`seq.block_table` 同步为紧凑后的块列表

#### Scenario: 所有层布局一致

- **WHEN** 对一条序列执行一次压缩
- **THEN** 该序列每一层的 KV 都按同一份 slot 置换搬运，压缩后所有层共用同一个 `block_table` 与同一个 `context_len`

#### Scenario: gather 结果数值正确

- **WHEN** 对一条仅保留部分 KV 的序列继续 decode 一个 token
- **THEN** 新 token 对保留 KV 的注意力输出，等价于"在完整 KV 上计算后仅取保留 KV 子集"的参考结果（被丢弃 KV 不参与），且与压缩前相比保留 KV 的向量值逐位一致

### Requirement: num_kv 元数据解耦

`Sequence` SHALL 维护 `num_kv` 表示 cache 内实际保留的 KV 数。压缩后引擎 MUST 用 `num_kv` 而非 `len(seq)` 计算 `context_lens`、新 token 的 `slot_mapping`、以及 `may_append`/`can_append` 的开块判断；新 token 的 RoPE 位置 MUST 仍使用真实生成步 `len(seq)-1`。

#### Scenario: context_lens 反映保留数

- **WHEN** 一次压缩完成，`num_kv` 减小
- **THEN** 下一 decode step 的 `context_lens` 等于该序列的 `num_kv`，flash-attn 仅读取保留的 KV，不越界

#### Scenario: 位置不重排

- **WHEN** 压缩后继续 decode
- **THEN** 新 token 的位置仍为 `len(seq)-1`（真实生成步），保留 KV 的位置不被重排为连续整数

#### Scenario: num_kv 随序列序列化到 TP worker

- **WHEN** 在 tensor parallel 下把序列状态发往 worker
- **THEN** `num_kv` 随 `__getstate__/__setstate__` 一并传递，worker 侧据其构建正确的 decode 张量

### Requirement: 压缩与前缀缓存隔离

被 gather 改写内容的 block，引擎 SHALL 立即从 `hash_to_block_id` 注销并置 `block.hash=-1`；压缩 MUST 只作用于 `ref_count==1` 的块。已压缩过的序列 SHALL 跳过后续前缀缓存注册。

#### Scenario: 改写块退出哈希复用

- **WHEN** 一个已注册进 `hash_to_block_id` 的 block 被 gather 改写
- **THEN** 该 block 不再可被任何后续序列经内容哈希命中复用

#### Scenario: 不压缩共享块

- **WHEN** 某序列待压缩，但其某 block 的 `ref_count > 1`（被其他序列共享）
- **THEN** 引擎跳过该序列的压缩或保护该共享块，不改写被共享的物理内容
