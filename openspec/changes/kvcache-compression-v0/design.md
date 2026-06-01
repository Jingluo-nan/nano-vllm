## Context

nano-vllm 的 KV-Cache 是 paged 的：切成 `block_size`（默认 256）大小的块，`block_table` 记录"逻辑块 i → 物理块号"。decode 每步给序列追加一个 token 的 KV，占用单调增长，空闲块耗尽即触发抢占 + recompute。

本设计是分阶段计划（v0/v1/v2）的 v0。已落实的关键调研结论（来自代码阅读）：

- **F1：RoPE 在写 cache 前施加**（`qwen3.py:92-94` 先 `rotary_emb(q,k)` 再 `attn()`）→ cache 里存的是**已旋转**的 K，带各自原始位置 j 的旋转。
- **F2：`len(seq)` 与"cache 内 KV 数"必须解耦**。当前 `context_lens`/`positions`/`slot_mapping`/`num_blocks` 全从 `len(seq)` 推导，隐含"cache KV 数 == 序列 token 数"，压缩打破此假设。
- **F3：压缩只需改 `prepare_decode` 产出的张量**（`set_context → _CONTEXT 全局 → attention` 读取）。
- **F4：前缀缓存在 `hash_blocks` 注册写满块、`deallocate` 保留 token_ids**；改写块内容会让 hash 失效。

约束：`kvcache_block_size` 为 256 倍数；decode 走预捕获 CUDA graph；采样 Gumbel-max（`temperature>1e-10`）。

## Goals / Non-Goals

**Goals:**
- 把每条序列 decode 期的 KV footprint 从"随生成线性增长"压成"恒定 ≈ (sink+recent) 块"，降抢占、提 batch、提 decode 吞吐。
- **建成物理 gather/compact 管线**，用 StreamingLLM 的确定性 keep-set 把它单独测对，为 v1 注意力 Top-K 复用。
- 默认关闭、可一键回退、零回归。

**Non-Goals:**
- 不做注意力打分（v1）、不做周期/阈值触发调优与 A/B（v2）。
- 不追求无损；接受可控质量下降，质量评估属验收。
- 不做"无限上下文 / cache 位置重编号"——本场景总长 ≪ RoPE 训练范围，无需重旋转。
- 不改 TP 协议、不引新依赖、第一版不强求 sub-block 精度。

## Decisions

### D1. v0 就建物理 gather kernel，而非 no-move 取巧
- **选择**：v0 即实现"按 keep_indices 物理搬紧幸存 KV、回收尾块"的 gather。
- **为什么**：曾考虑过 no-move（StreamingLLM 块对齐时只摘 block_table 中间项、不搬数据）。但 **v1 的注意力 Top-K 保留集是散落的**——每个块都有幸存者，不搬就一块都省不出来。若 v0 走 no-move，则完全没替 v1 验证最难的 gather 管线，违背"v0 为 v1 铺路"的初衷。故 v0 必须建 gather；StreamingLLM 在此只提供**确定性、可手算对拍**的 keep-set 来测这套机器。
- **代价**：比 no-move 多写一个搬运 kernel + hash 注销；但这是 v1 的刚需，提前还。

### D2. 所有层共用一份 block_table；keep-set 是"全序列级"的一份
- **选择**：`evict(seq, keep_indices)` 对该序列**所有层**用**同一份 keep_indices、同一份 slot 置换**做 gather；`block_table`/`context_lens` 不加层维度。
- **为什么**：nano-vllm 的 `block_table` 与全局 `Context.context_lens` 本就无层维度，所有层共享。若各层淘汰不同 token（尤其数量不同），各层想要的 block_table 长度就分叉，必须升成 per-layer block_table——伤筋动骨。v1 接注意力打分时走"**全局策略**"：各层/各 head 分数先聚合成一份全序列重要性，再统一淘汰，从而保住单 block_table。
- **代价**：放弃层间差异精度，换单 block_table / 单 context_len / 单次 gather。PyramidKV 式每层预算属 v2 之后。

### D3. copy-gather，不重旋转 → RoPE 自动正确
- **选择**：gather 只把 K/V 向量**按字节拷到新 slot**，不重新施加 RoPE；新 token 的 `positions` 仍用真实生成步 `len(seq)-1`。
- **为什么**：F1 表明 cache 内 K 已带原始位置旋转；flash-attn 不重旋转 cache、只 gather + causal mask。保留 K 拷到新 slot 后仍带原始旋转，query 用真实位置 → 相对距离 (i−j) 不变。**且本场景总长（~3072）≪ RoPE 训练范围（Qwen3 32k+, theta 1e6），不会 OOD**，故无需 StreamingLLM 论文式的 cache 位置重编号。
- **备选**：重编号 + 重旋转 K（无限流才需要）。放弃——本场景没必要，且会把 RoPE 命门变成真问题。

### D4. `num_kv` 解耦（F2 落地）
- **选择**：`Sequence` 新增 `num_kv` = cache 内实际保留 KV 数。`context_lens ← num_kv`；新 token `slot_mapping ← block_table[-1]*block_size + (num_kv-1)%block_size`；`may_append/can_append` 判断改用 `num_kv % block_size`；`positions` **保持 `len(seq)-1` 不变**。`__getstate__/__setstate__` 同步 `num_kv`（TP 热路径）。
- **为什么**：压缩后物理填充按 num_kv，不再等于 len(seq)；若不解耦，flash-attn 会按 len 读到越界/垃圾 KV。

### D5. 前缀缓存隔离 + v0 简化
- **选择**：gather 改写内容的 block 立即从 `hash_to_block_id` 注销、置 `hash=-1`；并**只对 `ref_count==1` 的块压缩**（不动被共享的前缀块）。**v0 进一步简化：对已压缩过的序列跳过后续 `hash_blocks` 注册**。
- **为什么**：改写块的 token_ids/hash 失效，别的序列命中会读脏数据。已压缩序列的块内容不再是干净前缀，维护其 hash 收益低、易错——v0 直接放弃其前缀复用价值（这些序列已在 decode 中后段，复用价值本就低）。
- **代价**：被压序列失去前缀缓存。可接受；若 v2 想找回，再单独处理。

### D6. gather 在 model forward 之外，CUDA graph 不受影响
- **选择**：压缩是 step 之间的元数据 + 显存搬运操作（引擎/调度层），发生在 `prepare_decode` 之前；不进 `attention.forward`，不碰 graph 捕获。
- **为什么**：CUDA graph 只认输入 buffer 形状；压缩改的是 buffer 里的值（更短的 block_table、更小的 context_len），copy 进固定 buffer 的机制不变。故 v0 **无需 eager 旁路**。

### D7. 块对齐的 StreamingLLM keep-set
- **选择**：保留 = 开头 `kv_sink_blocks` 整块 + 最近 `kv_recent_blocks` 整块；`streaming_keep_indices` 块对齐返回下标。触发：物理占用块数 > sink+recent 时压一次（v0 简单触发，周期/阈值留 v2）。
- **为什么**：块对齐让 keep-set 确定、易对拍；sink 用整块（256 token，论文说 4 个即够，整块更省特例）。

## Risks / Trade-offs

- **[gather slot 越界/对齐错]** → 严格 256 对齐；先写 CPU 索引参考实现对拍，再上 Triton；slot/block 索引加断言。
- **[hash 注销遗漏 → 脏数据]** → 改写块统一走注销路径；断言被注销块不在他序列 block_table 中；只压 ref_count==1。
- **[num_kv 解耦漏改某处]** → 全仓搜 `len(seq)`/`num_tokens` 在 decode 路径的使用点逐一核对（may_append/can_append/slot/context_len）。
- **[质量下降]** → 默认关闭；PPL + 任务级一致率对照；sink+recent 兜底。
- **[有损但任务依赖中段信息]** → 记录为已知局限；v0 只验证 PPL 与机制正确性，不承诺任务精度。

## Migration Plan

1. 先合阶段1度量（preemption/avg batch/decode tok_s）拿 baseline（独立 PR）。
2. 合 v0：config + num_kv + evict + gather + prepare_decode，默认关闭。
3. 开关开启在小规模验证 gather 正确性（CPU 对拍）+ PPL 没崩 + block 确实释放。
4. 通过后进 v1（仅换 keep_indices 算法）。
- **回滚**：`enable_kv_compression=False` 即回原行为。

## Open Questions

- v0 触发就用"超 sink+recent 即压"是否够？还是要加最小间隔避免抖动？倾向先简单，看 PPL/吞吐再说。
- `kv_sink_blocks`/`kv_recent_blocks` 默认值（1 / 3?）等度量数据出来后定。
- gather 首版 PyTorch 索引够快吗，还是 v0 就要 Triton？倾向先 PyTorch 对拍，慢再换。
