# `nanovllm/layers/` 逐文件分析

逐文件追加，每轮一个文件。顺序：attention → linear → embed_head → sampler → rotary_embedding → layernorm → activation。

---

## 1. `attention.py`

### 用途
- `layers/` 中最核心的算子，承担 **KV cache 读写** 与 **三路 flash-attn 分发**。
- 是「调度器侧状态」与「flash-attn 内核」之间的唯一桥梁——通过读 `Context` 全局拿到 prefill/decode 标志与所有索引张量。

### 核心成员
- `store_kvcache_kernel`（Triton JIT kernel）
  - 把当前步新算出的 K/V 按 `slot_mapping` 写入分页 KV cache。
  - `slot == -1` 时直接 return（用于 CUDA graph 的 padding 槽位）。
- `store_kvcache(key, value, k_cache, v_cache, slot_mapping)`（Python wrapper）
  - 做 stride 断言（要求 K/V 末维连续、`stride(1) == head_dim`），按 `N` 个 token 启动 kernel。
- `Attention(nn.Module)`
  - `__init__`：保存 `num_heads / head_dim / scale / num_kv_heads`；`k_cache` / `v_cache` 初始化为空 tensor，**由 `ModelRunner.allocate_kv_cache` 在外部 patch 进来**。
  - `forward(q, k, v)`：核心分发逻辑（见下）。

### 输入 / 输出（`forward`）
- 输入：
  - `q`：`[N_q_tokens, num_heads, head_dim]`
  - `k`、`v`：`[N_kv_tokens, num_kv_heads, head_dim]`
  - 其余索引（`slot_mapping`、`cu_seqlens_*`、`block_tables`、`context_lens`、`is_prefill`）**全部** 从 `get_context()` 取，不走参数。
- 输出：
  - `o`：与 `q` 同形，交给上游 `o_proj` 投影。

### 关键 shape / 约定
- KV cache 物理布局：每层 `[2, num_blocks, block_size, num_kv_heads, head_dim]`，被 split 成 `k_cache` / `v_cache`，stride(1) = `block_size * num_kv_heads * head_dim = D`（断言验证）。
- `slot_mapping`：长度 = 本步新增 token 数；元素是「block_id * block_size + offset」的全局槽位。
- `block_tables`：`[batch, max_blocks_per_seq]`，仅 decode 与「带前缀缓存的 prefill」用到。

### 三路分发
- **decode** (`is_prefill=False`)：`flash_attn_with_kvcache`，K/V 从 cache 取；`q` 需 `unsqueeze(1)`。
- **prefill + 前缀缓存** (`is_prefill=True && block_tables is not None`)：把传入的 `k, v` 直接替换成 `k_cache, v_cache`，调 `flash_attn_varlen_func` 并传 `block_table`。
- **纯 prefill**：直接对刚算出的 `k, v` 做 `flash_attn_varlen_func`，不读 cache。

### 与其它文件的关系
- **依赖**：
  - `utils/context.py`：`get_context()` 提供所有 per-step 索引。
  - `engine/model_runner.py`：在 `allocate_kv_cache` 中把每层 `Attention.k_cache / v_cache` 指向共享大 tensor；在 `prepare_prefill / prepare_decode` 里填好 `Context`。
- **被调用方**：
  - `models/qwen3.py` 的 `Qwen3Attention`：QKV 投影后调用本模块，再接 `o_proj`。
- **同目录关系**：
  - 上游：`linear.py`（QKV/输出投影）、`rotary_embedding.py`（在进入 `Attention` 前给 q/k 加 RoPE）。
  - 下游：无（结果直接回到 `qwen3.py` 的 decoder block）。

### 备注 / 易踩点
- `k_cache` 初始为空 tensor，`if k_cache.numel()`：用于跳过 **warmup forward pass**（彼时 KV cache 还没分配）。
- `causal=True` 在两条 prefill 路径都打开——带前缀缓存时 flash-attn 会按 `cu_seqlens_q` / `cu_seqlens_k` 的差自动定位「新 token 的因果窗口」。
- 整个 `forward` 没有任何 Python 级 batch 循环，全部交给 flash-attn 的 varlen / paged kernel。

---

## 2. `linear.py`

### 用途
- 实现 **张量并行（TP）线性层族**，是 Transformer 里所有 `Linear` 的替代。
- 同时承担 **权重加载分片** 职责：每个类暴露 `weight_loader`，由 `utils/loader.py` 在读 safetensors 时反射调用。

### 核心类（继承关系）
- `LinearBase`（抽象基类）
  - 持 `weight: [output_size, input_size]`，可选 `bias`。
  - 给 `weight.weight_loader = self.weight_loader`：让外部 loader 无须知道层类型即可调用正确的分片逻辑。
- `ReplicatedLinear`：不切分，全 rank 持完整权重；用于不参与 TP 的小层。
- `ColumnParallelLinear`：**沿 output 维（dim=0）切分**，每 rank 只持 `output_size / tp_size` 行；前向不通信。
- `MergedColumnParallelLinear(ColumnParallelLinear)`：把多个 `ColumnParallel` 段（如 `gate_proj` + `up_proj`）**融合到一个权重张量** 里；`weight_loader` 多一个 `loaded_shard_id: int` 参数指明写入哪一段。
- `QKVParallelLinear(ColumnParallelLinear)`：同上但专门做 Q/K/V 融合，支持 GQA（`num_kv_heads < num_heads`）；`weight_loader` 的 `loaded_shard_id` 取值为 `"q"/"k"/"v"`。
- `RowParallelLinear`：**沿 input 维（dim=1）切分**，前向后 `all_reduce(y)` 合并结果；bias 只在 `tp_rank==0` 加，避免重复累加。

### 输入 / 输出（`forward`）
- 通用输入：`x: [..., input_size_per_rank or full]`
- 输出：
  - Column 类：`[..., output_size / tp_size]`（**不通信**，由下游 RowParallel 合并）。
  - Row 类：`[..., output_size]`（内部 `all_reduce`）。
  - Replicated：`[..., output_size]`。

### 关键 shape / 约定
- `tp_dim` 指示该层 **权重的分片维度**（不是激活的维度）：Column = 0，Row = 1，Replicated = `None`。
- `QKVParallelLinear` 的融合输出布局：`[q | k | v]`，长度 = `(num_heads + 2 * num_kv_heads) * head_size`，再除以 `tp_size`。
- `MergedColumnParallelLinear.output_sizes`：未切分前各段的输出维度列表；`shard_offset / shard_size` 用 `// tp_size` 算分片后位置。
- `RowParallelLinear.weight_loader` 对 1D 张量（bias）走 `copy_` 全拷；权重才按 `tp_dim=1` 切。

### 与其它文件的关系
- **上游调用方**：
  - `models/qwen3.py`：QKV 用 `QKVParallelLinear`，gate+up 用 `MergedColumnParallelLinear`，`o_proj` / `down_proj` 用 `RowParallelLinear`。
  - `embed_head.py`：`ParallelLMHead` 在结构上是「按词表维切的列并行」，但走自己的 `weight_loader`，不复用本文件类。
- **权重加载**：
  - `utils/loader.py` 遍历 safetensors，对 `packed_modules_mapping`（在 `qwen3.py` 里定义）命中的 key 用 `weight_loader(param, loaded_weight, shard_id)` 三参数调用；其余用两参数调用。
- **通信依赖**：
  - `torch.distributed`：构造时拿 `tp_rank / tp_size`；`RowParallelLinear.forward` 在 `tp_size > 1` 时 `all_reduce`。

### 备注 / 易踩点
- **Column → Row 是 TP 的标准配对**：Column 的不通信输出正好是 Row 的分片输入，整段 attention/MLP 只在 Row 出口做一次 `all_reduce`。
- `LinearBase.__init__` 里 `self.weight = nn.Parameter(torch.empty(...))` 是 **未初始化** 的；权重必须由 `loader.py` 全量写入，否则就是脏内存。
- `MergedColumnParallelLinear` 和 `QKVParallelLinear` 的 `weight_loader` 签名比 `LinearBase` 多一个参数；调用方必须先在 `packed_modules_mapping` 里查到 `shard_id` 再调用。
- `divide` 的断言（整除）隐式约束：`num_heads`、`num_kv_heads`、`intermediate_size` 必须能被 `tp_size` 整除；否则 init 阶段就 fail。

---

## 3. `embed_head.py`

### 用途
- 模型的 **首尾两端**：输入词嵌入 + 输出 LM 头，二者都沿 **词表维（vocab）** 做 TP 切分。
- 注意：未复用 `linear.py` 的类——词表并行的通信模式（masked all-reduce / gather）与普通 Column/Row 并行不同，所以单独写。

### 核心类（继承关系）
- `VocabParallelEmbedding`
  - 把词表沿 dim=0 切给各 rank，每 rank 持 `[num_embeddings/tp_size, embedding_dim]`。
  - `forward`：对落在本 rank 区间外的 token 用 mask 置零，再 `F.embedding`，最后 `all_reduce` 合并各 rank 的嵌入向量。
- `ParallelLMHead(VocabParallelEmbedding)`
  - 复用同一份权重布局（实际项目中 Qwen3-0.6B 走 `tie_word_embeddings`，权重和 embedding 同源——见 `models/qwen3.py`）。
  - `forward`：先做 **last-token 抽取**（仅 prefill 步），再 `F.linear` 得到 logits，最后 `gather` 到 rank 0 拼回完整词表。

### 输入 / 输出
- `VocabParallelEmbedding.forward(x)`
  - 输入：`x: [N_tokens]`（int64 token ids）
  - 输出：`y: [N_tokens, embedding_dim]`（hidden states）
- `ParallelLMHead.forward(x)`
  - 输入：`x: [N_tokens, hidden_size]`（最后一层 RMSNorm 后的 hidden）
  - 输出（仅 rank 0 有值）：
    - prefill：`logits: [batch, vocab_size]`（每条序列只取最后一个 token）
    - decode：`logits: [batch, vocab_size]`

### 关键 shape / 约定
- `num_embeddings % tp_size == 0`（断言）。
- `vocab_start_idx / vocab_end_idx`：本 rank 负责的 token id 区间，越界 token 在 forward 里被 mask 成 0。
- prefill 的 last-token 抽取用 `cu_seqlens_q[1:] - 1`：拿 varlen batch 里每条 sequence 的末位置，避免对不参与采样的中间 token 浪费 LM 头计算。
- decode 步 `x` 本身就是「每序列一个 token」，不做抽取。

### 通信模式
- `VocabParallelEmbedding`：**masked + all-reduce**——每 rank 算自己词表段的嵌入（其余位置为 0），all-reduce 求和等价于完整 lookup。
- `ParallelLMHead`：**gather → cat**——`dist.gather` 到 rank 0，再沿最后一维拼接成完整 `[N, vocab_size]`；非 rank 0 返回 `None`。**只有 rank 0 拿到 logits**，与下游 `sampler.py` 只在 rank 0 跑相一致。

### 与其它文件的关系
- **依赖**：
  - `utils/context.py`：`ParallelLMHead` 通过 `get_context()` 拿 `is_prefill` 与 `cu_seqlens_q`。
  - `torch.distributed`：`all_reduce`（embedding）、`gather`（LM head）。
- **上游调用方**：
  - `models/qwen3.py`：`Qwen3Model` 头部用 `VocabParallelEmbedding`，`Qwen3ForCausalLM` 尾部用 `ParallelLMHead`；若 `tie_word_embeddings=True`，LM head 的权重指向 embedding 的权重。
- **下游消费者**：
  - `layers/sampler.py`：接 `ParallelLMHead` 的 `[batch, vocab]` logits。
- **加载器**：
  - `utils/loader.py`：通过 `weight.weight_loader`（同 `linear.py` 的约定）写入分片，`weight_loader` 只取自己那段。

### 备注 / 易踩点
- `ParallelLMHead.forward` 在非 rank 0 返回 `None`——`ModelRunner` 调用方必须只在 rank 0 处理后续采样逻辑。
- `assert not bias`：LM head 不允许 bias，简化 gather 路径。
- `tie_word_embeddings` 不在本文件处理，是在 `models/qwen3.py` 里把 `lm_head.weight = embed_tokens.weight` 完成的；loader 也据此跳过对 `lm_head.weight` 的二次加载。
- prefill 的 last-token 抽取是性能优化：避免对 prefill 期间所有中间 token 算 vocab 投影（vocab 通常很大）。

---

## 4. `sampler.py`

### 用途
- 模型 forward 之后的 **采样头**：把 `[batch, vocab]` logits 转成 `[batch]` 下一个 token id。
- 实现极简：唯一一个 `Sampler` 类、唯一一个 `forward`、共 ~10 行；用 **Gumbel-max trick** 替代显式 `multinomial`。

### 核心类 / 函数
- `Sampler(nn.Module)`
  - `forward(logits, temperatures)`：被 `@torch.compile` 装饰；无可训练参数，本质是一个函数封装在 module 里。

### 输入 / 输出
- 输入：
  - `logits: [batch, vocab_size]`（float，来自 `ParallelLMHead`，**仅 rank 0 持有**）
  - `temperatures: [batch]`（float，由 `ModelRunner` 从 `SamplingParams.temperature` 收集）
- 输出：
  - `sample_tokens: [batch]`（int64 token ids）

### 关键步骤（4 行）
1. `logits.float().div_(temperatures.unsqueeze(1))`：升精度 + 温度缩放（in-place）。
2. `softmax(dim=-1)`：得到概率分布。
3. `torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)`：采一份 Exp(1) 噪声并 clamp 防 0。
4. `probs.div_(noise).argmax(-1)`：Gumbel-max 等价采样——`argmax(p / E)` 与 `argmax(log p + Gumbel)` 同分布。

### 关键 shape / 约定
- 输入必须是二维 `[batch, vocab]`；不接受 prefill 中间步的 `[N_tokens, vocab]`（那个抽取已经在 `ParallelLMHead` 做完）。
- `temperatures` 必须 `> 1e-10`：在 `SamplingParams.__post_init__` 中断言，**不支持贪心**（temperature=0），因为 Gumbel-max 在 `T→0` 时除零。

### 与其它文件的关系
- **上游**：
  - `layers/embed_head.py::ParallelLMHead.forward` → 提供 logits。
  - `engine/model_runner.py`：在 `run_model` / `run` 末尾构造 `temperatures` tensor 并调用 `Sampler`。
- **下游**：
  - `engine/sequence.py` / `engine/scheduler.py`：采样结果通过 `Scheduler.postprocess` 追加到各 `Sequence.token_ids`。
- **配置约束**：
  - `sampling_params.SamplingParams`：`temperature > 1e-10` 的断言保护了本文件。

### 备注 / 易踩点
- **为什么不用 `torch.multinomial`**：Gumbel-max 只需一次 `exponential_` + `argmax`，对 `torch.compile` 更友好且无 host-device 同步。
- `@torch.compile` 在小 batch decode 步会被频繁触发，但 shape 稳定（batch 维由 CUDA-graph bucket 决定），compile cache 命中率高。
- 整段在 GPU 上运行；由于只 rank 0 进入这里（见 `embed_head.py` 的 gather 设计），TP 不会重复采样、也不需要广播采样结果给其它 rank（worker 只负责模型前向，下一步的 token 由 rank 0 通过 `Sequence` pickling 在调度时同步）。
- `clamp_min_(1e-10)`：防止 `exponential_` 的 0 样本导致 `div_` 出 inf。

---

## 5. `rotary_embedding.py`

### 用途
- **旋转位置编码（RoPE）**：在 `Attention` 之前对 Q/K 注入位置信息。
- 是无参数算子（仅一份 cos/sin lookup buffer），属于「辅助工具」类。

### 核心成员
- `apply_rotary_emb(x, cos, sin)`（纯函数）
  - 把 `x` 沿末维切两半 `x1, x2`，做经典的 `(x1·cos - x2·sin, x2·cos + x1·sin)` 旋转，再拼回。
  - 注意采用的是 **「前后两半」分组**（half-half），不是「相邻两两」分组。
- `RotaryEmbedding(nn.Module)`
  - `__init__`：预计算 `inv_freq`、外积 `t × inv_freq` 得到 `freqs`，再拼 `[cos, sin]` 缓存为 `cos_sin_cache: [max_pos, 1, rotary_dim]`，注册为 **non-persistent buffer**（不会进 state_dict）。
  - `forward(positions, query, key)`：`@torch.compile`，按 `positions` 索引 cache，对 Q/K 同时应用旋转。
- `get_rope(head_size, rotary_dim, max_position, base)`（`@lru_cache(1)`）
  - 进程级单例工厂；模型里每层都拿到同一份 RoPE 实例，避免重复构建 cache。

### 输入 / 输出（`forward`）
- 输入：
  - `positions: [N_tokens]`（int64，来自 `Context.positions`）
  - `query: [N_tokens, num_heads, head_dim]`
  - `key:   [N_tokens, num_kv_heads, head_dim]`
- 输出：
  - `(query', key')`，同 shape，已注入位置。

### 关键 shape / 约定
- `cos_sin_cache: [max_position_embeddings, 1, rotary_dim]`（`unsqueeze_(1)` 是为了能广播到 num_heads 维）。
- `rotary_dim == head_size` 由 `__init__` 中 `assert` 强制——意味着 **整个 head 都做 RoPE**，没有 partial-rotary 分支。
- `inv_freq` 长度 = `rotary_dim / 2`，`freqs[i,j] = positions[i] * inv_freq[j]`，最终拼成 `[..., rotary_dim]`（cos / sin 各占一半）。

### 与其它文件的关系
- **依赖**：仅 `torch`，无项目内依赖。
- **调用方**：
  - `models/qwen3.py::Qwen3Attention`：通过 `get_rope(...)` 拿到单例，在 `forward` 里 `q, k = rotary_emb(positions, q, k)`，紧接着送入 `Attention`。
- **数据来源**：
  - `positions` 由 `engine/model_runner.py::prepare_prefill/prepare_decode` 构造，经 `Context` 透传到 `Qwen3Model.forward`，再向下传给每层 attention。
- **同目录关系**：
  - 直接在 `attention.py` 之前调用——`attention.py` 不感知 RoPE。

### 备注 / 易踩点
- `cos_sin_cache` 用 `register_buffer(..., persistent=False)`：safetensors 里不会有这个键，loader 不会尝试加载/匹配。
- `@torch.compile` + cache 索引：`positions` 长度随 batch 变化但 dtype/shape 稳定，compile 友好。
- 「前后两半」旋转方案与某些 HF 实现一致（包括 Llama/Qwen），不同于「相邻两两」方案；若移植权重需要注意 layout 是否匹配。
- `lru_cache(1)`：参数完全相同才命中——多模型共用同一进程时若 `head_size/base` 不同，需要扩 cache 大小或换 key。

---

## 6. `layernorm.py`

### 用途
- **RMSNorm**：Qwen3/Llama 系标配的归一化层；替代 LayerNorm，无 bias、不减均值。
- 提供「带残差融合」分支：把 `x = norm(x + residual)` 合成一个算子，省一次 read/write。

### 核心类 / 函数
- `RMSNorm(nn.Module)`
  - 唯一参数：`weight: [hidden_size]`（初始化为全 1，由 loader 写入）。
  - `rms_forward(x)`：纯 RMSNorm；升 fp32 算 var，再降回 orig dtype 乘 `weight`。
  - `add_rms_forward(x, residual)`：先 `x = x + residual`，更新 `residual = x`（**新的残差出口**），再做 RMSNorm。
  - `forward(x, residual=None)`：根据 `residual` 是否为 `None` 分发到上面两个。两个内部函数都挂 `@torch.compile`。

### 输入 / 输出
- `forward(x)`：
  - 输入 `x: [N_tokens, hidden_size]`
  - 输出 `[N_tokens, hidden_size]`
- `forward(x, residual)`：
  - 输入 `x, residual: [N_tokens, hidden_size]`
  - 输出 `(out, new_residual)`：`new_residual = x + residual`（**未归一化**），`out = RMSNorm(new_residual)`

### 关键 shape / 约定
- `var = x.pow(2).mean(dim=-1, keepdim=True)`：在 hidden 维上求均方；输出乘 `rsqrt(var + eps)`。
- 全程在 fp32 里算 reduction，再降回原 dtype 乘 `weight`——避免 fp16/bf16 上方差下溢。
- `add_rms_forward` 返回的 `residual` 是 **加和后的值**（不是输入的 `residual`），下一个 block 直接拿它继续加。

### 与其它文件的关系
- **依赖**：仅 `torch`，无项目内依赖。
- **调用方**：
  - `models/qwen3.py`：
    - 每个 `Qwen3DecoderLayer` 用 `add_rms_forward` 形式做 pre-norm + 残差融合（attention 前、MLP 前各一次）。
    - `Qwen3Attention` 内部对 q/k 的 per-head norm（`q_norm` / `k_norm`）用单参数 `forward`。
    - 模型尾部 `final_norm` 也用单参数 `forward`。
- **加载**：
  - `weight` 通过普通 `nn.Parameter` 路径加载——`utils/loader.py` 对未挂 `weight_loader` 的参数走 `default_weight_loader`（直接 `param.data.copy_`）。

### 备注 / 易踩点
- **「带残差融合」的语义**：调用方约定第一个返回值是 norm 后的 hidden（进下一层算子），第二个返回值是新的残差（下一次再加）；qwen3 的 decoder block 严格按这个 pattern 写。
- 第一层 attention 前并没有可用的「前一层残差」——`Qwen3Model.forward` 显式构造 `residual = None`，第一层 `RMSNorm` 走 `rms_forward`（无融合）分支，从第二次起才用 `add_rms_forward`。
- `add_rms_forward` 里 `residual = x.to(orig_dtype)` 这一步是 **在乘 weight 之前** 落盘的——所以下一次拿到的残差不包含 `weight` 乘法的副本，保持原始 hidden 信号。
- `@torch.compile` 两个分支各编一份；shape 稳定（N_tokens 由 batch 决定，hidden 固定），cache 命中良好。

---

## 7. `activation.py`

### 用途
- **SwiGLU 激活的「合并版」**：和 `MergedColumnParallelLinear`（gate+up 融合权重）配套使用。
- 文件总共 4 行实质代码，最简单的辅助算子。

### 核心类 / 函数
- `SiluAndMul(nn.Module)`
  - 无参数。
  - `forward(x)`：把输入沿末维切两半 `x, y`，返回 `silu(x) * y`。
  - `@torch.compile` 装饰。

### 输入 / 输出
- 输入：`x: [N_tokens, 2 * intermediate_size_per_rank]`
  - 由 `gate_up_proj`（`MergedColumnParallelLinear`）输出，前一半是 gate，后一半是 up。
- 输出：`[N_tokens, intermediate_size_per_rank]`
  - 紧接着送入 `down_proj`（`RowParallelLinear`）做 reduce。

### 关键 shape / 约定
- 切分依据是「融合 linear 的输出 layout = `[gate | up]`」——必须与 `MergedColumnParallelLinear` 在 `weight_loader` 里使用的顺序（`shard_id=0` 写 gate，`shard_id=1` 写 up）一致。
- 仅末维变化（变为 1/2），其它维度保持。

### 与其它文件的关系
- **依赖**：仅 `torch.nn.functional.silu`。
- **上游**：
  - `layers/linear.py::MergedColumnParallelLinear`（`gate_up_proj`）→ 输出 `[..., 2*I]`。
- **下游**：
  - `layers/linear.py::RowParallelLinear`（`down_proj`）→ 把激活后的 `[..., I]` 投影回 `hidden_size` 并 all-reduce。
- **调用方**：
  - `models/qwen3.py::Qwen3MLP`：标准 `gate_up_proj → SiluAndMul → down_proj` 三段。

### 备注 / 易踩点
- 「为什么 gate 在前 / up 在前」是 layout 约定问题——必须与 `qwen3.py::packed_modules_mapping` 中 `gate_proj`、`up_proj` 列表顺序一致；颠倒就会算错。
- 没有别的激活实现（GeLU / ReLU / …）——目前仓库只支持 Qwen3 系列，全用 SwiGLU。如果加新模型且激活不同，需要在这里新增类。
- `@torch.compile` 的存在让 `silu(x) * y` 融合成一个 kernel，避免显存来回搬。

---

## 全部 7 个文件分析完成

已覆盖：`attention.py` / `linear.py` / `embed_head.py` / `sampler.py` / `rotary_embedding.py` / `layernorm.py` / `activation.py`。

如需进一步「逐文件交叉对照」「与 `engine/` 或 `models/qwen3.py` 的串接」或「画数据流图」可继续指示。
