# `Qwen3ForCausalLM` 分析大纲

## 1. 顶层封装：[Qwen3ForCausalLM](../nanovllm/models/qwen3.py#L186)
- `packed_modules_mapping`：q/k/v → qkv_proj，gate/up → gate_up_proj
- 子模块：`self.model`（Qwen3Model）、`self.lm_head`（ParallelLMHead）
- `tie_word_embeddings` 时 `lm_head.weight` 共享 embedding 权重
- `forward(input_ids, positions)` → hidden_states，shape `[num_tokens, hidden_size]`
- `compute_logits(hidden_states)` → logits，shape `[num_tokens, vocab_size]`(实际为 TP 后 all-gather 的全词表)

### 分析

> 形状约定：nano-vllm 不使用 `[B, T, ...]` 拼接 batch，而是把所有序列的 token 拼成一维 `N = ΣT_i`（其中 i 遍历 batch 内的每条序列，i.e. `N ≈ B·T` 当所有序列等长）。下文 `[N, H]` 即此含义。

- **作用**：因果语言模型的顶层封装，将主干网络与输出投影组合，对外只暴露 `forward`（出 hidden_states）与 `compute_logits`（出 logits）两个入口
- **类属性 `packed_modules_mapping`**：safetensors 权重命名到融合权重 + shard_id 的映射（`q/k/v_proj → qkv_proj`，`gate/up_proj → gate_up_proj`），由 `utils/loader.py` 在权重加载阶段读取
- **`__init__` 子模块**：
  - `self.model`：[Qwen3Model](../nanovllm/models/qwen3.py#L162) 主干
  - `self.lm_head`：`ParallelLMHead(vocab_size, hidden_size)`，词表维度按 TP 切分
- **权重绑定**：`config.tie_word_embeddings=True` 时执行 `self.lm_head.weight.data = self.model.embed_tokens.weight.data`，让输出投影与 embedding 共享同一块显存
- **`forward(input_ids, positions)` 输入**：
  - `input_ids`：`[N]`，所有序列 token 拼接的一维张量
  - `positions`：`[N]`，每个 token 对应的 RoPE 位置索引
- **`forward` 输出**：`hidden_states`，`[N, H]`，直接转发 `self.model(input_ids, positions)` 的返回值
- **forward 不计算 logits**：logits 由独立入口 `compute_logits` 按需调用（ModelRunner 通常只对每个序列最后一个 token 切片后再调用，避免对所有 prompt token 算 logits）
- **`compute_logits(hidden_states)` 输入**：`[N', H]`，`N'` 一般 ≤ N（已被切片）
- **`compute_logits` 输出**：`[N', V]`，由 `ParallelLMHead` 完成本地 matmul + 跨 TP all-gather，得到完整词表 logits
- **shape 变化总览**：
  - `forward`：`[N]` → `[N, H]`
  - `compute_logits`：`[N', H]` → `[N', V]`
- **与引擎层解耦**：本类不感知 paged KV cache、slot_mapping、block_tables，所有运行时调度信息通过全局 `Context` 进入 `Attention`
- **TP 透明性**：所有 TP 切分都下沉到子模块（VocabParallelEmbedding、QKV/Merged/Row/ColumnParallelLinear、ParallelLMHead），本类视角等价于单卡模型

## 2. 主干网络：[Qwen3Model](../nanovllm/models/qwen3.py#L162)
- `embed_tokens`：VocabParallelEmbedding，`[num_tokens] → [num_tokens, hidden_size]`
- `layers`：`num_hidden_layers` × Qwen3DecoderLayer
- `norm`：最终 RMSNorm（带 residual fuse）
- 前向流：embed → 逐层 (hidden_states, residual) 累积 → 最终 norm 融合 residual
- 输出 shape：`[num_tokens, hidden_size]`

### 分析

- **作用**：Qwen3 的解码主干，按 `embedding → L 层 DecoderLayer → 最终 RMSNorm` 顺序串联，输出隐状态（不含 logits）
- **`__init__` 子模块**：
  - `self.embed_tokens`：`VocabParallelEmbedding(V, H)`，词表维度按 TP 切分
  - `self.layers`：`nn.ModuleList`，包含 `L = config.num_hidden_layers` 个 [Qwen3DecoderLayer](../nanovllm/models/qwen3.py#L120)
  - `self.norm`：`RMSNorm(H)`，支持 `(hidden_states, residual)` 双输入的 fused add+norm
- **`forward(input_ids, positions)` 输入**：
  - `input_ids`：`[N]`
  - `positions`：`[N]`
- **`forward` 输出**：`hidden_states`，`[N, H]`（与输入 `N` 一致）
- **前向流程（按顺序）**：
  1. `hidden_states = self.embed_tokens(input_ids)` → `[N, H]`
  2. 初始化 `residual = None`，作为「首层无 residual」的哨兵
  3. 循环 `L` 层：`hidden_states, residual = layer(positions, hidden_states, residual)`，每层同时返回新的 hidden_states 与 residual
  4. 末尾 `hidden_states, _ = self.norm(hidden_states, residual)`，把最后一层的残差融合进归一化
- **residual 双通道**：层间显式传递 `(hidden_states, residual)` 两个张量，由各层内部完成 fused add+RMSNorm；首层 `residual is None` 触发不同分支（在 DecoderLayer 内处理，本章不展开）
- **不在主干内执行的逻辑**：QK-Norm、RoPE、paged attention、SiluAndMul MLP 等全部封装在 DecoderLayer 子模块内，主干仅做层间编排
- **shape 变化总览**：
  - embed：`[N]` → `[N, H]`
  - 每层：`[N, H]` → `[N, H]`（hidden_states 与 residual 维度同步保持 `[N, H]`）
  - 终态 norm：`[N, H]` → `[N, H]`
- **不输出 logits**：本类只产隐状态，logits 由上层 `Qwen3ForCausalLM.compute_logits` 调 `lm_head` 计算
- **TP 视角**：除 `embed_tokens` 走 vocab-TP 之外，主干本身对 TP 透明；切分逻辑全部在 DecoderLayer 内部的列/行并行线性层中完成

## 3. Decoder Block：[Qwen3DecoderLayer](../nanovllm/models/qwen3.py#L120)
- `input_layernorm`：RMSNorm，首层无 residual / 后续层 fused add+norm
- `self_attn`：Qwen3Attention
- `post_attention_layernorm`：RMSNorm，fused add+norm
- `mlp`：Qwen3MLP
- 前向流：(LN→Attn) → (Add+LN→MLP) → 返回 (hidden_states, residual) 给下一层
- 关键 shape：`[num_tokens, hidden_size]` 全程保持

### 分析

> 形状约定沿用第 1 章：`N = ΣT_i` 表示一个 step 内 batch 所有 token 拼接后的总长度，等价于用户符号体系下「打平后的 `B·T`」。下文还会用到 `H = hidden_size`、`I = intermediate_size`，attention 内部的 `A/K/D` 与 MLP 的 `I` 在第 4、5 章展开。

#### 模块组成

- **作用**：Qwen3 主干中重复 `L` 次的基本 block，承担「pre-norm 自注意力 + pre-norm MLP」的完整一轮变换，并通过显式 `(hidden_states, residual)` 双通道把残差留给下一层
- **`self.self_attn`**：[Qwen3Attention](../nanovllm/models/qwen3.py#L14)，多头注意力子模块，承担 QKV 投影 / QK-Norm / RoPE / paged attention / 输出投影
- **`self.mlp`**：[Qwen3MLP](../nanovllm/models/qwen3.py#L91)，SwiGLU 前馈网络（gate_up_proj + SiluAndMul + down_proj）
- **`self.input_layernorm`**：`RMSNorm(H, eps=config.rms_norm_eps)`，注意力前的归一化，**支持 1/2 入参两种调用形式**（首层 vs 后续层）
- **`self.post_attention_layernorm`**：`RMSNorm(H, eps=config.rms_norm_eps)`，MLP 前的 fused add+norm（恒为 2 入参形式）
- **pre-norm 结构**：与 LLaMA / Qwen2 同款，归一化在 sublayer 之前，残差走显式通道而非直接相加在 `hidden_states` 上
- **无 dropout / 无可选 bias**：本类不引入额外可训练参数，所有参数都在四个子模块内

#### forward 流程

- **入参**：
  - `positions`：`[N]`，RoPE 位置索引，仅透传给 `self_attn`，本类不使用
  - `hidden_states`：`[N, H]`，上一层（或 `embed_tokens`）的输出
  - `residual`：`[N, H]` 或 `None`，**`None` 仅出现在第 0 层**（由 `Qwen3Model.forward` 显式置 None）
- **出参**：`(hidden_states, residual)`，两者均为 `[N, H]`，交给下一层继续累积；最后一层的 residual 由 `Qwen3Model` 末尾的 `self.norm` 融合
- **步骤 1 — input_layernorm（attention 前归一化 + residual 切换）**：
  - 若 `residual is None`（首层）：`hidden_states, residual = input_layernorm(hidden_states), hidden_states` —— 把当前输入同时**当作新 residual 缓存**，并对原值做归一化
  - 否则（后续层）：`hidden_states, residual = input_layernorm(hidden_states, residual)` —— RMSNorm 内部先 `residual += hidden_states` 再对新 residual 做归一化（fused add+norm，写回的 `residual` 已包含本步加法结果）
- **步骤 2 — self_attn**：`hidden_states = self_attn(positions, hidden_states)`，对归一化后的张量做注意力，输出 shape 仍为 `[N, H]`（attention 内部细节本章不展开，参见第 4 章）
- **步骤 3 — post_attention_layernorm（MLP 前 fused add+norm）**：`hidden_states, residual = post_attention_layernorm(hidden_states, residual)`，恒为 2 入参形式，把 attention 输出加到 residual 上、再归一化
- **步骤 4 — mlp**：`hidden_states = mlp(hidden_states)`，对归一化后的张量做 SwiGLU 前馈，输出 shape 仍为 `[N, H]`（MLP 内部细节本章不展开，参见第 5 章）
- **返回**：`return hidden_states, residual` —— **注意此处 hidden_states 与 residual 还未相加**，最终一次加法发生在 `Qwen3Model` 末尾 `self.norm(hidden_states, residual)` 中

#### residual connection 位置

- **首层（layer 0）**：input_layernorm 步骤里完成「把原输入复制为 residual」的初始化，本层没有真正的「加」操作；attention 与 MLP 各自的残差通过下一次 fused add+norm 才合并
- **第 1 层起**：每次进入 RMSNorm（2 入参形式）时执行一次 `residual += hidden_states`，因此一层 decoder block 内实际发生 **2 次残差加法**：
  - `input_layernorm` 内：把上一层 MLP 的输出加到 residual
  - `post_attention_layernorm` 内：把本层 attention 的输出加到 residual
- **末层之后**：`Qwen3Model.norm(hidden_states, residual)` 再加一次，把最后一层 MLP 的输出合并入 residual，再做最终 RMSNorm
- **设计动机**：通过 `(hidden_states, residual)` 双通道把「加法」推迟并与下一次归一化融合，省一次显式 add kernel；只在层间传 2 个 `[N, H]` 张量，接口干净

#### shape 流程表

| 步骤 | 操作 | hidden_states | residual |
|---|---|---|---|
| 入口 | — | `[N, H]` | `[N, H]` 或 `None` |
| 1a | input_layernorm（首层，1 入参） | `[N, H]` | `[N, H]`（= 入口 hidden_states） |
| 1b | input_layernorm（后续层，2 入参 fused） | `[N, H]` | `[N, H]`（= 旧 residual + 旧 hidden_states） |
| 2 | self_attn(positions, hidden_states) | `[N, H]` | `[N, H]`（不变） |
| 3 | post_attention_layernorm（2 入参 fused） | `[N, H]` | `[N, H]`（= 旧 residual + attn 输出） |
| 4 | mlp(hidden_states) | `[N, H]` | `[N, H]`（不变） |
| 出口 | return | `[N, H]` | `[N, H]` |

- **全程 hidden 维保持 `H`**：本层内部不改变 token 数 `N` 也不改变 hidden 维 `H`，TP 切分发生在 `self_attn` / `mlp` 内部的列/行并行线性层中，对外形状透明
- **attention / mlp 内部的隐含 shape**（本章只列入口/出口，不展开过程）：
  - attention：`[N, H]` → 内部经 `[N, A·D/tp]` 等中间形态 → `[N, H]`（见第 4 章）
  - mlp：`[N, H]` → 内部经 `[N, 2I/tp]`、`[N, I/tp]` 中间形态 → `[N, H]`（见第 5 章）

#### 本层不展开、后续章节再分析的模块

- **Qwen3Attention（第 4 章）**：QKV 融合列并行投影、`q_norm/k_norm`（QK-Norm）、RoPE、paged attention 三路径（prefill / decode / 带 prefix 的 prefill）、`o_proj` 行并行 + all-reduce
- **Qwen3MLP（第 5 章）**：`gate_up_proj`（MergedColumnParallelLinear）+ `SiluAndMul` + `down_proj`（RowParallelLinear）
- **RoPE**：`rotary_emb` 由 `get_rope` 构造，作用于 q/k 的 head_dim 维（第 4 章一并展开）
- **Paged KV cache**：`Attention.forward` 通过全局 `Context` 读取 `slot_mapping / block_tables / cu_seqlens_*`，本类对此完全无感（第 4 章 + 第 7 章）
- **RMSNorm 双入参 fused 实现**：本章只用到「单入参 / 双入参」两种调用约定，kernel 实现细节不在本系列范围

## 4. 注意力模块：[Qwen3Attention](../nanovllm/models/qwen3.py#L14)
- TP 切分：`num_heads`、`num_kv_heads` 按 `tp_size` 切；`q_size`/`kv_size` 计算
- `qkv_proj`：QKVParallelLinear，融合 Q/K/V 列并行
- 拆分：`split([q_size, kv_size, kv_size], dim=-1)` 然后 `view(-1, h, head_dim)`
- `q_norm`/`k_norm`：head_dim 维度的 RMSNorm（Qwen3 特有的 QK-Norm，仅在无 qkv_bias 时启用）
- `rotary_emb`：RoPE 对 q, k 应用
- `attn`：Attention 层，对接 paged KV cache（prefill / decode / 带 prefix 的 prefill 三路径）
- `o_proj`：RowParallelLinear，输入做 all-reduce
- shape 流：`[N, hidden]` → `[N, (q+2kv)/tp]` → `[N, h, d] × 3` → `[N, h, d]` → `[N, h*d]` → `[N, hidden]`

### 分析（一）：模块组成与 forward 主流程

> 形状约定沿用前文：`N = ΣT_i`、`H = hidden_size`、`A = num_attention_heads`、`K = num_key_value_heads`、`D = head_dim`、`tp = dist.get_world_size()`。GQA 时 `A` 可整除 `K`，但 `K < A`。

#### 模块组成

- **作用**：实现 Qwen3 的多头自注意力子层，对外接收 `[N, H]` 隐状态、返回 `[N, H]` 注意力输出；内部串联 QKV 投影 → QK-Norm（可选）→ RoPE → paged attention → 输出投影
- **TP 头切分（`__init__` 内计算）**：
  - `self.num_heads = total_num_heads // tp`，`self.num_kv_heads = total_num_kv_heads // tp`（两者均要求被 `tp` 整除）
  - `self.head_dim`：优先取 config，否则回退到 `H // total_num_heads`
  - `self.q_size = num_heads · D`，`self.kv_size = num_kv_heads · D`（这两个数用于 forward 内的 `split`）
  - `self.scaling = D ** -0.5`（softmax 缩放因子）
- **`qkv_proj`**：`QKVParallelLinear(H, D, total_num_heads, total_num_kv_heads, bias=qkv_bias)` —— 把 Q/K/V 三个投影融合成一次列并行 GEMM，输出维度 `(q_size + 2·kv_size)`（已是本卡分片）
- **`o_proj`**：`RowParallelLinear(total_num_heads·D, H, bias=False)` —— 输出投影按输入维度切片，forward 末尾隐式 all-reduce 还原 `[N, H]`
- **`rotary_emb`**：`get_rope(D, rotary_dim=D, max_position, base=rope_theta)`，对 q/k 的最后一维做 RoPE；`rope_scaling` 存在时覆盖 `rope_theta`
- **`attn`**：`Attention(num_heads, head_dim, scaling, num_kv_heads)`，统一封装 prefill / decode / 带 prefix 的 prefill 三路径，从全局 `Context` 读取 `slot_mapping / block_tables / cu_seqlens_*`
- **QK-Norm（Qwen3 特有，仅 `qkv_bias=False` 时启用）**：`self.q_norm / self.k_norm = RMSNorm(D, eps=rms_norm_eps)`，对每个 head 的 `D` 维做归一化；若 `qkv_bias=True` 则不构造

#### forward 主流程

- **入参**：
  - `positions`：`[N]`，RoPE 位置索引（由 `ModelRunner.prepare_*` 准备）
  - `hidden_states`：`[N, H]`，来自 `input_layernorm` 的归一化输出
- **出参**：`[N, H]`，已 all-reduce 跨 TP 合并的注意力输出
- **步骤 1 — 融合 QKV 投影**：`qkv = self.qkv_proj(hidden_states)` → `[N, q_size + 2·kv_size]`（本卡分片）
- **步骤 2 — 三段切分**：`q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)`，得到 `[N, q_size]`、`[N, kv_size]`、`[N, kv_size]`
- **步骤 3 — reshape 成多头**：分别 `view(-1, num_heads, D)` / `view(-1, num_kv_heads, D)`，得到 `q: [N, A/tp, D]`，`k: [N, K/tp, D]`，`v: [N, K/tp, D]`
- **步骤 4 — QK-Norm（条件分支）**：若 `qkv_bias=False`，对 q、k 各自调用 `q_norm(q)` / `k_norm(k)`，沿 `D` 维归一化，shape 不变
- **步骤 5 — RoPE**：`q, k = self.rotary_emb(positions, q, k)`，对 q、k 的 `D` 维施加旋转位置编码，shape 不变（v 不动）
- **步骤 6 — paged attention**：`o = self.attn(q, k, v)`，内部根据 `Context.is_prefill` 与 `block_tables` 选择 flash-attn 三路径之一，同时把新 k/v 写入 paged KV cache；输出 `[N, A/tp, D]`
- **步骤 7 — 输出投影 + all-reduce**：`o.flatten(1, -1)` → `[N, A·D/tp]`，再 `self.o_proj(...)` 做行并行 GEMM + all-reduce，最终输出 `[N, H]`
- **shape 主线（简表）**：`[N, H]` → `[N, (q+2kv)/tp]` → `q/k/v 三路 [N, *, D]` → RoPE/QK-Norm 不变形 → `[N, A/tp, D]` → `[N, A·D/tp]` → `[N, H]`

#### shape 流程表

> 符号：`B = batch size`、`T = 当前 step 的 query 长度`（prefill 时 = prompt len，decode 时 = 1）、`S = key/value 总长度`（含 paged KV cache 里的历史 token，`S ≥ T`）、`H = hidden_size`、`A = num_attention_heads`、`K = num_key_value_heads`、`D = head_dim`、`G = A / K`（GQA 组大小，每 G 个 Q head 共享 1 个 KV head）、`tp = TP world size`。下表以「逻辑形状 `[B, T, ...]`」描述；nano-vllm 内部把 batch 维与 token 维打平为 `N = ΣT_i`（详见前文），TP 切分体现在 `/tp` 维度。

| # | 阶段 | 张量 | 单卡 shape | TP 还原后全局 shape |
|---|---|---|---|---|
| 0 | 入口 | `hidden_states` | `[B, T, H]` | `[B, T, H]` |
| 1 | `qkv_proj` 融合列并行 | `qkv` | `[B, T, (A + 2K)·D / tp]` | `[B, T, (A + 2K)·D]` |
| 2a | `split` → Q | `q` | `[B, T, A·D / tp]` | `[B, T, A·D]` |
| 2b | `split` → K | `k` | `[B, T, K·D / tp]` | `[B, T, K·D]` |
| 2c | `split` → V | `v` | `[B, T, K·D / tp]` | `[B, T, K·D]` |
| 3a | `view` 多头 Q | `q` | `[B, T, A / tp, D]` | `[B, T, A, D]` |
| 3b | `view` 多头 K | `k` | `[B, T, K / tp, D]` | `[B, T, K, D]` |
| 3c | `view` 多头 V | `v` | `[B, T, K / tp, D]` | `[B, T, K, D]` |
| 4 | QK-Norm（`D` 维，仅 `qkv_bias=False`） | `q`, `k` | 同 3a / 3b，不变 | 同上 |
| 5 | RoPE（作用于 `D` 维） | `q`, `k` | 同 3a / 3b，不变 | 同上 |
| 6a | 新 K/V 写入 paged cache（`slot_mapping`） | 写入槽 | 来自 `[B, T, K / tp, D]` | — |
| 6b | 从 paged cache 读取 K/V（含历史） | `k_cache`, `v_cache` | `[B, S, K / tp, D]` | `[B, S, K, D]` |
| 7a | GQA broadcast：每 G 个 Q head 共享 1 个 KV head | K/V 视图 | `[B, S, A / tp, D]`（KV 重复 G 次） | `[B, S, A, D]` |
| 7b | attention scores `Q·Kᵀ / √D` | `scores` | `[B, A / tp, T, S]` | `[B, A, T, S]` |
| 7c | softmax + `· V` | `o` | `[B, T, A / tp, D]` | `[B, T, A, D]` |
| 8 | `flatten(1, -1)` 合并头维 | `o` | `[B, T, A·D / tp]` | `[B, T, A·D]` |
| 9 | `o_proj` 行并行 + all-reduce | `output` | `[B, T, H]` | `[B, T, H]` |

- **GQA 关键点**：`A·D` ≠ `K·D`（除非 MHA `K = A`）；KV cache 只存 `K` 个 head，attention kernel 内部按 `G = A / K` 倍 broadcast 到 Q 的头数，省一半以上的 KV 显存
- **`S` 何时大于 `T`**：decode step `T = 1`，`S = past_len + 1`；带 prefix cache 的 prefill 中 `T = 本次新增 token 数`，`S = T + 已缓存 prefix 长度`；纯首次 prefill `S = T`

## 5. MLP 模块：[Qwen3MLP](../nanovllm/models/qwen3.py#L91)
- `gate_up_proj`：MergedColumnParallelLinear，融合 gate+up 列并行，输出 `[N, 2*intermediate/tp]`
- `act_fn`：SiluAndMul，门控 + 乘法，输出 `[N, intermediate/tp]`
- `down_proj`：RowParallelLinear，输入分片，输出 all-reduce 到 `[N, hidden]`
- 断言 `hidden_act == "silu"`

### 分析

> 形状约定沿用前文：`N = ΣT_i` 是 batch 打平后的 token 数（等价于用户符号下 `B·T`）。下表与正文用户符号 `B, T, H, I` 同时给出；`I = intermediate_size`，`tp = TP world size`。MLP 出口张量交由 [Qwen3DecoderLayer](../nanovllm/models/qwen3.py#L120) 的下一次 fused add+norm 完成残差（详见第 3 章）。

#### 模块作用

- **作用**：Qwen3 的逐位置前馈子层，实现 **SwiGLU** 变体：`down_proj( silu(gate_proj(x)) * up_proj(x) )`，对 attention 之后的隐状态做非线性升维-再降维变换
- **位置**：[Qwen3DecoderLayer](../nanovllm/models/qwen3.py#L120) 内 `post_attention_layernorm` 之后调用，输入已是归一化后的 `[N, H]`
- **不改变 token 维**：MLP 仅在 hidden 维上「升维 → 门控激活 → 降维」，token 数 `N` 全程不变
- **与 attention 的分工**：attention 做跨 token 的信息交互，MLP 做单 token 内特征非线性变换

#### 初始化参数与子模块

- **`hidden_size: int`（H）**：输入/输出维度，与主干 hidden 维一致
- **`intermediate_size: int`（I）**：MLP 内部升维后的宽度，Qwen3 中 `I > H`（典型 2–4 倍）
- **`hidden_act: str`**：被 `assert hidden_act == "silu"` 硬约束，本类不支持其他激活
- **`self.gate_up_proj`**：`MergedColumnParallelLinear(H, [I, I], bias=False)` —— **把 `gate_proj` 与 `up_proj` 两个 `H → I` 的列并行线性层融合为一次 GEMM**，输出维度 `2I`（按 TP 列方向均分到各卡）
- **`self.down_proj`**：`RowParallelLinear(I, H, bias=False)` —— `I → H` 的行并行线性层，输入维 `I` 按 TP 切片，输出在 forward 末尾隐式 all-reduce 还原 `H`
- **`self.act_fn`**：`SiluAndMul()` —— 一个 fused kernel，**同时完成 `silu(gate)` 与 `gate * up` 两个操作**，并把维度从 `2I` 收缩到 `I`
- **无可训练偏置**：`gate_up_proj` 与 `down_proj` 均 `bias=False`，整层只有两组权重矩阵

#### forward 输入输出

- **输入**：`x`，形状 `[N, H]`，来自 `post_attention_layernorm` 的归一化结果
- **输出**：`[N, H]`，由 `down_proj` 完成 all-reduce 后返回；交由 [Qwen3DecoderLayer](../nanovllm/models/qwen3.py#L120) 末尾返回给主干（**MLP 自身不做残差加法**，那一步推迟到下一层的 fused add+norm）
- **forward 体共 3 行**：`gate_up_proj` → `act_fn` → `down_proj`，无分支、无条件逻辑

#### 核心计算流程

- **步骤 1 — 融合升维**：`gate_up = self.gate_up_proj(x)` —— 一次列并行 GEMM 同时算出 gate 与 up，本卡输出 `[N, 2I / tp]`，前 `I/tp` 列为 gate 分片、后 `I/tp` 列为 up 分片
- **步骤 2 — gating 激活 + 乘法（融合）**：`x = self.act_fn(gate_up)` —— `SiluAndMul` 内部把 `gate_up` 沿最后一维切两半，**`out = silu(gate) * up` 一次性 fused 计算**，输出 `[N, I / tp]`（这里就是 SwiGLU 的「门控」语义所在）
- **步骤 3 — 降维 + all-reduce**：`x = self.down_proj(x)` —— 行并行 GEMM 在每张卡上算出本卡 `I/tp` 切片对 `H` 维的贡献，末尾跨 TP all-reduce 求和，得到完整 `[N, H]`
- **gate 与 up 的概念角色**：`gate` 经 silu 后充当**逐通道的软门控**（取值范围 ≈ (-0.28·x, ∞)），`up` 是被门控调制的「信息流」；二者按元素相乘
- **fused 实现的两层 fuse**：(a) `gate_proj` + `up_proj` 融合成一次 GEMM 减少 kernel launch；(b) `silu(·)` 与 `·*·` 融合成 `SiluAndMul` 减少一次 `2I` 中间张量的读写
- **TP 切分位置**：`gate_up_proj` 在 `I` 维（输出维）按 `tp` 切；`down_proj` 在 `I` 维（输入维）按 `tp` 切；二者天然配对，**中间张量 `[N, I/tp]` 各卡互不相同**，无需中间通信，只在 `down_proj` 末尾 all-reduce 一次

#### shape 流程表

| # | 阶段 | 张量 | 单卡 shape | TP 还原后全局 shape |
|---|---|---|---|---|
| 0 | 入口 `hidden_states`（来自 post_attention_layernorm） | `x` | `[B, T, H]` | `[B, T, H]` |
| 1 | `gate_up_proj` 融合列并行（一次 GEMM 出 gate+up） | `gate_up` | `[B, T, 2I / tp]` | `[B, T, 2I]` |
| 1a | 概念切片：`gate_proj` 输出（`gate_up[..., :I/tp]`） | `gate` | `[B, T, I / tp]` | `[B, T, I]` |
| 1b | 概念切片：`up_proj` 输出（`gate_up[..., I/tp:]`） | `up` | `[B, T, I / tp]` | `[B, T, I]` |
| 2a | activation：`silu(gate)`（SiluAndMul 内部） | `silu_gate` | `[B, T, I / tp]` | `[B, T, I]` |
| 2b | gating 乘法：`silu(gate) * up`（SiluAndMul 内部） | `gated` | `[B, T, I / tp]` | `[B, T, I]` |
| 3 | `down_proj` 行并行 + all-reduce | `mlp_out` | `[B, T, H]` | `[B, T, H]` |
| 4 | MLP 最终输出，交回 DecoderLayer | `hidden_states` | `[B, T, H]` | `[B, T, H]` |

#### 容易混淆的点

- **`gate_proj`/`up_proj` 在代码里不存在**：只有融合的 `gate_up_proj`；权重加载时通过 [Qwen3ForCausalLM.packed_modules_mapping](../nanovllm/models/qwen3.py#L187) 把 safetensors 里的 `gate_proj/up_proj` 名字重定向到 `gate_up_proj` 的两个 shard（参见第 1 章 / `utils/loader.py`）
- **`SiluAndMul` 一次做两件事**：很多教程把 SwiGLU 写成「先 silu 再乘」两步，本实现是一个 kernel，**不要把 `act_fn` 单独当作激活函数**——它的输入是 `2I`、输出是 `I`
- **gating 乘法的方向**：是 `silu(gate) * up`（**silu 只作用于 gate 通道**），不是 `silu(up) * gate`，也不是 `silu(gate * up)`
- **`down_proj` 后才 all-reduce**：MLP 内部中间张量 `[N, I/tp]` 在各卡上不同，只有最终输出经过 `RowParallelLinear` 的 all-reduce 才在 TP 维度上恒同
- **MLP 不做 residual**：本类 forward 末尾直接 `return x`，残差加法由 [Qwen3DecoderLayer](../nanovllm/models/qwen3.py#L120) 在**下一层**的 `input_layernorm` fused add+norm 中合并（参见第 3 章「residual connection 位置」）
- **`hidden_act` 仅支持 `"silu"`**：`assert hidden_act == "silu"` 写死；切换激活需要替换 `act_fn` 与对应 fused kernel，不能仅改 config

## 6. 输入/输出嵌入：embed_tokens & lm_head
- `VocabParallelEmbedding`：词表按 TP 切，本地 lookup + all-reduce
- `ParallelLMHead`：输出维度（vocab）按 TP 切，本地 matmul 后 gather
- 权重绑定路径（tie_word_embeddings）

## 7. 与引擎层的接口契约
- `forward` 输入由 `ModelRunner.prepare_prefill / prepare_decode` 构造
- `Attention` 不直接接收 cache，依赖全局 `Context`（slot_mapping、block_tables、cu_seqlens、is_prefill）
- `compute_logits` 在 `ModelRunner.run_model` 末尾按需对最后位置切片调用
- `packed_modules_mapping` 与 `utils/loader.py` 协同：safetensors 命名 → 融合权重 + shard_id

## 8. 完整 Shape 流程表（input_ids → logits）

> 符号：`B/T/S/V/H/L/A/K/D/I/G` 见用户约定。表中以教科书式 `[B, T, ...]` 描述；nano-vllm 内部把 batch 与 token 维打平为 `N = ΣT_i`（详见第 1 章），TP 切分体现为 `/tp` 维度。decoder layer 内部步骤标记「每层重复 L 次」，不展开 L 行。

| # | 阶段 | 操作 / 模块 | 输入 shape | 输出 shape | 说明 |
|---|---|---|---|---|---|
| 1 | input_ids | 外部输入 | — | `[B, T]` | nano-vllm 实际为 `[N]`，N=ΣT_i |
| 2 | token embedding | `VocabParallelEmbedding` | `[B, T]` | `[B, T, H]` | 词表按 TP 切，本地 lookup + all-reduce |
| 3 | causal mask | （隐式） | — | — | 由 flash-attn kernel 内部处理，无显式张量 |
| 4 | position_ids / RoPE 表 | `get_rope` + `rotary_emb` | positions `[B, T]` | cos/sin `[max_pos, D]` | RoPE 在每层 attention 内对 q/k 的 `D` 维施加，shape 不变 |
| 5 | decoder layer input | 层入口 | `[B, T, H]` | `[B, T, H]` | 每层重复 L 次；同时持有 residual `[B, T, H]` |
| 6 | input_layernorm | `RMSNorm`（首层 1 入参 / 后续层 fused add+norm） | `[B, T, H]`(+ residual) | `[B, T, H]` | 每层重复 L 次 |
| 7 | q_proj（概念） | `qkv_proj` 融合的 Q 分片 | `[B, T, H]` | `[B, T, A·D / tp]` | 实际为 `QKVParallelLinear` 一次 GEMM |
| 8 | k_proj（概念） | `qkv_proj` 融合的 K 分片 | `[B, T, H]` | `[B, T, K·D / tp]` | GQA：`K ≤ A` |
| 9 | v_proj（概念） | `qkv_proj` 融合的 V 分片 | `[B, T, H]` | `[B, T, K·D / tp]` | 与 K 同维 |
| 10 | reshape to heads | `view(-1, h, D)` | `[B, T, *·D / tp]` | q `[B, T, A/tp, D]` / k,v `[B, T, K/tp, D]` | 每层重复 L 次 |
| 11 | after QK-Norm + RoPE | `q_norm/k_norm` + `rotary_emb` | q,k `[B, T, *, D]` | shape 不变 | 仅 `qkv_bias=False` 时启用 QK-Norm；v 不动 |
| 12 | KV cache 更新 | `store_kvcache`（Triton） | 新 K/V `[B, T, K/tp, D]` | cache `[B, S, K/tp, D]` | `S ≥ T`；prefill 首步 `S = T`，decode `S = past_len + 1` |
| 13 | attention scores | `Q · Kᵀ / √D`（flash-attn 内部） | q `[B, T, A/tp, D]`, k `[B, S, K/tp, D]` | `[B, A/tp, T, S]` | GQA 按 `G = A/K` 倍 broadcast K |
| 14 | attention weights | softmax(scores) | `[B, A/tp, T, S]` | `[B, A/tp, T, S]` | 含 causal mask（kernel 内施加） |
| 15 | attention out（分头） | `weights · V` | weights `[B, A/tp, T, S]`, v `[B, S, K/tp, D]` | `[B, T, A/tp, D]` | flash-attn 直接返回此形 |
| 16 | attention out（合头） | `flatten(1, -1)` | `[B, T, A/tp, D]` | `[B, T, A·D / tp]` | 准备进入 o_proj |
| 17 | o_proj | `RowParallelLinear` + all-reduce | `[B, T, A·D / tp]` | `[B, T, H]` | 行并行末尾跨 TP all-reduce |
| 18 | attention residual output | 层内残差通道 | hidden `[B, T, H]`, residual `[B, T, H]` | 二者均 `[B, T, H]` | nano-vllm：加法延迟到下一次 fused add+norm，本步只透传 |
| 19 | post_attention_layernorm | `RMSNorm`（fused add+norm） | hidden+residual `[B, T, H]` | `[B, T, H]` | 每层重复 L 次 |
| 20 | gate_proj（概念） | `gate_up_proj` 前半 | `[B, T, H]` | `[B, T, I / tp]` | 实际与 up_proj 融合 |
| 21 | up_proj（概念） | `gate_up_proj` 后半 | `[B, T, H]` | `[B, T, I / tp]` | 融合 GEMM 一次出 `[B, T, 2I/tp]` |
| 22 | activation | `silu(gate)`（SiluAndMul 内部） | `[B, T, I / tp]` | `[B, T, I / tp]` | 仅作用于 gate 通道 |
| 23 | gated MLP intermediate | `silu(gate) * up`（SiluAndMul 融合） | `[B, T, 2I / tp]` | `[B, T, I / tp]` | 维度从 `2I` 收缩到 `I` |
| 24 | down_proj | `RowParallelLinear` + all-reduce | `[B, T, I / tp]` | `[B, T, H]` | 每层重复 L 次 |
| 25 | MLP residual out / layer 出口 | 层出口（返回 `(hidden, residual)`） | `[B, T, H]` | `[B, T, H]` | 每层重复 L 次；residual 仍 `[B, T, H]` |
| 26 | final norm | `Qwen3Model.norm`（fused add+norm） | hidden+residual `[B, T, H]` | `[B, T, H]` | 末层 MLP 输出在此并入 residual |
| 27 | lm_head | `ParallelLMHead`（本地 matmul + all-gather） | `[B, T', H]` | `[B, T', V]` | 通常先按序列末位切片，`T' ≤ T` |
| 28 | logits | `compute_logits` 返回 | — | `[B, T', V]` | TP 还原后的完整词表 |

**备注：**
1. 第 1 章已说明 nano-vllm 把 `[B, T]` 打平为 `N`，上表 `[B, T, ...]` 在源码中均为 `[N, ...]`。
2. 第 3 行「causal mask」与第 13/14 行的 scores/weights 在 nano-vllm 中**不显式构造张量**，由 flash-attn / `flash_attn_with_kvcache` 在 kernel 内完成；表中按教学逻辑保留。
3. 第 12 行 `S` 的取值：纯首次 prefill `S = T`；带 prefix cache 的 prefill `S = T + prefix_len`；decode `T = 1`, `S = past_len + 1`。
4. 第 17 / 24 行末尾的 all-reduce 是 TP 唯一的强制通信点；MLP 与 attention 各一次，layer 内共 2 次。
5. 第 27 行 `T'`：`ModelRunner` 仅对每条序列**最后位置**切片后调用 `compute_logits`，因此通常 `T' = B`（一条序列一个位置），不是 `T`。
