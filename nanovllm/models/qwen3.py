import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # qkv_proj 的输出维度已经是本卡分片了
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 对接 paged KV cache（prefill / decode / 带 prefix 的 prefill 三路径）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
    #shape 主线（简表）：[N, H] → [N, (q+2kv)/tp] → q/k/v 三路 [N, *, D] → RoPE/QK-Norm 不变形 → [N, A/tp, D] → [N, A·D/tp] → [N, H]
    def forward(
        self,
        positions: torch.Tensor, # RoPE位置索引
        hidden_states: torch.Tensor, # [N,H],来自 input_layernorm的归一化输出
    ) -> torch.Tensor:
        # [N, q_size + 2·kv_size] (本卡分片)，分片操作隐藏在下层算子
        qkv = self.qkv_proj(hidden_states) 
        # 得到 [N, q_size]、[N, kv_size]、[N, kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 得到 q: [N, A/tp, D]，k: [N, K/tp, D]，v: [N, K/tp, D]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 内部根据 Context.is_prefill 与 block_tables 选择 flash-attn 三路径之一，
        # 同时把新 k/v 写入 paged KV cache；输出 [N, A/tp, D]
        q, k = self.rotary_emb(positions, q, k)
        # o 的shape [total_tokens_q, num_heads, head_dim]
        o = self.attn(q, k, v)
        # o.flatten(1, -1) → [N, A·D/tp]，
        # 再 self.o_proj(...) 做行并行 GEMM + all-reduce，最终输出 [N, H]
        # output的输出是多头合并后的
        '''
        完整 attention 输出投影应该是：
        output = concat(attn_0, attn_1, ..., attn_{H-1}) @ W_o^T
        其中 concat(...) ∈ [N, H*d] 是所有头拼起来,W_o ∈ [hidden, H*d]。
        如果把 W_o 按列（输入维 = head 维）切：
        W_o = [ W_o^{(0)} | W_o^{(1)} ]            # tp_size=2 时
                └ 头 0..3 ┘  └ 头 4..7 ┘
        本卡形状: [hidden, 4*d]
        那么：
        output = (attn_0..3) @ W_o^{(0)}^T  +  (attn_4..7) @ W_o^{(1)}^T
         └─────────┬─────────┘       └─────────┬─────────┘
              rank 0 部分和                  rank 1 部分和
                  y_0                          y_1
        注意每个部分和 y_i 都是 [N, hidden] 形状——完整 hidden 维，但只是一个加项。最终：
        output = y_0 + y_1   ←  必须求和才正确

        attention 输出（沿头维分片）              本卡 W_o 分片
            [N, 4 heads]                          [hidden, 4 heads]
                │                                     │
                └───────────  本卡 GEMM  ─────────────┘
                                │
                        [N, hidden]  ← 这是部分和，不是答案
                                │
                ┌───── dist.all_reduce(sum) ─────┐
                │                                │
            rank 0 拿到                      rank 1 拿到
            完整 [N, hidden]                  完整 [N, hidden]

        '''
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int, # ：MLP 内部升维后的宽度，Qwen3 中 I > H（典型 2–4 倍）
        hidden_act: str,
    ) -> None:
        super().__init__()
        # MergedColumnParallelLinear(H, [I, I], bias=False) 把 gate_proj 与 up_proj 两个 H → I 的列并行线性层融合为一次 GEMM，
        # 输出维度 2I（按 TP 列方向均分到各卡）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # 输入分片，输出 all-reduce 到 [N, hidden]
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # 一个 fused kernel，同时完成 silu(gate) 与 gate * up 两个操作，并把维度从 2I 收缩到 I
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 输入形状 [N, H]，来自 post_attention_layernorm 的归一化结果

        #一次列并行 GEMM 同时算出 gate 与 up，本卡输出 [N, 2I / tp]，
        # 前 I/tp 列为 gate 分片、后 I/tp 列为 up 分片
        gate_up = self.gate_up_proj(x)

        # SiluAndMul 内部把 gate_up 沿最后一维切两半，
        # out = silu(gate) * up 一次性 fused 计算，输出 [N, I / tp]
        x = self.act_fn(gate_up)
        # 行并行 GEMM 在每张卡上算出本卡 I/tp 切片对 H 维的贡献，
        # 每张卡上的形状都是[N, H],末尾跨 TP all-reduce 求和，得到完整 [N, H]
        '''
        y = SiLU(gate) * up  @  W_down^T              (W_down ∈ [H, I])
        把中间激活 h = SiLU(gate) * up ∈ [N, I] 沿 I 维切两份（tp_size=2 时）：


        h         = [ h_0 | h_1 ]           h_i ∈ [N, I/2]
        W_down    = [ W_d^{(0)} | W_d^{(1)} ]   W_d^{(i)} ∈ [H, I/2]
        所以：


        y = h @ W_down^T
        = h_0 @ W_d^{(0)}^T  +  h_1 @ W_d^{(1)}^T
            └────────┬────────┘   └────────┬────────┘
            rank 0 的部分和          rank 1 的部分和
                    y_0                     y_1

        最终 y = y_0 + y_1       ←  必须求和
        注意每个 y_i 都是 [N, H]——形状完整，但只是一个加项。这一步在 linear.py:153-155 由 dist.all_reduce(y) 完成。
        '''
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            #  把当前输入同时当作新 residual 缓存，并对原值做归一化
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # residual += hidden_states
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        #(hidden_states, residual)，两者均为 [N, H]
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终RSMNorm，带残差
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    #输出形状[num_tokens, hidden_size]
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # embed_tokens 走 TP
        hidden_states = self.embed_tokens(input_ids)
        #首层无残差
        residual = None
        for layer in self.layers:
            # 每层同时返回新的 hidden_states和 residual
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        #最终RSMNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # 两个字模块 Qwen3, lm_head
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    #输出 shape [num_tokens, hidden_size]
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    #输出 shape [num_tokens, vocab_size](实际为 TP 后 all-gather 的全词表)
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
