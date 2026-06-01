"""task 5.3 单测：gather 后走 flash-attn 的 decode 注意力 vs CPU 手算参考对拍。

约定 / 设计：
- 给定 q 与完整 K/V（视为已烘焙 RoPE 的值），写入 paged kv_cache；
- 对两组 keep_indices（①块对齐 ②零散）：
  · 参考值 = 只在 K[keep]/V[keep] 上手算的注意力（fp32）；
  · 被测值 = compact_kv 物理 gather → 紧凑 block_table + cache_seqlens=len(keep)
            走 flash_attn_with_kvcache；
- torch.allclose 比对（fp16 flash vs fp32 参考，用 fp16 量级容差）。

注意：flash 分页路径要求 page block_size==256（CLAUDE.md），故此处 BS=256，
num_kv 跨多块以体现丢中段。compact_kv 只 copy 不重旋转，故 RoPE 不进入对比。
需 CUDA + flash-attn：
    python scratch/test_task5_3_gather_attention.py
"""
import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.engine.model_runner import ModelRunner

BS = 256                 # flash 分页路径要求 page block_size==256
NUM_LAYERS = 1
NUM_HEADS = 4
NUM_KV_HEADS = 4         # MHA（GQA 不在本测试范围）
HEAD_DIM = 128
SCALE = HEAD_DIM ** -0.5
NUM_BLOCKS = 6           # 物理块池
DTYPE = torch.float16
torch.manual_seed(0)


class _Dummy:
    """compact_kv 只需 block_size / kv_cache 两个属性。"""


def _build(num_kv, block_table):
    """构造 q、完整 K/V，并按 block_table 写入 paged kv_cache。返回 (q, K, V, kv_cache)。"""
    q = torch.randn(NUM_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    K = torch.randn(num_kv, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    V = torch.randn(num_kv, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

    kv_cache = torch.zeros(2, NUM_LAYERS, NUM_BLOCKS, BS, NUM_KV_HEADS, HEAD_DIM,
                           dtype=DTYPE, device="cuda")
    bt = torch.tensor(block_table, device="cuda")
    pos = torch.arange(num_kv, device="cuda")
    phys = bt[pos // BS]          # 逻辑块 → 物理块
    off = pos % BS # 在块中的偏移量
    kv_cache[0, 0, phys, off] = K
    kv_cache[1, 0, phys, off] = V
    return q, K, V, kv_cache


def _reference(q, K, V, keep_indices):
    """只在保留 KV 上手算注意力（fp32）。返回 [NUM_HEADS, HEAD_DIM]。"""
    idx = torch.tensor(keep_indices, device="cuda")
    kk = K.index_select(0, idx).float()    # [n, heads, d]
    vv = V.index_select(0, idx).float()
    qf = q.float()                         # [heads, d]
    # 每个 head 独立：scores[h, n] = q[h]·kk[n,h] * scale
    scores = torch.einsum("hd,nhd->hn", qf, kk) * SCALE
    attn = torch.softmax(scores, dim=-1)   # [heads, n]
    out = torch.einsum("hn,nhd->hd", attn, vv)  # [heads, d]
    return out


def _tested(q, kv_cache, block_table, keep_indices):
    """compact_kv gather → flash_attn_with_kvcache。返回 [NUM_HEADS, HEAD_DIM]。"""
    d = _Dummy()
    d.block_size = BS
    d.kv_cache = kv_cache
    ModelRunner.compact_kv(d, block_table, keep_indices)
    torch.cuda.synchronize()

    num_keep = len(keep_indices)
    new_num_blocks = (num_keep + BS - 1) // BS
    compacted_bt = block_table[:new_num_blocks]

    k_cache = kv_cache[0, 0]   # [num_blocks, BS, num_kv_heads, head_dim]
    v_cache = kv_cache[1, 0]
    q_in = q.unsqueeze(0).unsqueeze(0)   # [batch=1, seqlen=1, heads, d]
    bt = torch.tensor([compacted_bt], dtype=torch.int32, device="cuda")
    cache_seqlens = torch.tensor([num_keep], dtype=torch.int32, device="cuda")
    o = flash_attn_with_kvcache(q_in, k_cache, v_cache,
                                cache_seqlens=cache_seqlens, block_table=bt,
                                softmax_scale=SCALE, causal=True)
    return o.squeeze(0).squeeze(0)   # [heads, d]


def _run_case(num_kv, block_table, keep_indices, label):
    q, K, V, kv_cache = _build(num_kv, block_table)
    ref = _reference(q, K, V, keep_indices)
    out = _tested(q, kv_cache, block_table, keep_indices).float()
    max_abs = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, rtol=1e-2, atol=2e-2), \
        f"{label}: max_abs_diff={max_abs:.4e}"
    print(f"  {label}: max_abs_diff={max_abs:.4e}")


def test_block_aligned():
    # 3 个逻辑块（256,256,88），非恒等物理映射；保留块0 + 末块，丢中段块1
    num_kv = 600
    block_table = [4, 1, 3, 0, 2, 5][:3]   # 逻辑块0→4, 1→1, 2→3
    keep_indices = list(range(0, BS)) + list(range(2 * BS, num_kv))
    _run_case(num_kv, block_table, keep_indices, "block_aligned")


def test_scattered():
    # 零散下标（跨块、不块对齐），压通用 token 级 gather 路径
    num_kv = 600
    block_table = [4, 1, 3, 0, 2, 5][:3]
    keep_indices = sorted(set(
        list(range(0, 50)) + list(range(120, 130)) + [300, 301, 415, 500, 599]
    ))
    _run_case(num_kv, block_table, keep_indices, "scattered")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA"
    tests = [test_block_aligned, test_scattered]
    for t in tests:
        print(f"running {t.__name__}")
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
