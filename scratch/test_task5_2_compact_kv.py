"""task 5.2 单测：ModelRunner.compact_kv 物理 gather 逐位一致。

约定（model_runner.py）：
- token 级通用置换：new 位置 j 放原位置 keep_indices[j] 的 KV；
- slot = block_table[idx//bs]*bs + idx%bs（new 紧凑到 block_table 前部物理块）；
- 所有层 + k/v 两半一并搬；先 index_select().clone() 再 index_copy_（重叠安全）；
- copy 不重旋转（本测试只验证搬运逐位一致，RoPE 正确性见 5.3）。

需 CUDA。用轻量桩 + 哨兵编码反查每个 slot 内容：
    python scratch/test_task5_2_compact_kv.py
"""
import torch

from nanovllm.engine.model_runner import ModelRunner

BS = 4
NUM_BLOCKS = 6
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
K2 = 2
TOTAL_SLOTS = NUM_BLOCKS * BS


class _Dummy:
    """只需 block_size / kv_cache 两个属性即可调 compact_kv。"""


def _val(k: int, layer: int, slot: int) -> int:
    # 唯一编码：含 (k, layer, slot)，反查时据此断言无串层/串半
    return (k * NUM_LAYERS + layer) * TOTAL_SLOTS + slot


def _make_cache() -> torch.Tensor:
    # [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    cache = torch.empty(K2, NUM_LAYERS, NUM_BLOCKS, BS, NUM_KV_HEADS, HEAD_DIM,
                        dtype=torch.float32, device="cuda")
    flat = cache.view(K2, NUM_LAYERS, TOTAL_SLOTS, NUM_KV_HEADS, HEAD_DIM)
    for k in range(K2):
        for layer in range(NUM_LAYERS):
            for slot in range(TOTAL_SLOTS):
                flat[k, layer, slot] = _val(k, layer, slot)
    return cache


def _slots(block_table, indices):
    return [block_table[p // BS] * BS + p % BS for p in indices]


def _run_and_check(block_table, keep_indices, label):
    d = _Dummy()
    d.block_size = BS
    d.kv_cache = _make_cache()

    old_slots = _slots(block_table, keep_indices)
    new_slots = _slots(block_table, range(len(keep_indices)))

    ModelRunner.compact_kv(d, block_table, keep_indices)
    torch.cuda.synchronize()

    flat = d.kv_cache.view(K2, NUM_LAYERS, TOTAL_SLOTS, NUM_KV_HEADS, HEAD_DIM)
    for k in range(K2):
        for layer in range(NUM_LAYERS):
            for j in range(len(keep_indices)):
                got = flat[k, layer, new_slots[j]]
                expected = _val(k, layer, old_slots[j])
                assert torch.all(got == expected), (
                    f"{label}: k={k} layer={layer} j={j} "
                    f"new_slot={new_slots[j]} got={got.flatten()[0].item()} "
                    f"expected(old_slot={old_slots[j]})={expected}"
                )


def test_block_aligned_gather():
    # 非恒等 block_table 制造物理间接；保留块0 + 末块
    block_table = [3, 1, 4, 2, 0]
    keep_indices = list(range(0, 4)) + list(range(16, 20))  # 块0 + 块4
    _run_and_check(block_table, keep_indices, "block_aligned")


def test_scattered_gather():
    # 零散下标，压通用 token 级置换路径（不等 v1）
    block_table = [3, 1, 4, 2, 0]
    keep_indices = [0, 2, 5, 7, 9, 13, 18]
    _run_and_check(block_table, keep_indices, "scattered")


def test_overlap_safe():
    # old/new slot 大量重叠（前部内重排）→ .clone() 保证不自我污染
    block_table = [3, 1, 4, 2, 0]
    # 保留这几个逻辑slot id对应的cache
    keep_indices = [1, 0, 3, 2, 6, 5, 4]  # 前 8 槽内乱序
    _run_and_check(block_table, keep_indices, "overlap")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA"
    tests = [
        test_block_aligned_gather,
        test_scattered_gather,
        test_overlap_safe,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
