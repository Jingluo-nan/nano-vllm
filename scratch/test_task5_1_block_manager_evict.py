"""task 5.1 单测：BlockManager.evict 回收尾部整块 + free_block_ids 增量。

约定（block_manager.py）：
- 紧凑块数 new_num_blocks = ceil(num_keep / block_size)；
- 释放 block_table[new_num_blocks:]（ref_count-=1，归零才进 free_block_ids）；
- del block_table[new_num_blocks:]；
- num_dropped_kv = num_tokens - num_keep - 1（pending last_token 时序约定）；
- assert 0 < new_num_blocks <= len(block_table)。
纯 CPU（BlockManager 是纯 Python 记账），零依赖：
    python scratch/test_task5_1_block_manager_evict.py
"""
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BS = 4  # 测试用小 block_size


def _make_seq(num_tokens: int) -> Sequence:
    seq = Sequence(list(range(num_tokens)), SamplingParams(temperature=0.6))
    return seq


def _occupy(mgr: BlockManager, block_id: int, ref_count: int = 1):
    """把一个物理块标记为占用（从 free 移到 used，设 ref_count）。"""
    mgr.free_block_ids.remove(block_id)
    mgr.used_block_ids.add(block_id)
    mgr.blocks[block_id].ref_count = ref_count


def test_basic_tail_reclaim_and_free_increment():
    # 5 块全 ref_count=1，num_tokens=20，keep 前 8 个 KV → new_num_blocks=2，还 3 块
    mgr = BlockManager(num_blocks=10, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [0, 1, 2, 3, 4]
    for b in seq.block_table:
        _occupy(mgr, b)

    free_before = len(mgr.free_block_ids)
    tail = seq.block_table[2:]  # [2,3,4] 预期被还
    mgr.evict(seq, keep_indices=list(range(8)))

    assert seq.block_table == [0, 1], seq.block_table
    # free_block_ids 增量 == 被还块数
    assert len(mgr.free_block_ids) - free_before == 3
    # 被还的正是原尾部 3 块
    for b in tail:
        assert b in mgr.free_block_ids, b
        assert b not in mgr.used_block_ids, b
        assert mgr.blocks[b].ref_count == 0
    # 保留块不动
    assert {0, 1} <= mgr.used_block_ids
    # num_dropped_kv = num_tokens - num_keep - 1 = 20 - 8 - 1
    #  -1 就是给 pending last_token 留的位置——它没被丢弃,只是此刻还没落进 cache,下一步就会补上,所以不能算进 num_dropped_kv
    assert seq.num_dropped_kv == 20 - 8 - 1, seq.num_dropped_kv


def test_half_full_last_block_block_count_ceil():
    # num_keep=9，block_size=4 → ceil(9/4)=3，保留 3 块（不是 2）
    mgr = BlockManager(num_blocks=10, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [0, 1, 2, 3, 4]
    for b in seq.block_table:
        _occupy(mgr, b)

    mgr.evict(seq, keep_indices=list(range(9)))
    assert seq.block_table == [0, 1, 2], seq.block_table
    assert {3, 4} <= set(mgr.free_block_ids)
    assert seq.num_dropped_kv == 20 - 9 - 1


def test_shared_block_not_reclaimed():
    # 尾块被别的序列共享 (ref_count=2) → 只 -1，不归还
    mgr = BlockManager(num_blocks=10, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [0, 1, 2, 3, 4]
    for b in [0, 1, 2, 3]:
        _occupy(mgr, b, ref_count=1)
    _occupy(mgr, 4, ref_count=2)  # 块4被共享

    free_before = len(mgr.free_block_ids)
    mgr.evict(seq, keep_indices=list(range(8)))  # 还块 2,3,4
    # 块2,3 归还，块4 因 ref_count 2→1 不归还
    assert mgr.blocks[4].ref_count == 1
    assert 4 not in mgr.free_block_ids
    assert 4 in mgr.used_block_ids
    assert len(mgr.free_block_ids) - free_before == 2  # 只还了 2,3
    assert {2, 3} <= set(mgr.free_block_ids)


def test_no_reclaim_when_keep_fills_all_blocks():
    # new_num_blocks == len(block_table) → 不删任何块，free 不变
    mgr = BlockManager(num_blocks=10, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [0, 1, 2, 3, 4]
    for b in seq.block_table:
        _occupy(mgr, b)

    free_before = len(mgr.free_block_ids)
    mgr.evict(seq, keep_indices=list(range(17)))  # ceil(17/4)=5 == len
    assert seq.block_table == [0, 1, 2, 3, 4]
    assert len(mgr.free_block_ids) == free_before
    assert seq.num_dropped_kv == 20 - 17 - 1


def test_num_dropped_kv_formula_various():
    for num_tokens, num_keep in [(20, 4), (20, 12), (33, 5), (100, 9)]:
        mgr = BlockManager(num_blocks=64, block_size=BS)
        seq = _make_seq(num_tokens)
        nblocks = (num_tokens + BS - 1) // BS
        seq.block_table = list(range(nblocks))
        for b in seq.block_table:
            _occupy(mgr, b)
        mgr.evict(seq, keep_indices=list(range(num_keep)))
        assert seq.num_dropped_kv == num_tokens - num_keep - 1, (num_tokens, num_keep)


if __name__ == "__main__":
    tests = [
        test_basic_tail_reclaim_and_free_increment,
        test_half_full_last_block_block_count_ceil,
        test_shared_block_not_reclaimed,
        test_no_reclaim_when_keep_fills_all_blocks,
        test_num_dropped_kv_formula_various,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
