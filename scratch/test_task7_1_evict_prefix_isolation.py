"""task 7.1 单测：evict 的前缀缓存隔离（KV 压缩 v0 第 7 步 / 设计 D5）。

约定（block_manager.evict）：
- gather 改写的前部块（keep_indices 非 identity 段所在块）从 hash_to_block_id 注销、置 hash=-1；
- sink 段（keep_indices[j]==j 的前缀）所在块未被搬动 → hash 保留，可继续供他序列复用；
- 断言被改写块 ref_count==1，被共享则 raise（该序列不应压缩）；
- 改写发生在物理层（compact_kv），本测试只验 hash 记账与断言，不触 CUDA。

纯 CPU（BlockManager 纯 Python 记账），零依赖：
    python scratch/test_task7_1_evict_prefix_isolation.py
"""
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BS = 4  # 测试用小 block_size


def _make_seq(num_tokens: int) -> Sequence:
    return Sequence(list(range(num_tokens)), SamplingParams(temperature=0.6))


def _occupy(mgr: BlockManager, block_id: int, ref_count: int = 1):
    mgr.free_block_ids.remove(block_id)
    mgr.used_block_ids.add(block_id)
    mgr.blocks[block_id].ref_count = ref_count


def _register_hash(mgr: BlockManager, block_id: int, h: int):
    """模拟 hash_blocks 注册：给块设 hash 并登记到 hash_to_block_id。"""
    mgr.blocks[block_id].hash = h
    mgr.hash_to_block_id[h] = block_id


def test_rewritten_blocks_unregistered_sink_kept():
    # num_tokens=20，物理 KV=19，5 块 [10,11,12,13,14]（bs=4）。
    # keep = sink 块0(0..3) + recent(12..18) → num_keep=11, new_num_blocks=ceil(11/4)=3。
    # identity 前缀=4(=keep[0..3]) → first_rewritten=4//4=1 → 改写块=压缩后下标1,2=物理11,12。
    # sink 块（物理10）未搬，hash 应保留。
    mgr = BlockManager(num_blocks=20, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [10, 11, 12, 13, 14]
    _occupy(mgr, 10, ref_count=2)  # sink 块被共享（前缀复用）——不该被断言/注销
    for b in [11, 12, 13, 14]:
        _occupy(mgr, b, ref_count=1)
    # 五个 kept/tail 块都登记了 hash
    for b, h in [(10, 100), (11, 111), (12, 122), (13, 133), (14, 144)]:
        _register_hash(mgr, b, h)

    keep_indices = [0, 1, 2, 3] + list(range(12, 19))  # 4 + 7 = 11
    mgr.evict(seq, keep_indices)

    # 改写块 11,12 不再是 hash_to_block_id 的 value（核心验证）
    assert 11 not in mgr.hash_to_block_id.values()
    assert 12 not in mgr.hash_to_block_id.values()
    assert 111 not in mgr.hash_to_block_id
    assert 122 not in mgr.hash_to_block_id
    assert mgr.blocks[11].hash == -1
    assert mgr.blocks[12].hash == -1
    assert mgr.blocks[11].token_ids == []
    # sink 块 10 未改写：hash 保留、仍登记
    assert mgr.blocks[10].hash == 100
    assert mgr.hash_to_block_id.get(100) == 10
    # 尾块 13,14 被释放（现有逻辑保留其 hash 供复用，不在本任务范围）
    assert seq.block_table == [10, 11, 12]
    assert 13 in mgr.free_block_ids and 14 in mgr.free_block_ids
    # num_dropped_kv = 20 - 11 - 1
    assert seq.num_dropped_kv == 20 - 11 - 1


def test_assert_when_rewritten_block_shared():
    # 改写块（物理11）ref_count=2 → evict 应断言失败
    mgr = BlockManager(num_blocks=20, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [10, 11, 12, 13, 14]
    _occupy(mgr, 10, ref_count=1)
    _occupy(mgr, 11, ref_count=2)  # 被改写块却被共享 → 非法
    for b in [12, 13, 14]:
        _occupy(mgr, b, ref_count=1)

    keep_indices = [0, 1, 2, 3] + list(range(12, 19))
    try:
        mgr.evict(seq, keep_indices)
    except AssertionError:
        pass
    else:
        raise AssertionError("被改写块 ref_count>1 时 evict 应断言失败")
    # 断言在任何注销之前触发：block_table 未被截断
    assert seq.block_table == [10, 11, 12, 13, 14]


def test_no_rewritten_block_when_all_identity():
    # keep = 纯前缀 [0..10]（全 identity，无 token 被前移）→ 无改写块，无注销、无断言。
    # （非 streaming 形态，仅验证边界：identity 段恰好填满 kept 块时不误伤。）
    mgr = BlockManager(num_blocks=20, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [10, 11, 12, 13, 14]
    for b in seq.block_table:
        _occupy(mgr, b, ref_count=2)  # 全共享：若误判有改写块会触发断言
    for b, h in [(10, 100), (11, 111), (12, 122)]:
        _register_hash(mgr, b, h)

    keep_indices = list(range(11))  # ceil(11/4)=3 块全 identity
    mgr.evict(seq, keep_indices)  # 不应抛断言

    # 全部 hash 保留（无块被改写）
    assert mgr.hash_to_block_id.get(100) == 10
    assert mgr.hash_to_block_id.get(111) == 11
    assert mgr.hash_to_block_id.get(122) == 12
    assert seq.block_table == [10, 11, 12]


def test_zero_regression_unrelated_hashes_intact():
    # 其他序列登记的 hash（指向不被本次压缩触碰的块）不受影响
    mgr = BlockManager(num_blocks=20, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [10, 11, 12, 13, 14]
    _occupy(mgr, 10, ref_count=1)
    for b in [11, 12, 13, 14]:
        _occupy(mgr, b, ref_count=1)
    _occupy(mgr, 7, ref_count=1)  # 别的序列的块
    _register_hash(mgr, 7, 777)
    _register_hash(mgr, 11, 111)

    mgr.evict(seq, [0, 1, 2, 3] + list(range(12, 19)))
    assert mgr.hash_to_block_id.get(777) == 7  # 无关 hash 原样
    assert 111 not in mgr.hash_to_block_id     # 改写块 hash 被注销


if __name__ == "__main__":
    tests = [
        test_rewritten_blocks_unregistered_sink_kept,
        test_assert_when_rewritten_block_shared,
        test_no_rewritten_block_when_all_identity,
        test_zero_regression_unrelated_hashes_intact,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
