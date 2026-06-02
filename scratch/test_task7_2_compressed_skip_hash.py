"""task 7.2 单测：已压缩序列跳过 hash_blocks 注册（KV 压缩 v0 第 7 步 / 设计 D5）。

约定：
- Sequence 新增标志位 kv_compressed（默认 False，evict 末尾置 True）；
- BlockManager.hash_blocks 开头若 seq.kv_compressed 则直接 return，不写 hash_to_block_id；
- 压缩关闭/未压缩时行为不变（零回归）。

纯 CPU（BlockManager 纯 Python 记账），零依赖：
    python scratch/test_task7_2_compressed_skip_hash.py
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


def _setup_two_full_blocks(mgr: BlockManager, seq: Sequence):
    """让 seq 有 2 个写满的块待注册（num_cached=0, scheduled=8, bs=4）。"""
    seq.block_table = [0, 1]
    for b in seq.block_table:
        _occupy(mgr, b)
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 8  # 跨满块 0、1


def test_default_flag_false():
    seq = _make_seq(8)
    assert seq.kv_compressed is False


def test_compressed_seq_skips_hash_blocks():
    mgr = BlockManager(num_blocks=8, block_size=BS)
    seq = _make_seq(8)
    _setup_two_full_blocks(mgr, seq)
    seq.kv_compressed = True  # 标记已压缩

    mgr.hash_blocks(seq)
    # 不应写入任何 hash
    assert mgr.hash_to_block_id == {}
    assert mgr.blocks[0].hash == -1
    assert mgr.blocks[1].hash == -1


def test_uncompressed_seq_registers_zero_regression():
    mgr = BlockManager(num_blocks=8, block_size=BS)
    seq = _make_seq(8)
    _setup_two_full_blocks(mgr, seq)
    assert seq.kv_compressed is False

    mgr.hash_blocks(seq)
    # 两个满块都登记
    assert len(mgr.hash_to_block_id) == 2
    assert mgr.blocks[0].hash != -1 and mgr.blocks[1].hash != -1
    assert mgr.hash_to_block_id[mgr.blocks[0].hash] == 0
    assert mgr.hash_to_block_id[mgr.blocks[1].hash] == 1


def test_evict_sets_flag_then_hash_blocks_noop():
    # evict 末尾置 kv_compressed=True；之后 hash_blocks 应 no-op
    mgr = BlockManager(num_blocks=16, block_size=BS)
    seq = _make_seq(20)
    seq.block_table = [10, 11, 12, 13, 14]
    _occupy(mgr, 10, ref_count=1)
    for b in [11, 12, 13, 14]:
        _occupy(mgr, b, ref_count=1)

    assert seq.kv_compressed is False
    mgr.evict(seq, keep_indices=[0, 1, 2, 3] + list(range(12, 19)))
    assert seq.kv_compressed is True

    # 压缩后即便人为摆出可注册的满块，hash_blocks 也不再写入
    before = dict(mgr.hash_to_block_id)
    seq.num_cached_tokens = 0
    seq.num_scheduled_tokens = 8
    mgr.hash_blocks(seq)
    assert mgr.hash_to_block_id == before  # 无新增


if __name__ == "__main__":
    tests = [
        test_default_flag_false,
        test_compressed_seq_skips_hash_blocks,
        test_uncompressed_seq_registers_zero_regression,
        test_evict_sets_flag_then_hash_blocks_noop,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
