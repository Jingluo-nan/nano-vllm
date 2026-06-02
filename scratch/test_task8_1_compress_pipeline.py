"""task 8.1 单测：decode 压缩管线的门控 + 触发决策 + 块账（KV 压缩 v0 第 8 步）。

端到端（开启压缩跑通长生成）须 GPU + flash-attn，本地无法实跑；这里覆盖可纯 CPU
验证的三段（compact_kv 的物理 gather 已在 task5.2/5.3 GPU 对拍过）：
- can_evict：gather 前门控，被改写块须 ref_count==1（共享 sink 不影响）；
- 触发决策：should_compress + streaming_keep_indices 在何时压、保留集是真子集；
- 管线块账：can_evict→（跳过 GPU 的 compact_kv）→evict 后 block_table 收缩、
  num_dropped_kv/kv_compressed 正确、尾块释放。

纯 CPU（BlockManager 纯 Python 记账）：
    python scratch/test_task8_1_compress_pipeline.py
"""
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_compression import streaming_keep_indices, should_compress
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BS = 4
SINK, RECENT = 1, 3  # 与默认含义一致：1 块 sink + 3 块 recent（此处用小 bs）


def _make_seq(num_tokens: int) -> Sequence:
    return Sequence(list(range(num_tokens)), SamplingParams(temperature=0.6))


def _occupy(mgr: BlockManager, block_id: int, ref_count: int = 1):
    mgr.free_block_ids.remove(block_id)
    mgr.used_block_ids.add(block_id)
    mgr.blocks[block_id].ref_count = ref_count


# —— can_evict 门控 ——
def test_can_evict_all_owned_true():
    mgr = BlockManager(num_blocks=32, block_size=BS)
    seq = _make_seq(24)
    seq.block_table = [20, 21, 22, 23, 24, 25]
    for b in seq.block_table:
        _occupy(mgr, b, 1)
    keep = streaming_keep_indices(23, SINK, RECENT, BS)  # physical_kv=23
    assert mgr.can_evict(seq, keep) is True


def test_can_evict_shared_rewritten_false():
    mgr = BlockManager(num_blocks=32, block_size=BS)
    seq = _make_seq(24)
    seq.block_table = [20, 21, 22, 23, 24, 25]
    for b in seq.block_table:
        _occupy(mgr, b, 1)
    mgr.blocks[21].ref_count = 2  # 被改写块（compacted idx 1）被共享 → 门控拦下
    keep = streaming_keep_indices(23, SINK, RECENT, BS)
    assert mgr.can_evict(seq, keep) is False


def test_can_evict_shared_sink_ok():
    mgr = BlockManager(num_blocks=32, block_size=BS)
    seq = _make_seq(24)
    seq.block_table = [20, 21, 22, 23, 24, 25]
    for b in seq.block_table:
        _occupy(mgr, b, 1)
    mgr.blocks[20].ref_count = 2  # sink 块（compacted idx 0）共享是合法的
    keep = streaming_keep_indices(23, SINK, RECENT, BS)
    assert mgr.can_evict(seq, keep) is True


# —— 触发决策：should_compress + streaming 是真子集 ——
def test_trigger_decision():
    # 块数 ≤ sink+recent(=4) 不压；> 4 才压
    for physical_kv in range(1, 40):
        num_blocks = (physical_kv + BS - 1) // BS
        keep = streaming_keep_indices(physical_kv, SINK, RECENT, BS)
        if should_compress(num_blocks, SINK, RECENT):
            assert len(keep) < physical_kv, physical_kv         # 确有可丢
            assert keep == sorted(set(keep))                    # 有序去重
            assert keep[0] == 0 and keep[-1] == physical_kv - 1 # 含 sink 头与最新
        else:
            assert keep == list(range(physical_kv))             # 不压则全保留


# —— 管线块账：evict 后状态（compact_kv 的 GPU 搬迁此处跳过）——
def test_pipeline_block_accounting():
    mgr = BlockManager(num_blocks=32, block_size=BS)
    seq = _make_seq(24)                      # num_tokens=24, num_dropped_kv=0 → num_kv=24
    seq.block_table = [20, 21, 22, 23, 24, 25]
    for b in seq.block_table:
        _occupy(mgr, b, 1)
    free_before = len(mgr.free_block_ids)

    physical_kv = seq.num_kv - 1             # =23
    num_blocks = (physical_kv + BS - 1) // BS
    assert should_compress(num_blocks, SINK, RECENT)
    keep = streaming_keep_indices(physical_kv, SINK, RECENT, BS)  # len=15
    assert mgr.can_evict(seq, keep)
    # （真实管线此处先 model_runner.call("compact_kv", seq.block_table, keep)，GPU 搬迁）
    mgr.evict(seq, keep)

    new_num_blocks = (len(keep) + BS - 1) // BS  # ceil(15/4)=4
    assert len(seq.block_table) == new_num_blocks == 4
    assert {24, 25} <= set(mgr.free_block_ids)            # 尾部 2 块释放
    assert len(mgr.free_block_ids) - free_before == 2
    assert seq.num_dropped_kv == 24 - len(keep) - 1       # =8
    assert seq.num_kv == len(keep) + 1                    # 压缩后 =16
    assert seq.kv_compressed is True


def test_short_seq_not_triggered():
    # 块数 ≤ sink+recent 的短序列：不触发（physical_kv 占 4 块=sink+recent）
    mgr = BlockManager(num_blocks=32, block_size=BS)
    seq = _make_seq(17)               # num_kv=17 → physical_kv=16 → 4 块
    physical_kv = seq.num_kv - 1
    num_blocks = (physical_kv + BS - 1) // BS
    assert num_blocks == 4
    assert should_compress(num_blocks, SINK, RECENT) is False


if __name__ == "__main__":
    tests = [
        test_can_evict_all_owned_true,
        test_can_evict_shared_rewritten_false,
        test_can_evict_shared_sink_ok,
        test_trigger_decision,
        test_pipeline_block_accounting,
        test_short_seq_not_triggered,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
