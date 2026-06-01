"""task 4.1 单测：kv_compression.streaming_keep_indices 块对齐保留集 + 边界。

约定（kv_compression.py）：
- 返回当前 cache 布局中要保留的位置下标，取值 [0, num_kv)，有序、去重；
- 块对齐保留「开头 sink_blocks 块 + 最近 recent_blocks 块」；
- 当 sink+recent 已覆盖全部块时返回全部下标（无可丢中段）。
零依赖：python scratch/test_task4_1_streaming_keep_indices.py
"""
from nanovllm.engine.kv_compression import streaming_keep_indices


def _check_invariants(keep, num_kv):
    # [0, num_kv) 范围内
    assert all(0 <= i < num_kv for i in keep), keep
    # 有序
    assert keep == sorted(keep), keep
    # 去重
    assert len(keep) == len(set(keep)), keep


def test_drops_middle_blocks_full_blocks():
    # block_size=4, num_kv=20 → 5 个满块；sink=1, recent=1
    # 保留块0([0,4)) + 末块([16,20))，中段块1/2/3 丢弃
    keep = streaming_keep_indices(num_kv=20, sink_blocks=1, recent_blocks=1, block_size=4)
    assert keep == list(range(0, 4)) + list(range(16, 20)), keep
    _check_invariants(keep, 20)


def test_last_block_half_full():
    # num_kv=18, block_size=4 → 5 块，末块只有 2 个 token(下标16,17)
    # sink=1, recent=1 → [0,4) + [16,18)
    keep = streaming_keep_indices(num_kv=18, sink_blocks=1, recent_blocks=1, block_size=4)
    assert keep == [0, 1, 2, 3, 16, 17], keep
    _check_invariants(keep, 18)


def test_covers_all_blocks_returns_all():
    # 3 块，sink+recent=3 >= num_blocks → 全保留
    keep = streaming_keep_indices(num_kv=12, sink_blocks=2, recent_blocks=1, block_size=4)
    assert keep == list(range(12)), keep
    _check_invariants(keep, 12)


def test_sink_plus_recent_exceeds_blocks_returns_all():
    # sink+recent 超过块数，仍返回全部（不越界、不重叠）
    keep = streaming_keep_indices(num_kv=10, sink_blocks=5, recent_blocks=5, block_size=4)
    assert keep == list(range(10)), keep
    _check_invariants(keep, 10)


def test_just_one_block_droppable():
    # 4 块，sink=1 recent=2 → 只能丢中间 1 块(块1)
    # 保留 块0([0,4)) + 块2,3([8,16))
    keep = streaming_keep_indices(num_kv=16, sink_blocks=1, recent_blocks=2, block_size=4)
    assert keep == list(range(0, 4)) + list(range(8, 16)), keep
    _check_invariants(keep, 16)
    # 边界：恰好等于覆盖时不应触发丢弃（sink+recent==num_blocks-? 这里 num_blocks=4 > 3）
    assert 4 not in keep and 5 not in keep  # 中段块1确实被丢


def test_no_overlap_between_sink_and_recent():
    # 大量块，sink/recent 不重叠且中段连续丢弃
    keep = streaming_keep_indices(num_kv=40, sink_blocks=2, recent_blocks=1, block_size=4)
    # sink: [0,8); recent: 末块 [36,40)
    assert keep == list(range(0, 8)) + list(range(36, 40)), keep
    _check_invariants(keep, 40)


if __name__ == "__main__":
    tests = [
        test_drops_middle_blocks_full_blocks,
        test_last_block_half_full,
        test_covers_all_blocks_returns_all,
        test_sink_plus_recent_exceeds_blocks_returns_all,
        test_just_one_block_droppable,
        test_no_overlap_between_sink_and_recent,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
