"""task 4.2 单测：kv_compression.should_compress 触发谓词。

约定（kv_compression.py）：
- 物理占用块数 num_kv_blocks > sink_blocks + recent_blocks 时才触发压缩；
- 短序列（块数 <= sink+recent）无可回收整块，返回 False。
零依赖：python scratch/test_task4_2_should_compress.py
"""
from nanovllm.engine.kv_compression import should_compress


def test_below_threshold_no_compress():
    # 块数不足，没有中段可丢
    assert should_compress(num_kv_blocks=3, sink_blocks=1, recent_blocks=2) is False


def test_equal_threshold_no_compress():
    # 边界：恰好等于 sink+recent 不触发（无可回收整块）
    assert should_compress(num_kv_blocks=4, sink_blocks=1, recent_blocks=3) is False


def test_just_above_threshold_compress():
    # 边界：刚超过一块即触发
    assert should_compress(num_kv_blocks=5, sink_blocks=1, recent_blocks=3) is True


def test_well_above_threshold_compress():
    assert should_compress(num_kv_blocks=100, sink_blocks=1, recent_blocks=3) is True


def test_zero_blocks_no_compress():
    assert should_compress(num_kv_blocks=0, sink_blocks=1, recent_blocks=1) is False


if __name__ == "__main__":
    tests = [
        test_below_threshold_no_compress,
        test_equal_threshold_no_compress,
        test_just_above_threshold_compress,
        test_well_above_threshold_compress,
        test_zero_blocks_no_compress,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
