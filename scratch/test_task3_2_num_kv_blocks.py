"""task 3.2 单测：Sequence.num_kv_blocks / last_kv_block_num_tokens（KV 压缩 v0 第 3 步）。

约定（sequence.py）：
- num_kv_blocks = ceil(num_kv / block_size)；
- last_kv_block_num_tokens = num_kv - (num_kv_blocks - 1) * block_size（末块实际 token 数，1..block_size）；
- 压缩关闭时 num_dropped_kv==0 → 两者分别等于 num_blocks / last_block_num_tokens（零回归）。
重点覆盖跨块边界：整除（末块满）、半满末块、单块。

Sequence.block_size 是类属性(默认 256)，测小块边界须同步对齐为 BS。
纯 CPU 零依赖：
    python scratch/test_task3_2_num_kv_blocks.py
"""
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BS = 4
Sequence.block_size = BS  # 对齐类属性，使 num_kv_blocks/last_kv_block_num_tokens 按 BS 算


def _make_seq(num_tokens: int, num_dropped_kv: int = 0) -> Sequence:
    seq = Sequence(list(range(num_tokens)), SamplingParams(temperature=0.6))
    seq.num_dropped_kv = num_dropped_kv
    return seq


def test_exact_block_boundary_last_block_full():
    # num_kv 是 BS 整数倍 → 末块满，last_kv_block_num_tokens == BS
    for num_kv in (4, 8, 12, 16):
        seq = _make_seq(num_kv)  # num_dropped_kv=0 → num_kv == num_tokens
        assert seq.num_kv == num_kv
        assert seq.num_kv_blocks == num_kv // BS, (num_kv, seq.num_kv_blocks)
        assert seq.last_kv_block_num_tokens == BS, (num_kv, seq.last_kv_block_num_tokens)


def test_half_full_last_block():
    # num_kv 非整数倍 → 末块半满，余数 = num_kv % BS
    cases = {1: (1, 1), 5: (2, 1), 7: (2, 3), 9: (3, 1), 10: (3, 2), 11: (3, 3)}
    for num_kv, (exp_blocks, exp_last) in cases.items():
        seq = _make_seq(num_kv)
        assert seq.num_kv_blocks == exp_blocks, (num_kv, seq.num_kv_blocks)
        assert seq.last_kv_block_num_tokens == exp_last, (num_kv, seq.last_kv_block_num_tokens)


def test_single_block():
    seq = _make_seq(3)
    assert seq.num_kv_blocks == 1
    assert seq.last_kv_block_num_tokens == 3


def test_based_on_num_kv_not_num_tokens():
    # 压缩后：num_tokens=20, dropped=12 → num_kv=8 → 2 块、末块满 4
    seq = _make_seq(20, num_dropped_kv=12)
    assert seq.num_kv == 8
    assert seq.num_kv_blocks == 2          # 基于 num_kv(8)，不是 num_tokens(20→5 块)
    assert seq.last_kv_block_num_tokens == 4
    # 对照：基于 num_tokens 的旧属性仍是 5 块
    assert seq.num_blocks == 5

    # num_kv=9 → 3 块、末块 1
    seq2 = _make_seq(20, num_dropped_kv=11)
    assert seq2.num_kv == 9
    assert seq2.num_kv_blocks == 3
    assert seq2.last_kv_block_num_tokens == 1


def test_zero_regression_equals_num_blocks():
    # num_dropped_kv==0 时，num_kv_blocks==num_blocks 且 last_kv_block_num_tokens==last_block_num_tokens
    for num_tokens in range(1, 40):
        seq = _make_seq(num_tokens)
        assert seq.num_kv_blocks == seq.num_blocks, num_tokens
        assert seq.last_kv_block_num_tokens == seq.last_block_num_tokens, num_tokens


def test_last_block_num_tokens_range_invariant():
    # 不变量：1 <= last_kv_block_num_tokens <= BS
    for num_kv in range(1, 30):
        seq = _make_seq(num_kv)
        assert 1 <= seq.last_kv_block_num_tokens <= BS, (num_kv, seq.last_kv_block_num_tokens)


if __name__ == "__main__":
    tests = [
        test_exact_block_boundary_last_block_full,
        test_half_full_last_block,
        test_single_block,
        test_based_on_num_kv_not_num_tokens,
        test_zero_regression_equals_num_blocks,
        test_last_block_num_tokens_range_invariant,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
