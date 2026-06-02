"""task 3.1 单测：Sequence.num_kv 属性（KV 压缩 v0 第 3 步）。

约定（sequence.py）：
- num_kv = num_tokens - num_dropped_kv（cache 内实际保留的 KV 数）；
- decode（append_token）后 num_tokens+1，num_dropped_kv 不变 → num_kv 自动 +1；
- 压缩关闭时 num_dropped_kv==0 → num_kv 恒等于 num_tokens（零回归）。
纯 CPU 零依赖：
    python scratch/test_task3_1_num_kv.py
"""
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _make_seq(num_tokens: int) -> Sequence:
    return Sequence(list(range(num_tokens)), SamplingParams(temperature=0.6))


def test_num_kv_closed_equals_num_tokens():
    # 压缩关闭（默认 num_dropped_kv==0）：num_kv 恒等于 num_tokens
    for n in range(1, 50):
        seq = _make_seq(n)
        assert seq.num_dropped_kv == 0
        assert seq.num_kv == seq.num_tokens == n, (seq.num_kv, n)


def test_num_kv_subtracts_dropped():
    seq = _make_seq(20)
    seq.num_dropped_kv = 7
    assert seq.num_kv == 20 - 7 == 13


def test_num_kv_auto_increments_on_decode():
    # decode：append_token 使 num_tokens+1，num_dropped_kv 不变 → num_kv +1
    seq = _make_seq(10)
    seq.num_dropped_kv = 4          # num_kv = 6
    assert seq.num_kv == 6
    seq.append_token(999)           # decode 一步
    assert seq.num_tokens == 11
    assert seq.num_dropped_kv == 4  # 不变
    assert seq.num_kv == 7          # 自动 +1
    seq.append_token(998)
    assert seq.num_kv == 8


def test_num_kv_closed_auto_increments():
    # 关闭压缩时 decode 后 num_kv 仍恒等 num_tokens
    seq = _make_seq(5)
    for _ in range(5):
        before = seq.num_kv
        seq.append_token(0)
        assert seq.num_kv == before + 1
        assert seq.num_kv == seq.num_tokens


if __name__ == "__main__":
    tests = [
        test_num_kv_closed_equals_num_tokens,
        test_num_kv_subtracts_dropped,
        test_num_kv_auto_increments_on_decode,
        test_num_kv_closed_auto_increments,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
