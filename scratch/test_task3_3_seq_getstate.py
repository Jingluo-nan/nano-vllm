"""task 3.3 单测：Sequence.__getstate__/__setstate__ 对 num_dropped_kv 的 round-trip。

约定（sequence.py）：
- num_dropped_kv 被编入序列化元组；
- prefill 时 last_state 是完整 token_ids，decode 时只有 last_token；
- round-trip 后 num_dropped_kv / num_tokens 等字段保持一致。
零依赖（不需要 pytest / CUDA）：直接 python scratch/test_task3_3_seq_getstate.py
"""
import pickle

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _roundtrip(seq: Sequence) -> Sequence:
    return pickle.loads(pickle.dumps(seq))


def test_prefill_roundtrip_preserves_dropped_kv():
    seq = Sequence([10, 11, 12, 13], SamplingParams(temperature=0.6))
    seq.is_prefill = True
    seq.num_dropped_kv = 2  # 模拟已发生压缩
    seq.block_table = [5, 6]

    out = _roundtrip(seq)

    assert out.num_dropped_kv == 2, out.num_dropped_kv
    assert out.num_tokens == 4
    assert out.token_ids == [10, 11, 12, 13]  # prefill 带完整 token_ids
    assert out.last_token == 13
    assert out.block_table == [5, 6]
    # num_kv 属性据 num_dropped_kv 推导
    assert out.num_kv == 4 - 2


def test_decode_roundtrip_preserves_dropped_kv():
    seq = Sequence([10, 11, 12, 13], SamplingParams(temperature=0.6))
    seq.append_token(14)
    seq.is_prefill = False
    seq.num_dropped_kv = 3
    seq.block_table = [7]

    out = _roundtrip(seq)

    assert out.num_dropped_kv == 3, out.num_dropped_kv
    assert out.num_tokens == 5
    assert out.token_ids == []          # decode 不带完整 token_ids
    assert out.last_token == 14         # 只带 last_token
    assert out.block_table == [7]
    assert out.num_kv == 5 - 3


def test_default_dropped_kv_is_zero_after_roundtrip():
    # 压缩关闭路径：num_dropped_kv 恒为 0，round-trip 后 num_kv == num_tokens
    seq = Sequence([1, 2, 3], SamplingParams(temperature=0.6))
    seq.is_prefill = True
    out = _roundtrip(seq)
    assert out.num_dropped_kv == 0
    assert out.num_kv == out.num_tokens == 3


if __name__ == "__main__":
    tests = [
        test_prefill_roundtrip_preserves_dropped_kv,
        test_decode_roundtrip_preserves_dropped_kv,
        test_default_dropped_kv_is_zero_after_roundtrip,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
