"""task 6 单测：prepare_decode / block_manager 改用 num_kv（KV 压缩 v0 第 6 步）。

覆盖三个子任务的纯索引算术（不触 CUDA / flash-attn）：
- 6.1 context_lens ← seq.num_kv
- 6.2 slot_mapping ← block_table[-1]*block_size + (num_kv-1)%block_size；positions = len(seq)-1
- 6.3 can_append/may_append 改用 num_kv % block_size == 1

6.2 的真 `prepare_decode` 要 .cuda()，本地无法实跑 → 这里**复刻**其逐条公式做对拍：
  既验证压缩开启时落点正确，又验证 num_dropped_kv==0 时与旧公式逐位一致（零回归）。
6.1/6.3 用真 BlockManager + Sequence（纯 Python 记账）实跑。

纯 CPU 零依赖：
    python scratch/test_task6_prepare_decode_num_kv.py
"""
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

BS = 4  # 测试用小 block_size


def _make_seq(num_tokens: int, num_dropped_kv: int = 0) -> Sequence:
    seq = Sequence(list(range(num_tokens)), SamplingParams(temperature=0.6))
    seq.num_dropped_kv = num_dropped_kv
    seq.is_prefill = False
    return seq


# —— 6.2：复刻 prepare_decode 的逐条公式（新 / 旧），用于对拍 ——
def _decode_fields_new(seq: Sequence, block_size: int):
    pos = len(seq) - 1
    ctx = seq.num_kv
    slot = seq.block_table[-1] * block_size + seq.last_kv_block_num_tokens - 1
    return pos, ctx, slot


def _decode_fields_old(seq: Sequence, block_size: int):
    # 压缩接入前的原始公式（基于 num_tokens）
    pos = len(seq) - 1
    ctx = len(seq)
    slot = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
    return pos, ctx, slot


def test_6_2_zero_regression_when_no_compression():
    # num_dropped_kv==0 → 新公式必须与旧公式逐位一致（覆盖块边界各种余数）
    for num_tokens in range(1, 40):
        seq = _make_seq(num_tokens)
        nblocks = (num_tokens + BS - 1) // BS
        seq.block_table = list(range(10, 10 + nblocks))
        assert _decode_fields_new(seq, BS) == _decode_fields_old(seq, BS), num_tokens


def test_6_2_slot_lands_in_compacted_layout():
    # 压缩场景：num_tokens=20, num_dropped_kv=12 → num_kv=8。
    # 紧凑布局：8 个 KV 占块内偏移，新 token(第 8 个,0-indexed=7) 落在第 8//BS=1 块、偏移 7%4=3。
    # evict 后 block_table 只剩 ceil(8/4)=2 块。
    seq = _make_seq(20, num_dropped_kv=12)
    seq.block_table = [30, 31]  # 紧凑后保留 2 块
    pos, ctx, slot = _decode_fields_new(seq, BS)
    assert pos == 19              # 逻辑位置不变 = len-1
    assert ctx == 8              # context_lens = num_kv
    # last_kv_block_num_tokens = 8 - (2-1)*4 = 4 → slot = 31*4 + 4 - 1 = 127
    assert slot == 31 * BS + 4 - 1 == 127, slot
    # 等价校验：slot 块内偏移 == (num_kv-1)%BS
    assert slot - seq.block_table[-1] * BS == (seq.num_kv - 1) % BS


def test_6_2_just_after_evict_pending_token_starts_new_block():
    # evict 后 num_kv = num_keep+1。取 num_keep=8(BS 整数倍) → num_kv=9，
    # 新 token 落点 = (9-1)%4 = 0 → 起新块；ceil(9/4)=3 块。
    seq = _make_seq(20, num_dropped_kv=20 - 8 - 1)  # num_kv = 9
    seq.block_table = [40, 41, 42]
    pos, ctx, slot = _decode_fields_new(seq, BS)
    assert ctx == 9
    assert (seq.num_kv - 1) % BS == 0
    assert slot == 42 * BS + 0, slot  # 落在新尾块偏移 0


# —— 6.3：can_append / may_append 改用 num_kv ——
def test_6_3_may_append_uses_num_kv():
    # num_tokens 较大但 num_kv 刚好 %BS==1 → 应开新块
    mgr = BlockManager(num_blocks=16, block_size=BS)
    seq = _make_seq(20, num_dropped_kv=20 - 9)  # num_kv = 9, 9%4 != 1 → 不开
    seq.block_table = [0, 1, 2]
    for b in seq.block_table:
        mgr.free_block_ids.remove(b); mgr.used_block_ids.add(b); mgr.blocks[b].ref_count = 1
    before = len(seq.block_table)
    mgr.may_append(seq)
    assert len(seq.block_table) == before, "9%4!=1 不该开块"

    # num_kv = 13 → 13%4==1 → 开新块
    seq2 = _make_seq(20, num_dropped_kv=20 - 13)
    seq2.block_table = [5, 6, 7]
    for b in seq2.block_table:
        mgr.free_block_ids.remove(b); mgr.used_block_ids.add(b); mgr.blocks[b].ref_count = 1
    before2 = len(seq2.block_table)
    mgr.may_append(seq2)
    assert len(seq2.block_table) == before2 + 1, "13%4==1 应开块"


def test_6_3_zero_regression_may_append():
    # num_dropped_kv==0 时，触发开块的 num_tokens 与原 len(seq)%BS==1 完全一致
    for num_tokens in range(1, 40):
        mgr = BlockManager(num_blocks=64, block_size=BS)
        seq = _make_seq(num_tokens)  # num_kv == num_tokens
        nblocks = (num_tokens + BS - 1) // BS
        seq.block_table = list(range(nblocks))
        for b in seq.block_table:
            mgr.free_block_ids.remove(b); mgr.used_block_ids.add(b); mgr.blocks[b].ref_count = 1
        before = len(seq.block_table)
        mgr.may_append(seq)
        expected_new = 1 if (num_tokens % BS == 1) else 0
        assert len(seq.block_table) - before == expected_new, num_tokens


def test_6_3_can_append_uses_num_kv():
    # free 仅剩 0 块时，num_kv%BS==1（需新块）→ False；否则 True
    mgr = BlockManager(num_blocks=4, block_size=BS)
    # 占满所有块使 free 为空
    for b in list(mgr.free_block_ids):
        mgr.free_block_ids.remove(b); mgr.used_block_ids.add(b); mgr.blocks[b].ref_count = 1
    assert len(mgr.free_block_ids) == 0

    seq_need = _make_seq(20, num_dropped_kv=20 - 13)  # num_kv=13, %4==1 需新块
    assert mgr.can_append(seq_need) is False
    seq_ok = _make_seq(20, num_dropped_kv=20 - 9)     # num_kv=9, %4!=1 不需新块
    assert mgr.can_append(seq_ok) is True


if __name__ == "__main__":
    tests = [
        test_6_2_zero_regression_when_no_compression,
        test_6_2_slot_lands_in_compacted_layout,
        test_6_2_just_after_evict_pending_token_starts_new_block,
        test_6_3_may_append_uses_num_kv,
        test_6_3_zero_regression_may_append,
        test_6_3_can_append_uses_num_kv,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)} passed")
