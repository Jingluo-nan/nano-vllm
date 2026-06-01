"""KV-Cache 压缩的保留集（keep-set）计算 —— v0: StreamingLLM。

约定：keep_indices 是"当前 cache 布局里要保留的 KV 位置下标"，
取值范围 [0, num_kv)，有序、去重。下游 evict/gather 据此搬运。

v0 用确定性的 StreamingLLM 策略（sink + recent，块对齐，无需注意力打分）。
v1 在此追加 attention_topk_keep_indices(...)，下游 evict 接口不变。
"""

# 覆盖不了"块内少数 token 搬迁"，是块对齐的
def streaming_keep_indices(
    num_kv: int,
    sink_blocks: int,
    recent_blocks: int,
    block_size: int,
) -> list[int]:
    """块对齐地保留"开头 sink_blocks 块 + 最近 recent_blocks 块"。

    返回当前 cache 布局中要保留的位置下标（[0, num_kv)，有序、去重）。
    当 sink+recent 已覆盖全部块时返回全部下标（无需压缩）。
    """
    num_blocks = (num_kv + block_size - 1) // block_size
    # sink+recent 覆盖了所有块 → 没有中段可丢，全保留
    if sink_blocks + recent_blocks >= num_blocks:
        return list(range(num_kv))

    keep: list[int] = []
    # sink：前 sink_blocks 块。因为后面还有 recent 块，这些 sink 块一定是满的
    keep.extend(range(0, sink_blocks * block_size))
    # recent：最后 recent_blocks 块，从 recent_start 到 num_kv（末块可能半满）
    recent_start = (num_blocks - recent_blocks) * block_size
    keep.extend(range(recent_start, num_kv))
    return keep


def should_compress(
    num_kv_blocks: int,
    sink_blocks: int,
    recent_blocks: int,
) -> bool:
    """v0 触发条件：物理占用块数超过 sink+recent 时才压一次。

    短序列（块数不超过 sink+recent）没有可回收的整块，跳过。
    """
    return num_kv_blocks > sink_blocks + recent_blocks
