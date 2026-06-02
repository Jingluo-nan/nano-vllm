import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

# —— KV 压缩开关 ——
# True 开启 StreamingLLM 压缩。注意触发条件：num_kv_blocks > sink + recent，
# 默认 block_size=256 / sink=1 / recent=3 → 需 physical_kv > 1024 token 才会压一次，
# 故下面用长生成 + ignore_eos 才能真正触发（否则开了也等于没开）。
ENABLE_COMPRESSION = True
KV_SINK_BLOCKS = 1
KV_RECENT_BLOCKS = 3


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_kv_compression=ENABLE_COMPRESSION,
        kv_sink_blocks=KV_SINK_BLOCKS,
        kv_recent_blocks=KV_RECENT_BLOCKS,
    )

    # —— 统计压缩是否真触发：包裹 evict 计数 + 跟踪 block 用量 ——
    bm = llm.scheduler.block_manager
    total_blocks = len(bm.blocks)
    stats = {"evicts": 0, "used_peak": len(bm.used_block_ids)}
    _orig_evict = bm.evict

    def _evict(seq, keep_indices):
        stats["evicts"] += 1
        return _orig_evict(seq, keep_indices)

    bm.evict = _evict
    _orig_step = llm.step

    def _step():
        out = _orig_step()
        stats["used_peak"] = max(stats["used_peak"], len(bm.used_block_ids))
        return out

    llm.step = _step

    # 长生成 + ignore_eos：确保跨过压缩触发阈值（>1024 token）
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1400, ignore_eos=True)
    prompts = [
        "Write a detailed essay about the history and future of artificial intelligence.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    # —— 压缩信息 ——
    print("\n" + "=" * 60)
    print(f"压缩开启: {ENABLE_COMPRESSION} (sink={KV_SINK_BLOCKS}, recent={KV_RECENT_BLOCKS})")
    print(f"evict 触发次数: {stats['evicts']}")
    print(f"used_block 峰值: {stats['used_peak']} / {total_blocks}")
    print(f"当前 free_block: {len(bm.free_block_ids)} / {total_blocks}")
    if ENABLE_COMPRESSION and stats["evicts"] == 0:
        print("⚠️ 压缩未触发：生成长度未跨过阈值（需 physical_kv > sink+recent 块）")
    print("=" * 60)


if __name__ == "__main__":
    main()
