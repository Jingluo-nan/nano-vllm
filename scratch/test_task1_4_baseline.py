"""task 1.4：默认配置（压缩关闭）跑 baseline，记录三项基线度量。

输出 LLMEngine.metrics 的三项：preemptions / avg_decode_batch / decode_tok_s。
固定工作负载 + 固定种子，可复现，供 task 9.3 收益对照参照。
需 CUDA + flash-attn + 权重 ~/huggingface/Qwen3-0.6B/。
    python scratch/test_task1_4_baseline.py
"""
import os
import torch
from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
NUM_SEQS = 8           # 常规并发，不刻意制造 KV 压力（baseline = 默认正常运行）
MAX_TOKENS = 512       # 常规生成长度
GPU_UTIL = 0.9         # 默认配置

if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA"
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    # 默认配置：enable_kv_compression=False（不传即默认关闭）
    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1, gpu_memory_utilization=GPU_UTIL)
    prompts = [[100 + i] + list(range(i, i + 12)) for i in range(NUM_SEQS)]
    sp = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)
    m = llm.metrics

    print("=" * 56)
    print(f"BASELINE（默认配置，压缩关闭）")
    print(f"工作负载: {NUM_SEQS} 序列 × {MAX_TOKENS} token, gpu_util={GPU_UTIL}, seed=0")
    print("-" * 56)
    print(f"  preemptions      = {m['preemptions']}")
    print(f"  avg_decode_batch = {m['avg_decode_batch']:.2f}")
    print(f"  decode_tok_s     = {m['decode_tok_s']:.1f}")
    print("=" * 56)
    # 进程组由 LLM 退出时自动销毁，无需手动 destroy（重复销毁会断言失败）
