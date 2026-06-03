"""task 9.3 验收：收益对照（开/关压缩对比 preemption↓、平均 decode batch↑）。

依赖第 1 步基线度量（LLMEngine.metrics: preemptions / avg_decode_batch / decode_tok_s）。
制造 KV 紧张：多序列并发长生成 + 调小 gpu_memory_utilization 缩小 KV cache。
压缩把每序列 KV 钉在 sink+recent 块 → 更多序列并发存活 → 抢占↓、平均 decode batch↑。

每配置独立子进程。需 CUDA + flash-attn + 权重 ~/huggingface/Qwen3-0.6B/。
    python scratch/test_task9_3_benefit.py
"""
import os
import sys
import json
import subprocess

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
NUM_SEQS = 16          # 并发序列数
MAX_TOKENS = 2000      # 长生成让"压缩稳态"主导：off 16×⌈2000/256⌉=128≫91 全程抢占；
                       # on 16×4=64<91 爬坡后稳住 → preemption 显著下降
GPU_UTIL = 0.9         # 与其它实跑一致以保证 KV cache 能分配；靠并发+长度制造压力


def _child(enable: bool):
    from nanovllm import LLM, SamplingParams

    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=GPU_UTIL,
              enable_kv_compression=enable, kv_sink_blocks=1, kv_recent_blocks=3)
    bm = llm.scheduler.block_manager
    total = len(bm.blocks)
    peak = {"used": 0}
    orig_step = llm.step
    def wrapped_step():
        out = orig_step()
        peak["used"] = max(peak["used"], len(bm.used_block_ids))
        return out
    llm.step = wrapped_step

    # 各序列用不同前缀，避免前缀缓存共享块而稀释压力
    prompts = [[100 + i] + list(range(i, i + 12)) for i in range(NUM_SEQS)]
    sp = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS, ignore_eos=True)
    llm.generate(prompts, sp, use_tqdm=False)
    m = llm.metrics
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    print("RESULT_JSON " + json.dumps({
        "preemptions": m["preemptions"],
        "avg_decode_batch": m["avg_decode_batch"],
        "decode_tok_s": m["decode_tok_s"],
        "used_peak": peak["used"], "total": total,
    }))


def _run(enable):
    tag = "on" if enable else "off"
    p = subprocess.run([sys.executable, __file__, tag], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")]
    if not lines:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"子进程 {tag} 无结果")
    return json.loads(lines[-1][len("RESULT_JSON "):])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("on", "off"):
        _child(sys.argv[1] == "on")
        sys.exit(0)

    print(f"工作负载: {NUM_SEQS} 序列并发, max_tokens={MAX_TOKENS}, gpu_util={GPU_UTIL}")
    off = _run(False)
    on = _run(True)

    print(f"\n{'='*60}")
    off_used = f"{off['used_peak']}/{off['total']}"
    on_used = f"{on['used_peak']}/{on['total']}"
    print(f"{'指标':<22}{'关闭压缩':>16}{'开启压缩':>16}")
    print(f"{'preemptions':<22}{off['preemptions']:>16}{on['preemptions']:>16}")
    print(f"{'avg_decode_batch':<22}{off['avg_decode_batch']:>16.2f}{on['avg_decode_batch']:>16.2f}")
    print(f"{'decode_tok_s':<22}{off['decode_tok_s']:>16.1f}{on['decode_tok_s']:>16.1f}")
    print(f"{'used_block 峰值':<20}{off_used:>16}{on_used:>16}")
    print('='*60)

    # —— 断言：收益方向正确 ——
    assert on["preemptions"] <= off["preemptions"], \
        f"preemption 未下降: on={on['preemptions']} off={off['preemptions']}"
    assert on["avg_decode_batch"] >= off["avg_decode_batch"], \
        f"avg_decode_batch 未上升: on={on['avg_decode_batch']:.2f} off={off['avg_decode_batch']:.2f}"
    print(f"\nPASS: preemptions {off['preemptions']}→{on['preemptions']}, "
          f"avg_decode_batch {off['avg_decode_batch']:.2f}→{on['avg_decode_batch']:.2f}")
