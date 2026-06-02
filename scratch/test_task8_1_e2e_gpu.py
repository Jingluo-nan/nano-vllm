"""task 8.1 端到端实跑（GPU + flash-attn）：开启 KV 压缩长生成不崩 + 压缩确实触发。

验证点（tasks.md 8.1）：
- 开启 enable_kv_compression，长生成（>sink+recent 块，触发压缩）端到端跑通，不报错/不崩/不非法内存；
- 压缩确实触发：evict 被调用、free_block 低水位回升；
- 与关闭压缩对照：开启时 used_block 峰值更低（block 确实被回收）。

每个配置在独立子进程跑（init_process_group + KV 显存预算不可在同进程内重建两次）。
需 CUDA + flash-attn + 权重 ~/huggingface/Qwen3-0.6B/。enforce_eager=True 隔离 CUDA graph。
    python scratch/test_task8_1_e2e_gpu.py
"""
import os
import sys
import json
import subprocess

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
# block_size=256, sink=1+recent=3 → physical_kv>1024 才触发，故生成 >1024 token
MAX_TOKENS = 1400


def _child(enable: bool):
    """子进程：跑单个配置，把统计以 JSON 打到 stdout 最后一行。"""
    from nanovllm import LLM, SamplingParams

    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1,
              enable_kv_compression=enable, kv_sink_blocks=1, kv_recent_blocks=3)
    bm = llm.scheduler.block_manager
    stats = {"evict_calls": 0, "free_low": len(bm.free_block_ids),
             "used_peak": len(bm.used_block_ids), "total": len(bm.free_block_ids)}
    orig_evict = bm.evict
    def wrapped_evict(seq, keep_indices):
        stats["evict_calls"] += 1
        return orig_evict(seq, keep_indices)
    bm.evict = wrapped_evict
    orig_step = llm.step
    def wrapped_step():
        out = orig_step()
        stats["free_low"] = min(stats["free_low"], len(bm.free_block_ids))
        stats["used_peak"] = max(stats["used_peak"], len(bm.used_block_ids))
        return out
    llm.step = wrapped_step

    sp = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS, ignore_eos=True)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]  # ignore_eos 保证生成满 max_tokens
    outputs = llm.generate([prompt], sp, use_tqdm=False)
    o = outputs[0]
    stats["gen_len"] = len(o["token_ids"]) if isinstance(o, dict) and "token_ids" in o else len(o)
    print("RESULT_JSON " + json.dumps(stats))


def _run_child(enable: bool):
    tag = "on" if enable else "off"
    p = subprocess.run([sys.executable, __file__, tag], capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")]
    if not line:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"子进程 {tag} 未产出结果（见上）")
    return json.loads(line[-1][len("RESULT_JSON "):])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("on", "off"):
        _child(sys.argv[1] == "on")
        sys.exit(0)

    print("=== 关闭压缩 (baseline) ===")
    off = _run_child(False)
    print(f"  生成 {off['gen_len']} token; used_peak={off['used_peak']} "
          f"free_low={off['free_low']} evict={off['evict_calls']}")

    print("=== 开启压缩 ===")
    on = _run_child(True)
    print(f"  生成 {on['gen_len']} token; used_peak={on['used_peak']} "
          f"free_low={on['free_low']} evict={on['evict_calls']}")

    assert on["gen_len"] >= MAX_TOKENS, f"未生成满 max_tokens: {on['gen_len']}"
    assert on["evict_calls"] > 0, "压缩从未触发（evict 未被调用）"
    assert on["used_peak"] <= off["used_peak"], \
        f"开启压缩 used_peak 未下降: on={on['used_peak']} off={off['used_peak']}"
    print(f"\nPASS: 端到端跑通、压缩触发 {on['evict_calls']} 次、"
          f"used_peak {off['used_peak']}→{on['used_peak']}（回收 {off['used_peak']-on['used_peak']} 块）")
