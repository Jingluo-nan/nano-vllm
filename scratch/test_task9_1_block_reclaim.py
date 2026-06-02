"""task 9.1 验收：block 确实释放（开/关压缩对照 + 压缩点 free_block 回升）。

验证点（tasks.md 9.1）：
- 开启压缩时 used_block_ids 峰值更低（block 被回收，不随生成无限增长）；
- 每次压缩点 free_block_ids 回升（evict 当场把尾块还回池）；
- 计数证明回收（无泄漏：序列结束 free 复原）。

每个配置独立子进程跑。需 CUDA + flash-attn + 权重 ~/huggingface/Qwen3-0.6B/。
    python scratch/test_task9_1_block_reclaim.py
"""
import os
import sys
import json
import subprocess

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_TOKENS = 1400  # >1024 才跨过 sink1+recent3 触发阈值


def _child(enable: bool):
    from nanovllm import LLM, SamplingParams

    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1,
              enable_kv_compression=enable, kv_sink_blocks=1, kv_recent_blocks=3)
    bm = llm.scheduler.block_manager
    total = len(bm.blocks)
    rec = {"total": total, "free_start": len(bm.free_block_ids),
           "used_peak": len(bm.used_block_ids), "evict_events": []}
    orig_evict = bm.evict

    def wrapped_evict(seq, keep_indices):
        free_before = len(bm.free_block_ids)
        r = orig_evict(seq, keep_indices)
        free_after = len(bm.free_block_ids)
        rec["evict_events"].append({"free_before": free_before, "free_after": free_after,
                                    "reclaimed": free_after - free_before})
        return r

    bm.evict = wrapped_evict
    orig_step = llm.step

    def wrapped_step():
        out = orig_step()
        rec["used_peak"] = max(rec["used_peak"], len(bm.used_block_ids))
        return out

    llm.step = wrapped_step

    sp = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS, ignore_eos=True)
    out = llm.generate([[1, 2, 3, 4, 5, 6, 7, 8]], sp, use_tqdm=False)[0]
    rec["gen_len"] = len(out["token_ids"]) if isinstance(out, dict) and "token_ids" in out else len(out)
    rec["free_end"] = len(bm.free_block_ids)
    print("RESULT_JSON " + json.dumps(rec))


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

    off = _run(False)
    on = _run(True)

    off_free = f"{off['free_start']}/{off['free_end']}"
    on_free = f"{on['free_start']}/{on['free_end']}"
    print(f"\n{'='*64}")
    print(f"{'指标':<22}{'关闭压缩':>14}{'开启压缩':>14}")
    print(f"{'生成 token':<22}{off['gen_len']:>14}{on['gen_len']:>14}")
    print(f"{'used_block 峰值':<22}{off['used_peak']:>14}{on['used_peak']:>14}")
    print(f"{'free 起始/结束':<22}{off_free:>14}{on_free:>14}")
    print(f"{'evict 次数':<22}{len(off['evict_events']):>14}{len(on['evict_events']):>14}")
    print('='*64)
    print("开启压缩 —— 各压缩点 free_block 回升：")
    for i, e in enumerate(on["evict_events"]):
        print(f"  evict#{i+1}: free {e['free_before']} → {e['free_after']}  (+{e['reclaimed']} 块)")

    # —— 断言 ——
    assert on["used_peak"] < off["used_peak"], \
        f"used_peak 未下降: on={on['used_peak']} off={off['used_peak']}"
    assert len(on["evict_events"]) > 0, "压缩未触发"
    assert all(e["reclaimed"] > 0 for e in on["evict_events"]), \
        f"存在未回升的压缩点: {on['evict_events']}"
    assert on["free_end"] == on["total"], f"序列结束有块泄漏: free_end={on['free_end']}/{on['total']}"
    print(f"\nPASS: used_peak {off['used_peak']}→{on['used_peak']}；"
          f"{len(on['evict_events'])} 个压缩点 free 均回升；结束无泄漏（{on['free_end']}/{on['total']}）")
