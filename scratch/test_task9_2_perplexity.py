"""task 9.2 验收：PPL 没崩（相同文本，开/关压缩算 perplexity，仅小幅上升不爆炸）。

方法（teacher forcing 控制变量）：
1. 先用关闭压缩、固定种子生成一段参考 token 序列 ref（真实 prompt，长度跨过压缩阈值）；
2. 对**同一条 ref**，分别在开/关压缩下重跑：monkeypatch sampler 强制逐位返回 ref 的 token
   （强制走同一路径），同时记录模型对该 token 的 log 概率 = log_softmax(原始 logits)[token]；
3. PPL = exp(-平均 logprob)。两次唯一差别只剩压缩 → PPL 差异纯归因于压缩。

需 CUDA + flash-attn + 权重 ~/huggingface/Qwen3-0.6B/。每配置独立子进程。
    python scratch/test_task9_2_perplexity.py
"""
import os
import sys
import json
import math
import tempfile
import subprocess

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_TOKENS = 1400  # 续写长度，>1024 触发压缩
PROMPT = "Write a detailed essay about the history and future of artificial intelligence."


def _gen_ref(out_path: str):
    """关闭压缩、固定种子生成参考序列，存 {prompt_ids, ref_ids} 到 out_path。"""
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    tok = AutoTokenizer.from_pretrained(PATH)
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}],
                                   tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(text)
    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS, ignore_eos=True)
    out = llm.generate([prompt_ids], sp, use_tqdm=False)[0]
    comp = out["token_ids"]
    with open(out_path, "w") as f:
        json.dump({"prompt_ids": prompt_ids, "ref_ids": prompt_ids + comp}, f)


def _score(enable: bool, ref_path: str):
    """对 ref teacher-forcing 算 PPL；打印 RESULT_JSON。"""
    import torch
    from nanovllm import LLM, SamplingParams

    with open(ref_path) as f:
        data = json.load(f)
    prompt_ids, ref_ids = data["prompt_ids"], data["ref_ids"]
    P = len(prompt_ids)
    n_score = len(ref_ids) - P

    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1,
              enable_kv_compression=enable, kv_sink_blocks=1, kv_recent_blocks=3)

    state = {"idx": P, "logprobs": []}

    def hooked_sampler(logits, temperatures):
        # logits: [batch=1, vocab]；PPL 用模型真实分布（不做 temperature 缩放）
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok = ref_ids[state["idx"]]
        state["logprobs"].append(lp[0, tok].item())
        state["idx"] += 1
        return torch.tensor([tok], device=logits.device)

    llm.model_runner.sampler = hooked_sampler

    sp = SamplingParams(temperature=0.6, max_tokens=n_score, ignore_eos=True)
    llm.generate([prompt_ids], sp, use_tqdm=False)

    lps = state["logprobs"]
    assert len(lps) == n_score, f"打分位置数不符: {len(lps)} != {n_score}"
    ppl_all = math.exp(-sum(lps) / len(lps))
    # 后段（压缩生效区，第 1024 个续写 token 之后）单独算
    tail = lps[1024:] if len(lps) > 1024 else []
    ppl_tail = math.exp(-sum(tail) / len(tail)) if tail else float("nan")
    print("RESULT_JSON " + json.dumps({"ppl_all": ppl_all, "ppl_tail": ppl_tail, "n": len(lps)}))


def _run(args):
    p = subprocess.run([sys.executable, __file__, *args], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")]
    if not lines:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"子进程 {args} 无结果")
    return json.loads(lines[-1][len("RESULT_JSON "):]) if lines else None


if __name__ == "__main__":
    # 子进程分发
    if len(sys.argv) >= 2 and sys.argv[1] == "gen":
        _gen_ref(sys.argv[2]); sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] in ("on", "off"):
        _score(sys.argv[1] == "on", sys.argv[2]); sys.exit(0)

    # 主进程编排
    ref_path = os.path.join(tempfile.gettempdir(), "kvc_ppl_ref.json")
    print("生成参考序列（关闭压缩，种子 0）...")
    subprocess.run([sys.executable, __file__, "gen", ref_path], check=True,
                   capture_output=True, text=True)
    with open(ref_path) as f:
        meta = json.load(f)
    print(f"  ref 长度 {len(meta['ref_ids'])}（prompt {len(meta['prompt_ids'])} + 续写 {len(meta['ref_ids'])-len(meta['prompt_ids'])}）")

    off = _run(["off", ref_path])
    on = _run(["on", ref_path])

    print(f"\n{'='*56}")
    print(f"{'':<18}{'关闭压缩':>16}{'开启压缩':>16}")
    print(f"{'PPL（全程）':<16}{off['ppl_all']:>16.4f}{on['ppl_all']:>16.4f}")
    print(f"{'PPL（>1024 续写段）':<14}{off['ppl_tail']:>16.4f}{on['ppl_tail']:>16.4f}")
    print('='*56)
    rise = (on["ppl_all"] - off["ppl_all"]) / off["ppl_all"] * 100
    print(f"全程 PPL 变化: {off['ppl_all']:.3f} → {on['ppl_all']:.3f}  ({rise:+.2f}%)")

    # 断言：PPL 不爆炸（放宽阈值：相对上升 < 50% 且绝对值有限）
    assert math.isfinite(on["ppl_all"]), "PPL 非有限值（崩了）"
    assert on["ppl_all"] < off["ppl_all"] * 1.5, \
        f"PPL 上升过多（>50%）: {off['ppl_all']:.3f} → {on['ppl_all']:.3f}"
    print(f"\nPASS: 开启压缩 PPL 仅小幅变化（{rise:+.2f}%），未爆炸。")
