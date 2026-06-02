"""质量观察（非任务单测）：开/关 KV 压缩，同种子真实 prompt 长生成，解码成文本对比。

同种子下两者在压缩触发前应逐 token 一致，触发后开始分化 —— 看分化后文本是否仍连贯。
每个配置独立子进程跑（init_process_group / KV 显存预算不可同进程重建两次）。
    python scratch/inspect_compression_quality.py
"""
import os
import sys
import json
import subprocess

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
MAX_TOKENS = 1400          # >1024 才跨过 sink1+recent3 触发阈值
PROMPT = "Write a detailed essay about the history and future of artificial intelligence."


def _child(enable: bool):
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    tokenizer = AutoTokenizer.from_pretrained(PATH)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)

    llm = LLM(PATH, enforce_eager=True, tensor_parallel_size=1,
              enable_kv_compression=enable, kv_sink_blocks=1, kv_recent_blocks=3)
    bm = llm.scheduler.block_manager
    evicts = {"n": 0}
    orig = bm.evict
    bm.evict = lambda seq, ki: (evicts.__setitem__("n", evicts["n"] + 1), orig(seq, ki))[1]

    sp = SamplingParams(temperature=0.6, max_tokens=MAX_TOKENS, ignore_eos=True)
    out = llm.generate([text], sp, use_tqdm=False)[0]
    print("RESULT_JSON " + json.dumps(
        {"evicts": evicts["n"], "token_ids": out["token_ids"], "text": out["text"]}))


def _run(enable):
    tag = "on" if enable else "off"
    p = subprocess.run([sys.executable, __file__, tag], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")]
    if not lines:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"子进程 {tag} 无结果")
    return json.loads(lines[-1][len("RESULT_JSON "):])


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("on", "off"):
        _child(sys.argv[1] == "on")
        sys.exit(0)

    off = _run(False)
    on = _run(True)
    div = _first_divergence(off["token_ids"], on["token_ids"])

    print(f"\n{'='*70}\n关闭压缩: {len(off['token_ids'])} token, evict={off['evicts']}")
    print(f"开启压缩: {len(on['token_ids'])} token, evict={on['evicts']}")
    print(f"首次分化于第 {div} 个生成 token（此前两者逐 token 一致 = 压缩尚未影响）")
    print('='*70)
    print("\n----- 关闭压缩，全文 -----\n")
    print(off["text"])
    print("\n----- 开启压缩，全文 -----\n")
    print(on["text"])
    print("\n----- 开启压缩：分化点之后约 300 字符（重点看连贯性）-----\n")
    # 用分化 token 之前的文本长度近似定位
    tail = on["text"][-1500:]
    print("...(末尾片段)...\n" + tail)
