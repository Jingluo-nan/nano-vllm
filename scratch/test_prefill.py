import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context

from my_model_runner import ModelRunner

def main():
    # 单卡环境准备：nano-vllm 的网络层在构造时要查 world_size，
    # 这里先初始化一个单进程组（多卡版后续介绍）
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)

    # 本地路径（Config 要求 model 是已存在的目录）
    model_path = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    config = Config(model_path, enforce_eager=True, max_model_len=4096)
    runner = ModelRunner(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # prompt → token ids（chat template）
    msgs = [{"role": "user", "content": "你是谁"}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    prompt_ids = tokenizer(text).input_ids
    # 造一条 Sequence，整段 prompt 一次 prefill
    seq = Sequence(prompt_ids, SamplingParams(temperature=0.6))
    seq.num_scheduled_tokens = len(seq)               # 整段都要算
    seq.block_table = list(range(seq.num_blocks))
    input_ids, positions = runner.prepare_prefill([seq])
    ctx = get_context()
    print("prompt tokens :", len(prompt_ids))
    print("input_ids[:8] :", input_ids[:8].tolist())
    print("positions[:8] :", positions[:8].tolist())
    print("cu_seqlens_q  :", ctx.cu_seqlens_q.tolist())
    print("slot_mapping[:8]:", ctx.slot_mapping[:8].tolist())
    # 端到端：一批 Sequence → 第一个 next token
    token_ids = runner.run([seq], is_prefill=True)
    print("first next token id :", token_ids[0])
    print("decoded             :", tokenizer.decode(token_ids))

    # 收尾，避免 NCCL 进程组泄漏的告警
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()