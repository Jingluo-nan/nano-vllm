import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import get_context, reset_context

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
    first_token = runner.run([seq], is_prefill = True)[0]
    seq.append_token(first_token) # 把第一个token接回序列

    print("prompt: ", len(prompt_ids))
    print("first token   :", first_token, "->", tokenizer.decode([first_token]))
    print("seq length now:", len(seq))

    input_ids, positions = runner.prepare_decode([seq])
    ctx = get_context()
    print("input_ids   :", input_ids.tolist())        # [last_token]，每条一个
    print("positions   :", positions.tolist())        # [len-1]
    print("context_lens:", ctx.context_lens.tolist()) # [len]
    print("slot_mapping:", ctx.slot_mapping.tolist()) 
    print("block_tables:", ctx.block_tables.tolist())
    reset_context()

    for _ in range(16):
        token = runner.run([seq], is_prefill = False)[0]
        if token == tokenizer.eos_token_id:
            break
        seq.append_token(token)

    # 解码出 prompt 之后续写的部分
    print("completion:", tokenizer.decode(seq.completion_token_ids))

    # 收尾，避免 NCCL 进程组泄漏的告警
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()