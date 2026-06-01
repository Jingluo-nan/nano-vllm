import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            # 把请求加入队列
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # —— 基线度量：累计 decode 阶段的步数/batch/token/耗时，跑完算平均 ——
        self.scheduler.num_preemptions = 0
        decode_steps = 0
        decode_batch_sum = 0      # 各 decode step 的 batch size 之和 → 平均 decode batch
        decode_tokens = 0         # decode 阶段生成的 token 总数（每序列每步 1 个）
        decode_time = 0.          # decode 阶段累计耗时
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            dt = perf_counter() - t
            if num_tokens > 0:
                prefill_throughput = num_tokens / dt
            else:
                decode_throughput = -num_tokens / dt
                decode_steps += 1
                decode_batch_sum += -num_tokens
                decode_tokens += -num_tokens
                decode_time += dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 汇总三项基线指标，存到 self.metrics 供外部读取，并在 verbose 时打印
        self.metrics = {
            "preemptions": self.scheduler.num_preemptions,
            "avg_decode_batch": decode_batch_sum / decode_steps if decode_steps else 0.,
            "decode_tok_s": decode_tokens / decode_time if decode_time else 0.,
        }
        if use_tqdm:
            print(
                f"[metrics] preemptions={self.metrics['preemptions']} "
                f"avg_decode_batch={self.metrics['avg_decode_batch']:.1f} "
                f"decode_tok_s={self.metrics['decode_tok_s']:.1f}"
            )
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
