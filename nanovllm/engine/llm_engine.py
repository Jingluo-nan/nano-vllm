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
from nanovllm.engine.kv_compression import streaming_keep_indices, should_compress


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
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
        # KV 压缩(v0)：仅 decode 阶段、postprocess 之后触发（此时 last_token 已 append、
        # 其 KV 尚未写入 cache，符合 evict 的"物理 KV 数 = num_kv-1"时序约定）。
        if not is_prefill and self.config.enable_kv_compression:
            self.maybe_compress_kv(seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def maybe_compress_kv(self, seqs: list[Sequence]):
        """对触发条件成立的 decode 序列做 StreamingLLM 压缩：keep-set → 门控 → gather → 回收。

        管线（设计 D1/D5）：streaming_keep_indices 算确定性保留集 → can_evict 在 gather 前
        判被改写块独占（否则跳过该序列，避免改坏共享块）→ compact_kv 物理搬迁（所有 rank
        各搬自己的 kv_cache 分片）→ evict 回收尾块 + 注销 hash + 置 num_dropped_kv。
        接口固定 evict(seq, keep_indices)，v1 仅替换 keep_indices 来源。
        """
        config = self.config
        bs = config.kvcache_block_size
        block_manager = self.scheduler.block_manager
        for seq in seqs:
            if seq.is_finished:
                continue
            # 物理 cache 内 KV 数 = num_kv - 1（pending last_token 的 KV 尚未写入）
            physical_kv = seq.num_kv - 1
            if physical_kv <= 0:
                continue
            num_kv_blocks = (physical_kv + bs - 1) // bs
            if not should_compress(num_kv_blocks, config.kv_sink_blocks, config.kv_recent_blocks):
                continue
            keep_indices = streaming_keep_indices(physical_kv, config.kv_sink_blocks, config.kv_recent_blocks, bs)
            if len(keep_indices) >= physical_kv:
                continue  # 无可丢（理论上 should_compress 已挡住，防御性）
            # gather 前门控：被改写块须独占，否则跳过该序列（数据一旦搬迁无法回滚）
            if not block_manager.can_evict(seq, keep_indices):
                continue
            # 物理 gather 须在 evict 截断 block_table 之前（compact_kv 读旧 block_table）
            self.model_runner.call("compact_kv", seq.block_table, keep_indices)
            block_manager.evict(seq, keep_indices)

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
