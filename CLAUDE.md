# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a from-scratch, ~1,200-line reimplementation of the vLLM offline inference engine. The public API mirrors vLLM (`LLM`, `SamplingParams`, `LLM.generate`) but the codebase is intentionally small enough to read end-to-end. Currently only the Qwen3 model family is implemented (`nanovllm/models/qwen3.py`).

## Common Commands

Install (editable, for local development):
```bash
pip install -e .
```

Run the example (expects model weights at `~/huggingface/Qwen3-0.6B/`):
```bash
python example.py
```

Run the throughput benchmark:
```bash
python bench.py
```

Download model weights:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

There is no test suite, no linter config, and no build step beyond `pip install`. `flash-attn` and CUDA are required at runtime; the engine is GPU-only (`torch.set_default_device("cuda")` in `ModelRunner.__init__`).

## Architecture

The engine is structured as three layers that communicate via a small set of dataclasses (`Config`, `Sequence`, `Context`).

### Engine loop (`nanovllm/engine/`)

`LLMEngine.generate` is a synchronous loop calling `step()` until `Scheduler.is_finished()`. Each step does: `Scheduler.schedule()` → `ModelRunner.run()` → `Scheduler.postprocess()`. Critically, **a single step is either all-prefill or all-decode** — `Scheduler.schedule()` returns `(seqs, is_prefill)` and only mixes sequence types within the same phase.

- `Scheduler` maintains `waiting` and `running` deques. It prefers prefill: it drains `waiting` up to `max_num_batched_tokens` / `max_num_seqs` first, and only schedules decode if no prefill was scheduled. **Chunked prefill is allowed only for the first sequence in a batch** (see the `if remaining < num_tokens and scheduled_seqs` guard in `scheduler.py`); a long prompt is split across multiple steps and stays at the head of `waiting` until fully prefilled. During decode, if KV cache is exhausted, sequences are preempted (moved back to `waiting` with `is_prefill=True`, blocks freed).
- `BlockManager` implements **paged KV cache with prefix caching** (block size = `kvcache_block_size`, default 256, must be a multiple of 256). Blocks are content-addressed via xxhash chained over (prev_hash, token_ids). On allocate, `can_allocate` walks the prefix to find already-cached blocks and returns `num_cached_blocks`; matched blocks have their `ref_count` bumped instead of being recopied. `hash_blocks` (called from `Scheduler.postprocess`) registers newly-filled blocks into `hash_to_block_id` so subsequent sequences can hit the cache.
- `Sequence` carries scheduler bookkeeping (`num_cached_tokens`, `num_scheduled_tokens`, `block_table`, `is_prefill`). It has custom `__getstate__`/`__setstate__` to minimize pickled size when shipped to TP workers — only `last_token` is sent during decode, full `token_ids` only during prefill.

### Model runner (`engine/model_runner.py`)

One `ModelRunner` per GPU. Rank 0 runs in the main process; ranks ≥1 run in `mp.Process`es spawned by `LLMEngine.__init__`. They communicate via a single `SharedMemory` segment (`name="nanovllm"`, 1 MiB) plus per-worker `mp.Event`s — rank 0 pickles `(method_name, args)`, sets all events; workers wake, read, execute. Workers loop in `ModelRunner.loop()` until they receive `"exit"`. NCCL is initialized via `tcp://localhost:2333`.

`ModelRunner.run` has two paths:
- **Prefill or eager or batch > 512** → run the model directly.
- **Decode** → replay a captured CUDA graph. Graphs are pre-captured in `capture_cudagraph()` for batch sizes `[1, 2, 4, 8, 16, 32, ..., max_bs]`; at runtime it picks the smallest graph ≥ current batch size and copies inputs into the persistent `graph_vars` buffers before `graph.replay()`. Disable with `enforce_eager=True`.

KV cache is allocated *after* a warmup forward pass: the runner measures peak memory, then sizes `num_kvcache_blocks` to fill `gpu_memory_utilization` of the remaining VRAM. Each `Attention` module has its `k_cache`/`v_cache` attributes patched to point into the shared `self.kv_cache` tensor (`allocate_kv_cache`).

`prepare_prefill` / `prepare_decode` build the per-step tensors (`input_ids`, `positions`, `slot_mapping`, `cu_seqlens_*`, `block_tables`, `context_lens`) on pinned CPU memory then `.cuda(non_blocking=True)`, and stash them in a module-global `Context` (`utils/context.py`) that `Attention.forward` reads. This is how scheduler state reaches the attention kernel without threading args through every layer.

### Attention and the prefill/decode split (`layers/attention.py`)

`Attention.forward` reads `get_context()` and dispatches to one of three flash-attn paths:
1. **Decode** (`is_prefill=False`): `flash_attn_with_kvcache` with `block_table` and `context_lens`.
2. **Prefill with prefix cache** (`is_prefill=True` and `block_tables is not None`): `flash_attn_varlen_func` reading K/V *from the cache* (`k, v = k_cache, v_cache`) using `block_table`. The scheduler sets this up when `cu_seqlens_k[-1] > cu_seqlens_q[-1]` (i.e. some tokens are already cached).
3. **Pure prefill** (warmup, or no cached prefix): `flash_attn_varlen_func` on the freshly-computed K/V.

In all paths a Triton kernel `store_kvcache` writes new K/V into the paged cache at `slot_mapping`. Slots of `-1` are skipped (used for CUDA-graph padding).

### Tensor parallelism (`layers/linear.py`, `layers/embed_head.py`)

TP is implemented at the linear-layer level. `ColumnParallelLinear` shards output dim, `RowParallelLinear` shards input dim and all-reduces output, `QKVParallelLinear` and `MergedColumnParallelLinear` fuse the standard QKV / gate+up shards into a single sharded weight. Each parallel layer exposes a `weight_loader(param, loaded_weight, [shard_id])` method; `utils/loader.py` walks `*.safetensors`, applies `Qwen3ForCausalLM.packed_modules_mapping` (which renames `q_proj/k_proj/v_proj` → `qkv_proj`, `gate_proj/up_proj` → `gate_up_proj`), and dispatches to the right loader. To add a new model, define an analogous `packed_modules_mapping`.

## Conventions to preserve

- Prefer pinned-CPU → `.cuda(non_blocking=True)` for any per-step tensor (see `prepare_prefill`/`prepare_decode`).
- The `Context` global is the channel between scheduler and attention; do not pass these tensors as function arguments.
- `Sequence.__getstate__` is a hot path during TP — keep it minimal. Don't add fields that need to round-trip to workers unless you also add them here.
- `kvcache_block_size` must be a multiple of 256 (asserted in `Config.__post_init__`); flash-attn paged kernels assume this.
- Greedy sampling is intentionally rejected (`SamplingParams.__post_init__` asserts `temperature > 1e-10`); the sampler uses Gumbel-max.
