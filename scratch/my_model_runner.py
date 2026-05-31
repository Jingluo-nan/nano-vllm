import torch
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

class ModelRunner:
    def __init__(self, config: Config):
        self.config = config
        self.block_size = config.kvcache_block_size
        Sequence.block_size = self.block_size

        # 确保模型、kv cache 都加载到GPU中
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_config.dtype)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_kv_cache(num_blocks=8)

        # 还原device/dtype,否则 prepare_* 里
        # torch.tensor(,pin_memory=True)会报错
        torch.set_default_dtype(default_dtype)
        torch.set_default_device("cpu")

    def allocate_kv_cache(self, num_blocks: int):
        """为模型分配kv cache"""
        hf = self.config.hf_config
        nkv = hf.num_key_value_heads
        # num_attention_heads是Q头的数量
        # head_dim不一定等于 hf.hidden_size // hf.num_attention_heads
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        print(head_dim)

        self.kv_cache = torch.empty(
            2, hf.num_hidden_layers, num_blocks,
            self.block_size, nkv, head_dim,
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # Sequence -> 张量
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill
            else self.prepare_decode(seqs)
        )

        # 在 GPU 上进行采样，所以 sampler 参数也需要存到 GPU 显存中
        # 同步拷贝
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype = torch.float32, device = "cuda",
        )

        logits = self.run_model(input_ids, positions)
        token_ids = self.sampler(logits, temperatures).tolist()

        # 模型跑完后，KV cache 已写入；趁 context 还没 reset 看一下
        # self.print_kv_cache(layer_id=0)

        reset_context()
        return token_ids

    def print_kv_cache(self, layer_id: int = 0):
        """打印 KV cache 的整体形状，以及本步写入槽位上的 K/V 内容（指定层）"""
        torch.cuda.synchronize()    # GPU 异步，先同步保证读到的是写入后的值
        ctx = get_context()
        slots = ctx.slot_mapping    # 本步每个 token 写入的全局槽位下标

        print("=" * 60)
        print("[KV cache] 整体形状 :", tuple(self.kv_cache.shape),
              "(2=K/V, 层数, block数, block_size, kv头数, head_dim)")
        print(f"[KV cache] 本步写入 {slots.numel()} 个 token，槽位: {slots.tolist()}")

        # 取出第 layer_id 层的 K/V cache，并展平 block 维度便于按槽位索引
        # 形状 [block数, block_size, kv头数, head_dim] -> [总槽位数, kv头数, head_dim]
        k = self.kv_cache[0, layer_id].flatten(0, 1)
        v = self.kv_cache[1, layer_id].flatten(0, 1)
        print(f"[KV cache] 第 {layer_id} 层，按槽位展平后形状: K={tuple(k.shape)} V={tuple(v.shape)}")

        # 看本步第一个 token 写进 cache 的 K/V 向量（只取前 8 维，避免刷屏）
        first_slot = int(slots[0])
        print(f"[KV cache] slot {first_slot} 的 K[head0, :8]:",
              k[first_slot, 0, :8].float().cpu().tolist())
        print(f"[KV cache] slot {first_slot} 的 V[head0, :8]:",
              v[first_slot, 0, :8].float().cpu().tolist())
        # 写入区域的非零元素占比，可粗略确认 cache 确实被写过
        written = k[slots]
        print(f"[KV cache] 已写入槽位的 K 非零元素占比: "
              f"{(written != 0).float().mean().item():.3f}")
        print("=" * 60)

    @torch.inference_mode()
    def run_model(self, input_ids, positions):
        hidden = self.model(input_ids, positions)
        return self.model.compute_logits(hidden)
    

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 最长的 block_table,决定矩形张量的列数
        max_len = max(len(seq.block_table) for seq in seqs)

        bt = [seq.block_table + [-1] * (max_len - len(seq.block_table))
              for seq in seqs]
        
        table = torch.tensor(bt, dtype=torch.int32, pin_memory=True)
        return table.cuda(non_blocking=True)

    def prepare_prefill(self ,seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0

        slot_mapping = []
        block_tables = None

        for seq in seqs:
            start = seq.num_cached_tokens   # 已经缓存的前缀长度
            seqlen_q = seq.num_scheduled_tokens # 本步要算的新token长度
            end = start + seqlen_q
            seqlen_k = end # key 总长 = 前缀 + 新

            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)

            start_block = start // self.block_size
            # end 向上取整，是开区间 
            # 16 / 10 -> 1 + 1 -> 2
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                base = seq.block_table[i] * self.block_size
                slot_start = base
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1 :
                    slot_end = base + self.block_size
                else :
                    slot_end = base + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))


        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)


        def to_cuda(x, dtype):
            t = torch.tensor(x, dtype = dtype, pin_memory = True)
            return t.cuda(non_blocking = True)
        
        input_ids = to_cuda(input_ids, torch.int64)
        positions = to_cuda(positions, torch.int64)
        cu_seqlens_q = to_cuda(cu_seqlens_q, torch.int32)
        cu_seqlens_k = to_cuda(cu_seqlens_k, torch.int32)
        slot_mapping = to_cuda(slot_mapping, torch.int32)

        set_context(True,cu_seqlens_q,cu_seqlens_k,
                    max_seqlen_q,max_seqlen_k,slot_mapping,None,block_tables)

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = [] # 新token要写到哪个slot
        context_lens = [] # 每条seq要会看的 kv 长度

        for seq in seqs:
            # decode 每条只取上一步产出的那个token
            input_ids.append(seq.last_token)
            # 它的位置
            positions.append(len(seq) - 1)
            # 注意力要回看的kv总长 = 整条序列长度
            context_lens.append(len(seq))
            slot = (seq.block_table[-1] * self.block_size +
                    seq.last_block_num_tokens -1)
            slot_mapping.append(slot)
        
        def to_cuda(x, dtype):
            t = torch.tensor(x, dtype = dtype, pin_memory = True)
            return t.cuda(non_blocking = True)

        input_ids = to_cuda(input_ids, torch.int64)
        positions = to_cuda(positions, torch.int64)
        context_lens = to_cuda(context_lens, torch.int32)
        slot_mapping = to_cuda(slot_mapping, torch.int32)

        block_tables = self.prepare_block_tables(seqs)

        set_context(False, slot_mapping=slot_mapping,context_lens=context_lens,block_tables=block_tables)

        return input_ids, positions