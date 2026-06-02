from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 查询是否有可复用的物理块，并预估"可复用几块、要新几块"
    # 返回 -1（装不下）或可复用块数
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break #不可再复用了
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1 # 装不下
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks): # range不包含 num_cached_blocks
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1  # 已释放但 hash 未失效 → 取回再用
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        #先释放后缀块，那么前缀块被其他人分走的概率就小。可以包含前缀块的缓存价值
        #块被归还时，并不会清空其中的tokens，这样可以保留块的缓存价值
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def _rewritten_block_ids(self, seq: Sequence, keep_indices: list[int]) -> list[int]:
        """gather 会改写内容的前部块（物理块号列表）。

        keep_indices[j]==j 的最长前缀是"原样未搬"的 token（sink 段）；首个 keep_indices[j]!=j
        起，后续 token 都被前移、所在块被改写。下标 j=压缩后紧凑布局的新位，
        keep_indices[j]=压缩前原物理槽。num_identity//block_size 即首个含被改写 token 的块；
        全 identity（无 token 前移）则无改写块。
        """
        num_keep = len(keep_indices)
        new_num_blocks = (num_keep + self.block_size - 1) // self.block_size
        num_identity = 0
        while num_identity < num_keep and keep_indices[num_identity] == num_identity:
            num_identity += 1
        first_rewritten = new_num_blocks if num_identity == num_keep else num_identity // self.block_size
        return [seq.block_table[b] for b in range(first_rewritten, new_num_blocks)]

    def can_evict(self, seq: Sequence, keep_indices: list[int]) -> bool:
        """gather **之前**的门控（设计 D5）：被改写块须 ref_count==1（独占）才可压缩。

        必须在 `ModelRunner.compact_kv` 之前调用——一旦 gather 改了数据就无法回滚，
        evict 里的同名断言只是事后兜底。共享块（典型是被前缀复用的 sink）不在改写集内，
        不影响判定。
        """
        return all(self.blocks[bid].ref_count == 1 for bid in self._rewritten_block_ids(seq, keep_indices))

    def evict(self, seq: Sequence, keep_indices: list[int]):
        """回收序列尾部不再需要的整块（v0 压缩的块账部分，token 级通用、含零散）。

        前置：须在 `ModelRunner.compact_kv` 完成物理 gather（把幸存 KV 紧凑到前部 slot）
        之后调用。gather 后空出的一定是尾部整块，无 sub-block 碎片回收。

        keep_indices 索引压缩前**物理** cache 的 KV 位置（[0, 物理KV数)，不含尚未写入
        的 pending last_token）。

        时序约定（⏸ 需在 GPU 环境核对）：压缩在 step 的 postprocess 之后、下个
        prepare_decode 之前触发。此时 last_token 已 append 进 token_ids 但其 KV 尚未写入
        cache → 物理 KV 数 = num_tokens-1-旧num_dropped_kv = seq.num_kv-1。压缩后下个
        decode step 会把 pending token 写到第 len(keep_indices) 个 slot，使
        num_kv(=context_lens) = len(keep_indices)+1，故
        num_dropped_kv = num_tokens - len(keep_indices) - 1。

        前缀缓存隔离（step7 / 设计 D5）：gather 把幸存 KV 前移，被改写内容的前部块的
        token_ids/hash 随之失效——若仍留在 hash_to_block_id，其他序列前缀命中会读到脏数据。
        故对被改写块统一注销 hash、置 hash=-1；并断言这些块 ref_count==1（独占），
        被共享则该序列不应压缩（调用方 step8 触发须先保证，否则此处断言拦截）。
        keep_indices 开头与原位重合的 sink 段（keep_indices[j]==j）未被搬动 → 其块内容
        不变、hash 仍有效，保留以继续供他序列前缀复用。
        """
        num_keep = len(keep_indices)
        new_num_blocks = (num_keep + self.block_size - 1) // self.block_size
        assert 0 < new_num_blocks <= len(seq.block_table)
        # —— D5：注销被 gather 改写的前部块的前缀缓存 hash ——
        rewritten_blocks = self._rewritten_block_ids(seq, keep_indices)
        # 先统一断言，避免部分注销后失败留下不一致状态
        for block_id in rewritten_blocks:
            ref = self.blocks[block_id].ref_count
            assert ref == 1, f"被改写块 {block_id} 被共享(ref_count={ref})，该序列不应压缩"
        for block_id in rewritten_blocks:
            block = self.blocks[block_id]
            if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
            block.hash = -1
            block.token_ids = []
        # 释放尾部整块（这些块在 gather 后只剩被丢弃的 KV）
        for block_id in seq.block_table[new_num_blocks:]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        del seq.block_table[new_num_blocks:]
        # num_kv 解耦：见上方时序约定的 -1（pending last_token 尚未入 cache）
        seq.num_dropped_kv = seq.num_tokens - num_keep - 1
        # D5：标记已压缩，后续 hash_blocks 跳过该序列（块内容已非干净前缀）
        seq.kv_compressed = True

    # KV 压缩(v0)：开新块的判断改用 num_kv（紧凑后 cache 占用），而非逻辑 token 数。
    # 新 token 落在紧凑布局的第 num_kv-1 个 slot；num_kv % block_size == 1 即它起一个新块。
    # 压缩关闭时 num_kv==len(seq)，与原行为完全一致。
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (seq.num_kv % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if seq.num_kv % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # D5：已压缩序列的块内容已非干净前缀，跳过注册，避免他序列命中读脏数据
        if seq.kv_compressed:
            return
        # 整数除法的语义是”向下取整”,end 自然只数已写满的块,半满的尾块对应的余数被舍去,不进入循环范围
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
