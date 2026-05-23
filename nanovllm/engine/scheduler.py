from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    '''
    先尝试调度 prefill;只要 prefill 调出了哪怕一条序列,
    本 step 就只做 prefill,RUNNING 队列里的 decode 全部等下一 step;
    只有 prefill 一条都没调出时,本 step 才做 decode
    '''
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            # 本 step 剩余的 token 预算
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # 这条 seq 第一次被调度,block_table 还是空
            if not seq.block_table:
                # 这里看是否可以allocate，不是只看一个chunk需要的block
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                #  总 prompt token 减去 prefix cache 已经覆盖的部分
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 这条 seq 之前已经被调度过,但 token 预算不够,只 prefill 了一部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # remaining 是本 step 还能算多少 token、num_tokens 是这条 seq 还要算多少 token
            # 本step装不下 seq
            # chunkprefill只对队首生效，注意  and scheduled_seqs 这个条件
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING # RUNNING一定是decode阶段
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode，  因为decode一次只操作一个token，所以很难到达max_num_batched_tokens的限制
        # 更容易到达max_num_seqs的限制
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 抢队尾
                    self.preempt(self.running.pop())
                else:
                    # 自抢占
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # may_append 可能会分配block
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
