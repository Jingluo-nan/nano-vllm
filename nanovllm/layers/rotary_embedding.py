from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    '''
    x 是 query 或 key,形状 (token数, head数, 128)。沿最后一维对半切:


    x: [d0 d1 d2 ... d63 | d64 d65 ... d127]
            x1 (前64)          x2 (后64)
    x1 = 前 64 维,x2 = 后 64 维,形状都是 (..., 64)
    配对方式:x1[i] 和 x2[i] 组成第 i 对 (x1[i], x2[i]),共 64 对
    .float():升到 fp32 算旋转,避免 fp16 精度损失(最后再转回去)
    注意:这里是「前半 vs 后半」配对(GPT-NeoX 风格),和原始 RoPE 论文「相邻两维 d0-d1 配对」是等价的另一种排布,nano-vllm/HF 用的是这种。
    '''
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    '''
    平面上一个点 (a, b) 绕原点转角度 θ,新坐标是:
    a' = a·cosθ − b·sinθ
    b' = a·sinθ + b·cosθ

    x1的形状是(token数, head_size, 64)
    cos的形状是(token数, 1, 64)
    逐元素相乘时，cos的第二维度会做广播
    '''
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size # 128维度全部参与旋转

        # 64根指针各自的转速（频率）
        # 第i个 = 1 / base ^ (2i / d) ，i越大转的越慢
        # inv_freq : (64,) 
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # max_position_embeddings 是预计算 cos/sin表覆盖的最大位置 （40960）
        # t        : (40960,)   
        t = torch.arange(max_position_embeddings, dtype=torch.float)
       
        # ③ 外积 = 一张「位置 × 频率」乘法表:
        #    freqs[位置, 频率] = 位置 × 该频率
        #                     = 这个位置这根指针转过的角度
        # freqs    : (40960, 64)     ← 位置 × 频率 的表
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

         # ④ 旋转要用 cosθ、sinθ,先都算好
        cos = freqs.cos()
        sin = freqs.sin()

        # ⑤ 每行拼成 [64 个 cos | 64 个 sin] = 128 个数
        #    unsqueeze_(1) 插一个长度 1 的维度,
        #      forward 时会广播(自动复制)到每个 head
        '''
        (40960, 128)  ──unsqueeze_(1)──►  (40960, 1, 128)
        位置  数值                        位置  ?  数值
        中间多出来的这个长度 1 的维度,是给 注意力头(head) 留的广播位。
        为什么要插它? query/key 的形状是 (token数, head数, head_dim),带 head 维。
        而 RoPE 的旋转角度只跟位置有关、和哪个 head 无关——所有 head 用同一套 cos/sin。所以:
        cos_sin 查表后:  (token数,   1,    128)
        query:          (token数, head数, 128)
                                    ↑
                    长度 1 → 广播时自动“复制”到每个 head
        '''
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # register_buffer:把 cache 挂在模型上、跟随模型加载到 GPU,
        #   但它不需要训练的权重,只是一块「只读数据」
        # persistent=False:不写进权重文件,用配置随时能重算
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 形状为 (40960, 1, 128)
        # 这个查表操作会启动一个cuda kernel
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

# @lru_cache(1) 把唯一一次构造结果缓存住,之后相同入参直接返回缓存的那个实例。
# 28层共用同一个RotaryEmbedding实际
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
