import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- 1. RMSNorm (均方根归一化) ---
# 相比传统的 LayerNorm，RMSNorm 移除了计算均值的步骤（即不进行中心化），
# 只计算均方根来进行缩放。这样计算效率更高，且在多模态和 LLM 中表现很好。
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # 防止除以 0 的极小值
        # 可学习的缩放权重参数，初始化为全 1，维度与输入特征维度一致
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算特征维度（最后一个维度）的均方根的倒数 (rsqrt = 1 / sqrt(x^2.mean + eps))
        # 然后将原张量乘以这个倒数完成归一化
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 强制将输入转为 fp32 进行归一化计算，防止在混合精度训练 (fp16/bf16) 时发生数值溢出
        output = self._norm(x.float()).type_as(x)
        # 乘以可学习的缩放参数
        return output * self.weight


# --- 2. 差分注意力辅助函数 ---
def lambda_init_fn(depth):
    # 根据 Transformer 层的深度初始化 lambda 参数。
    # 随着深度增加，lambda 的初始值会逐渐减小，用于稳定深层网络的差分注意力训练。
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 用于支持 Grouped-Query Attention (GQA) 或 Multi-Query Attention (MQA)
    # 如果 n_rep == 1，说明是标准的多头注意力，直接返回原始张量
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1: return x
    # 如果 n_rep > 1，则将 K 和 V 的头数复制扩展，以匹配 Q 的头数
    return x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim).reshape(bs, n_kv_heads * n_rep, slen,
                                                                                     head_dim)


# --- 3. 跨模态差分注意力核心 (Cross Multihead Differential Attention) ---
class CrossMultiheadDiffAttn(nn.Module):
    """专门为跨模态定制的交叉差分注意力"""

    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        # 注意：这里的 num_heads 实际上是“差分对”的数量。
        # 外部传入的 num_heads 已经除以 2 了，因为我们需要两组注意力来做减法(差分)
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.n_rep = 1  # 默认使用标准 MHA，不进行 KV 复制扩展

        # 每个头的维度。因为要在内部生成两倍的头(用于差分)，所以再除以 2
        self.head_dim = embed_dim // num_heads // 2
        # 注意力计算时的缩放因子 (1 / sqrt(d_k))
        self.scaling = self.head_dim ** -0.5

        # 将输入映射为 Q, K, V。因为是偏置设置为 False，不使用 bias
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # 最终输出的线性映射层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 初始化控制差分力度的 lambda 基准值
        self.lambda_init = lambda_init_fn(depth)

        # 定义 4 个可学习的向量，用于动态生成 lambda_1 和 lambda_2
        # normal_(mean=0, std=0.1) 进行高斯分布初始化
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        # 使用 RMSNorm 替换 LayerNorm。
        # 这里针对每个 Head 的内部进行归一化 (Headwise Normalization)，维度是 2 * head_dim
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(self, query_x, kv_x):
        # bsz: 批次大小 (Batch Size)
        # tgt_len: 目标序列长度 (比如文本的 token 数)
        bsz, tgt_len, _ = query_x.size()
        # src_len: 源序列长度 (比如图像特征的 patch 数)
        src_len = kv_x.size(1)

        # 1. 线性映射得到 Q, K, V
        q = self.q_proj(query_x)
        k = self.k_proj(kv_x)
        v = self.v_proj(kv_x)

        # 2. 将张量 Reshape 成多头机制的形状。
        # 注意这里的 2 * self.num_heads，我们将通道切分为双倍的头数，用于构建两个平行的注意力图
        # [B, tgt_len, 2*heads, head_dim]
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        # v 只需要 num_kv_heads 的数量，但每个头的维度是 2 * head_dim
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        # 3. 调整维度顺序，方便后续进行矩阵乘法 (B, Heads, Seq_len, Head_dim)
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)

        # Q 乘以缩放因子
        q *= self.scaling

        # 4. 计算注意力权重: Q 和 K 的转置相乘
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        # 强制在 fp32 下计算 softmax，防止极端值导致 nan，然后再转回原本的数据类型
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        # 5. 核心差分逻辑：计算动态差分系数 lambda
        # 通过 q1, k1 和 q2, k2 计算出当前特征的具体 lambda 值，并结合深度的初始值 lambda_init
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 将注意力的头拆分成两组 (正向注意力图 和 负向/消除噪声注意力图)
        # 形状变为 [B, heads, 2, tgt_len, src_len]
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)

        # 核心减法：图 0 (主注意力) 减去 lambda加权后的图 1 (背景/冗余噪声注意力)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        # 6. 将差分后的权重与 V 相乘，聚合特征
        attn = torch.matmul(attn_weights, v)

        # 7. 对聚合后的特征进行 Head-wise 的 RMSNorm 归一化
        attn = self.subln(attn)
        # 乘以缩放系数 (1 - lambda_init)，保持数值范围稳定
        attn = attn * (1 - self.lambda_init)

        # 8. 还原维度结构：合并所有的头 [B, Heads, L, D] -> [B, L, Heads, D] -> [B, L, C]
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        # 9. 通过最后的线性层映射输出
        return self.out_proj(attn)


# --- 4. 文本与低频图像交互模块 ---
class TextLowFreqInteraction(nn.Module):
    """
    实现: 图像低频特征 (Low Freq) -> 文本自注意力特征 (Text) 的交叉融合，最终输出 Agent A
    """

    def __init__(self, dim=768, num_heads=8, dropout=0.1, depth=1):
        super().__init__()

        # 1. 文本自注意力分支 (Text Self-Attention)
        self.text_norm = RMSNorm(dim)
        # 适配差分注意力：将原本的头数减半传入
        text_diff_heads = max(1, num_heads // 2)
        self.text_self_diff_attn = CrossMultiheadDiffAttn(embed_dim=dim, depth=depth, num_heads=text_diff_heads)

        # 2. 图像特征入口的归一化
        self.img_norm = RMSNorm(dim)

        # 3. 交叉注意力分支 (Cross Attention)
        diff_heads = max(1, num_heads // 2)
        self.cross_norm = RMSNorm(dim)
        self.cross_diff_attn = CrossMultiheadDiffAttn(embed_dim=dim, depth=depth, num_heads=diff_heads)

        # 4. 前馈神经网络 (Feed Forward Network, FFN)，用于进一步提取非线性特征
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.ff_norm = RMSNorm(dim)

    def forward(self, text_emb, img_low_freq):
        """
        参数:
        text_emb: 文本特征，形状 [B, L_text, C] (例如 [B, 24, 768])
        img_low_freq: 图像低频特征，形状 [B, N_img, C] (例如 [B, 49, 768])
        """
        # --- 步骤 1: 文本的自注意力交互 ---
        text_res = text_emb  # 保存残差
        text_emb = self.text_norm(text_emb)  # 归一化
        # Q=text, K=text, V=text (纯文本内部分析)
        text_feat = self.text_self_diff_attn(query_x=text_emb, kv_x=text_emb)
        # 添加残差连接 (Residual Connection)
        text_feat = text_feat + text_res

        # --- 步骤 2: 图像低频特征归一化，准备作为 K 和 V ---
        img_kv = self.img_norm(img_low_freq)

        # --- 步骤 3: 跨模态差分注意力 ---
        # Query = 文本特征 (text_feat)，Key/Value = 图像低频特征 (img_kv)
        # 也就是让文本去图像中寻找相关的高亮区域
        agent_a = self.cross_diff_attn(query_x=text_feat, kv_x=img_kv)

        # 再次添加残差连接 (加上输入的文本特征) 并归一化
        agent_a = self.cross_norm(agent_a + text_feat)

        # --- 步骤 4: 通过 FFN 层进行通道混淆与特征强化 ---
        agent_a = self.ff_norm(agent_a + self.feed_forward(agent_a))

        return agent_a


# --- 5. 多尺度 Agent 生成器 ---
class HierarchicalTextAgentGenerator(nn.Module):
    def __init__(self, in_channels_list=[192, 384], embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        # 1. 图像特征投影层 (将不同尺度的图像通道统一映射到 embed_dim=768)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),  # 1x1 卷积降/升维
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            ) if c != embed_dim else nn.Identity()  # 如果通道已经一致则跳过 (Identity)
            for c in in_channels_list
        ])

        # 2. 文本特征更新器 (为每个尺度赋予专属的文本先验偏好)
        self.text_updaters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in in_channels_list
        ])

        # 3. 实例化刚才定义的差分交互模块 (每个尺度对应一个)
        self.interactions = nn.ModuleList([
            TextLowFreqInteraction(dim=embed_dim, num_heads=num_heads, dropout=dropout, depth=1)
            for _ in in_channels_list
        ])

        # 4. Agent 融合网络 (将多尺度生成的 Agent 融合为一个标准的输出 Agent)
        self.agent_fusion = nn.Sequential(
            nn.Linear(embed_dim * len(in_channels_list), embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        # 💡 [新增核心改进] 5. Agent 内部共识层 (Standard Self-Attention)
        # 用途：让融合后的 24 个多尺度 Agent Token 互相开会交换情报，消除尺度冲突，建立全局共识。
        self.agent_self_attn = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, text_emb, img_feats):
        agents = []
        # 遍历每一种尺度的特征 (比如 layer3 和 layer4)
        for proj, text_upd, interact, feat in zip(self.projections, self.text_updaters, self.interactions, img_feats):
            # 1. 将该层图像特征映射到统一的通道数 (embed_dim)
            feat_proj = proj(feat)
            B, C, H, W = feat_proj.shape

            # 2. 展平图像的空间维度，适配 Transformer 的输入格式
            # [B, C, H, W] -> [B, C, H*W] -> 转置为 [B, H*W, C] (即 [批次, 序列长度, 特征维度])
            feat_flat = feat_proj.view(B, C, -1).transpose(1, 2)

            # 3. 计算特定尺度下的专属文本特征 (基础文本 + 更新后的残差)
            scale_specific_text = text_emb + text_upd(text_emb)

            # 4. 让该尺度的文本与该尺度的图像低频特征进行【差分交互】
            agent_i = interact(text_emb=scale_specific_text, img_low_freq=feat_flat)

            # 收集该尺度生成的 Agent
            agents.append(agent_i)

        # 5. 在特征维度 (dim=-1) 拼接所有尺度的 Agent
        # 例如两个尺度，原本是 [B, L, 768]，拼接后变成 [B, L, 1536]
        agents_cat = torch.cat(agents, dim=-1)

        # 6. 经过融合网络，将拼接后的特征重新压缩回标准维度 [B, L, 768]
        final_agent = self.agent_fusion(agents_cat)

        # 💡 [新增核心改进] 7. 共识建立阶段 (Consensus Building)
        # 让 24 个 Agent 经过普通的 Self-Attention，统一多尺度提取出的情报，输出高质量的代理特征
        final_agent = self.agent_self_attn(final_agent)

        return final_agent