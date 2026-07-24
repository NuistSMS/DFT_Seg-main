import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import numpy as np

class AgentAttentionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 agent_num=1, if_dwc=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 缩放因子：1 / sqrt(d)
        self.scale = qk_scale or head_dim ** -0.5

        # ========================================================
        # 核心创新 1: Process A - Low Freq (Q) 去 Query Agent (K)
        # 初始两个 Q1, Q2 和 两个 K1, K2
        # ========================================================
        # 输出维度为 dim * 2，后续切分为双分支
        self.q_lf = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_ag = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # 将两个注意力图 (Q1K1^T 和 Q2K2^T) 拼起来后做 MLP 降维
        self.mlp_attn_a = nn.Sequential(
            nn.Linear(2, 1)
        )

        # ========================================================
        # 核心创新 2: Process B - Agent (Q) 去 Query High Freq (K)
        # 初始两个 Q1, Q2 和 两个 K1, K2
        # ========================================================
        self.q_ag = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_hf = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # V 只需要一个原本的维度即可
        self.v_hf = nn.Linear(dim, dim, bias=qkv_bias)

        # 将两个注意力图 (Q1K1^T 和 Q2K2^T) 拼起来后做 MLP 降维
        self.mlp_attn_b = nn.Sequential(
            nn.Linear(2, 1)
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.if_dwc = if_dwc
        if self.if_dwc:
            self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, attn, agent_input):
        """
        x: Low Frequency Features (作为 Query，代表低频主干结构) [B, N, C]
        attn: High Frequency Features (作为 Key/Value，提供高频细节检索) [B, N, C]
        agent_input: 外部传入的 Agent (多尺度先验加权后的文本) [B, N_agent, C]
        """
        B, N, C = x.shape
        _, N_agent, _ = agent_input.shape  # N_agent 对应你的 24 (文本 token 数)
        head_dim = C // self.num_heads

        # =========================================================
        # 阶段 1: Low-Freq -> Agent
        # 低频特征(Q) 去 文本Agent(K) 中寻找匹配
        # =========================================================
        # 1. 生成双重 Q_lf 和 双重 K_ag (通过 reshape 和 permute 拆成两份)
        q_lf = self.q_lf(x).reshape(B, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q_lf1, q_lf2 = q_lf[0], q_lf[1]  # shape: [B, num_heads, N, head_dim]

        k_ag = self.k_ag(agent_input).reshape(B, N_agent, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k_ag1, k_ag2 = k_ag[0], k_ag[1]  # shape: [B, num_heads, N_agent, head_dim]

        # 2. 计算并行的两个 Attention Map (分别除以根号d)
        attn_a1 = (q_lf1 @ k_ag1.transpose(-2, -1)) * self.scale
        attn_a2 = (q_lf2 @ k_ag2.transpose(-2, -1)) * self.scale

        # 3. 拼起来做一个 MLP 到原来的维度，再做 softmax
        attn_a_cat = torch.stack([attn_a1, attn_a2], dim=-1)  # shape: [B, num_heads, N, N_agent, 2]
        attn_agent = self.mlp_attn_a(attn_a_cat).squeeze(-1)  # shape: [B, num_heads, N, N_agent]
        attn_agent = attn_agent.softmax(dim=-1)
        attn_agent = self.attn_drop(attn_agent)

        # =========================================================
        # 阶段 2: Agent -> High-Freq
        # 文本Agent(Q) 去 高频特征(K) 中检索高频纹理细节(V)
        # =========================================================
        # 1. 生成双重 Q_ag 和 双重 K_hf
        q_ag = self.q_ag(agent_input).reshape(B, N_agent, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q_ag1, q_ag2 = q_ag[0], q_ag[1]  # shape: [B, num_heads, N_agent, head_dim]

        k_hf = self.k_hf(attn).reshape(B, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k_hf1, k_hf2 = k_hf[0], k_hf[1]  # shape: [B, num_heads, N, head_dim]

        # 2. 生成 V_hf (正常维度)
        v_hf = self.v_hf(attn).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # 3. 计算并行的两个 Attention Map
        attn_b1 = (q_ag1 @ k_hf1.transpose(-2, -1)) * self.scale
        attn_b2 = (q_ag2 @ k_hf2.transpose(-2, -1)) * self.scale

        # 4. 拼起来做一个 MLP 融合，做 softmax
        attn_b_cat = torch.stack([attn_b1, attn_b2], dim=-1)  # shape: [B, num_heads, N_agent, N, 2]
        attn_v = self.mlp_attn_b(attn_b_cat).squeeze(-1)  # shape: [B, num_heads, N_agent, N]
        attn_v = attn_v.softmax(dim=-1)
        attn_v = self.attn_drop(attn_v)

        # =========================================================
        # 阶段 3: 特征聚合 (再乘 V，再投射回原空间)
        # =========================================================
        # 提取高频细节：Agent 聚合 High-Freq V
        x_s = (attn_v @ v_hf)  # shape: [B, num_heads, N_agent, head_dim]

        # 将包含高频细节的 Agent 特征，映射回 Low-Freq 空间
        x_out = (attn_agent @ x_s)  # shape: [B, num_heads, N, head_dim]

        # 恢复形状: [B, N, C]
        x_out = x_out.transpose(1, 2).reshape(B, N, C)

        # 可选的 Depth-wise Conv
        if self.if_dwc:
            x_out = x_out + self.dwc(x_out.permute(0, 2, 1).reshape(B, C, int(N ** 0.5), int(N ** 0.5))).flatten(
                2).transpose(1, 2)

        # 最终线性投影
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out