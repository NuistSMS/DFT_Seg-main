import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

class DFLM(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 agent_num=1, if_dwc=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Process A - Low Freq (Q) to Query Agent (K)
        self.q_lf = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_ag = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.mlp_attn_a = nn.Sequential(nn.Linear(2, 1))

        # Process B - Agent (Q) to Query High Freq (K)
        self.q_ag = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.k_hf = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v_hf = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_attn_b = nn.Sequential(nn.Linear(2, 1))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.if_dwc = if_dwc
        if self.if_dwc:
            self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, attn, agent_input):
        B, N, C = x.shape
        _, N_agent, _ = agent_input.shape
        head_dim = C // self.num_heads


        q_lf = self.q_lf(x).reshape(B, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q_lf1, q_lf2 = q_lf[0], q_lf[1]

        k_ag = self.k_ag(agent_input).reshape(B, N_agent, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k_ag1, k_ag2 = k_ag[0], k_ag[1]

        attn_a1 = (q_lf1 @ k_ag1.transpose(-2, -1)) * self.scale
        attn_a2 = (q_lf2 @ k_ag2.transpose(-2, -1)) * self.scale

        attn_a_cat = torch.stack([attn_a1, attn_a2], dim=-1)
        attn_agent = self.mlp_attn_a(attn_a_cat).squeeze(-1)
        attn_agent = attn_agent.softmax(dim=-1)
        attn_agent = self.attn_drop(attn_agent)


        q_ag = self.q_ag(agent_input).reshape(B, N_agent, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q_ag1, q_ag2 = q_ag[0], q_ag[1]

        k_hf = self.k_hf(attn).reshape(B, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k_hf1, k_hf2 = k_hf[0], k_hf[1]

        v_hf = self.v_hf(attn).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn_b1 = (q_ag1 @ k_hf1.transpose(-2, -1)) * self.scale
        attn_b2 = (q_ag2 @ k_hf2.transpose(-2, -1)) * self.scale

        attn_b_cat = torch.stack([attn_b1, attn_b2], dim=-1)
        attn_v = self.mlp_attn_b(attn_b_cat).squeeze(-1)
        attn_v = attn_v.softmax(dim=-1)
        attn_v = self.attn_drop(attn_v)


        x_s = (attn_v @ v_hf)
        x_out = (attn_agent @ x_s)
        x_out = x_out.transpose(1, 2).reshape(B, N, C)

        if self.if_dwc:
            x_out = x_out + self.dwc(x_out.permute(0, 2, 1).reshape(B, C, int(N ** 0.5), int(N ** 0.5))).flatten(
                2).transpose(1, 2)

        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out