import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1: return x
    return x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim).reshape(bs, n_kv_heads * n_rep, slen,
                                                                                     head_dim)


class CrossMultiheadDiffAttn(nn.Module):
    # (逻辑完全不变)
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.n_rep = 1
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5)

    def forward(self, query_x, kv_x):
        bsz, tgt_len, _ = query_x.size()
        src_len = kv_x.size(1)

        q = self.q_proj(query_x)
        k = self.k_proj(kv_x)
        v = self.v_proj(kv_x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = repeat_kv(k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim).transpose(1, 2), self.n_rep)
        v = repeat_kv(v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim).transpose(1, 2), self.n_rep)

        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        return self.out_proj(attn)


class MSM(nn.Module):

    def __init__(self, dim=768, num_heads=8, dropout=0.1, depth=1):
        super().__init__()
        self.text_norm = RMSNorm(dim)
        text_diff_heads = max(1, num_heads // 2)
        self.text_self_diff_attn = CrossMultiheadDiffAttn(embed_dim=dim, depth=depth, num_heads=text_diff_heads)

        self.img_norm = RMSNorm(dim)
        diff_heads = max(1, num_heads // 2)
        self.cross_norm = RMSNorm(dim)
        self.cross_diff_attn = CrossMultiheadDiffAttn(embed_dim=dim, depth=depth, num_heads=diff_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.ff_norm = RMSNorm(dim)

    def forward(self, text_emb, img_low_freq):
        text_res = text_emb
        text_emb = self.text_norm(text_emb)
        text_feat = self.text_self_diff_attn(query_x=text_emb, kv_x=text_emb)
        text_feat = text_feat + text_res

        img_kv = self.img_norm(img_low_freq)
        agent_a = self.cross_diff_attn(query_x=text_feat, kv_x=img_kv)
        agent_a = self.cross_norm(agent_a + text_feat)
        agent_a = self.ff_norm(agent_a + self.feed_forward(agent_a))

        return agent_a


class PGMSM(nn.Module):

    def __init__(self, in_channels_list=[192, 384], embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            ) if c != embed_dim else nn.Identity()
            for c in in_channels_list
        ])

        self.text_updaters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in in_channels_list
        ])

        self.interactions = nn.ModuleList([
            MSM(dim=embed_dim, num_heads=num_heads, dropout=dropout, depth=1)
            for _ in in_channels_list
        ])

        self.agent_fusion = nn.Sequential(
            nn.Linear(embed_dim * len(in_channels_list), embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )

        self.agent_self_attn = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, text_emb, img_feats):
        agents = []
        for proj, text_upd, interact, feat in zip(self.projections, self.text_updaters, self.interactions, img_feats):
            feat_proj = proj(feat)
            B, C, H, W = feat_proj.shape
            feat_flat = feat_proj.view(B, C, -1).transpose(1, 2)

            scale_specific_text = text_emb + text_upd(text_emb)
            agent_i = interact(text_emb=scale_specific_text, img_low_freq=feat_flat)
            agents.append(agent_i)

        agents_cat = torch.cat(agents, dim=-1)
        final_agent = self.agent_fusion(agents_cat)
        final_agent = self.agent_self_attn(final_agent)
        return final_agent