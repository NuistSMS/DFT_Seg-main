import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from net.agent_attention import AgentAttentionBlock


class TextAdapter(nn.Module):
    def __init__(self, embed_dim=768, bottleneck_dim=192, dropout=0.1, enabled=True):
        super().__init__()
        self.enabled = enabled
        if enabled:
            self.down = nn.Linear(embed_dim, bottleneck_dim)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.up = nn.Linear(bottleneck_dim, embed_dim)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if not self.enabled:
            return x
        return self.norm(x + self.up(self.dropout(self.act(self.down(x)))))


class TextEncoderWithAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        text_cfg = cfg["TEXT"]
        os.environ.setdefault("HF_MODULES_CACHE", str(Path(".hf_modules_cache").resolve()))
        self.encoder = AutoModel.from_pretrained(text_cfg["bert_type"], output_hidden_states=True, trust_remote_code=True)
        if bool(text_cfg.get("freeze_text_encoder", True)):
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.adapter = TextAdapter(
            embed_dim=int(cfg["MODEL"].get("text_dim", 768)),
            bottleneck_dim=int(text_cfg.get("adapter_bottleneck_dim", 192)),
            dropout=float(text_cfg.get("adapter_dropout", 0.1)),
            enabled=bool(text_cfg.get("adapter_enabled", True)),
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output["hidden_states"][-1]
        hidden = self.adapter(hidden)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden, pooled


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Encoder3D(nn.Module):
    def __init__(self, in_channels, base):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base * 4, base * 8)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        return [e1, e2, e3, b]


class TextCrossFuse3D(nn.Module):
    def __init__(self, channels, text_dim=768, num_heads=4, dropout=0.1):
        super().__init__()
        self.text_update_norm = nn.LayerNorm(text_dim)
        self.text_update_mlp = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Linear(text_dim * 2, text_dim),
        )
        self.text_project = nn.Linear(text_dim, channels)
        self.vis_norm = nn.LayerNorm(channels)
        self.txt_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, volume, text_emb):
        b, c, d, h, w = volume.shape
        txt = text_emb + self.text_update_mlp(self.text_update_norm(text_emb))
        txt = self.txt_norm(self.text_project(txt))
        vis = volume.flatten(2).transpose(1, 2)
        fused, _ = self.cross_attn(query=self.vis_norm(vis), key=txt, value=txt)
        fused = self.out_norm(vis + fused)
        return fused.transpose(1, 2).reshape(b, c, d, h, w)


class DecoderStage3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, text_dim=768, num_heads=4):
        super().__init__()
        self.text_fuse = TextCrossFuse3D(in_channels, text_dim=text_dim, num_heads=num_heads)
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, text_emb):
        x = self.text_fuse(x, text_emb)
        x = self.up(x)
        x = _match_size(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class BraTS3DSegModel(nn.Module):
    """3D DFTSeg-style model: low/high wavelet branches + text-agent fusion."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["MODEL"]
        in_channels = int(model_cfg["in_channels"])
        num_classes = int(model_cfg["num_classes"])
        base = int(model_cfg.get("base_channels", 16))
        text_dim = int(model_cfg.get("text_dim", 768))

        heads = int(model_cfg.get("num_heads", 4))
        self.text_encoder = TextEncoderWithAdapter(cfg)
        self.encoder_l = Encoder3D(in_channels, base)
        self.encoder_h = Encoder3D(in_channels, base)
        self.text_agent_project = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, base * 8),
            nn.GELU(),
            nn.Linear(base * 8, base * 8),
        )
        self.high_freq_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.fusion_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.fusion_agent_attn = AgentAttentionBlock(dim=base * 8, window_size=(1, 1, 1), num_heads=heads, agent_num=0)
        self.decoder3 = DecoderStage3D(base * 8, base * 4, base * 4, text_dim=text_dim, num_heads=heads)
        self.decoder2 = DecoderStage3D(base * 4, base * 2, base * 2, text_dim=text_dim, num_heads=heads)
        self.decoder1 = DecoderStage3D(base * 2, base, base, text_dim=text_dim, num_heads=heads)
        self.out = nn.Conv3d(base, num_classes, kernel_size=1)

    def parameter_groups(self, cfg):
        train_cfg = cfg["TRAIN"]
        return [
            {"params": [p for p in self.text_encoder.encoder.parameters() if p.requires_grad], "lr": float(train_cfg["lr_text_encoder"]), "name": "text_encoder"},
            {"params": self.text_encoder.adapter.parameters(), "lr": float(train_cfg["lr_text_adapter"]), "name": "text_adapter"},
            {
                "params": list(self.encoder_l.parameters())
                + list(self.encoder_h.parameters())
                + list(self.text_agent_project.parameters())
                + list(self.fusion_agent_attn.parameters())
                + list(self.decoder3.parameters())
                + list(self.decoder2.parameters())
                + list(self.decoder1.parameters())
                + list(self.out.parameters())
                + [self.high_freq_scale, self.fusion_scale],
                "lr": float(train_cfg["lr_image"]),
                "name": "wavelet_text_fusion_segmentation_3d",
            },
        ]

    def forward(self, image_low, image_high, text):
        text_emb, _ = self.text_encoder(text["input_ids"], text["attention_mask"])
        feat_l = self.encoder_l(image_low)
        feat_h = self.encoder_h(image_high)

        b_l = feat_l[-1]
        b_h = feat_h[-1]
        b, c, d, h, w = b_l.shape
        flat_l = b_l.flatten(2).transpose(1, 2)
        flat_h = b_h.flatten(2).transpose(1, 2)
        agent_tokens = self.text_agent_project(text_emb)
        fused_flat = self.fusion_agent_attn(x=flat_l, attn=flat_h, agent_input=agent_tokens)
        final_bottleneck = flat_l * self.high_freq_scale + fused_flat * self.fusion_scale
        x = final_bottleneck.transpose(1, 2).reshape(b, c, d, h, w)

        x = self.decoder3(x, feat_l[2], text_emb)
        x = self.decoder2(x, feat_l[1], text_emb)
        x = self.decoder1(x, feat_l[0], text_emb)
        logits = self.out(x)
        return logits


def _match_size(x, ref):
    if x.shape[2:] == ref.shape[2:]:
        return x
    return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)
