import torch
import torch.nn as nn
from einops import rearrange, repeat
from net.decoder import Decoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoModel
from timm.models.layers import to_2tuple

from net.agent_attention import AgentAttentionBlock
from net.fusion import HierarchicalTextAgentGenerator


# ==========================================
# 💡 [新增] 全局文本适配器 (TextAdapterQKV)
# 作用：将 BERT 输出的文本投影到解码器专属的新空间
# ==========================================
class TextAdapterQKV(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 核心：利用 Self-Attention 生成 QKV 并投影
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # x shape: [B, L, C]
        res = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(query=x_norm, key=x_norm, value=x_norm)
        # Q, K, V 都是文本自身，计算自注意力
        x_attn, _ = self.attn(query=x_norm, key=x_norm, value=x_norm)
        x = res + x_attn  # 残差连接

        res = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = res + x_mlp
        return x


class TextDomainAdapter(nn.Module):
    def __init__(self, embed_dim=768, bottleneck_dim=192, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.adapter(x)


# ==========================================
# 💡 用于提取先验掩码特征的 ResNet 编码器
# ==========================================
class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                            return_dict=True)
        last_hidden_states = torch.stack(
            [output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]])
        embed = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1)
        embed = self.project_head(embed)
        return {'feature': output['hidden_states'], 'project': embed}


class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Linear(768, project_dim)

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature": output['hidden_states'], "project": project}


class SegModel(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=512, max_text_len=24):
        super(SegModel, self).__init__()

        self.encoder_h = VisionModel(vision_type, project_dim)
        self.encoder_l = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.text_domain_adapter = TextDomainAdapter(embed_dim=768)

        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]

        self.hierarchical_agent_generator = HierarchicalTextAgentGenerator(
            in_channels_list=[feature_dim[3], feature_dim[2], feature_dim[1], feature_dim[0]],
            embed_dim=768,
            num_heads=8
        )

        self.fusion_agent_attn = AgentAttentionBlock(dim=768, window_size=to_2tuple(7), num_heads=8, agent_num=49)

        self.high_freq_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.fusion_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 💡 [新增] 实例化解码器的全局 QKV 适配器
        self.decoder_text_adapter = TextAdapterQKV(embed_dim=768)

        self.decoder16 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24,
                                 input_text_len=max_text_len)
        self.decoder8 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12,
                                input_text_len=max_text_len)
        self.decoder4 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9,
                                input_text_len=max_text_len)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data, train_mask=None):
        image2, image, text = data[:3]

        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
            image2 = repeat(image2, 'b 1 h w -> b c h w', c=3)

        text_out = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_emb = self.text_domain_adapter(text_out['feature'][-1])

        out_l = self.encoder_h(image)
        feat_l_list = out_l['feature']
        feat_l_bottleneck = feat_l_list[-1]

        out_h = self.encoder_l(image2)
        feat_h_list = out_h['feature']
        feat_h_bottleneck = feat_h_list[-1]

        flat_l = rearrange(feat_l_bottleneck, 'b c h w -> b (h w) c')
        flat_h = rearrange(feat_h_bottleneck, 'b c h w -> b (h w) c')

        agent_tokens = self.hierarchical_agent_generator(text_emb=text_emb, img_feats=feat_l_list[-4:])

        fused_feat = self.fusion_agent_attn(x=flat_l, attn=flat_h, agent_input=agent_tokens)

        final_bottleneck_flat = flat_l * self.high_freq_scale + fused_feat * self.fusion_scale

        skips = feat_l_list[-4:-1]
        skips = [rearrange(item, 'b c h w -> b (h w) c') if len(item.shape) == 4 else item for item in skips]

        # 💡 [修改] 在送入 Decoder 之前，文本先经过 QKV 进行全局重投影
        decoder_text_emb = self.decoder_text_adapter(text_emb)

        # 💡 [修改] 传递 decoder_text_emb 给每一层 Decoder
        os16 = self.decoder16(final_bottleneck_flat, skips[2], decoder_text_emb)
        os8 = self.decoder8(os16, skips[1], decoder_text_emb)
        os4 = self.decoder4(os8, skips[0], decoder_text_emb)

        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()

        return out, out, train_mask
