import torch
import torch.nn as nn
from einops import rearrange, repeat
from net.decoder import Decoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel
from timm.models.layers import to_2tuple
import torchvision.models as models


from net.dflm import DFLM
from net.pgmsm import PGMSM


class PSR_Global(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        res = x
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(query=x_norm, key=x_norm, value=x_norm)
        x = res + x_attn

        res = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = res + x_mlp
        return x


class PriorMaskGenerator(nn.Module):
    """对应论文中的先验特征提取网络 (原 PriorResNetEncoder)"""
    def __init__(self, resnet_type='resnet18', pretrained=True):
        super(PriorMaskGenerator, self).__init__()
        if resnet_type == 'resnet18':
            basemodel = models.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            basemodel = models.resnet34(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            basemodel = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet type")

        self.stem = nn.Sequential(basemodel.conv1, basemodel.bn1, basemodel.relu, basemodel.maxpool)
        self.layer1 = basemodel.layer1
        self.layer2 = basemodel.layer2
        self.layer3 = basemodel.layer3
        self.layer4 = basemodel.layer4

    def forward(self, x):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class BERTModel(nn.Module):
    # (保持不变)
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
    # (保持不变)
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True, trust_remote_code=True)
        self.project_head = nn.Linear(768, project_dim)

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature": output['hidden_states'], "project": project}


class DFTSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):
        super(DFTSeg, self).__init__()

        self.encoder_h = VisionModel(vision_type, project_dim)
        self.encoder_l = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.prior_encoder = PriorMaskGenerator('resnet50', pretrained=True)

        self.spatial_dim = [7, 14, 28, 56]
        feature_dim = [768, 384, 192, 96]

        self.w = nn.Parameter(torch.tensor([0.0]))


        self.pgmsm = PGMSM(
            in_channels_list=[256, 512, 1024, 2048],
            embed_dim=768,
            num_heads=8
        )

        self.dflm = DFLM(dim=768, window_size=to_2tuple(7), num_heads=8, agent_num=49)

        self.high_freq_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.fusion_scale = nn.Parameter(torch.ones(1) * 0.1)


        self.psr_global = PSR_Global(embed_dim=768)

        self.decoder16 = Decoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], 24)
        self.decoder8 = Decoder(feature_dim[1], feature_dim[2], self.spatial_dim[1], 12)
        self.decoder4 = Decoder(feature_dim[2], feature_dim[3], self.spatial_dim[2], 9)
        self.decoder1 = SubpixelUpsample(2, feature_dim[3], 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data, train_mask=None):
        image2, image, text, prior_mask = data

        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)
            image2 = repeat(image2, 'b 1 h w -> b c h w', c=3)
            prior_mask = prior_mask.expand_as(image)

        weight = torch.sigmoid(self.w)
        highlighted_image = image * prior_mask
        blended_image = weight * image + (1.0 - weight) * highlighted_image

        text_out = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_emb = text_out['feature'][-1]

        out_l = self.encoder_h(image)
        feat_l_list = out_l['feature']
        feat_l_bottleneck = feat_l_list[-1]

        out_h = self.encoder_l(image2)
        feat_h_list = out_h['feature']
        feat_h_bottleneck = feat_h_list[-1]

        prior_feats = self.prior_encoder(blended_image)

        flat_l = rearrange(feat_l_bottleneck, 'b c h w -> b (h w) c')
        flat_h = rearrange(feat_h_bottleneck, 'b c h w -> b (h w) c')

        # 使用重命名后的 PGMSM
        agent_tokens = self.pgmsm(text_emb=text_emb, img_feats=prior_feats)

        # 使用重命名后的 DFLM
        fused_feat = self.dflm(x=flat_l, attn=flat_h, agent_input=agent_tokens)

        final_bottleneck_flat = flat_l * self.high_freq_scale + fused_feat * self.fusion_scale

        skips = feat_l_list[-4:-1]
        skips = [rearrange(item, 'b c h w -> b (h w) c') if len(item.shape) == 4 else item for item in skips]

        # 使用重命名后的 PSR
        decoder_text_emb = self.psr_global(text_emb)

        os16 = self.decoder16(final_bottleneck_flat, skips[2], decoder_text_emb)
        os8 = self.decoder8(os16, skips[1], decoder_text_emb)
        os4 = self.decoder4(os8, skips[0], decoder_text_emb)

        os4 = rearrange(os4, 'B (H W) C -> B C H W', H=self.spatial_dim[-1], W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)
        out = self.out(os1).sigmoid()

        return out, out, train_mask