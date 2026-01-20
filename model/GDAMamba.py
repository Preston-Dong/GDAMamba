from model.VSS_module import VSSLayer
import math
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from einops import rearrange


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveGatedFusion, self).__init__()
        self.gate_A = nn.Conv2d(in_channels, 1, kernel_size=1)  # 为A生成门控
        self.gate_B = nn.Conv2d(in_channels, 1, kernel_size=1)  # 为B生成门控

    def forward(self, x_lists):
        A = x_lists[0]
        B = x_lists[1]
        gate_A = torch.sigmoid(self.gate_A(A))  # [B, 1, H, W]
        gate_B = torch.sigmoid(self.gate_B(B))  # [B, 1, H, W]
        fused = gate_A * A + gate_B * B
        return fused


class TDspeMamba(nn.Module):
    def __init__(self, channels, use_residual=True):
        super(TDspeMamba, self).__init__()
        self.channels = channels
        self.use_residual = use_residual
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self.channels,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.SiLU()
        )

    def forward(self, x_lists):
        x1 = x_lists[0]
        x2 = x_lists[1]
        B, C, H, W = x1.shape
        x_input = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1)
        x = rearrange(x_input, 'b c h w n -> (b h w) n c')
        x_flat = self.mamba(x)
        x_proj = self.proj(x_flat)
        x_recon = rearrange(x_proj, '(b h w) n c -> b c h w n', b=B, h=H, w=W, n=2)
        if self.use_residual:
            out = x_input + x_recon
            return out[:, :, :, :, 0], out[:, :, :, :, 1]
        else:
            return x_recon[:, :, :, :, 0], x_recon[:, :, :, :, 1]


class GDspaMamba(nn.Module):
    def __init__(self, hidden_dim, patch_size=1):
        super(GDspaMamba, self).__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.vss = VSSLayer(dim=hidden_dim, depth=1)
        if self.patch_size != 1:
            self.patch_embedding = nn.Sequential(
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                          kernel_size=patch_size, stride=patch_size, padding=0),
                nn.GroupNorm(4, hidden_dim),
                nn.SiLU()
            )

            self.up_sample = nn.Upsample(scale_factor=patch_size, mode='bilinear')
        
    def forward(self, x):
        if self.patch_size != 1:
            x = self.patch_embedding(x)
        x_input = rearrange(x, 'b c h w -> b h w c')
        x_spa = self.vss(x_input)
        x_spa = rearrange(x_spa, 'b h w c -> b c h w')
        if self.patch_size != 1:
            x_spa = self.up_sample(x_spa)
        return x_spa


class GDAModule(nn.Module):
    def __init__(self, hidden_dim, patch_size=2):
        super(GDAModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.spe = TDspeMamba(channels=self.hidden_dim)
        self.spa = GDspaMamba(hidden_dim=hidden_dim, patch_size=patch_size)
        self.fusion = AdaptiveGatedFusion(in_channels=hidden_dim)

    def forward(self, x_lists):
        x_spa = x_lists[0]
        x_spa = self.spa(x_spa)

        x1 = x_lists[1]
        x2 = x_lists[2]
        x1, x2 = self.spe([x1, x2])
        x_dd = x2 - x1
        x_fusion = self.fusion([x_spa, x_dd])
        return x_fusion, x1, x2


class GDAMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_classes=2, patch_size=1, group_num=4, blocks=3):
        super(GDAMamba, self).__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [int(hidden_dim) for i_layer in range(blocks)]
            patch_size = [int(patch_size) for i_layer in range(blocks)]
        self.down_ratio = max(patch_size)

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim[0], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim[0]),
            nn.SiLU()
        )
        self.patch_embedding_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim[0], kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim[0]),
            nn.SiLU()
        )
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim[-1], out_channels=num_classes, kernel_size=1, stride=1, padding=0),
        )
        num_blocks = blocks  # 假设 blocks 传入的数量
        gda_modules_list = [
            GDAModule(hidden_dim=hidden_dim[i], patch_size=patch_size[i])
            for i in range(num_blocks)
        ]

        self.GDA_modules = nn.ModuleList(gda_modules_list)

    def forward(self, x1, x2):
        x = x2 - x1
        _, _, h, w = x.shape

        pad_1 = math.ceil(h / self.down_ratio) * self.down_ratio - h
        pad_2 = math.ceil(w / self.down_ratio) * self.down_ratio - w

        x_patch = F.pad(x, (0, pad_2, 0, pad_1), mode="reflect")
        x_patch_1 = F.pad(x1, (0, pad_2, 0, pad_1), mode="reflect")
        x_patch_2 = F.pad(x2, (0, pad_2, 0, pad_1), mode="reflect")

        x = self.patch_embedding(x_patch)
        x1 = self.patch_embedding_1(x_patch_1)
        x2 = self.patch_embedding_1(x_patch_2)
        for layer in self.GDA_modules:
            [x, x1, x2] = layer([x, x1, x2])
        x = x[:, :, :h, :w]

        results = self.cls_head(x)
        return results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = 1
    w = 450
    h = 140
    channels = 154
    xd = torch.randn(batch, channels, w, h).to(device)
    x1 = torch.randn(batch, channels, w, h).to(device)
    x2 = torch.randn(batch, channels, w, h).to(device)
    model = GDAMamba(input_dim=154, hidden_dim=32).to(device)
    features = model(x1, x2)
    print(features[0].shape)
