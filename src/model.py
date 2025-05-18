import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaGN(nn.Module):
    def __init__(self, feats_in, c_hidden):
        super().__init__()
        self.norm = nn.GroupNorm(8, feats_in)
        self.proj = nn.Linear(c_hidden, 2 * feats_in)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, t):
        params = self.proj(t)  # [B, 2*feats_in]
        scale, shift = params.chunk(2, dim=1)  # Each is [B, feats_in]
        # Reshape for broadcasting
        scale = scale.view(-1, scale.shape[1], 1, 1)  # [B, C, 1, 1]
        shift = shift.view(-1, shift.shape[1], 1, 1)  # [B, C, 1, 1]
        x = self.norm(x)
        return x * (scale + 1) + shift

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.ada_gn = AdaGN(out_ch, time_emb_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.ada_gn(h, t)
        h = self.relu(h)
        h = self.conv2(h)
        return h + self.residual(x)

class SimpleUnet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Downsampling
        self.down1 = Block(in_channels, model_channels, time_emb_dim)
        self.down2 = Block(model_channels, model_channels*2, time_emb_dim)
        self.down3 = Block(model_channels*2, model_channels*4, time_emb_dim)

        # Bottleneck
        self.bottleneck1 = Block(model_channels*4, model_channels*4, time_emb_dim)
        self.bottleneck2 = Block(model_channels*4, model_channels*4, time_emb_dim)

        # Upsampling
        self.up1 = Block(model_channels*8, model_channels*2, time_emb_dim)
        self.up2 = Block(model_channels*4, model_channels, time_emb_dim)
        self.up3 = Block(model_channels*2, model_channels, time_emb_dim)

        # Final conv
        self.final = nn.Conv2d(model_channels, in_channels, 1)

        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)

        # Downsampling
        d1 = self.down1(x, t)
        d2 = self.down2(self.pool(d1), t)
        d3 = self.down3(self.pool(d2), t)

        # Bottleneck
        b1 = self.bottleneck1(self.pool(d3), t)
        b2 = self.bottleneck2(b1, t)

        # Upsampling with skip connections
        u1 = self.up1(torch.cat([self.upsample(b2), d3], dim=1), t)
        u2 = self.up2(torch.cat([self.upsample(u1), d2], dim=1), t)
        u3 = self.up3(torch.cat([self.upsample(u2), d1], dim=1), t)

        return self.final(u3)
