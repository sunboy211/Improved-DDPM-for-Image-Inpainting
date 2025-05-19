import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
refer to Unet and attention class:"https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/repaint.py"
"""
class TimeEmbedding(nn.Module) :
    def __init__(self,dim) :
        super().__init__()
        half_dim = dim//2
        self.embedding=nn.Sequential(nn.Linear(1,half_dim),nn.SiLU(),nn.Linear(half_dim,dim))

    def forward(self,t):
        t=t.unsqueeze(-1).float()
        return self.embedding(t)

class UNet(nn.Module):
    def __init__(self,in_channels,out_channels=3, time_dim=256,dropout_p=0.1) :
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)

        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.e2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.e3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.self_attn = SpatialSelfAttention(256, heads=4)

        # Middle block
        self.middle_time = nn.Linear(time_dim, 256)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        #Decoder
        self.d1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.d3 = nn.Conv2d(64, out_channels, 3, padding=1)  # 最后一层通常不加 BN 和 ReLU

    def forward(self, x, t):
        x1 = self.e1(x) # [B, 64, 64, 64]
        x2 = self.e2(x1)  # [B, 128, 32, 32]
        x3 = self.e3(x2)  # [B, 256, 16, 16]

        #Time embedding integration
        t_emb = self.time_mlp(t)
        t_emb = self.middle_time(t_emb)[:, :, None, None]
        x = x3 + t_emb
        x = self.middle_conv(x)

        # Attention Pooling
        x = x + self.self_attn(x)

        x = self.middle_conv(x)

        # Decoder with skip connections
        x= F.interpolate(x, scale_factor=2, mode='nearest')
        x= torch.cat([x,x2], dim=1)
        x= self.d1(x)
        x= F.interpolate(x, scale_factor=2, mode='nearest')
        x=torch.cat([x, x1], dim=1)

        x=self.d2(x)
        return self.d3(x)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (in_channels // heads) ** -0.5

        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Flatten spatial dimension
        q = q.reshape(B, self.heads, C // self.heads, H * W)  # (B, heads, C//heads, HW)
        k = k.reshape(B, self.heads, C // self.heads, H * W)
        v = v.reshape(B, self.heads, C // self.heads, H * W)

        # Attention
        attn = (q.transpose(-1, -2) @ k) * self.scale  # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)
        out = attn @ v.transpose(-1, -2)  # (B, heads, HW, C//heads)
        out = out.transpose(-1, -2).reshape(B, C, H, W)

        return self.out_proj(out)
