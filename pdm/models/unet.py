import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim // 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_c, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.norm1 = nn.GroupNorm(32, out_c)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_c)
        self.dropout = nn.Dropout(dropout)
        self.time_proj = nn.Linear(time_c, out_c)
        self.shortcut = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        # 时间嵌入注入
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        h = self.dropout(h)
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv.unbind(1) # [B, Heads, C_h, N]
        
        attn = (q.transpose(-2, -1) @ k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(out)

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
    def forward(self, x):
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, base_c=64):
        super().__init__()
        self.time_mlp = TimeEmbedding(base_c * 4)

        # === Encoder ===
        self.inc = nn.Conv2d(in_channels, base_c, 3, 1, 1)
        
        # Level 1
        self.down1_res1 = ResBlock(base_c, base_c, base_c*4)
        self.down1_res2 = ResBlock(base_c, base_c, base_c*4)
        self.down1_pool = DownSample(base_c)
        
        # Level 2
        self.down2_res1 = ResBlock(base_c, base_c*2, base_c*4)
        self.down2_res2 = ResBlock(base_c*2, base_c*2, base_c*4)
        self.down2_pool = DownSample(base_c*2)
        
        # Level 3
        self.down3_res1 = ResBlock(base_c*2, base_c*4, base_c*4)
        self.down3_res2 = ResBlock(base_c*4, base_c*4, base_c*4)
        self.down3_pool = DownSample(base_c*4)
        
        # === Bottleneck ===
        self.bot1 = ResBlock(base_c*4, base_c*8, base_c*4)
        self.attn = AttentionBlock(base_c*8)
        self.bot2 = ResBlock(base_c*8, base_c*4, base_c*4) # Channel 减半回 4倍

        # === Decoder ===
        # Level 3
        self.up3_sample = UpSample(base_c*4) # 尺寸变大 32->64
        self.up3_res1 = ResBlock(base_c*8, base_c*4, base_c*4) # Concat后通道是 4+4=8
        self.up3_res2 = ResBlock(base_c*4, base_c*4, base_c*4)
        
        # Level 2
        self.up2_sample = UpSample(base_c*4) # 尺寸变大 64->128
        self.up2_res1 = ResBlock(base_c*6, base_c*2, base_c*4) # Concat后通道是 4+2=6
        self.up2_res2 = ResBlock(base_c*2, base_c*2, base_c*4)
        
        # Level 1
        self.up1_sample = UpSample(base_c*2) # 尺寸变大 128->256
        self.up1_res1 = ResBlock(base_c*3, base_c, base_c*4) # Concat后通道是 2+1=3
        self.up1_res2 = ResBlock(base_c, base_c, base_c*4)
        
        self.outc = nn.Sequential(
            nn.GroupNorm(32, base_c), 
            nn.SiLU(), 
            nn.Conv2d(base_c, out_channels, 3, 1, 1)
        )

    def forward(self, x_noisy, t, condition_hazy, condition_phys):
        # 1. 拼接输入
        x = torch.cat([x_noisy, condition_hazy, condition_phys], dim=1) # [B, 9, H, W]
        t_emb = self.time_mlp(t)

        skips = []
        
        # === Encoder ===
        x = self.inc(x)
        skips.append(x) # Skip 0
        
        # Down 1
        x = self.down1_res1(x, t_emb)
        x = self.down1_res2(x, t_emb)
        skips.append(x) # Skip 1 (Before Downsample)
        x = self.down1_pool(x)
        
        # Down 2
        x = self.down2_res1(x, t_emb)
        x = self.down2_res2(x, t_emb)
        skips.append(x) # Skip 2
        x = self.down2_pool(x)
        
        # Down 3
        x = self.down3_res1(x, t_emb)
        x = self.down3_res2(x, t_emb)
        skips.append(x) # Skip 3
        x = self.down3_pool(x)

        # === Bottleneck ===
        x = self.bot1(x, t_emb)
        x = self.attn(x)
        x = self.bot2(x, t_emb)

        # === Decoder ===
        # Up 3
        x = self.up3_sample(x) # 先上采样，尺寸变大
        skip = skips.pop()     # 拿出 Skip 3
        # 如果尺寸因为 padding 还是不一致，强制对齐
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up3_res1(x, t_emb)
        x = self.up3_res2(x, t_emb)
        
        # Up 2
        x = self.up2_sample(x)
        skip = skips.pop()     # 拿出 Skip 2
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up2_res1(x, t_emb)
        x = self.up2_res2(x, t_emb)
        
        # Up 1
        x = self.up1_sample(x)
        skip = skips.pop()     # 拿出 Skip 1 (Skip 0 还没用)
        # 注意：这里 Skip 1 是 down1 结束时的特征，Skip 0 是 inc 的特征
        # 实际上我们通常只需要其中一个，或者再做一层。
        # 这里逻辑对齐: up1_sample 对应 down1_pool
        # Skip 1 尺寸是 128 (假设256输入)，Up1输出也是 128
        # Wait, Up1 sample 128 -> 256
        # Skip 1 是 256. 
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up1_res1(x, t_emb)
        x = self.up1_res2(x, t_emb)
        
        # 还有一个 inc 的 skip (Skip 0) 没用到？
        # 一般 U-Net 3层 Down 对应 3层 Up。
        # 这里最外层的 Skip 0 可以不用，或者你可以加一个 final resblock。
        # 目前的代码结构已经完整闭环。
        
        return self.outc(x)