# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt


# class FourierUnit(nn.Module):
#     def __init__(self, dim):
#         super(FourierUnit, self).__init__()

#         self.conv = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, 1), nn.ReLU(True),
#                                   nn.Conv2d(dim * 2, dim * 2, 1))

#     def forward(self, x):
#         fft_dim = (-2, -1)
#         ffted = torch.fft.rfftn(x, dim=fft_dim)
#         ffted = torch.cat((ffted.real, ffted.imag), 1)
#         ffted = self.conv(ffted)
#         real, imag = torch.chunk(ffted, 2, dim=1)
#         ffted = torch.complex(real, imag)
#         ifft_shape_slice = x.shape[-2:]
#         output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim)
#         return output

# class Att(nn.Module):
#     def __init__(self, dim):
#         super(Att, self).__init__()
#         self.four = FourierUnit(dim)
#         self.conv1 = nn.Conv2d(dim * 2, dim, 1)
#         self.conv2 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())
#         self.gate = Gate(dim)
#         self.softmax = nn.Softmax(dim=1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, dim, 1, bias=True),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 1, bias=True)
#         )
#         # Channel Attention
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # Pixel Attention
#         self.pa = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect', groups=dim, bias=True),
#             nn.Conv2d(dim, dim // 8, 1, padding=0),
#             nn.ReLU(True),
#             nn.Conv2d(dim // 8, dim, 1, padding=0),
#             nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect', groups=dim, bias=True),
#         )

#     def forward(self, x, y):
#         res = self.conv1(torch.cat([x, y], 1))
#         att1 = self.mlp(self.avg_pool(res)).expand_as(res) + self.mlp(self.max_pool(res)).expand_as(res) + self.pa(res)
#         att2 = self.four(res)
#         att = self.gate(att2, att1)
#         att = self.conv2(att)
#         out = att * res
#         return out


# class ResBlock(nn.Module):
#     def __init__(self, dim):
#         super(ResBlock, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True, padding_mode='reflect'), nn.ReLU(True),
#                                   nn.Conv2d(dim, dim, 3, 1, 1, bias=True, padding_mode='reflect'))

#     def forward(self, x):
#         return self.conv(x) + x


# class Gate(nn.Module):
#     def __init__(self, dim):
#         super(Gate, self).__init__()
#         self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

#     def forward(self, x, y):
#         res = self.conv(x - y) * x + y
#         return res


# class PatchUnEmbed(nn.Module):
#     def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
#         super().__init__()
#         self.out_chans = out_chans
#         self.embed_dim = embed_dim

#         if kernel_size is None:
#             kernel_size = 1

#         self.proj = nn.Sequential(
#             nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
#                       padding=kernel_size // 2, padding_mode='reflect'),
#             nn.PixelShuffle(patch_size)
#         )

#     def forward(self, x):
#         x = self.proj(x)
#         return x


# class FFC_ResnetBlock(nn.Module):
#     def __init__(self, dim):
#         super(FFC_ResnetBlock, self).__init__()
#         self.ffc1 = FourierUnit(dim)
#         self.ffc2 = FourierUnit(dim)
#         self.res1 = ResBlock(dim)
#         self.res2 = ResBlock(dim)
#         self.att = Att(dim)
#         self.cat = nn.Conv2d(dim * 2, dim, 1)
#         self.g1 = Gate(dim)
#         self.g2 = Gate(dim)
#         #out = α ⊙ x + (1 − α) ⊙ y
#         #其中 α = sigmoid( 1×1_conv(x − y) )   ⊙ 表示逐像素乘

#     def forward(self, x):
#         x1 = self.res1(x)
#         x2 = self.ffc1(x)
#         res1 = self.g1(x2, x1)
#         res2 = self.g2(x1, x2)
#         res1 = self.res2(res1)
#         res2 = self.ffc2(res2)
#         out = self.att(res1, res2) + x
#         return out


# class PatchEmbed(nn.Module):
#     def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
#         super().__init__()
#         self.in_chans = in_chans
#         self.embed_dim = embed_dim

#         if kernel_size is None:
#             kernel_size = patch_size

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
#                               padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

#     def forward(self, x):
#         x = self.proj(x)
#         return x


# class SKFusion(nn.Module):
#     def __init__(self, dim, height=2, reduction=8):
#         super(SKFusion, self).__init__()

#         self.height = height
#         d = max(int(dim / reduction), 4)

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(dim, d, 1, bias=False),
#             nn.ReLU(True),
#             nn.Conv2d(d, dim * height, 1, bias=False)
#         )

#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, in_feats):
#         B, C, H, W = in_feats[0].shape

#         in_feats = torch.cat(in_feats, dim=1)
#         in_feats = in_feats.view(B, self.height, C, H, W)

#         feats_sum = torch.sum(in_feats, dim=1)
#         attn = self.mlp(self.avg_pool(feats_sum))
#         attn = self.softmax(attn.view(B, self.height, C, 1, 1))

#         out = torch.sum(in_feats * attn, dim=1)
#         return out


# class Net(nn.Module):
#     def __init__(self, dim=24):
#         super(Net, self).__init__()
#         self.in_conv = nn.Sequential(nn.ReflectionPad2d(3),
#                                      nn.Conv2d(3, dim, kernel_size=7, padding=0))
#         self.down1 = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, padding_mode='reflect'))

#         self.down2 = nn.Sequential(
#             nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1, padding_mode='reflect'))

#         self.up2 = nn.Sequential(
#             nn.Conv2d(dim * 4, dim * 2 * 4, kernel_size=1,
#                       padding=0, padding_mode='reflect'),
#             nn.PixelShuffle(2)
#         )

#         self.up3 = nn.Sequential(
#             nn.Conv2d(dim * 2, dim * 4, kernel_size=1,
#                       padding=0, padding_mode='reflect'),
#             nn.PixelShuffle(2)
#         )

#         self.conv = nn.Sequential(nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect'))

#         blocks1 = [FFC_ResnetBlock(dim) for _ in range(3)]
#         self.g1 = nn.Sequential(*blocks1)

#         blocks2 = [FFC_ResnetBlock(dim * 2) for _ in range(3)]
#         self.g2 = nn.Sequential(*blocks2)

#         blocks3 = [FFC_ResnetBlock(dim * 4) for _ in range(6)]
#         self.g3 = nn.Sequential(*blocks3)

#         blocks4 = [FFC_ResnetBlock(dim * 2) for _ in range(3)]
#         self.g4 = nn.Sequential(*blocks4)

#         blocks5 = [FFC_ResnetBlock(dim) for _ in range(3)]
#         self.g5 = nn.Sequential(*blocks5)

#         self.mix1 = SKFusion(dim * 2)
#         self.mix2 = SKFusion(dim)

#     def forward(self, x):
#         res1 = self.g1(self.in_conv(x))
#         res2 = self.g2(self.down1(res1))
#         res3 = self.g3(self.down2(res2))
#         res4 = self.g4(self.mix1([self.up2(res3), res2]))
#         res5 = self.g5(self.mix2([self.up3(res4), res1]))
#         out = self.conv(res5)
#         return out + x


#=======================================================================================================================
'''
SFRDP-Net.Net 的 Docstring
 └─ UnrolledPhysicsDehazeNet
     ├─ J_phys
     ├─ t_final
     └─ g_final
          ↓
   Conditional Fourier-Net
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
def conv_block(in_c, out_c, k=3, s=1, p=1, act=True):
    layers: List[nn.Module] = [nn.Conv2d(in_c, out_c, k, s, p)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InitialPhysicsBlock(nn.Module):
    def __init__(self, base=32):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(3, base),
            conv_block(base, base),
            conv_block(base, base)
        )

        self.t_head = nn.Sequential(
            conv_block(base, base),
            nn.Conv2d(base, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.g_head = nn.Sequential(
            conv_block(base + 1, base),
            conv_block(base, base),
            nn.Conv2d(base, 3, 3, 1, 1)
        )

    def forward(self, I):
        feat = self.encoder(I)

        t0 = torch.clamp(self.t_head(feat), 0.05, 1.0)
        g0 = self.g_head(torch.cat([feat, 1 - t0], dim=1))
        g0 = torch.clamp(g0, 0.0, 1.0)

        return t0, g0


class RefinementBlock(nn.Module):
    def __init__(self, base_t=16, base_g=32):
        super().__init__()

        self.t_net = nn.Sequential(
            conv_block(1, base_t),
            conv_block(base_t, base_t),
            nn.Conv2d(base_t, 1, 3, 1, 1)
        )

        self.g_net = nn.Sequential(
            conv_block(3, base_g),
            conv_block(base_g, base_g),
            nn.Conv2d(base_g, 3, 3, 1, 1)
        )

        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, g):
        t = torch.clamp(t + self.beta * self.t_net(t), 0.05, 1.0)
        g = torch.clamp(g + self.beta * self.g_net(g), 0.0, 1.0)
        return t, g


def physics_dehaze(I, t, g, eps=1e-3):
    J = (I - g) / torch.clamp(t, eps, 1.0)
    return torch.clamp(J, 0.0, 1.0)


class UnrolledPhysicsDehazeNet(nn.Module):
    def __init__(self, num_iters=6):
        super().__init__()

        self.init_block = InitialPhysicsBlock()
        self.refine_blocks = nn.ModuleList(
            [RefinementBlock() for _ in range(num_iters)]
        )

    def forward(self, I):
        t, g = self.init_block(I)

        for blk in self.refine_blocks:
            t, g = blk(t, g)

        J = physics_dehaze(I, t, g)
        return J, t, g


class FourierUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.ReLU(True),
            nn.Conv2d(dim * 2, dim * 2, 1)
        )

    def forward(self, x):
        ffted = torch.fft.rfftn(x, dim=(-2, -1))
        ffted = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted = self.conv(ffted)
        real, imag = torch.chunk(ffted, 2, dim=1)
        ffted = torch.complex(real, imag)
        return torch.fft.irfftn(ffted, s=x.shape[-2:], dim=(-2, -1))


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x, y):
        return self.conv(x - y) * x + y

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class FourierResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect')
        )
        self.fourier = FourierUnit(dim)
        self.gate = Gate(dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.se = SEBlock(dim)

    def forward(self, x):
        local_feat = self.local(x)
        freq_feat = self.fourier(x)
        local = self.gate(freq_feat, local_feat)
        freq = self.gate(local_feat, freq_feat)
        res =(1-self.alpha)*local + self.alpha * freq
        res = self.se(res)
        return x + res

    
class ResBlockSE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect')
        )
        self.se = SEBlock(dim)

    def forward(self, x):
        res = self.conv(x)
        res = self.se(res)
        return x + res
    
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect'),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1, padding_mode='reflect')
        )

    def forward(self, x):
        return x + self.conv(x)


class Net(nn.Module):
    def __init__(self, dim=24, in_chans=7):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, dim, kernel_size=7)
        )

        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1, padding_mode='reflect')
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1, padding_mode='reflect')

        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 1),
            nn.PixelShuffle(2)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1),
            nn.PixelShuffle(2)
        )

        self.g1 = nn.Sequential(*[ResBlock(dim) for _ in range(3)])
        self.g2 = nn.Sequential(*[ResBlock(dim * 2) for _ in range(3)])
        self.g3 = nn.Sequential(*[ResBlock(dim * 4) for _ in range(6)])
        self.g4 = nn.Sequential(*[ResBlock(dim * 2) for _ in range(3)])
        self.g5 = nn.Sequential(*[ResBlock(dim) for _ in range(3)])

        self.out_conv = nn.Conv2d(dim, 3, 3, 1, 1, padding_mode='reflect')

    def forward(self, x):
        r1 = self.g1(self.in_conv(x))
        r2 = self.g2(self.down1(r1))
        r3 = self.g3(self.down2(r2))
        r4 = self.g4(self.up2(r3) + r2)
        r5 = self.g5(self.up3(r4) + r1)
        return self.out_conv(r5)


class PhysicsFourierFusionNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.physics = UnrolledPhysicsDehazeNet(num_iters=6)
        self.refine  = Net(dim=24, in_chans=7)
  
    def forward(self, I, return_phys=False):
        J_phys, t, g = self.physics(I)
        x = torch.cat([J_phys, t, g], dim=1)
        # x= torch.cat([J_phys,I], dim=1)
        J_res = self.refine(x)
        J_final = J_phys + J_res

        # if return_phys:
        #     return J_phys, t, g, J_final
        # else:
        #     return J_final
        return J_phys, t, g, J_final

