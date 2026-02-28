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




class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.Sigmoid())

    def forward(self, x, y):
        return self.conv(x - y) * x + y




    

    
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


class RefineNet(nn.Module):
    def __init__(self, dim=24, in_chans=7):
        super().__init__()
        # 用一个大感受野的首层卷积提取对去雾至关重要的低频上下文特征
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


class AdvancedPhysicsNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.physics = UnrolledPhysicsDehazeNet(num_iters=6)
        self.refine  = RefineNet(dim=24, in_chans=7)
  
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

