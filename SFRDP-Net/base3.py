'''
wu_quwu.base3 的 Docstring
I
 └── Initial Estimation Net → t0, g0
       └── Unrolled Refinement × K → {t_k, g_k}
             └── Physics Solver → J

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


class FourierLowFreqGuidance(nn.Module):
    """
    Enforce low-frequency update in Fourier domain
    """
    def __init__(self, keep_ratio=0.25):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        fft = torch.fft.fft2(x, norm="ortho")
        fft = torch.fft.fftshift(fft)

        h_keep = int(H * self.keep_ratio)
        w_keep = int(W * self.keep_ratio)

        mask = torch.zeros_like(fft)
        h0 = H // 2 - h_keep // 2
        h1 = H // 2 + h_keep // 2
        w0 = W // 2 - w_keep // 2
        w1 = W // 2 + w_keep // 2

        mask[:, :, h0:h1, w0:w1] = 1.0

        fft_low = fft * mask
        fft_low = torch.fft.ifftshift(fft_low)
        out = torch.fft.ifft2(fft_low, norm="ortho").real

        return out

class InitialPhysicsBlock(nn.Module):
    """
    Estimate initial t0(x) and g0(x)
    """
    def __init__(self, base=32):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(3, base),
            conv_block(base, base),
            conv_block(base, base)
        )

        # transmission t
        self.t_head = nn.Sequential(
            conv_block(base, base),
            nn.Conv2d(base, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # airlight-related term g
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
    """
    One iteration of physics-guided refinement
    """
    def __init__(self, base_t=16, base_g=32):
        super().__init__()

        # t refinement
        self.t_net = nn.Sequential(
            conv_block(1, base_t),
            conv_block(base_t, base_t),
            nn.Conv2d(base_t, 1, 3, 1, 1)
        )

        # g refinement (low-frequency preferred)
        self.g_net = nn.Sequential(
            conv_block(3, base_g),
            conv_block(base_g, base_g),
            nn.Conv2d(base_g, 3, 3, 1, 1)
        )

        # learnable step size (important!)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, g):
        dt = self.t_net(t)
        dg = self.g_net(g)

        t_new = torch.clamp(t + self.beta * dt, 0.05, 1.0)
        g_new = torch.clamp(g + self.beta * dg, 0.0, 1.0)

        return t_new, g_new
class MSFreqRefinementBlock(nn.Module):
    """
    Multi-scale + Fourier-guided refinement block
    """
    def __init__(self, base_t=16, base_g=32):
        super().__init__()

        # -------- t branch (multi-scale spatial) --------
        self.t_conv = nn.Sequential(
            conv_block(1, base_t),
            conv_block(base_t, base_t),
            nn.Conv2d(base_t, 1, 3, 1, 1)
        )

        # -------- g branch (multi-scale spatial) --------
        self.g_conv = nn.Sequential(
            conv_block(3, base_g),
            conv_block(base_g, base_g),
            nn.Conv2d(base_g, 3, 3, 1, 1)
        )

        # frequency guidance for g
        self.freq_guidance = FourierLowFreqGuidance(keep_ratio=0.25)

        # learnable step size
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, g):
        # ===== Multi-scale t =====
        t_half = F.interpolate(t, scale_factor=0.5, mode="bilinear", align_corners=False)
        t_quarter = F.interpolate(t, scale_factor=0.25, mode="bilinear", align_corners=False)

        dt = self.t_conv(t)
        dt_half = F.interpolate(self.t_conv(t_half), size=t.shape[-2:], mode="bilinear", align_corners=False)
        dt_quarter = F.interpolate(self.t_conv(t_quarter), size=t.shape[-2:], mode="bilinear", align_corners=False)

        dt_total = (dt + dt_half + dt_quarter) / 3.0

        # ===== Multi-scale g =====
        g_half = F.interpolate(g, scale_factor=0.5, mode="bilinear", align_corners=False)
        g_quarter = F.interpolate(g, scale_factor=0.25, mode="bilinear", align_corners=False)

        dg = self.g_conv(g)
        dg_half = F.interpolate(self.g_conv(g_half), size=g.shape[-2:], mode="bilinear", align_corners=False)
        dg_quarter = F.interpolate(self.g_conv(g_quarter), size=g.shape[-2:], mode="bilinear", align_corners=False)

        dg_spatial = (dg + dg_half + dg_quarter) / 3.0

        # ===== Fourier low-frequency constraint =====
        dg_freq = self.freq_guidance(dg_spatial)

        # ===== Update with projection =====
        t_new = torch.clamp(t + self.beta * dt_total, 0.05, 1.0)
        g_new = torch.clamp(g + self.beta * dg_freq, 0.0, 1.0)

        return t_new, g_new

def physics_dehaze(I, t, g, eps=1e-3):
    """
    J = (I - g) / t
    """
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
        t_list, g_list,alp_list = [], [],[1,1,1,1,1,1]

        # initial estimation
        t, g = self.init_block(I)
        t_list.append(t)
        g_list.append(g)

        # unrolled refinement
        for block in self.refine_blocks:
            t, g = block(t, g)
            t_list.append(t)
            g_list.append(g)

        # physics solver (ONCE)
        J = physics_dehaze(I, t, g)

        return J, t_list, g_list,alp_list
