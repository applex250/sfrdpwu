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
    def __init__(self, num_iters=8):
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



class PhysicsNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.physics = UnrolledPhysicsDehazeNet(num_iters=6)
        
  
    def forward(self, I):
        J_phys, t, g = self.physics(I)
        
        J_final = J_phys


        return J_phys, t, g, J_final

import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (保留你原有的 conv_block, InitialPhysicsBlock, RefinementBlock, physics_dehaze, UnrolledPhysicsDehazeNet) ...
# 请将你原来的代码保留在上面，下面是新增和修改的部分

class ContextBlock(nn.Module):
    """
    一个简单的残差块，用于提取深层特征
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) # 细化阶段BN有助于收敛
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

class FrequencyRefinerBlock(nn.Module):
    """
    频域分支：专门处理全局色彩和低频信息，修正色偏
    """
    def __init__(self, channels):
        super().__init__()
        # 1x1 卷积在频域等价于全局感受野，处理幅度谱和相位谱
        self.conv_real = nn.Conv2d(channels, channels, 1, 1, 0)
        self.conv_imag = nn.Conv2d(channels, channels, 1, 1, 0)
        self.fuse = nn.Conv2d(channels, channels, 1, 1, 0)
        
    def forward(self, x):
        # x: [B, C, H, W]
        # FFT
        fft_x = torch.fft.rfft2(x, norm='ortho') # [B, C, H, W/2+1] Complex
        
        # 处理实部虚部
        real = fft_x.real
        imag = fft_x.imag
        
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        
        # iFFT
        fft_out = torch.complex(real_out, imag_out)
        out = torch.fft.irfft2(fft_out, s=x.shape[-2:], norm='ortho')
        
        return self.fuse(out) + x # 残差连接

class DualDomainRefiner(nn.Module):
    def __init__(self, base_c=32):
        super().__init__()
        in_channels = 3 + 3 + 1 + 3 # J_phys, I, t, g
        
        # 1. 浅层特征提取
        self.entry = nn.Conv2d(in_channels, base_c, 3, 1, 1)
        
        # 2. 空域分支 (保留你原来的 U-Net 结构，为了简洁这里简写)
        self.spatial_unet = nn.Sequential(
            ContextBlock(base_c),
            nn.Conv2d(base_c, base_c, 3, 1, 1) # 简单的占位，实际请用你之前的 U-Net
        )
        
        # 3. 频域分支 (新增)
        # 在低分辨率下做频域处理效率更高，效果更好
        self.freq_branch = nn.Sequential(
            nn.AvgPool2d(2), # 下采样
            FrequencyRefinerBlock(base_c),
            FrequencyRefinerBlock(base_c),
            nn.Upsample(scale_factor=2, mode='bilinear') # 上采样
        )
        
        # 4. 融合与输出
        self.fusion = nn.Conv2d(base_c * 2, base_c, 1)
        self.tail = nn.Conv2d(base_c, 3, 3, 1, 1)
        
    def forward(self, I, J_phys, t, g):
        if g.shape[-1] == 1: g = g.expand_as(I)
        x_in = torch.cat([J_phys, I, t, g], dim=1)
        
        feat = self.entry(x_in)
        
        # 并行处理
        out_spatial = self.spatial_unet(feat)
        out_freq = self.freq_branch(feat)
        
        # 融合
        out = torch.cat([out_spatial, out_freq], dim=1)
        out = self.fusion(out)
        
        res = self.tail(out)
        return res

class PhysicsFourierFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 你的物理模块 (负责去薄雾，恢复整体结构)
        self.physics = UnrolledPhysicsDehazeNet(num_iters=6)
        
        # 2. 新增的细化模块 (负责修补浓雾，提升PSNR)
        self.refine = DualDomainRefiner(base_c=32)
        
    def forward(self, I,return_phys=False):
        # 第一阶段：物理去雾
        # 这里的 J_phys 也就是你现在的 23dB 结果
        J_phys, t, g = self.physics(I)
        
        # 阻断梯度？视情况而定。
        # 如果显存不够，可以 J_phys.detach()，只训练 Refiner。
        # 如果想端到端刷分，建议保留梯度。
        
        # 第二阶段：残差细化
        # 网络根据 t 的大小，自动决定每个像素需要加多少残差
        res = self.refine(I, J_phys, t, g)
        
        # 最终结果 = 物理结果 + 神经网络修补
        J_final = J_phys + res
        
        # 再次截断，保证图像有效
        J_final = torch.clamp(J_final, 0.0, 1.0)
        
        return J_phys, t, g, J_final

# ==========================================
# 训练 Loss 建议 (这对刷分至关重要！)
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, J_phys, J_final, t, g, Target, I_input):
        # 1. 最终输出的重建 Loss (最重要)
        loss_rec = self.l1(J_final, Target)
        
        # 2. 物理中间层的 Loss (保证物理分支不退化)
        loss_phys = self.l1(J_phys, Target)
        
        # 3. 频域 Loss / FFT Loss (针对你提到的 PSNR 提升)
        # 浓雾去除往往需要频域约束，防止高频噪声
        fft_pred = torch.fft.rfft2(J_final, norm='ortho')
        fft_gt = torch.fft.rfft2(Target, norm='ortho')
        loss_fft = self.l1(torch.abs(fft_pred), torch.abs(fft_gt))
        
        # 总 Loss
        return loss_rec + 0.5 * loss_phys + 0.1 * loss_fft