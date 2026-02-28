import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

# ==========================================
# Part 1: Basic Components (SOTA Style)
# ==========================================

class LayerNorm2d(nn.Module):
    """用于图像的 LayerNorm，替代 BatchNorm，对去雾更稳定"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    """NAFNet 中的核心门控机制"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FourierUnit(nn.Module):
    """
    频域处理单元：在频率域中进行卷积，捕捉全局信息
    """
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        # 处理实部和虚部的卷积
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        # 1. FFT 变换 (Real -> Complex)
        # norm='ortho' 保证能量守恒
        ffted = torch.fft.rfft2(x, norm='ortho') 
        
        # 2. 拼接实部和虚部 (Batch, C*2, H, W/2+1)
        ffted_cat = torch.cat([ffted.real, ffted.imag], dim=1)
        
        # 3. 频域特征交互
        ffted_cat = self.conv_layer(ffted_cat)
        ffted_cat = self.relu(ffted_cat)
        
        # 4. 拆分实部虚部并恢复复数
        f_real, f_imag = torch.chunk(ffted_cat, 2, dim=1)
        ffted_res = torch.complex(f_real, f_imag)
        
        # 5. IFFT 逆变换
        output = torch.fft.irfft2(ffted_res, s=(h, w), norm='ortho')
        return output

class SpectralNAFBlock(nn.Module):
    """
    结合了 NAFNet (空间域) 和 FFT (频域) 的混合 Block
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        
        # --- Spatial Branch (NAFNet style) ---
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel) # Depthwise
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0),
        )
        self.sg = SimpleGate()
        
        # FFN Part
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # --- Frequency Branch ---
        self.fft_block = FourierUnit(c, c)
        self.fft_beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # 1. Spatial Processing
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta
        
        # 2. FFN Processing
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        spatial_out = y + x * self.gamma
        
        # 3. Spectral Processing & Fusion
        # 使用残差连接加入频域信息
        fft_out = self.fft_block(inp)
        
        return spatial_out + fft_out * self.fft_beta

# ==========================================
# Part 2: Improved Physics Module
# ==========================================

class ContextPhysicsBlock(nn.Module):
    """改进的物理参数估计，加入全局上下文"""
    def __init__(self, base=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, base, 3, 1, 1)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True))
        
        # Global Context (捕捉大气光A)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv2d(base, base, 1),
            nn.ReLU(True),
            nn.Conv2d(base, base, 1)
        )

        # Heads
        self.t_head = nn.Sequential(nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, 1, 3, 1, 1), nn.Sigmoid())
        self.g_head = nn.Sequential(nn.Conv2d(base+1, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, I):
        x = F.relu(self.enc1(I))
        feat = self.enc2(x)
        
        # Inject Global Context
        gl_feat = self.global_mlp(self.global_pool(feat))
        feat = feat + gl_feat 
        
        t0 = torch.clamp(self.t_head(feat), 0.01, 1.0)
        # g0 (airlight term) 依赖于 t0
        g0 = self.g_head(torch.cat([feat, 1-t0], dim=1))
        return t0, g0

class RefinementBlock(nn.Module):
    """物理参数微调"""
    def __init__(self, base=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, base, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(base, 4, 3, 1, 1) # Output delta_t (1) and delta_g (3)
        )
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, g):
        inp = torch.cat([t, g], dim=1)
        res = self.net(inp)
        dt, dg = torch.split(res, [1, 3], dim=1)
        
        t = torch.clamp(t + self.beta * dt, 0.01, 1.0)
        g = torch.clamp(g + self.beta * dg, 0.0, 1.0)
        return t, g

class UnrolledPhysicsDehazeNet(nn.Module):
    def __init__(self, num_iters=6):
        super().__init__()
        self.init_block = ContextPhysicsBlock()
        self.refine_blocks = nn.ModuleList([RefinementBlock() for _ in range(num_iters)])

    def forward(self, I):
        t, g = self.init_block(I)
        for blk in self.refine_blocks:
            t, g = blk(t, g)
        
        # Physics Reconstruction
        # J = (I - g) / t 
        # Note: In standard ASM, I = J*t + A(1-t), so J = (I - A(1-t))/t.
        # Here 'g' approximates 'A(1-t)'.
        J = (I - g) / (t + 1e-6)
        J = torch.clamp(J, 0.0, 1.0)
        return J, t, g

# ==========================================
# Part 3: Fourier Refinement Net
# ==========================================

# class SpectralRefineNet(nn.Module):
#     def __init__(self, dim=32, in_chans=7): # 3(J) + 1(t) + 3(g) = 7
#         super().__init__()
        
#         # Input Projection
#         self.in_conv = nn.Conv2d(in_chans, dim, 3, 1, 1)
        
#         # Encoder
#         self.enc1 = nn.Sequential(SpectralNAFBlock(dim), SpectralNAFBlock(dim))
#         self.down1 = nn.Conv2d(dim, dim*2, 2, 2)
        
#         self.enc2 = nn.Sequential(SpectralNAFBlock(dim*2), SpectralNAFBlock(dim*2))
#         self.down2 = nn.Conv2d(dim*2, dim*4, 2, 2)
        
#         # Bottleneck (Pure Frequency Learning here helps a lot)
#         self.middle = nn.Sequential(
#             SpectralNAFBlock(dim*4),
#             SpectralNAFBlock(dim*4)
#         )
        
#         # Decoder
#         self.up2 = nn.Sequential(nn.Conv2d(dim*4, dim*8, 1), nn.PixelShuffle(2))
#         self.dec2 = nn.Sequential(SpectralNAFBlock(dim*2), SpectralNAFBlock(dim*2))
        
#         self.up1 = nn.Sequential(nn.Conv2d(dim*2, dim*4, 1), nn.PixelShuffle(2))
#         self.dec1 = nn.Sequential(SpectralNAFBlock(dim), SpectralNAFBlock(dim))
        
#         # Output
#         self.out_conv = nn.Conv2d(dim, 3, 3, 1, 1)
        
#     def forward(self, x):
#         # Encoder
#         x1 = self.enc1(self.in_conv(x))
#         x2 = self.enc2(self.down1(x1))
        
#         # Middle
#         xm = self.middle(self.down2(x2))
        
#         # Decoder with Skip Connections
#         d2 = self.up2(xm) + x2
#         d2 = self.dec2(d2)
        
#         d1 = self.up1(d2) + x1
#         d1 = self.dec1(d1)
        
#         return self.out_conv(d1)


# class SpectralRefineNet(nn.Module):
#     def __init__(self, dim=24, in_chans=7): # 参照 RefineNet 默认 dim=24
#         super().__init__()
        
#         # 1. 大感受野首层：提取全局低频上下文特征
#         self.in_conv = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_chans, dim, kernel_size=7)
#         )

#         # 2. 下采样层：采用 ReflectionPad 减少边界伪影
#         self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1, padding_mode='reflect')
#         self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1, padding_mode='reflect')

#         # 3. 上采样层：使用 PixelShuffle 保持高效推理
#         self.up2 = nn.Sequential(
#             nn.Conv2d(dim * 4, dim * 8, 1),
#             nn.PixelShuffle(2)
#         )

#         self.up3 = nn.Sequential(
#             nn.Conv2d(dim * 2, dim * 4, 1),
#             nn.PixelShuffle(2)
#         )

#         # 4. 核心处理块：将 ResBlock 替换为性能更强的 SpectralNAFBlock
#         # 深度分布参照 RefineNet: 3 -> 3 -> 6 (Bottleneck) -> 3 -> 3
#         self.g1 = nn.Sequential(*[SpectralNAFBlock(dim) for _ in range(3)])
#         self.g2 = nn.Sequential(*[SpectralNAFBlock(dim * 2) for _ in range(3)])
#         self.g3 = nn.Sequential(*[SpectralNAFBlock(dim * 4) for _ in range(6)])
#         self.g4 = nn.Sequential(*[SpectralNAFBlock(dim * 2) for _ in range(3)])
#         self.g5 = nn.Sequential(*[SpectralNAFBlock(dim) for _ in range(3)])

#         # 5. 输出层
#         self.out_conv = nn.Conv2d(dim, 3, 3, 1, 1, padding_mode='reflect')

#     def forward(self, x):
#         # Encoder 路径
#         r1 = self.g1(self.in_conv(x))           # Level 1 (Original Res)
#         r2 = self.g2(self.down1(r1))           # Level 2 (1/2 Res)
#         r3 = self.g3(self.down2(r2))           # Level 3 (1/4 Res, Bottleneck)
        
#         # Decoder 路径 (带 Skip Connection)
#         r4 = self.g4(self.up2(r3) + r2)        # 融合 Level 2 
#         r5 = self.g5(self.up3(r4) + r1)        # 融合 Level 1
        
#         return self.out_conv(r5)


class SpectralRefineNet(nn.Module):
    def __init__(self, dim=32, in_chans=7):
        super().__init__()
        
        # 1. 大感受野首层 (来自 RefineNet 的优点)
        # 7x7 卷积能立刻捕捉图像的全局亮度/雾气分布
        self.in_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, dim, kernel_size=7, padding=0) 
        )

        # 2. 下采样模块 (使用 ReflectionPad 减少边缘伪影)
        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1, padding_mode='reflect')
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1, padding_mode='reflect')

        # 3. 上采样模块 (PixelShuffle 保持高效)
        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 1),
            nn.PixelShuffle(2)
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1),
            nn.PixelShuffle(2)
        )

        # 4. 核心处理模块 (使用 SpectralNAFBlock)
        # 参考 RefineNet 的深度设置: [3, 3, 6, 3, 3]
        # 这种非对称结构 (瓶颈层最深) 是当前 SOTA 的标配
        
        # Encoder Stage 1
        self.enc1 = nn.Sequential(*[
            SpectralNAFBlock(dim) for _ in range(3)
        ])
        
        # Encoder Stage 2
        self.enc2 = nn.Sequential(*[
            SpectralNAFBlock(dim * 2) for _ in range(3)
        ])
        
        # Bottleneck (最深的部分，频域处理在这里最有效，因为感受野最大)
        self.middle = nn.Sequential(*[
            SpectralNAFBlock(dim * 4) for _ in range(6)
        ])
        
        # Decoder Stage 2
        self.dec2 = nn.Sequential(*[
            SpectralNAFBlock(dim * 2) for _ in range(3)
        ])
        
        # Decoder Stage 1
        self.dec1 = nn.Sequential(*[
            SpectralNAFBlock(dim) for _ in range(3)
        ])

        # 5. 输出层 (Reflect Pad)
        self.out_conv = nn.Conv2d(dim, 3, 3, 1, 1, padding_mode='reflect')

    def forward(self, x):
        # --- Encoding Path ---
        # Level 1
        x_in = self.in_conv(x)
        feat1 = self.enc1(x_in)      # Skip connection 1
        
        # Level 2
        x_down1 = self.down1(feat1)
        feat2 = self.enc2(x_down1)   # Skip connection 2
        
        # --- Bottleneck ---
        x_down2 = self.down2(feat2)
        feat_mid = self.middle(x_down2)
        
        # --- Decoding Path ---
        # Level 2 Reconstruction
        # 先上采样，然后与 Skip Connection 相加
        up2 = self.up2(feat_mid)
        # 这里的相加融合了深层语义(up2)和浅层细节(feat2)
        feat_dec2 = self.dec2(up2 + feat2) 
        
        # Level 1 Reconstruction
        up1 = self.up1(feat_dec2)
        feat_dec1 = self.dec1(up1 + feat1)
        
        # Output
        return self.out_conv(feat_dec1)

# ==========================================
# Part 4: Final Assembled Model
# ==========================================

class PhysicsFourierFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.physics = UnrolledPhysicsDehazeNet(num_iters=3)
        self.refine = SpectralRefineNet(dim=32, in_chans=7)
        
    def forward(self, I, return_phys=False):
        # 1. Physics Branch Estimation
        J_phys, t, g = self.physics(I)
        
        # 2. Prepare Refinement Input (Physics priors)
        # Detach physics gradients if you want stable 2-stage training, 
        # or keep them for end-to-end. End-to-end is usually better for SOTA.
        refine_in = torch.cat([J_phys, t, g], dim=1) 
        
        # 3. Residual Refinement
        J_res = self.refine(refine_in)
        
        # 4. Final Addition
        J_final = J_phys + J_res
        J_final = torch.clamp(J_final, 0.0, 1.0)
        
        return J_phys, t, g, J_final