import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Part 1: Frequency Domain Components
# ==========================================

class FrequencySelectionUnit(nn.Module):
    """
    频域选择单元：
    只处理全局信息，不触碰高频纹理。
    """
    def __init__(self, in_channels):
        super().__init__()
        # 使用 1x1 卷积在频域处理实部和虚部
        # 1x1 卷积在频域等价于在空域进行全局卷积，能极好地捕捉雾的分布
        self.conv_real = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.conv_imag = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        
        # 门控机制：决定频域信息对空间域的影响程度
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. FFT 变换 (Real -> Complex)
        # rfft2 只计算一般频谱，节省一半显存
        fft_x = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 提取实部和虚部
        real = fft_x.real
        imag = fft_x.imag
        
        # 3. 频域特征交互 (仅做通道间混合，不改变频率位置)
        # 这样避免了在频域做 3x3 卷积导致的空间混叠
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)
        
        # 4. 激活函数 (可选，通常频域操作是线性的，但在特征层可以加非线性)
        real_out = self.relu(real_out)
        imag_out = self.relu(imag_out)
        
        # 5. 逆变换 iFFT
        fft_out = torch.complex(real_out, imag_out)
        output = torch.fft.irfft2(fft_out, s=(H, W), norm='ortho')
        
        # 6. 门控融合：让网络自己决定要多少频域信息
        return output * self.gate(output)

# ==========================================
# Part 2: Spatial Components (Gated ResBlock)
# ==========================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

# 修复 SpatialGatedBlock
class SpatialGatedBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 【修复2】加入 LayerNorm
        self.norm = nn.LayerNorm(c) 
        self.conv1 = nn.Conv2d(c, c * 2, 3, 1, 1, padding_mode='reflect') 
        self.sg = SimpleGate() 
        self.conv2 = nn.Conv2d(c, c, 3, 1, 1, padding_mode='reflect')
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        # LayerNorm 需要 permute 因为它默认处理最后一个维度
        inp = x.permute(0, 2, 3, 1) # B, H, W, C
        inp = self.norm(inp)
        inp = inp.permute(0, 3, 1, 2) # B, C, H, W
        
        out = self.conv1(inp)
        out = self.sg(out)
        out = self.conv2(out)
        return x + out * (1.0 + self.beta)

# ==========================================
# Part 3: The Hybrid Block (Spatial + FFT)
# ==========================================

class DualDomainBlock(nn.Module):
    """
    【核心创新点】
    并行处理：
    1. Spatial Path: 保持纹理清晰 (ResNet/Gated 风格)
    2. Freq Path: 去除全局雾气背景 (FFT 风格)
    """
    def __init__(self, dim):
        super().__init__()
        self.spatial = SpatialGatedBlock(dim)
        self.freq = FrequencySelectionUnit(dim)
        
        # 融合层
        self.fusion = nn.Conv2d(dim * 2, dim, 1, 1, 0)

    def forward(self, x):
        # 路径 1: 空间域处理 (保持细节)
        x_spatial = self.spatial(x)
        
        # 路径 2: 频域处理 (捕捉全局上下文)
        # 残差连接：x + FFT(x)
        x_freq = x + self.freq(x) 
        
        # 融合
        out = torch.cat([x_spatial, x_freq], dim=1)
        return self.fusion(out)

# ==========================================
# Part 4: Complete Network
# ==========================================

# class DeepPhysicsNet(nn.Module):
#     # ... (保持之前的深层物理模块不变，这是地基) ...
#     def __init__(self, num_iters=6): 
#         super().__init__()
#         self.init_block = InitialPhysicsBlock() # 需要把之前的 InitialPhysicsBlock 代码拷过来
#         self.steps = nn.ModuleList([PhysicsRefineStep() for _ in range(num_iters)]) # 同上

#     def forward(self, I):
#         t, g = self.init_block(I)
        
#         for step in self.steps:
#             t, g = step(t, g)
            
#         t_safe = torch.clamp(t, min=0.01, max=1.0)
#         # 【修复1】避免直接除法导致的梯度爆炸
#         # 方法 A: 梯度截断 (推荐)
#         # 在反向传播时，如果 1/t 的梯度过大，将其截断
#         inv_t = 1.0 / t_safe
#         if self.training:
#             # 这里的 hook 可以防止 1/t 的导数过大
#             inv_t.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        
#         J = (I - g) * inv_t
#         return torch.clamp(J, 0.0, 1.0), t, g

class DeepPhysicsNet(nn.Module):
    def __init__(self, num_iters=6): 
        super().__init__()
        self.init_block = InitialPhysicsBlock()
        self.steps = nn.ModuleList([PhysicsRefineStep() for _ in range(num_iters)])

    def forward(self, I):
        t, g = self.init_block(I)
        
        # 循环迭代
        for step in self.steps:
            t, g = step(t, g)
            
        # 【关键修改 3】延迟约束 (Deferred Constraint)
        # 只有在最后输出用于物理计算时，才进行硬截断或 Sigmoid
        # 这里的 clamp 处于计算图末端，不会阻断中间层的梯度回传
        
        # 处理 t: 保证 t > 0，避免除零
        # 建议使用 Softplus 或 Sigmoid 的变体来代替硬 clamp，更平滑
        # 但为了保留物理意义，这里用 clamp 配合梯度 hook (见上一条回答) 也是可以的
        t_final = torch.clamp(t, min=0.01, max=1.0)
        g_final = torch.clamp(g, 0.0, 1.0)
        
        # 物理公式重构
        # 加入上一条建议的梯度保护
        inv_t = 1.0 / t_final
        if self.training:
            inv_t.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        J = (I - g_final) * inv_t
        return torch.clamp(J, 0.0, 1.0), t_final, g_final

# 辅助类定义 (为了代码完整性)
class InitialPhysicsBlock(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True))
        self.t_head = nn.Sequential(nn.Conv2d(base, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, 1, 3, 1, 1), nn.Sigmoid())
        self.g_head = nn.Sequential(nn.Conv2d(base + 1, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, 3, 3, 1, 1), nn.Sigmoid())
    def forward(self, I):
        feat = self.encoder(I)
        t = torch.clamp(self.t_head(feat), 0.01, 1.0)
        g = self.g_head(torch.cat([feat, 1-t], dim=1))
        return t, g

# class PhysicsRefineStep(nn.Module):
#     def __init__(self, base=16):
#         super().__init__()
#         self.net = nn.Sequential(nn.Conv2d(4, base, 3, 1, 1), nn.ReLU(True), nn.Conv2d(base, 4, 3, 1, 1))
#         self.beta = nn.Parameter(torch.tensor(0.1))
#     def forward(self, t, g):
#         res = self.net(torch.cat([t, g], dim=1))
#         dt, dg = torch.split(res, [1, 3], dim=1)
#         return torch.tanh(t+self.beta*dt, 0.01, 1.0), torch.clamp(g+self.beta*dg, 0.0, 1.0)
class PhysicsRefineStep(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        # 增加隐层通道数，稍微增强一点非线性能力
        self.net = nn.Sequential(
            nn.Conv2d(4, base, 3, 1, 1), 
            nn.ReLU(True), 
            nn.Conv2d(base, 4, 3, 1, 1)
        )
        # 初始化 beta 为一个较小的值，防止初始震荡
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, t, g):
        # 输入: t, g (此时它们可能是无界的特征)
        inp = torch.cat([t, g], dim=1)
        res = self.net(inp)
        dt, dg = torch.split(res, [1, 3], dim=1)
        
        # 【关键修改 1】使用 Tanh 限制更新步长
        # 这样每次更新最多只改变 ±beta 的幅度
        # 既防止了梯度爆炸，又保留了平滑的梯度
        dt = torch.tanh(dt)
        dg = torch.tanh(dg)
        
        # 【关键修改 2】移除 torch.clamp
        # 让数值在循环过程中自由流动，避免梯度消失
        t_new = t + self.beta * dt
        g_new = g + self.beta * dg
        
        return t_new, g_new
    
    
class HybridRefineNet(nn.Module):
    def __init__(self, dim=24, in_chans=7):
        super().__init__()
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_chans, dim, 7))
        self.down1 = nn.Conv2d(dim, dim * 2, 3, 2, 1)
        self.down2 = nn.Conv2d(dim * 2, dim * 4, 3, 2, 1)

        self.up2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 1), nn.PixelShuffle(2))
        self.up1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 1), nn.PixelShuffle(2))

        # 使用 DualDomainBlock 替代纯 ResBlock
        # 只在 bottleneck 和最深层使用 FFT，避免在浅层破坏纹理
        self.enc1 = nn.Sequential(SpatialGatedBlock(dim), SpatialGatedBlock(dim),SpatialGatedBlock(dim))
        self.enc2 = nn.Sequential(SpatialGatedBlock(dim*2), SpatialGatedBlock(dim*2),DualDomainBlock(dim*2)) # 中层开始引入 FFT
        self.middle = nn.Sequential(DualDomainBlock(dim*4), DualDomainBlock(dim*4), DualDomainBlock(dim*4),DualDomainBlock(dim*4), DualDomainBlock(dim*4), DualDomainBlock(dim*4)) # 深层强依赖 FFT
        self.dec2 = nn.Sequential(DualDomainBlock(dim*2), DualDomainBlock(dim*2),SpatialGatedBlock(dim*2))
        self.dec1 = nn.Sequential(SpatialGatedBlock(dim), SpatialGatedBlock(dim),SpatialGatedBlock(dim)) # 回到浅层只用 Spatial

        self.out_conv = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        x_in = self.in_conv(x)
        f1 = self.enc1(x_in)
        f2 = self.enc2(self.down1(f1))
        f_mid = self.middle(self.down2(f2))
        
        f_dec2 = self.dec2(self.up2(f_mid) + f2)
        f_dec1 = self.dec1(self.up1(f_dec2) + f1)
        return self.out_conv(f_dec1)

class PhysicsFourierFusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.physics = DeepPhysicsNet(num_iters=6) # 保持深层物理
        self.refine = HybridRefineNet()
        
    def forward(self, I, return_phys=False):
        J_p, t, g = self.physics(I)
        inp = torch.cat([J_p, t, g], dim=1)
        J_res = self.refine(inp)
        J_res =torch.clamp(J_p + J_res, 0.0, 1.0)
        return J_p, t, g, J_res

# 测试
if __name__ == "__main__":
    net = PhysicsFourierFusionNet().cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    print(f"Output shape: {net(x).shape}")