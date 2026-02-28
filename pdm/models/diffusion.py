import torch
import torch.nn as nn
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Linear Schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, hazy, j_phys):
        """
        Inference: 从随机噪声 -> 清晰图
        """
        B, C, H, W = hazy.shape
        img = torch.randn((B, 3, H, W), device=hazy.device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps, leave=False):
            t = torch.full((B,), i, device=hazy.device, dtype=torch.long)
            
            # 预测噪声
            pred_noise = self.model(img, t, hazy, j_phys)
            
            # 还原 x_{t-1}
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = 0
            
            # DDPM Update Rule
            term1 = 1 / torch.sqrt(alpha)
            term2 = (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            mean = term1 * (img - term2)
            img = mean + torch.sqrt(beta) * noise
            
        return img