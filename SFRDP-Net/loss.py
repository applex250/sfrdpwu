import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from torchvision import models
from torchvision.models import VGG19_Weights
class fftLoss(nn.Module):
    # def __init__(self):
    #     super(fftLoss, self).__init__()

    # def forward(self, x, y):
    #     diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
    #     loss = torch.mean(abs(diff))
    #     #loss = (1/n) * Σ|diff_i|loss = (1/n) * Σ|diff_i|
    #     return loss
    def __init__(self, loss_weight=0.1):
        super(fftLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        # 自动适配 device，不要写死 cuda:0
        # 使用 rfft2 处理实数图像，效率高一倍
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # 幅度谱 Loss (Amplitude): 恢复清晰度，去除伪影
        loss_amp = self.criterion(pred_fft.abs(), target_fft.abs())
        
        # 相位谱 Loss (Phase): 恢复结构和边缘
        loss_pha = self.criterion(pred_fft.angle(), target_fft.angle())
        
        # 经验公式：幅度权重 1.0，相位权重 0.5
        return self.loss_weight * (loss_amp + 0.5 * loss_pha)
# ==========================================
# 3. 改进版 Charbonnier Loss (比 L1 更鲁棒)
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
# ==========================================
# 2. 新增 Color Loss (解决偏蓝/偏紫)
# ==========================================
class ColorLoss(nn.Module):
    def __init__(self, loss_weight=0.5):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        # 强制 RGB 向量方向一致
        # 加上 1e-8 防止除零
        loss = 1 - self.cos(pred, target).mean()
        return self.loss_weight * loss   

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, a, p):
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)
        loss = 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            contrastive = d_ap
            loss += self.weights[i] * contrastive
        return loss


class PFDC(nn.Module):
    def __init__(self):
        super(PFDC, self).__init__()
        self.vgg = Vgg19().cuda()
        self.smooth_l1 = nn.SmoothL1Loss().cuda()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def loss_formulation(self, x, y):
        B, C, H, W = x.shape
        x_mean = x.mean(dim=[2,3],keepdim=True)
        y_mean = y.mean(dim=[2, 3], keepdim=True)
        dis = torch.abs(x_mean-y_mean)
        dis_max =  torch.max(dis,dim=1)[0].view(B,1,1,1)
        dis = dis / dis_max
        dis = torch.exp(dis / 0.2)-0.3
        #γ=0.3  τ = 0.2
        dis = dis.detach()
        return dis

    def forward(self, out, y):
        out_vgg, y_vgg = self.vgg(out), self.vgg(y)
        loss = 0
        for i in range(len(out_vgg)):
            w = self.loss_formulation(out_vgg[i], y_vgg[i].detach())
            contrastive = self.smooth_l1(out_vgg[i] * w, y_vgg[i].detach() * w)
            loss += self.weights[i] * contrastive
        return loss


# ==========================================
# 4. SSIM Loss (结构相似性损失)
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim_loss(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight=0.15):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        # Clamp to [0, 1]
        img1 = torch.clamp(img1, min=0, max=1)
        img2 = torch.clamp(img2, min=0, max=1)

        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        ssim_val = _ssim_loss(img1, img2, window, self.window_size, channel, self.size_average)
        return self.loss_weight * (1 - ssim_val)
