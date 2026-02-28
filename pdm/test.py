import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# --- 导入你的模型 ---
# 请根据实际文件名修改 import 路径
from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion
from models.physics_net import AdvancedPhysicsNet  # 假设你的物理模型叫这个

# --- 配置参数 ---
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'img_size': 256,            # 必须是 16 的倍数，推荐 256
    'phys_weight': '/home/user/sfrdpwu/pdm/best_s1/best_model.pth',
    'diff_weight': '/home/user/sfrdpwu/pdm/checkpoints/diff_epoch_100.pth',
    'test_hazy_dir': '/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_moderate/dataset/test/hazy',
    'test_gt_dir': '/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_moderate/dataset/test/GT',  # 如果没有 GT，设为 None
    'save_dir': './pdm/results',
    'sample_steps': 1000        # 采样步数，和训练时保持一致
}

def calc_metrics(img_tensor, gt_tensor):
    """计算单张图片的 PSNR 和 SSIM"""
    # Tensor [0, 1] -> Numpy [0, 255] (H, W, C)
    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    gt = gt_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    img = (img * 255).astype(np.uint8)
    gt = (gt * 255).astype(np.uint8)
    
    p = psnr_loss(gt, img, data_range=255)
    # win_size 必须小于图片尺寸，对于小图可能需要调整，这里设为 7 或 11
    s = ssim_loss(gt, img, data_range=255, channel_axis=2, win_size=11)
    return p, s

@torch.no_grad()
def test():
    # 1. 准备目录
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = CONFIG['device']
    
    print(f"Loading models on {device}...")
    
    # 2. 加载 Stage 1: 物理模型
    phys_net = AdvancedPhysicsNet().to(device)
    phys_net.load_state_dict(torch.load(CONFIG['phys_weight'], map_location=device))
    phys_net.eval()
    
    # 3. 加载 Stage 2: Diffusion 模型
    unet = ConditionalUNet(in_channels=9, out_channels=3).to(device)
    # 注意：如果训练时用了 DataParallel，加载权重时可能需要去掉 'module.' 前缀
    unet.load_state_dict(torch.load(CONFIG['diff_weight'], map_location=device))
    unet.eval()
    
    diffusion = GaussianDiffusion(unet, timesteps=CONFIG['sample_steps'])
    diffusion.to(device) # 如果你的 diffusion 类有 buffer，需要 to(device)

    # 4. 数据预处理
    transform = T.Compose([
        T.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到 [-1, 1]
    ])
    
    files = sorted(os.listdir(CONFIG['test_hazy_dir']))
    avg_psnr, avg_ssim = 0, 0
    count = 0
    
    print(f"Start inference on {len(files)} images...")
    
    for name in tqdm(files):
        # --- A. 读取与预处理 ---
        hazy_path = os.path.join(CONFIG['test_hazy_dir'], name)
        hazy_img = Image.open(hazy_path).convert('RGB')
        hazy_tensor = transform(hazy_img).unsqueeze(0).to(device) # [1, 3, H, W], Range [-1, 1]
        
        # --- B. Stage 1 推理 (Physics) ---
        # 物理模型通常接受 [0, 1] 输入，所以要反归一化一下
        phys_input = (hazy_tensor + 1) * 0.5
        j_phys, _, _, _ = phys_net(phys_input)
        
        # 将 Stage 1 结果转回 [-1, 1] 给 Diffusion 用
        j_phys_norm = j_phys * 2.0 - 1.0
        
        # --- C. Stage 2 推理 (Diffusion Sampling) ---
        # 使用 DDPM 采样，条件是 (Hazy + J_phys)
        j_final_norm = diffusion.p_sample_loop(hazy_tensor, j_phys_norm)
        
        # --- D. 后处理 ---
        # 转回 [0, 1]
        j_final = (j_final_norm + 1) * 0.5
        hazy_vis = (hazy_tensor + 1) * 0.5
        
        # --- E. 计算指标 (如果有 GT) ---
        metric_str = ""
        vis_list = [hazy_vis, j_phys, j_final] # 有雾，物理结果，最终结果
        
        if CONFIG['test_gt_dir']:
            gt_path = os.path.join(CONFIG['test_gt_dir'], name)
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).convert('RGB')
                gt_tensor = transform(gt_img).unsqueeze(0).to(device)
                gt_vis = (gt_tensor + 1) * 0.5
                
                # 计算 PSNR/SSIM
                p, s = calc_metrics(j_final, gt_vis)
                avg_psnr += p
                avg_ssim += s
                count += 1
                metric_str = f" | PSNR: {p:.2f} SSIM: {s:.4f}"
                
                vis_list.append(gt_vis) # 把 GT 也拼上去
        
        # --- F. 保存结果 ---
        # 将图片横向拼接: Hazy | Physics | Diffusion | GT
        combined = torch.cat(vis_list, dim=3) 
        save_path = os.path.join(CONFIG['save_dir'], name)
        save_image(combined, save_path)
        
        print(f"Saved: {name}{metric_str}")

    if count > 0:
        print(f"\nAverage PSNR: {avg_psnr / count:.4f}")
        print(f"Average SSIM: {avg_ssim / count:.4f}")

if __name__ == '__main__':
    test()