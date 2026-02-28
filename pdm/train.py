import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os

# 导入你的模块
from models.unet import ConditionalUNet
from models.diffusion import GaussianDiffusion
from data.dataset import DehazeDataset, run_stage1
# 假设你的物理模型定义在 physics_net.py
from models.physics_net import AdvancedPhysicsNet 


'''
Project/
├── models/
│   ├── unet.py          # Conditional U-Net 网络定义
│   ├── diffusion.py     # 高斯扩散数学逻辑
│   └── physics_net.py   # (你之前的物理模型代码放这里)
├── data/
│   └── dataset.py       # 数据加载与预处理 (自动调用 Stage 1)
├── train.py             # 训练主脚本
└── inference.py         # 推理脚本
'''

def train():
    # --- Configs ---
    device = 'cuda'
    lr = 1e-4
    batch_size = 4 # 根据显存调整，Diffusion 比较吃显存
    epochs = 100
    save_dir = '/home/user/sfrdpwu/pdm/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # --- Models ---
    # 1. 加载 Stage 1 (Pre-trained)
    phys_net = AdvancedPhysicsNet().to(device)
    phys_net.load_state_dict(torch.load('/home/user/sfrdpwu/pdm/best_s1/best_model.pth'))
    phys_net.eval() # 冻结！
    
    # 2. 初始化 Stage 2 (Diffusion)
    unet = ConditionalUNet(in_channels=9, out_channels=3).to(device)
    diffusion = GaussianDiffusion(unet).to(device)
    
    # --- Optimizer & Scaler ---
    optimizer = optim.AdamW(unet.parameters(), lr=lr)
    scaler = GradScaler() # 混合精度
    
    # --- Data ---
    dataset = DehazeDataset('/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_moderate/dataset/train/hazy', '/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_moderate/dataset/train/GT', img_size=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- Loop ---
    print("Start Training Conditional Diffusion...")
    for epoch in range(epochs):
        unet.train()
        epoch_loss = 0
        
        for hazy, gt in loader:
            hazy = hazy.to(device)
            gt = gt.to(device)
            
            # 1. 现场生成 Stage 1 结果 (J_phys)
            with torch.no_grad():
                j_phys = run_stage1(phys_net, hazy)
            
            optimizer.zero_grad()
            
            # 2. 随机采样时间步 t
            t = torch.randint(0, 1000, (hazy.size(0),), device=device).long()
            
            # 3. 前向加噪 & 预测噪声 (Mixed Precision)
            with autocast():
                # 加噪 GT
                x_t, noise = diffusion.q_sample(gt, t)
                # 预测噪声 (Condition: Hazy + J_phys)
                pred_noise = unet(x_t, t, hazy, j_phys)
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

            # 4. 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.6f}")
        
        # Save
        if (epoch+1) % 10 == 0:
            torch.save(unet.state_dict(), f"{save_dir}/diff_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    train()