from tqdm import tqdm
from torch.backends import cudnn
from torch import optim
from torch.optim.adamw import AdamW
from pytorch_msssim import *
from metrics import *
import torch.nn as nn
import torch
from loss import *
import numpy as np
import torchvision.utils as vutils
import os
import torch.fft
import torchvision
import copy

class Model(nn.Module):
    def __init__(self, net, opts):
        super().__init__()
        self.opt = opts
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler('cuda')
        self.model = net.to(self.device)
        self.optimizer = AdamW(params=filter(lambda x: x.requires_grad, self.model.parameters()), lr=opts.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, amsgrad=False, weight_decay=0.01)

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.total_epoch,
                                                              eta_min=self.opt.lr * 0.05)
        self.set_seed(opts.seed)
        #----------------------------------------------------
        self.criterion1 = CharbonnierLoss().to(self.device)
        self.criterion2 = fftLoss().to(self.device)
        self.criterion3 = PFDC().to(self.device)
        self.criterion4 = ColorLoss().to(self.device)
        self.criterion5 = SSIMLoss(loss_weight=0.15).to(self.device)  # 新增 SSIM Loss
        self.criterion = nn.SmoothL1Loss(beta=1.0).to(self.device)

        #----------------------------------------------------
        # EMA (Exponential Moving Average)
        self.ema_decay = 0.999
        self.model_ema = copy.deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.requires_grad = False


    def set_seed(self, seed):
        seed = int(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def update_ema(self):
        """更新 EMA 模型权重"""
        with torch.no_grad():
            for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                param_ema.data = self.ema_decay * param_ema.data + (1 - self.ema_decay) * param.data

    def save_network(self, epoch, now_psnr, now_ssim):
        if self.best_psnr < now_psnr and self.best_ssim < now_ssim:
            self.best_psnr = now_psnr
            self.best_ssim = now_ssim
            model_path = os.path.join(self.opt.model_Savepath, 'best_model.pth')
            opt_path = os.path.join(self.opt.optim_Savepath, 'best_opt.pth')
        elif epoch % self.opt.save_fre_step == 0:
            model_path = os.path.join(self.opt.model_Savepath, 'E{}_model.pth'.format(epoch))
            opt_path = os.path.join(self.opt.optim_Savepath, 'E{}_opt.pth'.format(epoch))
        else:
            return

        # model_save
        torch.save(
            self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            model_path)
        # optim_save
        opt_state = {'epoch': epoch, 'ssim': now_ssim, 'psnr': now_psnr,
                     'scheduler': self.scheduler.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(opt_state, opt_path)

    def load_network(self):
        model_path = self.opt.model_loadPath
        opt_path = self.opt.opt_loadPath
        self.model.load_state_dict(torch.load(model_path))
        optim_state = torch.load(opt_path,weights_only=False)
        self.start_epoch = optim_state['epoch']
        self.best_psnr = optim_state['psnr']
        self.best_ssim = optim_state['ssim']
        self.optimizer.load_state_dict(optim_state['optimizer'])
        self.scheduler.load_state_dict(optim_state['scheduler'])
        print(self.best_psnr)
        print(self.best_ssim)
        return self.start_epoch

    def optimize_parameters(self, train_loader, epoch, i):
        self.model.train()
        total_loss = 0

        # ColorLoss 权重调度: 前 50 epochs 从 0.1 逐步增加到 0.6
        color_weight = min(0.6, 0.1 + 0.5 * epoch / 50)

        for hazy, gt in train_loader:
            hazy = hazy.to(self.device)
            gt   = gt.to(self.device)

            self.optimizer.zero_grad()

            J_phys, t, g, J_final = self.model(hazy, return_phys=True)

            # reconstruction
            loss_rec = self.criterion1(J_final, gt)

            # physics consistency
            loss_phys = self.criterion1(J_phys, gt)

            # transmission smoothness
            loss_t = (
                torch.mean(torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])) +
                torch.mean(torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :]))
            )

            # g low-frequency prior
            g_lp = torch.nn.functional.avg_pool2d(g, 15, 1, 7)
            loss_g = torch.mean(torch.abs(g - g_lp))

            # 物理一致性检查: 重建的有雾图像应该与输入一致
            # I = t * J + (1 - t) * g
            I_reconstructed = t * J_final + (1 - t) * g
            loss_consistency = self.criterion1(I_reconstructed, hazy)

            # 增强 t/g 约束 + 添加物理一致性损失
            lp = loss_phys + 0.15 * loss_t + 0.08 * loss_g + 0.05 * loss_consistency

            # loss_p2 = self.criterion2(J_phys, J_final)
            # loss_p2 = self.criterion2(J_phys, gt)
            # loss_p3 = self.criterion3(J_phys, gt)
            l1 = self.criterion1(J_final, gt)
            l2 = self.criterion2(J_final, gt)
            l3 = self.criterion3(J_final, gt)
            l4 = color_weight * (1 - self.criterion4.cos(J_final, gt).mean())  # ColorLoss 调度
            l5 = self.criterion5(J_final, gt)  # SSIM Loss
            # pred_fft = torch.fft.rfft2(J_final, norm='ortho')
            # target_fft = torch.fft.rfft2(gt, norm='ortho')
            # lfft= self.criterion(torch.abs(pred_fft), torch.abs(target_fft))

            # 调整精炼损失权重：增加频域和感知损失权重
            lr = l1 + 0.15 * l2 + 0.08 * l3 + 0.5 * l4 + l5  # 添加 SSIM Loss
            # lp = loss_phys +0.1 * loss_t +0.05 * loss_g +  0.001*loss_p2+ 0.01*loss_p3
            loss = 0
            if i == 0:
                # 阶段 0: 只有物理损失
                loss = lp
            elif i == 1:
                # 阶段 1: 精炼损失 + 轻微物理约束（10%）
                loss = lr + 0.1 * lp
            elif i == 2:
                # 阶段 2: 联合训练，增加物理约束权重
                loss = 0.6 * lr + 0.4 * lp


            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.8)
            self.optimizer.step()

            # 更新 EMA
            self.update_ema()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    # def optimize_parameters(self, train_dataloader, epoch):
    #     self.model.train()
    #     total_loss = 0.0
    #     for idx, (input_img, label_img) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=True)):
    #         input_img = input_img.to(self.device)
    #         label_img = label_img.to(self.device)
    #         #----------------------------------------------------
    #         output = self.model(input_img)
    #         loss = self.criterion1(output, label_img)
    # +0.001*self.criterion2(output, label_img)
    # +0.01*self.criterion3(output, label_img)
    #         #----------------------------------------------------
    #         #loss = self.model(label_img,input_img)
    #         loss.backward()
    #         total_loss += loss.item()
    #         self.optimizer.step()
    #         self.optimizer.zero_grad(set_to_none=True)
    #     return total_loss / len(train_dataloader)

    def test(self, test_dataloder, i, use_ema=False):
        # 选择使用 EMA 模型还是原始模型
        model_to_use = self.model_ema if use_ema else self.model
        model_to_use.eval()
        # torch.cuda.empty_cache()
        ssims = []
        psnrs = []
        for step, (inputs, targets) in enumerate(test_dataloder):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = model_to_use(inputs)
            if isinstance(pred, tuple):
                if i == 0:
                    pred = pred[0]
                else:
                    pred = pred[-1]
                       # 只要 J_final
            else:
                pred = pred
            ssims.append(ssim(pred, targets).item())
            psnrs.append(psnr(pred, targets))
        ssim_mean = np.mean(ssims)
        psnr_mean = np.mean(psnrs)
        return psnr_mean, ssim_mean

    def print_model(self):
        pytorch_total_params = sum(p.nelement() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: ==> {}".format(pytorch_total_params / 1e6))
# --------------------------------
# import torch
# import torch.nn as nn
# from torch import optim
# from torch.optim.adamw import AdamW
# import numpy as np
# import os
# from tqdm import tqdm

# # 假设这些是你项目中的辅助文件
# # 如果没有，请确保你有对应的实现或替换为标准 Loss
# from pytorch_msssim import ssim 
# from metrics import psnr # 假设你有一个 metrics.py 计算 psnr
# from loss import fftLoss, PFDC # 假设这是你自定义的 Loss

# class Model(nn.Module):
#     def __init__(self, net, opts):
#         super().__init__()
#         self.opt = opts
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
#         # 1. 网络初始化
#         self.model = net.to(self.device)
        
#         # 2. 优化器 (FP32 标准设置)
#         # 移除 scaler，回归最稳健的训练方式
#         self.optimizer = AdamW(
#             params=filter(lambda x: x.requires_grad, self.model.parameters()), 
#             lr=opts.lr,
#             betas=(0.9, 0.999),
#             eps=1e-08, 
#             weight_decay=1e-4 # 稍微增加一点 weight decay 有助于稳定
#         )

#         # 3. 学习率调度器
#         self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
#             self.optimizer, 
#             T_max=self.opt.total_epoch,
#             eta_min=self.opt.lr * 0.01
#         )
        
#         # 4. 初始化 Loss 函数
#         self.criterion_l1 = nn.SmoothL1Loss(beta=1.0).to(self.device)
#         self.criterion_fft = fftLoss().to(self.device) 
#         self.criterion_percep = PFDC().to(self.device)
        
#         # 5. 记录指标
#         self.best_psnr = 0
#         self.best_ssim = 0
#         self.start_epoch = 0
        
#         # 设置随机种子
#         self.set_seed(opts.seed)

#     def set_seed(self, seed):
#         seed = int(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False # 关闭 benchmark 以保证复现性和稳定性

#     def optimize_parameters(self, train_loader, epoch, stage):
#         """
#         :param stage: 
#             0 -> 物理预热 (Physics Warmup)
#             1 -> 精修训练 (Refinement Training)
#             2 -> 联合微调 (Joint Finetuning)
#         """
#         self.model.train()
#         total_loss = 0
#         count = 0
        
#         # 使用 tqdm 包装 loader，实时显示 Loss
#         loop = tqdm(train_loader, desc=f"Epoch {epoch} [Stage {stage}]")

#         for hazy, gt in loop:
#             hazy = hazy.to(self.device)
#             gt   = gt.to(self.device)

#             # --- 1. 清空梯度 ---
#             self.optimizer.zero_grad()

#             # --- 2. 前向传播 (FP32) ---
#             # 这里的 model 应该是我们之前定义的 HybridDehazeNet
#             J_phys, t, g, J_final = self.model(hazy, return_phys=True)

#             # --- 3. Loss 计算 ---
            
#             # [A] 物理 Loss 部分
#             loss_phys = self.criterion_l1(J_phys, gt)
            
#             # 透射率 t 的平滑约束 (TV Loss)
#             loss_t = (torch.mean(torch.abs(t[:, :, :, :-1] - t[:, :, :, 1:])) +
#                       torch.mean(torch.abs(t[:, :, :-1, :] - t[:, :, 1:, :])))
            
#             # 大气光 g 的平滑约束 (避免 g 出现高频噪声)
#             g_blur = torch.nn.functional.avg_pool2d(g, 15, 1, 7)
#             loss_g = torch.mean(torch.abs(g - g_blur))

#             # [B] Refine/最终 Loss 部分
#             loss_rec = self.criterion_l1(J_final, gt)
            
#             # FFT Loss (频域一致性)
#             pred_fft = torch.fft.rfft2(J_final, norm='ortho')
#             gt_fft   = torch.fft.rfft2(gt, norm='ortho')
#             loss_fft = self.criterion_l1(torch.abs(pred_fft), torch.abs(gt_fft))
            
#             # 感知 Loss
#             loss_pfdc = self.criterion_percep(J_final, gt)

#             # --- 4. 阶段性 Loss 组合策略 ---
#             loss = 0.0
            
#             # 定义组件包
#             L_physics = loss_phys + 0.1 * loss_t + 0.1 * loss_g
#             L_refine  = loss_rec + 0.05 * loss_fft + 0.04 * loss_pfdc

#             if stage == 0:
#                 # 【阶段 0】只练物理分支，不管 Refine
#                 # 目的：让 t 和 g 先收敛到一个合理范围，给后续打基础
#                 loss = L_physics
            
#             elif stage == 1:
#                 # 【阶段 1】主练 Refine，辅练物理
#                 # 目的：提升画质，同时保持物理参数不跑偏
#                 loss = L_refine + 0.1 * L_physics
                
#             elif stage == 2:
#                 # 【阶段 2】全开
#                 loss = L_refine + 0.2 * L_physics

#             # --- 5. 安全检查 (NaN 熔断) ---
#             if torch.isnan(loss):
#                 print(f"[Warning] Loss is NaN at Epoch {epoch}. Skipping batch.")
#                 continue

#             # --- 6. 反向传播 (FP32) ---
#             loss.backward()

#             # --- 7. 梯度裁剪 (核心维稳步骤) ---
#             # 对 ResNet+DeepPhysics 结构至关重要，防止梯度爆炸
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

#             # --- 8. 参数更新 ---
#             self.optimizer.step()

#             total_loss += loss.item()
#             count += 1
            
#             # 更新进度条
#             loop.set_postfix(loss=loss.item())

#         return total_loss / (count + 1e-6)

#     def test(self, test_loader, stage):
#         self.model.eval()
#         ssims = []
#         psnrs = []
        
#         # 测试时不需要计算梯度，节省显存
#         with torch.no_grad():
#             for inputs, targets in test_loader:
#                 inputs = inputs.to(self.device)
#                 targets = targets.to(self.device)
                
#                 # 获取输出
#                 J_phys, t, g, J_final = self.model(inputs, return_phys=True)
                
#                 # 根据阶段选择看哪张图
#                 # 如果是纯物理预热阶段，看 J_phys 的效果
#                 # 否则看 J_final
#                 if stage == 0:
#                     pred = J_phys
#                 else:
#                     pred = J_final

#                 # 强制约束到 [0, 1] 范围，这对指标计算很重要
#                 pred = torch.clamp(pred, 0.0, 1.0)
                
#                 ssims.append(ssim(pred, targets).item())
#                 psnrs.append(psnr(pred, targets)) # 确保你的 psnr 函数支持 tensor 输入

#         return np.mean(psnrs), np.mean(ssims)

#     def save_network(self, epoch, now_psnr, now_ssim):
#         # 保存最佳模型
#         if now_psnr > self.best_psnr:
#             self.best_psnr = now_psnr
#             self.best_ssim = now_ssim
#             save_path = os.path.join(self.opt.model_Savepath, 'best_model.pth')
#             torch.save(self.model.state_dict(), save_path)
#             print(f"Saved Best Model at Epoch {epoch} with PSNR: {self.best_psnr:.2f}")

#         # 定期保存
#         if epoch % self.opt.save_fre_step == 0:
#             save_path = os.path.join(self.opt.model_Savepath, f'model_epoch_{epoch}.pth')
#             torch.save(self.model.state_dict(), save_path)

#     def load_network(self):
#         load_path = self.opt.model_loadPath
#         if load_path is not None and os.path.exists(load_path):
#             print(f"Loading model from {load_path}...")
#             # weights_only=False 是为了兼容旧版 pytorch 保存的权重
#             state_dict = torch.load(load_path, map_location=self.device)
            
#             # 兼容性处理：如果保存的是 module.xxx (DataParallel) 但现在是单卡
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 name = k[7:] if k.startswith('module.') else k
#                 new_state_dict[name] = v
                
#             self.model.load_state_dict(new_state_dict)
#             print("Model loaded successfully.")
#         else:
#             print("No pre-trained model found. Training from scratch.")

#     def print_model(self):
#         params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print(f"Model Structure: {self.model.__class__.__name__}")
#         print(f"Total Trainable Parameters: {params / 1e6:.2f} M")