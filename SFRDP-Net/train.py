# from model import Model
# from option import *
# from data_utils import get_dataloader
# from tensorboardX import SummaryWriter
# from Net import Net as Net
# from math import ceil
# import torch
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_loader,test_loader = get_dataloader(opt)
#     model = Model(Net,opt)
#     # print(model.model) 
#     model.print_model()
#     # 断点续训：检查是否需要加载 checkpoint
#     start_epoch = 0  # 默认从 epoch 0 开始
#     #if opt.model_loadPath and os.path.exists(opt.model_loadPath) and  opt.opt_loadPath and os.path.exists(opt.opt_loadPath):
#         #print(f"Loading checkpoint from: {opt.model_loadPath}")
#         #start_epoch = model.load_network()
#         #print(f"Resume training from epoch: {start_epoch + 1}")  # 下一个 epoch 开始

#     writer = SummaryWriter(log_dir=opt.logdir)
#     b_Psnr = 0
#     b_Ssim = 0
#     b_epoch = 0
#     for epoch in range(start_epoch, opt.total_epoch):
        
#         lr = model.scheduler.get_last_lr()[0]
#         loss = model.optimize_parameters(train_loader,epoch)
#         model.scheduler.step()
        

#         with torch.no_grad():
#             psnr,ssim = model.test(test_loader)
#             if psnr> b_Psnr:
#                 b_epoch = epoch
#                 b_Psnr = psnr
#                 b_Ssim = ssim
#             print("epoch:", epoch)
#             print("epoch:", epoch, "psnr:",psnr,"ssim:",ssim)
#             print("best_epoch:",b_epoch, "b_Psnr:", b_Psnr, "b_Ssim:", b_Ssim)
#             print("loss:",loss)

#             writer.add_scalar('lr', lr, epoch)
#             writer.add_scalar('psnr', psnr, epoch)
#             writer.add_scalar('ssim', ssim, epoch)
#             writer.add_scalar('train_loss', loss, epoch)
#             model.save_network(epoch,psnr,ssim)
#     writer.close()


# if __name__ == "__main__":
#     main()


#========================================================================================================================
import time
import torch
from option import opt
from data_utils import get_dataloader
from tensorboardX import SummaryWriter
from model import Model
from Net2 import PhysicsFourierFusionNet
import os


def freeze_physics(model, freeze=True):
    for p in model.physics.parameters():
        p.requires_grad = not freeze
# model.physics.parameters()
def freeze_re(model, freeze=True):
    for p in model.refine.parameters():
        p.requires_grad = not freeze
# model.physics.parameters()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===============================
    # dataloader
    # ===============================
    train_loader, test_loader = get_dataloader(opt)

    # ===============================
    # model wrapper
    # ===============================
    net = PhysicsFourierFusionNet().to(device)
    model = Model(net, opt)

    model.print_model()

    # ===============================
    # tensorboard
    # ===============================
    writer = SummaryWriter(log_dir=opt.logdir)

    # ===============================
    # resume (optional)
    # ===============================
    start_epoch = 0
    # if opt.model_loadPath and os.path.exists(opt.model_loadPath):
    #     start_epoch = model.load_network()
    #     print(f"Resume training from epoch {start_epoch}")

    # ===============================
    # best record
    # ===============================
    best_psnr = 0
    best_ssim = 0
    best_epoch = 0

    # ===============================
    # training loop
    # ===============================
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())   # 时间
    with open('./best.txt', 'a') as f:
        f.write(f'{t}\n')
    for epoch in range(start_epoch, opt.total_epoch):

        # ---------- stage control ----------
        # tempi = 0
        # if epoch < opt.freeze_phys_epoch:
        #     tempi = 0
        #     freeze_physics(model.model, freeze=False)
        #     freeze_re(model.model, freeze=True)
        # elif epoch < 500:
        #     tempi = 1
        #     freeze_physics(model.model, freeze=True)
        #     freeze_re(model.model, freeze=False)
        # else:
        # tempi = 2
        # freeze_physics(model.model, freeze=False)
        # freeze_re(model.model, freeze=False)

        # ---------- stage control ----------
        tempi = 0
# 000000000000000000000000000000000000000000000
        if epoch < opt.freeze_phys_epoch:
            tempi = 0
            freeze_physics(model.model, freeze=False)
            freeze_re(model.model, freeze=True)
        else:
            tempi = 2
            freeze_physics(model.model, freeze=False)
            freeze_re(model.model, freeze=False)
# 000000000000000000000000000000000000000000
        # ---------- train ----------

        # if epoch < opt.freeze_phys_epoch:
        #     for p in model.model.physics.parameters():
        #         p.requires_grad = False
        # else:
        #     for p in model.model.physics.parameters():
        #         p.requires_grad = True
        # ---------- train ----------

        # 学习率预热: 前 10 epochs 线性增加
        warmup_epochs = 10
        if epoch < warmup_epochs:
            warmup_lr = opt.lr * (epoch + 1) / warmup_epochs
            for param_group in model.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            lr = warmup_lr
        else:
            lr = model.scheduler.get_last_lr()[0]

        train_loss = model.optimize_parameters(train_loader, epoch, i=tempi)
        if epoch >= warmup_epochs:
            model.scheduler.step()
        train_loss = model.optimize_parameters(train_loader, epoch,i=tempi)
        model.scheduler.step()

        # ---------- eval ----------
        with torch.no_grad():
            psnr, ssim = model.test(test_loader, i=tempi, use_ema=True)  # 使用 EMA 模型测试

        # ---------- record ----------
        if psnr > best_psnr:
            best_psnr = psnr
            best_ssim = ssim
            best_epoch = epoch
            with open('./best.txt', 'a') as f:
                f.write(f'{best_psnr:.4f} {best_ssim:.4f} {best_epoch}\n')

        print(
            f"[Epoch {epoch}] "
            f"LR: {lr:.2e} | "
            f"Loss: {train_loss:.4f} | "
            f"PSNR: {psnr:.3f} | "
            f"SSIM: {ssim:.4f} | "
            f"Best: {best_psnr:.3f} @ {best_epoch}"
        )

        # ---------- tensorboard ----------
        writer.add_scalar("lr", lr, epoch)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("psnr", psnr, epoch)
        writer.add_scalar("ssim", ssim, epoch)

        # ---------- save ----------
        model.save_network(epoch, psnr, ssim)

    writer.close()


if __name__ == "__main__":
    main()
