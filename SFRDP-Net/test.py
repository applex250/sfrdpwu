# from model import Model
# from option import *
# from data_utils import get_dataloader
# from tensorboardX import SummaryWriter
# from Net import Net
# from PIL import Image
# from torchvision import transforms
# import torchvision.utils as vutils
# import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter(opt.logdir)
# train_loader,test_loader = get_dataloader(opt)

# # --- 关键修改 ---
# net_instance = Net(opt)        # 实例化网络
# model = Model(net_instance, opt)  # 传入实例
# model.load_network()



# with torch.no_grad():
#     psnr,ssim = model.test(test_loader)
#     print("psnr:",psnr,"ssim:",ssim)

# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转换为张量
# ])

# image_path = "/home/shared_dir/yao_ce/statehaze1k/Haze1k/Haze1k_thin/dataset/test/hazy/382.png"
# input_image = Image.open(image_path).convert('RGB')
# input_tensor = transform(input_image).unsqueeze(0).to(device)
# # 模型推理
# model = model.model
# model.load_state_dict(torch.load("./checkpoints/best_model.pth"))
# model.eval()  # 切换到评估模式
# with torch.no_grad():
#     output_tensor = model(input_tensor)
# output_path = "RDDTF.png"  # 指定输出路径
# vutils.save_image(output_tensor.cpu(), output_path)

import os
import torch
from torchvision import utils as vutils
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from data_utils import get_dataloader
from model import Model
from Net2 import PhysicsFourierFusionNet
from option import opt
from metrics import *
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ===============================
    # dataloader
    # ===============================
    _, test_loader = get_dataloader(opt)

    # ===============================
    # model
    # ===============================
    net = PhysicsFourierFusionNet().to(device)
    model = Model(net, opt)

    # 加载训练好的权重
    checkpoint_path = "./checkpoints/best_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.model.load_state_dict(state_dict)
    model.model.eval()

    # ===============================
    # 输出目录
    # ===============================
    output_dir = "./test_results"
    os.makedirs(output_dir, exist_ok=True)

    # ===============================
    # 测试集逐图推理
    # ===============================
    psnr_list = []
    ssim_list = []
    psnr_p_list = []
    ssim_p_list = []
    psnr_plus_list = []
    ssim_plus_list = []

    with torch.no_grad():
        for i, (input_tensor, gt_tensor) in enumerate(test_loader):
            input_tensor = input_tensor.to(device)
            gt_tensor = gt_tensor.to(device)

            # 推理
            output = model.model(input_tensor)
            output_tensor = output[-1]
            output_tensor_p = output[0]

            # 计算 PSNR 和 SSIM
            psnr_val = psnr(gt_tensor, output_tensor)
            ssim_val = ssim(gt_tensor, output_tensor)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            #----------------------------
            psnr_val_p = psnr(gt_tensor, output_tensor_p)
            ssim_val_p = ssim(gt_tensor, output_tensor_p)
            psnr_p_list.append(psnr_val_p)
            ssim_p_list.append(ssim_val_p)
            psnr_plus_list.append(psnr_val-psnr_val_p)
            ssim_plus_list.append(ssim_val-ssim_val_p)

            #----------------------------
            # 打印结果
            print(f"[{i+1}/{len(test_loader.dataset)}] PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.4f}")
            print(f"[{i+1}/{len(test_loader.dataset)}] PSNR_p: {psnr_val_p:.3f}, SSIM_p: {ssim_val_p:.4f}")
            print(f"[{i+1}/{len(test_loader.dataset)}] PSNR+: {psnr_val-psnr_val_p:.3f}, SSIM+: {ssim_val-ssim_val_p:.4f}")


            # 保存输出图片
            output_path = os.path.join(output_dir, f"result_{i+1:04d}.png")
            vutils.save_image(output_tensor.cpu(), output_path)
            vutils.save_image(output_tensor_p.cpu(), output_path.replace(".png", "_p.png"))
    # 平均指标
    mean_psnr = sum(psnr_list) / len(psnr_list)
    mean_ssim = sum(ssim_list) / len(ssim_list)
    mean_psnr_p = sum(psnr_p_list) / len(psnr_p_list)
    mean_ssim_p = sum(ssim_p_list) / len(ssim_p_list)
    mean_psnr_plus = sum(psnr_plus_list) / len(psnr_plus_list)
    mean_ssim_plus = sum(ssim_plus_list) / len(ssim_plus_list)
    print(f"\n=== Average PSNR: {mean_psnr:.3f}, Average SSIM: {mean_ssim:.4f} ===")
    print(f"=== Average PSNR_p: {mean_psnr_p:.3f}, Average SSIM_p: {mean_ssim_p:.4f} ===")
    print(f"=== Average PSNR+: {mean_psnr_plus:.3f}, Average SSIM+: {mean_ssim_plus:.4f} ===")

if __name__ == "__main__":
    main()
