import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as T

# 导入模块同上...

@torch.no_grad()
def inference(image_path, phys_net, diffusion_net, device='cuda'):
    # Preprocess
    img = Image.open(image_path).convert('RGB').resize((256, 256))
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
    hazy = transform(img).unsqueeze(0).to(device) # [1, 3, H, W]
    
    # 1. Stage 1
    j_phys = run_stage1(phys_net, hazy)
    
    # 2. Stage 2 (Diffusion Sampling)
    # 输入: Hazy (原图) + J_phys (物理引导)
    # 输出: 干净图
    result = diffusion_net.p_sample_loop(hazy, j_phys)
    
    # Denormalize & Save
    result = (result + 1) * 0.5
    save_image(result, 'dehazed_output.png')

# 使用示例
# 加载模型...
# inference('test_hazy.jpg', phys_net, diffusion, device='cuda')