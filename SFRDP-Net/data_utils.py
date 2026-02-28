import os
import random
import cv2
import numpy as np
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rotate

def imread_uint(path: str, n_channels: int = 3) -> np.ndarray:
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G) dtype:uint8
    if n_channels == 1:
        img = cv2.imread(path, 0)
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise NotImplementedError
    return img

def uint2tensor3(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    tensor: torch.Tensor = torch.from_numpy(np.copy(img)).permute(
        2, 0, 1).float().div(255.)
    return tensor

def center_crop(image, target_size):
    h, w = image.shape[:2]
    th, tw = target_size
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    return image[max(y1, 0):min(y1 + th, h), max(x1, 0):min(x1 + tw, w)]

class TrainDataset(Dataset):
    def __init__(self, haze_root, clear_root, crop_size=256):
        super(TrainDataset, self).__init__()
        '''
        :param haze_root:  存放雾霾图像数据的路径
        :param clear_root: 存放清晰图像数据的路径
        :param crop_size:  裁剪大小
        '''
        self.haze_root = haze_root
        self.clear_root = clear_root
        self.crop_size = crop_size
        
        # 获取所有图像文件名
        self.hazy_img_names = sorted(os.listdir(haze_root))
        self.clear_img_names = sorted(os.listdir(clear_root))
        
        # 确保雾霾图像和清晰图像数量一致
        # assert len(self.hazy_img_names) == len(self.clear_img_names), \
        #     f"Number of hazy images ({len(self.hazy_img_names)}) doesn't match clear images ({len(self.clear_img_names)})"

    def aug_data(self, data, target):
        trans = A.Compose([
            #A.RandomCrop(self.crop_size, self.crop_size, always_apply=True),
            A.RandomCrop(self.crop_size, self.crop_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),  # 新增颜色抖动
        ], additional_targets={'target': 'image'})
        
        augmented = trans(image=data, target=target)
        data = augmented['image']
        target = augmented['target']
        
        data = uint2tensor3(data)
        target = uint2tensor3(target)
        
        return data, target

    def __getitem__(self, idx):
        # 获取图像路径
        haze_path = os.path.join(self.haze_root, self.hazy_img_names[idx])
        clear_path = os.path.join(self.clear_root, self.clear_img_names[idx])
        
        # 读取图像
        hazy_img = imread_uint(haze_path, 3)
        clear_img = imread_uint(clear_path, 3)
        
        # 确保图像大小一致
        clear_img = center_crop(clear_img, hazy_img.shape[:2])
        
        # 数据增强
        hazy_img, clear_img = self.aug_data(hazy_img, clear_img)
        
        return hazy_img, clear_img

    def __len__(self):
        return len(self.hazy_img_names)

class TrainDatasetDownscale(Dataset):
    def __init__(self, haze_root, clear_root, crop_size=256):
        super(TrainDatasetDownscale, self).__init__()
        '''
        :param haze_root:  存放雾霾图像数据的路径
        :param clear_root: 存放清晰图像数据的路径
        :param crop_size:  裁剪大小
        '''
        self.haze_root = haze_root
        self.clear_root = clear_root
        self.crop_size = crop_size
        self.downscale_factors = [0.5, 0.7, 1.0]
        
        # 获取所有图像文件名
        self.hazy_img_names = sorted(os.listdir(haze_root))
        self.clear_img_names = sorted(os.listdir(clear_root))
        
        assert len(self.hazy_img_names) == len(self.clear_img_names), \
            f"Number of hazy images ({len(self.hazy_img_names)}) doesn't match clear images ({len(self.clear_img_names)})"

    def aug_data(self, data, target):
        # 随机选择下采样因子
        downscale_factor = random.choice(self.downscale_factors)
        data = self.downsample(data, downscale_factor)
        target = self.downsample(target, downscale_factor)

        # 如果图像已经是目标大小，跳过裁剪
        if data.shape[0] == self.crop_size and data.shape[1] == self.crop_size:
            trans = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ], additional_targets={'target': 'image'})
        else:
            trans = A.Compose([
                A.RandomCrop(self.crop_size, self.crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ], additional_targets={'target': 'image'})
        
        augmented = trans(image=data, target=target)
        data = augmented['image']
        target = augmented['target']
        
        data = uint2tensor3(data)
        target = uint2tensor3(target)
        
        return data, target

    def downsample(self, img, scale):
        '''通过指定比例下采样图像'''
        height, width = img.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    def __getitem__(self, idx):
        # 获取图像路径
        haze_path = os.path.join(self.haze_root, self.hazy_img_names[idx])
        clear_path = os.path.join(self.clear_root, self.clear_img_names[idx])
        
        # 读取图像
        hazy_img = imread_uint(haze_path, 3)
        clear_img = imread_uint(clear_path, 3)
        
        # 确保图像大小一致
        clear_img = center_crop(clear_img, hazy_img.shape[:2])
        
        # 数据增强
        hazy_img, clear_img = self.aug_data(hazy_img, clear_img)
        
        return hazy_img, clear_img

    def __len__(self):
        return len(self.hazy_img_names)

class TestDataset(Dataset):
    def __init__(self, haze_root, clear_root):
        super(TestDataset, self).__init__()
        '''
        :param haze_root:  存放雾霾图像数据的路径
        :param clear_root: 存放清晰图像数据的路径
        '''
        self.haze_root = haze_root
        self.clear_root = clear_root
        
        # 获取所有图像文件名
        self.hazy_img_names = sorted(os.listdir(haze_root))
        self.clear_img_names = sorted(os.listdir(clear_root))
        
        assert len(self.hazy_img_names) == len(self.clear_img_names), \
            f"Number of hazy images ({len(self.hazy_img_names)}) doesn't match clear images ({len(self.clear_img_names)})"

    def __getitem__(self, idx):
        # 获取图像路径
        haze_path = os.path.join(self.haze_root, self.hazy_img_names[idx])
        clear_path = os.path.join(self.clear_root, self.clear_img_names[idx])
        
        # 读取图像
        hazy_img = imread_uint(haze_path, 3)
        clear_img = imread_uint(clear_path, 3)
        
        # 确保图像大小一致
        clear_img = center_crop(clear_img, hazy_img.shape[:2])
        
        # 转换为tensor
        hazy_img = uint2tensor3(hazy_img)
        clear_img = uint2tensor3(clear_img)
        
        return hazy_img, clear_img

    def __len__(self):
        return len(self.hazy_img_names)

class UnsupervisedDataset(Dataset):
    def __init__(self, haze_root, clear_root, crop_size=None):
        super(UnsupervisedDataset, self).__init__()
        '''
        :param haze_root:  存放雾霾图像数据的路径
        :param clear_root: 存放清晰图像数据的路径
        :param crop_size:  裁剪大小
        '''
        self.haze_root = haze_root
        self.clear_root = clear_root
        self.crop_size = crop_size
        
        # 获取所有图像文件名
        self.hazy_img_names = sorted(os.listdir(haze_root))
        self.clear_img_names = sorted(os.listdir(clear_root))
        
        self.len = len(self.hazy_img_names)
        self.all_indices = list(range(self.len))

    def aug_data(self, data):
        trans = A.Compose([
            A.RandomCrop(self.crop_size, self.crop_size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Flip(p=0.5),
        ])
        augmented = trans(image=data)
        data = augmented['image']
        data = uint2tensor3(data)
        return data

    def __getitem__(self, idx):
        # 获取雾霾图像
        haze_path = os.path.join(self.haze_root, self.hazy_img_names[idx])
        hazy_img = imread_uint(haze_path, 3)
        
        # 随机选择一个不同的清晰图像（无监督训练）
        c_idx = np.random.choice(self.all_indices)
        while c_idx == idx:
            c_idx = np.random.choice(self.all_indices)
        
        clear_path = os.path.join(self.clear_root, self.clear_img_names[c_idx])
        clear_img = imread_uint(clear_path, 3)
        
        # 确保图像大小一致
        clear_img = center_crop(clear_img, hazy_img.shape[:2])
        
        # 数据增强
        if self.crop_size is not None:
            hazy_img = self.aug_data(hazy_img)
            clear_img = self.aug_data(clear_img)
        else:
            hazy_img = uint2tensor3(hazy_img)
            clear_img = uint2tensor3(clear_img)
        
        return hazy_img, clear_img

    def __len__(self):
        return self.len

def get_dataloader(opt, use_downscale=True):
    """
    获取数据加载器
    :param opt: 包含数据路径和参数的配置对象
    :param use_downscale: 是否使用下采样版本的数据集（默认启用多尺度训练）
    :return: train_loader, test_loader
    """
    # 训练数据集
    if use_downscale:
        trainDataset = TrainDatasetDownscale(
            opt.train_haze_root,
            opt.train_clear_root,
            opt.crop_size
        )
    else:
        trainDataset = TrainDataset(
            opt.train_haze_root,
            opt.train_clear_root,
            opt.crop_size
        )
    
    # 测试数据集
    testDataset = TestDataset(
        opt.test_haze_root,
        opt.test_clear_root
    )
    
    # 数据加载器
    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        dataset=testDataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, test_loader

def get_unsupervised_dataloader(opt):
    """
    获取无监督学习的数据加载器
    :param opt: 包含数据路径和参数的配置对象
    :return: train_loader, test_loader
    """
    # 训练数据集
    trainDataset = UnsupervisedDataset(
        opt.train_haze_root,
        opt.train_clear_root,
        opt.crop_size
    )
    
    # 测试数据集（无监督测试时也使用无监督数据集，但不进行随机选择）
    testDataset = UnsupervisedDataset(
        opt.test_haze_root,
        opt.test_clear_root,
        crop_size=None  # 测试时不裁剪
    )
    
    # 数据加载器
    train_loader = DataLoader(
        dataset=trainDataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        dataset=testDataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 模拟配置对象
    class Opt:
        def __init__(self):
            self.train_haze_root = "data/train/haze"
            self.train_clear_root = "data/train/clear"
            self.test_haze_root = "data/test/haze"
            self.test_clear_root = "data/test/clear"
            self.crop_size = 256
            self.batch_size = 8
            self.num_workers = 4
    
    opt = Opt()
    
    # 获取有监督数据加载器
    train_loader, test_loader = get_dataloader(opt, use_downscale=False)
    print(f"有监督训练集大小: {len(train_loader.dataset)}")
    print(f"有监督测试集大小: {len(test_loader.dataset)}")
    
    # 获取下采样版本的数据加载器
    train_loader_down, test_loader_down = get_dataloader(opt, use_downscale=True)
    print(f"下采样训练集大小: {len(train_loader_down.dataset)}")
    
    # 获取无监督数据加载器
    unsup_train_loader, unsup_test_loader = get_unsupervised_dataloader(opt)
    print(f"无监督训练集大小: {len(unsup_train_loader.dataset)}")
    
    # 测试数据加载
    for haze, clear in train_loader:
        print(f"有监督训练数据形状: haze={haze.shape}, clear={clear.shape}")
        break
    
    for haze, clear in unsup_train_loader:
        print(f"无监督训练数据形状: haze={haze.shape}, clear={clear.shape}")
        break