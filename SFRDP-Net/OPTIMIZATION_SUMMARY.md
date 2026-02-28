# SFRDP-Net 优化总结

## 目标

将模型 PSNR 从 **27.09 dB** 提升至 **30 dB**（+2.9 dB）

## 问题诊断

| 当前配置 | 问题 | 影响 |
|---------|------|------|
| 无 SSIM Loss | 只优化像素误差，不优化结构相似性 | PSNR 损失约 0.5-1 dB |
| 无学习率预热 | 训练初期不稳定 | 收敛变慢 |
| ColorLoss=0.6（固定） | 早期权重过高，干扰物理模型学习 | 降低初始收敛效果 |
| 无多尺度训练 | 只处理 256×256 固定尺寸 | 泛化能力受限 |
| 无 EMA | 模型权重波动 | 损失约 0.3-0.5 dB |
| 数据增强简单 | 只有翻转和旋转 | 鲁棒性不足 |

## 优化方案

### 1. 添加 SSIM Loss

**文件**: `loss.py`

```python
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight=0.15):
        ...
```

### 2. 学习率预热

**文件**: `train.py`

```python
# 前 10 epochs 线性增加学习率
warmup_epochs = 10
if epoch < warmup_epochs:
    warmup_lr = opt.lr * (epoch + 1) / warmup_epochs
```

### 3. ColorLoss 权重调度

**文件**: `model.py`

```python
# 从 0.1 逐步增加到 0.6，避免早期干扰
color_weight = min(0.6, 0.1 + 0.5 * epoch / 50)
```

### 4. 启用多尺度训练

**文件**: `data_utils.py`

```python
def get_dataloader(opt, use_downscale=True):  # 默认启用
    ...
```

下采样因子: [0.5, 0.7, 1.0]

### 5. 实现 EMA

**文件**: `model.py`

```python
self.ema_decay = 0.999
self.model_ema = copy.deepcopy(self.model)

def update_ema(self):
    with torch.no_grad():
        for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            param_ema.data = self.ema_decay * param_ema.data + (1 - self.ema_decay) * param.data
```

### 6. 颜色抖动数据增强

**文件**: `data_utils.py`

```python
A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5)
```

### 7. 延长物理模块冻结期

**文件**: `option.py`

```python
--freeze_phys_epoch = 120  # 从 80 增加到 120
```

## 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `loss.py` | 添加 SSIMLoss 类（权重 0.15） |
| `model.py` | 添加 SSIM Loss、EMA、ColorLoss 调度 |
| `train.py` | 添加 10 epochs 学习率预热 |
| `data_utils.py` | 启用多尺度训练、添加颜色抖动 |
| `option.py` | freeze_phys_epoch 改为 120 |

## 预期效果

| 优化项 | 预期 PSNR 提升 |
|--------|---------------|
| SSIM Loss | +0.5 ~ +1.0 dB |
| 学习率预热 | +0.2 ~ +0.5 dB |
| ColorLoss 调度 | +0.3 ~ +0.5 dB |
| 多尺度训练 | +0.5 ~ +1.0 dB |
| EMA | +0.3 ~ +0.5 dB |
| 数据增强 | +0.2 ~ +0.4 dB |
| **总计** | **+2.0 ~ +4.0 dB** |

## 使用方法

### 训练

```bash
cd /home/user/sfrdpwu/SFRDP-Net
python train.py
```

### 测试

```bash
python test.py
```

### 监控训练

```bash
tensorboard --logdir ./logs
```

## 备份文件

原始文件已备份为：
- `train.py.bak`
- `data_utils.py.bak`
- `loss.py.bak`
- `option.py.bak`

## 训练时间估计

- 总 epochs: 1000
- 预计时间: 与原训练相当
