import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Model
from option import opt
from data_utils import get_dataloader
from Net import Net

def train():

    # Initialize model, dataloader, and optimizer
    model = Model(Net, opt)
    train_loader, test_loader = get_dataloader(opt)

    # Training loop
    for epoch in range(opt.total_epoch):
        print(f"Epoch {epoch+1}/{opt.total_epoch}")

        # Train step
        train_loss = model.optimize_parameters(train_loader, epoch)

        # Validation/Test step
        with torch.no_grad():
            psnr, ssim = model.test(test_loader)
            print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

        # Save checkpoint
        model.save_network(epoch, psnr, ssim)

if __name__ == "__main__":
    train()