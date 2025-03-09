import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List

from utils import train
from utils import set_seed, LabelSmoothingLoss, Cutout, RandomErasing
from models import ResNet, BasicBlock  # Adjust import based on your model file

# Custom dataset (replace with actual CIFAR10_4x if available)
from utils import CIFAR10_4x

def main():
    # Hyperparameters for ablation
    seed = 16
    run_id = "run_1"
    epochs = 100
    batch_size = 128
    learning_rate = 0.1
    augmentation = ["RandAugment", "Cutout", "Mixup"]
    normalization = "BatchNorm"
    gradient_clipping = 2.0  # Add gradient clipping

    # Set seed for reproducibility
    set_seed(seed)

    print('==> Preparing data..')
    # Data transforms
    mean = [125 / 255, 124 / 255, 115 / 255]
    std = [60 / 255, 59 / 255, 64 / 255]
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.08, 0.10)) if "RandomResizedCrop" in augmentation else transforms.Resize(128),
        transforms.RandAugment(num_ops=2, magnitude=5) if "RandAugment" in augmentation else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if "RandomHorizontalFlip" in augmentation else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16) if "Cutout" in augmentation else transforms.Lambda(lambda x: x),
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) if "RandomErasing" in augmentation else transforms.Lambda(lambda x: x),
        transforms.Normalize(mean, std)
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    trainset = CIFAR10_4x(root='./data', split="train", transform=train_transform)
    validset = CIFAR10_4x(root='./data', split="valid", transform=valid_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(validset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Define model
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)  # Adjust for your architecture

    # Define criterion, optimizer, and scheduler
    criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    # Paths for results and models
    results_path = os.path.join('ablation_results', run_id, 'metrics.csv')
    model_path = 'checkpoints'
    os.makedirs('ablation_results', exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Train the model
    trained_model = train(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        results_path=results_path,
        scheduler=scheduler,
        model_path=model_path,
        run_id=run_id,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        augmentation=augmentation,
        normalization=normalization,
        checkpoint_interval=10,
        gradient_clipping=gradient_clipping
    )

if __name__ == "__main__":
    main()