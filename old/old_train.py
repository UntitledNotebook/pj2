import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pandas as pd
from typing import Optional

from .utils.utils import evaluate, mixup_data, LabelSmoothingLoss, set_seed, progress_bar, clip_gradients

def train(model: torch.nn.Module,
          epochs: int,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          criterion: nn.Module,
          optimizer: optim.Optimizer,
          results_path: str,
          scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
          model_path: str = None,
          run_id: str = "run_1",
          seed: int = 42,
          batch_size: int = 32,
          learning_rate: float = 0.1,
          augmentation: list = ["RandAugment", "Cutout"],
          normalization: str = "BatchNorm",
          checkpoint_interval: int = 10,
          gradient_clipping: float = 1.0):
    """
    Enhanced training loop with Mixup, label smoothing, data augmentation,
    checkpoint saving, progress bar, and gradient clipping for CIFAR-10_4x.
    
    Args:
        model: The PyTorch model to be trained
        epochs: Number of training epochs
        train_loader: PyTorch DataLoader for training set
        valid_loader: PyTorch DataLoader for validation set
        criterion: Loss function
        optimizer: Optimizer for training
        results_path: Path to save results (CSV)
        scheduler: Learning rate scheduler (optional)
        model_path: Base path to save model checkpoints
        run_id: Unique identifier for this run (for ablation study)
        seed: Random seed for reproducibility
        batch_size: Training batch size
        learning_rate: Initial learning rate
        augmentation: List of augmentation techniques
        normalization: Type of normalization ("BatchNorm" or "LayerNorm")
        checkpoint_interval: Save checkpoint every N epochs
        gradient_clipping: Maximum norm for gradient clipping (default: 1.0)
    """
    
    # Set seed for reproducibility
    set_seed(seed)

    # Run on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Training loop
    cols = ['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'learning_rate']
    results_df = pd.DataFrame(columns=cols).set_index('epoch')
    print('Epoch \tBatch \tNLLLoss_Train \tTrain_Acc \tValid_Acc')

    # Best validation accuracy for checkpoint saving
    best_valid_acc = 0.0

    # Create directory for checkpoints
    os.makedirs(os.path.join(model_path, run_id), exist_ok=True)

    # Save ablation configuration (handled in main.py, but can be repeated here if needed)
    ablation_config = {
        "architecture": model.__class__.__name__,
        "block_type": getattr(model, 'block_type', 'unknown'),  # Adjust based on your model
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "augmentation": augmentation,
        "normalization": normalization,
        "seed": seed,
        "gradient_clipping": gradient_clipping
    }
    # Create directory for ablation results
    os.makedirs(os.path.join('ablation_results', run_id), exist_ok=True)
    with open(os.path.join('ablation_results', run_id, 'config.txt'), 'w') as f:
        for key, value in ablation_config.items():
            f.write(f"{key}: {value}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use progress bar for training batches
        train_bar = progress_bar(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader))
        for i, data in enumerate(train_bar, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply Mixup if enabled
            if "Mixup" in augmentation:
                inputs, (labels_a, labels_b), lam = mixup_data(inputs, labels, alpha=1.0)
                outputs = model(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Zero gradients, backward pass, optimize, and clip gradients
            optimizer.zero_grad()
            loss.backward()
            clip_gradients(model, gradient_clipping)
            optimizer.step()

            # Update running metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            if "Mixup" in augmentation:
                train_correct += (predicted == labels_a).sum().item() * lam + (predicted == labels_b).sum().item() * (1 - lam)
            else:
                train_correct += (predicted == labels).sum().item()

            # Update progress bar with current loss and accuracy
            train_bar.set_postfix({'Loss': running_loss / (i + 1), 'Acc': 100. * train_correct / train_total})

        if scheduler:
            scheduler.step()

        # Evaluate on training and validation sets
        train_acc, train_loss = evaluate(model, train_loader, device)
        valid_acc, valid_loss = evaluate(model, valid_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics
        results_df.loc[epoch + 1] = [train_loss, train_acc, valid_loss, valid_acc, current_lr]
        results_df.to_csv(results_path)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%, LR: {current_lr:.6f}')

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': train_loss,
                'valid_acc': valid_acc
            }
            torch.save(checkpoint, os.path.join(model_path, run_id, f'checkpoint_epoch_{epoch+1}.pth'))

        # Save best model based on validation accuracy
        if valid_acc > best_valid_acc and model_path:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(model_path, run_id, f'best_model_{run_id}.pth'))
            with open(os.path.join('ablation_results', run_id, 'best_summary.txt'), 'w') as f:
                f.write(f"Best Validation Accuracy: {best_valid_acc:.2f}%\n")
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Model Size: {os.path.getsize(os.path.join(model_path, run_id, f'best_model_{run_id}.pth')) / (1024 * 1024):.2f} MB\n")

    print('Finished Training')
    model.eval()

    # Save final model and metrics
    torch.save(model.state_dict(), os.path.join(model_path, run_id, 'final_model.pth'))
    model_size_mb = os.path.getsize(os.path.join(model_path, run_id, 'final_model.pth')) / (1024 * 1024)
    with open(os.path.join('ablation_results', run_id, 'final_summary.txt'), 'w') as f:
        f.write(f"Final Validation Accuracy: {valid_acc:.2f}%\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")

    return model