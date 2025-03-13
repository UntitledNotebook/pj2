import torch.nn.functional as F
from torch import nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        """
        Implements Label Smoothing Loss.
        
        Args:
            num_classes (int): Number of classes.
            smoothing (float): Smoothing factor in [0,1]. Default = 0.1.
            reduction (str): 'mean' (default), 'sum', or 'none'.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        """
        Computes the loss.
        
        Args:
            logits (Tensor): Raw model outputs (before softmax) of shape (batch_size, num_classes).
            target (Tensor): Hard labels (shape: [batch]) or soft labels (shape: [batch, num_classes]).

        Returns:
            loss (Tensor): Scalar loss value.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # Compute log-softmax

        # If target is a 1D tensor, convert to one-hot
        if target.dim() == 1:
            target = F.one_hot(target, num_classes=self.num_classes).float()

        # Apply label smoothing
        with torch.no_grad():
            smoothed_target = target * (1 - self.smoothing) + self.smoothing / self.num_classes

        # Compute loss
        loss = -(smoothed_target * log_probs).sum(dim=-1)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none' case


def create_criterion(config):
    if config['type'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**config['args'])
    elif config['type'] == 'LabelSmoothingLoss':
        return LabelSmoothingLoss(**config['args'])
