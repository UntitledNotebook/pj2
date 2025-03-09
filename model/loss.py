import torch.nn.functional as F
from torch import nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = classes
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        log_pred = torch.log_softmax(pred, dim=-1)
        smooth_label = torch.full_like(pred, self.smoothing / (self.cls - 1))
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)
        return self.criterion(log_pred, smooth_label)
