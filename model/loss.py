import torch.nn.functional as F
from torch import nn
import torch


def create_criterion(config):
    if config['type'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(**config['args'])
    else:
        raise NotImplementedError