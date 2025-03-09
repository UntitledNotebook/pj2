import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


from .resnet import resnet18
from .dla import dla60x_c