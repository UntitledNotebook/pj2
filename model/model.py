import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


from .resnet import resnet18
from .dla import dla60x_c, dla60x_c_new
from .efficientnet import effnetv2_s
from .densenet import densenet_cifar

def create_model(config):
    model = config['model']
    if model == 'resnet18':
        model = resnet18()
    elif model == 'dla60x_c':
        model = dla60x_c()
    elif model == 'dla60x_c_new':
        model = dla60x_c_new()
    elif model == 'efficientnet':
        model = effnetv2_s()
    elif model == 'densenet':
        model = densenet_cifar()
    else:
        raise NotImplementedError
    return model