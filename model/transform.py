from torchvision.transforms import v2

def create_transfrom(config):
    transforms = []
    for transform in config:
        args = transform['args']
        if transform['type'] == 'RandomResizedCrop':
            size, scale, ratio = args['size'], args['scale'], args['ratio']
            transforms.append(v2.RandomResizedCrop(size, scale, ratio))
        elif transform['type'] == 'RandomHorizontalFlip':
            p = args['p']
            transforms.append(v2.RandomHorizontalFlip(p))
        elif transform['type'] == 'ColorJitter':
            brightness, contrast, saturation, hue = args['brightness'], args['contrast'], args['saturation'], args['hue']
            transforms.append(v2.ColorJitter(brightness, contrast, saturation, hue))
        elif transform['type'] == 'RandomRotation':
            degrees = args['degrees']
            transforms.append(v2.RandomRotation(degrees))
        elif transform['type'] == 'AutoAugment':
            transforms.append(v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10))
        elif transform['type'] == 'RandomErasing':
            p, scale, ratio = args['p'], args['scale'], args['ratio']
            transforms.append(v2.RandomErasing(p, scale, ratio))
        elif transform['type'] == 'RandomRotation':
            degrees = args['degrees']
            transforms.append(v2.RandomRotation(degrees))
        
    transforms.extend([v2.ToTensor(),
    v2.Normalize([125 / 255, 124 / 255, 115 / 255], [60 / 255, 59 / 255, 64 / 255])])
    return v2.Compose(transforms)