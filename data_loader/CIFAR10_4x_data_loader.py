from base import BaseDataLoader
from .dataset import CIFAR10_4x
from typing import Optional, Callable

class CIFAR10_4x_DataLoader(BaseDataLoader):
    """
    DataLoader specifically designed for CIFAR10_4x dataset with predefined splits
    """
    def __init__(self,
                 root: str,
                 split: str = 'train',
                 batch_size: int = 128,
                 shuffle: bool = True,
                 num_workers: int = 2,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize CIFAR10_4x DataLoader
        
        Args:
            root (str): Root directory of dataset
            split (str): One of 'train', 'valid', or 'test'
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of subprocesses to use for data loading
            transform (callable, optional): Transform to be applied to images
            target_transform (callable, optional): Transform to be applied to targets
        """
        if split not in ['train', 'valid', 'test']:
            raise ValueError("split must be one of 'train', 'valid', or 'test'")

        # Create CIFAR10_4x dataset instance
        dataset = CIFAR10_4x(
            root=root,
            split=split,
            transform=transform,
            target_transform=target_transform
        )
        
        # Initialize parent BaseDataLoader
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def get_dataset_info(self) -> dict:
        """Return information about the dataset"""
        return {
            'split': self.dataset.split,
            'total_samples': len(self.dataset),
            'classes': self.dataset.classes,
            'batch_size': self.batch_size
        }