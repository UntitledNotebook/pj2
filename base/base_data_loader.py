import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from typing import Optional, Callable

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: int, 
                 shuffle: bool = True,
                 num_workers: int = 0,
                 collate_fn: Optional[Callable] = default_collate):
        self.shuffle = shuffle
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(**self.init_kwargs)