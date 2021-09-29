from typing import Optional, Callable

import gin
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from data.dataprovider import DataProvider, Loaders


@gin.configurable
class EncodedMNIST(MNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False, data_file_loc=None,
                 target_file_loc=None) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data_file_loc = data_file_loc
        self.target_file_loc = target_file_loc

    def _load_data(self):
        data = np.load(self.data_file_loc)
        targets = np.load(self.target_file_loc)
        return data, targets


@gin.configurable
class DataFrameLoader(DataProvider):
    def __init__(self, root, train_batchsize=128, test_batchsize=32):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.train_dataset = EncodedMNIST(
            root=root, train=True, transform=transform, download=True
        )

        self.test_dataset = EncodedMNIST(
            root=root, train=False, transform=transform, download=True
        )
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize

    def __call__(self, *args, **kwargs) -> Loaders:
        train_loader = DataLoader(
            self.train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        return Loaders(train_loader, test_loader)
