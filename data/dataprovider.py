import abc
from typing import NamedTuple

from torch.utils.data import DataLoader


class Loaders(NamedTuple):
    train_loader: DataLoader
    test_loader: DataLoader


class DataProvider(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs) -> Loaders:
        pass
