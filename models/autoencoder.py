from abc import ABC, abstractmethod

import torch
from torch import nn


class AutoEncoder(ABC, nn.Module):
    @abstractmethod
    def get_compressed_vec(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
