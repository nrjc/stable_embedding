import operator
from functools import reduce

import gin
import torch
from torch import nn

from models.autoencoder import AutoEncoder


@gin.configurable
class EfficientAutoEncoder(AutoEncoder):
    def __init__(self, before_compressed_dim=(4, 7, 7), compressed_dim=10, **kwargs):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.before_compressed_dim = before_compressed_dim
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.num_before_compressed_dim = reduce(operator.mul, before_compressed_dim, 1)
        self.linear_encoder = nn.Linear(self.num_before_compressed_dim, compressed_dim)
        self.linear_decoder = nn.Linear(compressed_dim, self.num_before_compressed_dim)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def get_compressed_vec(self, features: torch.Tensor) -> torch.Tensor:
        return self.get_internal_vec(features)

    def get_internal_vec(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(features))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(-1, self.num_before_compressed_dim)
        x = self.linear_encoder(x)
        x = self.relu(x)
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        code = self.get_internal_vec(features)
        x = self.linear_decoder(code)
        x = self.relu(x)
        x = x.reshape(-1, *self.before_compressed_dim)
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        return x
