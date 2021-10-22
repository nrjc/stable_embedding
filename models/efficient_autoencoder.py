import gin
import torch
from torch import nn

from models.autoencoder import AutoEncoder


@gin.configurable
class EfficientAutoEncoder(AutoEncoder):
    def __init__(self, compressed_dim=196, **kwargs):
        super().__init__()
        self.compressed_dim = compressed_dim
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def get_compressed_vec(self, features: torch.Tensor) -> torch.Tensor:
        return self.get_internal_vec(features).view(-1, self.compressed_dim)

    def get_internal_vec(self, features: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(features))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        code = self.get_internal_vec(features)
        x = self.relu(self.t_conv1(code))
        x = self.relu(self.t_conv2(x))
        return x
