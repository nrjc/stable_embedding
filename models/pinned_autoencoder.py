from typing import NamedTuple

import gin
import torch
from torch import nn

from models.autoencoder import AutoEncoder


class PinnedAutoEncoderOutput(NamedTuple):
    pinned_target: torch.Tensor
    compressed_vec: torch.Tensor
    reconstructed: torch.Tensor


@gin.configurable
class PinnedAutoEncoder(nn.Module):
    def __init__(self, internal_autoencoder: AutoEncoder = gin.REQUIRED, **kwargs):
        super().__init__()
        self.internal_autoencoder = internal_autoencoder

    def get_compressed_vec(self, features: torch.Tensor) -> torch.Tensor:
        return self.internal_autoencoder.get_compressed_vec(features)

    def forward(self, true_features: torch.Tensor, pinned_features: torch.Tensor,
                pinned_target: torch.Tensor) -> PinnedAutoEncoderOutput:
        reconstructed = self.internal_autoencoder.forward(true_features)
        compressed_vec = self.get_compressed_vec(pinned_features)
        return PinnedAutoEncoderOutput(pinned_target, compressed_vec, reconstructed)
