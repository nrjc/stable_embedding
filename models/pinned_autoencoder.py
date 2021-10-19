from typing import NamedTuple

import gin
import torch
from torch import nn


class PinnedAutoEncoderOutput(NamedTuple):
    pinned_target: torch.Tensor
    compressed_vec: torch.Tensor
    reconstructed: torch.Tensor


@gin.configurable
class PinnedAutoEncoder(nn.Module):
    def __init__(self, input_shape, inner_layer=128, pinned_loss_scaling_factor=1.0, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=inner_layer
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=inner_layer, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=input_shape
        )
        self.pinned_loss_scaling_factor = pinned_loss_scaling_factor

    def get_compressed_vec(self, features: torch.Tensor) -> torch.Tensor:
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code

    def forward(self, true_features: torch.Tensor, pinned_features: torch.Tensor,
                pinned_target: torch.Tensor) -> PinnedAutoEncoderOutput:
        code = self.get_compressed_vec(true_features)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        compressed_vec = self.get_compressed_vec(pinned_features)
        return PinnedAutoEncoderOutput(pinned_target, compressed_vec, reconstructed)
