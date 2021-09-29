from typing import NamedTuple

import gin
import torch
from torch import nn


class PinnedAutoEncoderOutput(NamedTuple):
    pinned_loss: torch.Tensor
    reconstructed: torch.Tensor


@gin.configurable
class PinnedAutoEncoder(nn.Module):
    def __init__(self, input_shape, pinned_loss_scaling_factor=1.0, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=input_shape
        )
        self.loss = nn.MSELoss()
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
        pinned_loss = self.loss(pinned_target, compressed_vec)
        return PinnedAutoEncoderOutput(pinned_loss * self.pinned_loss_scaling_factor, reconstructed)
