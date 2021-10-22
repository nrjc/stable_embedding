import gin
import torch
from torch import nn

from models.autoencoder import AutoEncoder


@gin.configurable
class NormalAutoEncoder(AutoEncoder):
    def __init__(self, input_shape, inner_layer=128, **kwargs):
        super().__init__()
        self.input_shape = input_shape
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

    def get_compressed_vec(self, features: torch.Tensor) -> torch.Tensor:
        batch_features = features.view(-1, self.input_shape)
        activation = self.encoder_hidden_layer(batch_features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        code = self.get_compressed_vec(features)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed.resize_as(features)
