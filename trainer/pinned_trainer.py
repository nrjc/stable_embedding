import gin
from torch import optim, nn

from data.dataframe_loader import DataFrameLoader
from data.dataprovider import DataProvider
from models.pinned_autoencoder import PinnedAutoEncoderOutput


@gin.configurable
class PinnedTrainer:
    model: nn.Module

    def __init__(self, data_provider_base: DataProvider, data_provider_encoded: DataFrameLoader, model: nn.Module,
                 epochs=20, lr=1e-3, viewed_shape=784, load_path=""):
        provider = data_provider_base()
        self.train_data_base, self.test_data_base = provider
        self.train_data_encoded = data_provider_encoded()
        self.epochs = epochs
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.viewed_shape = viewed_shape
        self.load_path = load_path

    def __call__(self, *args, **kwargs):
        for epoch in range(self.epochs):
            loss = 0
            for (batch_features, _), (encode_ft, encode_tar) in zip(self.train_data_base, self.train_data_encoded):
                # This is needed to massage the 128, 1, 28, 28 vector into something more congenial
                batch_features = batch_features.view(-1, self.viewed_shape)
                encoded_features = encode_ft.view(-1, self.viewed_shape)

                self.optimizer.zero_grad()

                # compute reconstructions
                outputs = self.model(batch_features, encoded_features, encode_tar)  # type: PinnedAutoEncoderOutput

                # compute training reconstruction loss
                train_loss = self.criterion(outputs.reconstructed, batch_features)
                total_loss = outputs.pinned_loss + train_loss

                # compute accumulated gradients
                total_loss.backward()

                # perform parameter update based on current gradients
                self.optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += total_loss.item()
            # compute the epoch training loss
            loss = loss / len(self.train_data_base)
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss))
