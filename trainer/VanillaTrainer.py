import itertools

import gin
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset

from data.dataprovider import DataProvider


@gin.configurable
class VanillaTrainer:
    def __init__(self, data_provider: DataProvider, model: nn.Module, epochs=20, lr=1e-3, viewed_shape=784,
                 save_path=""):
        provider = data_provider()
        self.train_data, self.test_data = provider
        self.epochs = epochs
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.viewed_shape = viewed_shape
        self.save_path = save_path

    def __call__(self, *args, **kwargs):
        for epoch in range(self.epochs):
            loss = 0
            for batch_features, _ in self.train_data:
                # This is needed to massage the 128, 1, 28, 28 vector into something more congenial
                batch_features = batch_features.view(-1, self.viewed_shape)

                self.optimizer.zero_grad()

                # compute reconstructions
                outputs = self.model(batch_features)

                # compute training reconstruction loss
                train_loss = self.criterion(outputs, batch_features)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                self.optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
            # compute the epoch training loss
            loss = loss / len(self.train_data)
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss))

    def run_all_and_save(self):
        batch_inputs = []
        outputs = []
        for batch_features, _ in itertools.islice(self.train_data, 3):
            # This is needed to massage the 128, 1, 28, 28 vector into something more congenial
            batch_features = batch_features.view(-1, self.viewed_shape)
            compressed_vec = self.model.get_compressed_vec(batch_features)
            batch_inputs.append(batch_features)
            outputs.append(compressed_vec)
        with torch.no_grad():
            input_tensor = torch.stack(batch_inputs)
            output_tensor = torch.stack(outputs)
            output_obj = TensorDataset(input_tensor, output_tensor)
            torch.save(output_obj, self.save_path)
