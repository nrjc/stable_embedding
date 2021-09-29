import gin
from torch import optim, nn

from data.dataprovider import DataProvider


@gin.configurable
class VanillaTrainer:
    def __init__(self, data_provider: DataProvider, model: nn.Module, epochs=20, lr=1e-3, viewed_shape=784):
        provider = data_provider()
        self.train_data, self.test_data = provider
        self.epochs = epochs
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.viewed_shape = viewed_shape

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
