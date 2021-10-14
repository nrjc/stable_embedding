import gin
import torch
from joblib import load
from matplotlib import pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics import accuracy_score
from torch import optim, nn

from data.dataframe_loader import DataFrameLoader
from data.dataprovider import DataProvider
from models.pinned_autoencoder import PinnedAutoEncoderOutput


@gin.configurable
class PinnedTrainer:
    model: nn.Module
    pca: PCA
    classifier: LinearClassifierMixin

    def __init__(self, data_provider_base: DataProvider, data_provider_encoded: DataFrameLoader, model: nn.Module,
                 epochs=20, lr=1e-3, viewed_shape=784, scaling_factor=1.0, pca_load_path=gin.REQUIRED,
                 classifier_load_path=gin.REQUIRED):
        provider = data_provider_base()
        self.train_data_base, self.test_data_base = provider
        self.train_data_encoded = data_provider_encoded()
        self.epochs = epochs
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.viewed_shape = viewed_shape
        self.pca = load(pca_load_path)
        self.scaling_factor = scaling_factor
        self.classifier = load(classifier_load_path)

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
                total_loss = self.criterion(outputs.reconstructed,
                                            batch_features) + self.scaling_factor * self.criterion(
                    outputs.pinned_target, outputs.compressed_vec)

                # compute accumulated gradients
                total_loss.backward()

                # perform parameter update based on current gradients
                self.optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss += total_loss.item()
            # compute the epoch training loss
            loss = loss / len(self.train_data_base)
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, self.epochs, loss))

    def evaluate(self):
        # Print old and new on same graph
        encoded_data_loader = self.train_data_encoded
        encoded_data = torch.cat([compressed for _, compressed in encoded_data_loader]).numpy()
        encoded_data_pca = self.pca.transform(encoded_data)
        plt.scatter(encoded_data_pca[:, 0], encoded_data_pca[:, 1])
        new_encoded_data = []
        classf = []
        for batch_features, output_class in self.train_data_base:
            batch_features = batch_features.view(-1, self.viewed_shape)
            compressed_vec = self.model.get_compressed_vec(batch_features)
            new_encoded_data.append(compressed_vec)
            classf.append(output_class)
        with torch.no_grad():
            new_encoded_data = torch.cat(new_encoded_data).numpy()
            pred = self.classifier.predict(new_encoded_data)
            score = accuracy_score(torch.cat(classf).numpy(), pred)
            print(f"Score new: {score}")
            new_encoded_data_pca = self.pca.transform(new_encoded_data)
            plt.scatter(new_encoded_data_pca[:, 0], new_encoded_data_pca[:, 1])
            plt.show()
