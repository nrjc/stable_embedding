import itertools
import logging
import os

import gin
import torch
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from torch import optim, nn
from torch.utils.data import TensorDataset

from data.dataprovider import DataProvider
from sklearn import svm
logger = logging.getLogger(__name__)


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
        self.pca = PCA(n_components=2)
        self.classifier = svm.LinearSVC()

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
        sample_classes = []
        for batch_features, sample_class in self.train_data:
            # This is needed to massage the 128, 1, 28, 28 vector into something more congenial
            batch_features = batch_features.view(-1, self.viewed_shape)
            compressed_vec = self.model.get_compressed_vec(batch_features)
            batch_inputs.append(batch_features)
            outputs.append(compressed_vec)
            sample_classes.append(sample_class)
        with torch.no_grad():
            input_tensor = torch.cat(batch_inputs)
            output_tensor = torch.cat(outputs)
            class_tensor = torch.cat(sample_classes)
            # Training classifier
            self.classifier.fit(output_tensor.numpy(), class_tensor.numpy())
            classf = self.classifier.predict(output_tensor.numpy())
            acc_score = accuracy_score(class_tensor.numpy(), classf)
            print(f"score: {acc_score}")
            # Training PCA dim reduction
            self.pca.fit(output_tensor.numpy())
            output_obj = TensorDataset(input_tensor, output_tensor)
            os.makedirs(self.save_path, exist_ok=True)
            dump(self.classifier, os.path.join(self.save_path, 'classifier.joblib'))
            dump(self.pca, os.path.join(self.save_path, 'pca.joblib'))
            torch.save(output_obj, os.path.join(self.save_path, 'tensordataset.pt'))
