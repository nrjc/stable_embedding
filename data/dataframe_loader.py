import gin
import torch
from torch.utils.data import DataLoader


@gin.configurable
class DataFrameLoader:
    def __init__(self, root, train_batchsize=1):
        self.train_dataset = torch.load(root)
        self.train_batchsize = train_batchsize

    def __call__(self, *args, **kwargs) -> DataLoader:
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.train_batchsize, shuffle=True, num_workers=4, pin_memory=True
        )

        return train_loader
