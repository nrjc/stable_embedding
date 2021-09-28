import gin
import torchvision
from torch.utils.data import DataLoader

from data.dataprovider import Loaders, DataProvider


@gin.configurable
class MNIST(DataProvider):
    def __init__(self, root, train_batchsize=128, test_batchsize=32):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.train_dataset = torchvision.datasets.MNIST(
            root=root, train=True, transform=transform, download=True
        )

        self.test_dataset = torchvision.datasets.MNIST(
            root=root, train=False, transform=transform, download=True
        )
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize

    def __call__(self, *args, **kwargs) -> Loaders:
        train_loader = DataLoader(
            self.train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        return Loaders(train_loader, test_loader)
