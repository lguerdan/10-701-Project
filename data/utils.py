from typing import Tuple, Optional, List
import torchvision
import torchvision.transforms as transforms
import torch

from continuum.datasets import PyTorchDataset
from continuum import TaskSet

DATA_ROOT_CIFAR = "data/datasets/cifar-10"
DATA_ROOT_MNIST = "data/datasets/mnist"


def load_data(dataset: str = 'cifar', devset=False):
    """Load CIFAR-10 (training and test set)."""

    if dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT_CIFAR, train=True, download=True,
                                                    transform=transform)
        testset = torchvision.datasets.CIFAR10(root=DATA_ROOT_CIFAR, train=False, download=True,
                                                   transform=transform)

    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(DATA_ROOT_MNIST, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(DATA_ROOT_MNIST, train=False, download=True, transform=transform)

    # Small sample of data for local testing
    if devset:
        select_ix = list(range(0, 320))
        trainset = torch.utils.data.Subset(trainset, select_ix)
        testset = torch.utils.data.Subset(testset, select_ix)

    return trainset, testset
