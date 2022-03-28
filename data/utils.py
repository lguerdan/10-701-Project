from typing import Tuple, Optional, List
import torchvision
import torchvision.transforms as transforms

DATA_ROOT = "data/datasets/cifar-10"
DATA_ROOT_MNIST = "data/datasets/mnist"

def load_data(dataset:str='cifar') -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """Load CIFAR-10 (training and test set)."""
    if dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        trainset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=transform
        )
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(DATA_ROOT_MNIST, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(DATA_ROOT_MNIST, train=False, download=True, transform=transform)
    return trainset, testset
    