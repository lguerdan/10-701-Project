from collections import OrderedDict
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import Tensor
from tqdm import tqdm

from models import cifar

def load_model(dataset: str):
    if dataset == 'cifar':
        return cifar.Net()

def train(
    net,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_epochs: int,
) -> List[Tuple[float, float]]:

    # Define loss and optimizer
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print(f"Training {n_epochs} epoch(s) w/ {len(trainloader)} batches each.", flush=True)
    results = []
    # Train the network
    for idx, epoch in enumerate(range(n_epochs)):
        running_loss = 0.0
        running_acc  = 0.0
        total = 0
        pbar = tqdm(trainloader, 0)

        for bx, data in enumerate(pbar):
            pbar.set_description(f'Epoch {epoch}: Training...')
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # collect statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_acc += (predicted == labels).sum().item()

        results.append((running_loss/total, running_acc/total))

    return results

def test(
    net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""

    net.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    total = 0
    loss = 0.0

    with torch.no_grad():
        pbar = tqdm(testloader)
        for idx, data in enumerate(pbar):
            pbar.set_description(f'Testing...')
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (loss/total, correct/total)
