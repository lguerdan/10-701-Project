from collections import OrderedDict
from typing import Tuple, Optional, List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils import clip_grad_norm_
from torch import Tensor

from models import cifar, mnist

def load_model(dataset: str):
    if dataset == 'cifar':
        return cifar.Net()
    elif dataset == 'mnist':
        return mnist.Net()

def train(
    model,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_epochs: int,
    opt_params
) -> List[Tuple[float, float]]:

    # Define loss and optimizer
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    #DP-SGD parameters
    lr = opt_params['lr']
    use_dp = opt_params['dp']
    C = opt_params['C']
    sigma = opt_params['sigma']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print(f"Training {n_epochs} epoch(s) w/ {len(trainloader)} batches each.", flush=True)
    results = []
    # Train the network
    for idx, epoch in enumerate(range(n_epochs)):
        
        running_loss = 0.0
        running_acc  = 0.0
        total = 0
        pbar = tqdm(trainloader, 0)

        for b_ix, sample in enumerate(pbar):
            pbar.set_description(f'Epoch {epoch}: Training...')

            image, label = sample[0].to(device), sample[1].to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            y_hat = model(image)
            loss = criterion(y_hat, label)
            loss.backward()

            if use_dp: 
                for param in model.parameters():     
                    clip_grad_norm_(param.grad, max_norm=C)
                    param = param - lr * param.grad
                    param += torch.normal(mean=torch.zeros(param.shape), std=sigma * C)
            else: 
                optimizer.step()

            # Collect statistics
            running_loss += loss.item()
            _, predicted = torch.max(y_hat.data, 1)
            total += label.size(0)
            running_acc += (predicted == label).sum().item()

        results.append((running_loss/total, running_acc/total))

    return results

def test(
    model,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0.0
    total = 0
    loss = 0.0

    with torch.no_grad():
        pbar = tqdm(testloader)
        for idx, data in enumerate(pbar):
            pbar.set_description(f'Testing...')
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (loss/total, correct/total)
