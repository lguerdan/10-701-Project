from collections import OrderedDict
from typing import Tuple, Optional, List
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils import clip_grad_norm_
from torch import Tensor

import helpers
from models import cifar, mnist
from data import utils

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(dataset: str):
    if dataset == 'cifar':
        return cifar.Net().to(DEVICE)
    elif dataset == 'mnist':
        return mnist.Net().to(DEVICE)


def run_exp(exp_name, params, use_devset=False):
    for benchmark in ['mnist', 'cifar']:
        print(f'Running: {exp_name}/{benchmark}')
        trainset, testset = utils.load_data(dataset=benchmark, devset=use_devset)
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=params['batch_size'], shuffle=False, drop_last=True)

        model = load_model(dataset=benchmark)
        train(exp_name=f'{exp_name}/{benchmark}', model=model, trainloader=trainloader, testloader=testloader,
              device=DEVICE, opt_params=params)


def train(
        exp_name: str,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        device: torch.device,
        opt_params
):
    n_epochs = opt_params['n_epochs']
    lr = opt_params['lr']
    momentum = opt_params['momentum']
    decay = opt_params['decay']

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    writer = SummaryWriter(f'runs/{exp_name}')
    log = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print(f"Training {n_epochs} epoch(s) w/ {len(trainloader)} batches each.", flush=True)
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, trainloader, device, optimizer, criterion, opt_params)
        test_loss, test_acc = test(model, testloader, device)

        # Write training metrics
        writer.add_scalar(tag='Train loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar(tag='Train accuracy', scalar_value=train_acc, global_step=epoch)

        # Write testing metrics
        writer.add_scalar(tag='Test loss', scalar_value=test_loss, global_step=epoch)
        writer.add_scalar(tag='Test accuracy', scalar_value=test_acc, global_step=epoch)
        log['epoch'].append(epoch)
        log['train_loss'].append(train_loss)
        log['train_acc'].append(train_acc)
        log['test_loss'].append(test_loss)
        log['test_acc'].append(test_acc)

    # export scalar data to JSON for external processing
    helpers.write_logs(exp_name, log, opt_params)
    writer.close()


def train_epoch(
        model,
        trainloader: torch.utils.data.DataLoader,
        device: torch.device,
        optimizer: torch.optim,
        criterion,
        opt_params
) -> List[Tuple[float, float]]:
    # DP-SGD parameters
    use_adaptive = opt_params['adaptive']
    num_microbatches = opt_params['num_microbatches']
    S = opt_params['S']
    z = opt_params['z']
    gamma = opt_params['gamma']  # Target quantile
    lr_c = opt_params['lr_c']
    sigma_b = 1.1 # Test value for sigma used in adaptive clipping
    sigma = z * S

    # Define loss and optimizer
    model.train()
    running_loss = 0.0
    total = 0.0
    correct = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # Indicator sum of gradient less than C for this batch
        b = 0.0

        x, y = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        y_hat = model(x)
        loss = criterion(y_hat, y)
        running_loss += torch.mean(loss).item()
        losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)

        saved_var = dict()
        for tensor_name, tensor in model.named_parameters():
            saved_var[tensor_name] = torch.zeros_like(tensor)

        for sample in losses:
            sample.backward(retain_graph=True)

            total_norm = 0.0
            for p in model.parameters():
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            b += 1 if total_norm ** 0.5 < S else 0
            b += torch.randn(1) * sigma_b

            torch.nn.utils.clip_grad_norm_(model.parameters(), S)

            for tensor_name, tensor in model.named_parameters():
                new_grad = tensor.grad.to(DEVICE)
                saved_var[tensor_name].add_(new_grad)
            model.zero_grad()

        b_t = b / num_microbatches
        if use_adaptive:
            # Linear clipping
            S = S - lr_c * (b_t - gamma)

            # Exponential clipping
            # S = S * torch.exp_(-lr_c * (b_t - gamma))

        for tensor_name, tensor in model.named_parameters():
            if device.type == 'cuda':
                noise = torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, sigma)
            else:
                noise = torch.FloatTensor(tensor.grad.shape).normal_(0, sigma)
            saved_var[tensor_name].add_(noise)
            tensor.grad = saved_var[tensor_name] / num_microbatches
        optimizer.step()

        # Collect statistics
        _, predicted = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    return running_loss / total, correct / total


def test(
        model,
        testloader: torch.utils.data.DataLoader,
        device: torch.device
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

    return loss / total, correct / total
