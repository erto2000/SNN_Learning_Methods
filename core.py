# imports
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    dim = train_dataset[0][0].shape[0]
    return train_loader, test_loader, dim


#  Network architecture
class SNN(torch.nn.Module):
    def __init__(self, input_dim, time_steps, beta, spike_grad):
        super().__init__()

        self.time_steps = time_steps
        self.net = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Linear(128, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    )

    def forward(self, data):
        spk_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        for step in range(self.time_steps):
            spk_out = self.net(data)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)


def batch_accuracy(loader, device, model):
    with torch.no_grad():
        total = 0
        acc = 0
        model.eval()

        for data, targets in iter(loader):
              data = data.to(device)
              targets = targets.to(device)
              spk_rec = model(data)

              acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
              total += spk_rec.size(1)

        return acc/total
