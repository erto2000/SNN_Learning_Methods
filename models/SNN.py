# imports
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn


class SNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, beta, spike_grad=None, neuron_type='lif'):
        super(SNNLayer, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.spk_rec = []
        if neuron_type == 'lif':
            self.neuron = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
            self.reset()
        else:
            self.neuron = None

    def update_weight(self, delta_weight):
        self.fc.weight -= delta_weight

    def reset(self):
        if self.neuron:
            utils.reset(self.neuron)
        self.spk_rec = []

    def get_spk_rec(self):
        return torch.stack(self.spk_rec)

    def forward(self, x):
        x = self.fc(x)
        if self.neuron:
            x = self.neuron(x)
        self.spk_rec.append(x)
        return x


class SNNDynamic(nn.Module):
    def __init__(self, structure, beta, spike_grad=None, output_neuron=False):
        super(SNNDynamic, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(structure) - 1):
            neuron_type = None if i == len(structure) - 2 and not output_neuron else 'lif'
            self.layers.append(SNNLayer(structure[i], structure[i + 1], beta, spike_grad, neuron_type))

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def run(self, time_series):
        for t in range(time_series.shape[0]):
            self(time_series[t])
        return self.layers[-1].get_spk_rec()


#  Network architecture
class SNN(torch.nn.Module):
    def __init__(self, input_dim, time_steps, beta, spike_grad, linear_layer=nn.Linear):
        super().__init__()

        self.time_steps = time_steps
        # self.net = nn.Sequential(
        #                 nn.Linear(input_dim, 128),
        #                 snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        #                 nn.Linear(128, 10),
        #                 snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        #             )
        self.fc1 = linear_layer(input_dim, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = linear_layer(128, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, data):
        # spk_rec = []
        # utils.reset(self.net[1])
        # utils.reset(self.net[3])
        #
        # for step in range(self.time_steps):
        #     spk_out = self.net(data)
        #     spk_rec.append(spk_out)
        #
        # return torch.stack(spk_rec)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        for step in range(self.time_steps):
            cur1 = self.fc1(data)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)


def accuracy_fn(spk_rec, targets):
    with torch.no_grad():
        return SF.accuracy_rate(spk_rec, targets)


def get_dynamic_snn_test_fn(model, time_steps=None):
    def test_fn(data, targets):
        with torch.no_grad():
            if time_steps:
                data = data.unsqueeze(0).repeat(time_steps, 1, 1)
            else:
                data = data.permute(1, 0, 2)
            model.reset()
            spk_rec = model.run(data)
            return accuracy_fn(spk_rec, targets)

    return test_fn


def get_snn_test_fn(model):
    def test_fn(data, targets):
        with torch.no_grad():
            spk_rec = model(data)
            return accuracy_fn(spk_rec, targets)

    return test_fn
