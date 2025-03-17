# imports
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn


class SNNPepita(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps, beta, spike_grad):
        super(SNNPepita, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.time_steps = time_steps

        self.h_sum = 0
        self.out_sum = 0

    def forward(self, x, use_first_dim_as_time=False):
        # Reset hidden states for the Leaky neurons
        utils.reset(self.lif1)

        # Accumulators for hidden and output spikes
        self.h_sum = 0
        self.out_sum = 0

        # Repeat for a number of time steps
        spk_rec = []
        time_steps  = x.shape[0] if use_first_dim_as_time else self.time_steps
        for t in range(time_steps):
            input = x[t] if use_first_dim_as_time else x
            h = self.lif1(self.fc1(input))
            out = self.fc2(h)
            self.h_sum += h  # accumulate hidden-layer spikes
            self.out_sum += out  # accumulate output spikes
            spk_rec.append(out)

        return torch.stack(spk_rec)


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


def get_snn_test_fn(model):
    def test_fn(data, targets):
        with torch.no_grad():
            spk_rec = model(data)
            return accuracy_fn(spk_rec, targets)

    return test_fn
