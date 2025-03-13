# imports
import snntorch as snn
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn


#  Network architecture
class SNN(torch.nn.Module):
    def __init__(self, input_dim, time_steps, beta, spike_grad, linear_layer=nn.Linear):
        super().__init__()

        self.time_steps = time_steps
        self.net = nn.Sequential(
                        linear_layer(input_dim, 128),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        linear_layer(128, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    )

    def forward(self, data):
        spk_rec = []
        utils.reset(self.net)  # resets hidden states for all LIF neurons in net

        for step in range(self.time_steps):
            spk_out = self.net(data)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)


def get_snn_accuracy_function(model):
    def test_fn(data, targets):
        with torch.no_grad():
            spk_rec = model(data)
            return SF.accuracy_rate(spk_rec, targets)

    return test_fn
