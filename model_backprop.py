from snntorch import functional as SF
import torch
from core import SNN


# Backpropagation model
def backprop_model(input_dim, time_steps, beta, spike_grad):
    model = SNN(input_dim, time_steps, beta, spike_grad)
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))

    def optimize_fn(spk_rec, targets):
        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    return {
        'name': 'Model_Fast_Sigmoid',
        'model': model,
        'optimize_fn': optimize_fn,
    }