from snntorch import functional as SF
import torch
from models.FeedbackLinear import FeedbackLinear
from models.SNN import SNN, get_snn_accuracy_function


# Random feedback model
def model_snn_random_feedback(name, input_dim, time_steps, beta, spike_grad):
    model = SNN(input_dim, time_steps, beta, spike_grad, linear_layer=FeedbackLinear)
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))

    def optimize_fn(data, targets):
        spk_rec = model(data)
        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        return loss_val.item()

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_snn_accuracy_function(model)
    }
