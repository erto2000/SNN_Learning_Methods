from snntorch import functional as SF
import torch
from core import SNN


# Random feedback model
def random_feedback_model(input_dim, time_steps, beta, spike_grad, device):
    model = SNN(input_dim, time_steps, beta, spike_grad)
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))

    # Generate random feedback weights
    feedback_weights = []
    for param in model.parameters():
        feedback_weights.append(torch.randn_like(param).to(device))

    def optimize_fn(spk_rec, targets):
        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()

        # Compute gradients using random feedback weights
        loss_val.backward()
        with torch.no_grad():
            for param, feedback in zip(model.parameters(), feedback_weights):
                if param.grad is not None:
                    param.grad = feedback * param.grad  # Apply random feedback weights

        optimizer.step()

    return {
        'name': 'Model_RF_Backprop',
        'model': model,
        'optimize_fn': optimize_fn,
    }
