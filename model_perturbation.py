from snntorch import functional as SF
import torch
from core import SNN


# Perturbation learning model
def model_perturbation(input_dim, time_steps, beta, spike_grad):
    model = SNN(input_dim, time_steps, beta, spike_grad)
    loss_fn = SF.ce_rate_loss()

    perturbation_scale = 0.01

    def optimize_fn(spk_rec, targets):
        with torch.no_grad():
            baseline_loss = loss_fn(spk_rec, targets)

            perturbations = []
            for param in model.parameters():
                perturbation = torch.randn_like(param) * perturbation_scale
                param.add_(perturbation)
                perturbations.append(perturbation)

            perturbed_loss = loss_fn(spk_rec, targets)
            delta_loss = perturbed_loss - baseline_loss

            for param, perturbation in zip(model.parameters(), perturbations):
                param.add_(-perturbation * delta_loss / perturbation_scale * 0.1)  # Apply update manually

    return {
        'name': 'Perturbation',
        'model': model,
        'optimize_fn': optimize_fn,
    }
