import torch
from snntorch import functional as SF
import copy
from SNN import SNN, get_snn_accuracy_function


# Perturbation learning model
def model_snn_perturbation(name, input_dim, time_steps, beta, spike_grad, perturbation_scale=0.01):
    model = SNN(input_dim, time_steps, beta, spike_grad)
    loss_fn = SF.ce_rate_loss()

    def optimize_fn(data, targets):
        # Deep copy the original state_dict to ensure all keys (including buffers) are preserved
        original_state = copy.deepcopy(model.state_dict())

        # Generate perturbations only for parameters (exclude buffers)
        perturbations = {
            name: torch.normal(0, perturbation_scale, size=param.size()).to(data.device)
            for name, param in model.named_parameters()
        }

        def compute_loss(state):
            model.load_state_dict(state)
            with torch.no_grad():
                output = model(data)
                return loss_fn(output, targets).item()

        # Compute original loss
        loss_orig = compute_loss(original_state)

        # Create perturbed state_dicts
        perturbed_pos = copy.deepcopy(original_state)
        perturbed_neg = copy.deepcopy(original_state)

        for name, perturb in perturbations.items():
            perturbed_pos[name] += perturb
            perturbed_neg[name] -= perturb

        # Compute losses for perturbed states
        loss_pos = compute_loss(perturbed_pos)
        loss_neg = compute_loss(perturbed_neg)

        # Determine the best perturbation
        if loss_pos < loss_neg and loss_pos < loss_orig:
            best_state = perturbed_pos
        elif loss_neg < loss_orig:
            best_state = perturbed_neg
        else:
            best_state = original_state

        # Update the model with the best state
        model.load_state_dict(best_state)

        return loss_orig

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_snn_accuracy_function(model)
    }
