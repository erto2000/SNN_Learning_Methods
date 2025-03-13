from models.ANN import ANN, get_ann_accuracy_function
from utility import init_model_weights, initialize_F_proj
import torch
import torch.nn.functional as F
import torch.nn as nn


def model_ann_pepita(name, structure, lr=0.01):
    model = ANN(structure, output_activation=nn.Softmax(dim=1))
    init_model_weights(model, init_method='default')
    f_proj = initialize_F_proj((structure[-1], structure[0]), init_method='default')

    def optimize_fn(data, targets):
        with torch.no_grad():
            f = f_proj.to(data.device)

            # Forward pass
            outputs = model(data, hold_nonlinear_activations=True)
            activations = [act.clone() for act in model.nonlinear_activations]
            target_onehot = F.one_hot(targets, num_classes=structure[-1]).float()

            # Compute error and projected error
            e = outputs - target_onehot
            proj_err = e @ f

            # Modulate input with projected error
            modulated_input = data + proj_err
            _ = model(modulated_input, hold_nonlinear_activations=True)
            modulated_activations = model.nonlinear_activations

            # Compute weight updates dynamically
            prev_activation = modulated_input
            for i, layer in enumerate(model.layers):
                if i < len(model.layers) - 1:  # Hidden layers
                    h = activations[i]  # Current activation
                    h_err = modulated_activations[i]  # Activation after perturbed forward pass

                    delta_w = (h - h_err).T @ prev_activation  # Weight update
                else:  # Last layer (uses error signal)
                    delta_w = e.T @ prev_activation

                # Apply weight update
                layer.weight -= lr * delta_w

                # Update for next iteration
                prev_activation = h

            return torch.norm(e).item()

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_ann_accuracy_function(model)
    }
