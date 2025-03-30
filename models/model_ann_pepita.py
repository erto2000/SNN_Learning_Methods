from models.ANN import ANN, get_ann_test_fn, accuracy_fn
from utility import init_model_weights, get_random_matrix
import torch
import torch.nn.functional as F
import torch.nn as nn


def get_model(name, structure, lr=0.01, init_method=None, multiplier=0.005, use_sign=False):
    model = ANN(structure, output_activation=nn.Softmax(dim=1))
    init_model_weights(model, init_method='default')
    f_proj = get_random_matrix((structure[-1], structure[0]), method=init_method, multiplier=multiplier)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    def optimize_fn(data, targets):
        with torch.no_grad():
            f = f_proj.to(data.device)

            # Forward pass
            outputs = model(data, hold_nonlinear_activations=True)
            activations = [act.clone() for act in model.nonlinear_activations]
            target_onehot = F.one_hot(targets, num_classes=structure[-1]).float()

            # Compute error and projected error
            e = outputs - target_onehot
            if use_sign:
                proj_err = torch.sign(e) @ f
            else:
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

                    grad = (h - h_err).T @ prev_activation  # Weight update
                else:  # Last layer (uses error signal)
                    grad = e.T @ prev_activation

                # Apply grad
                layer.weight.grad = grad

                # Update for next iteration
                prev_activation = h

            # Update weights
            optimizer.step()

            return torch.norm(e).item(), accuracy_fn(outputs, targets)

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_ann_test_fn(model)
    }


def get_trial_generator(input_dim, output_dim):
    def trial_generator(trial):
        init_method = trial.suggest_categorical("init_method", ["default", "he_uniform"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        multiplier = trial.suggest_float("multiplier", 1e-5, 1, log=True)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
        hidden_size = trial.suggest_int("hidden_size", 32, 1024)

        structure = [input_dim]
        for _ in range(hidden_layers):
            structure.append(hidden_size)
        structure.append(output_dim)

        return get_model('ANN_Pepita', structure, lr=lr, init_method=init_method, multiplier=multiplier)

    return trial_generator

