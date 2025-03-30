import torch
from models.SNN import SNNDynamic, get_dynamic_snn_test_fn, accuracy_fn
from utility import get_random_matrix


# Backpropagation model
def get_model(name, structure, time_steps, beta, output_neuron=False,
              lr=0.1, init_method=None, multiplier=0.005):
    model = SNNDynamic(structure, beta, output_neuron=output_neuron)

    f_proj = get_random_matrix((structure[-1], structure[0]), method=init_method, multiplier=multiplier)

    def optimize_fn(data, targets):
        with torch.no_grad():
            f = f_proj.to(data.device)

            model.reset()
            output_spk = model.repeat_run(data, time_steps)
            layer_spk_counts = [torch.sum(layer.get_spk_rec(), dim=0) for layer in model.layers]
            p = torch.softmax(layer_spk_counts[-1], dim=1)
            onehot_labels = torch.nn.functional.one_hot(targets, structure[-1]).float()
            e = p - onehot_labels  # shape: (batch, output_dim)

            projected_error = e @ f  # shape: (batch, input_dim)
            modulated_input = data + projected_error

            model.reset()
            _ = model.repeat_run(modulated_input, time_steps)

            # Compute weight updates dynamically
            prev_h = modulated_input
            for i, layer in enumerate(model.layers):
                if i < len(model.layers) - 1:  # Hidden layers
                    h = layer_spk_counts[i] / time_steps
                    h_err = torch.sum(layer.get_spk_rec(), dim=0) / time_steps  # Activation after perturbed forward pass

                    delta_w = (h - h_err).T @ prev_h  # Weight update
                else:  # Last layer (uses error signal)
                    delta_w = e.T @ prev_h

                # Apply weight update
                batch_size = data.shape[0]
                layer.update_weight(lr * delta_w / batch_size)

                # Update for next iteration
                prev_h = h_err

            return torch.norm(e).item(), accuracy_fn(output_spk, targets)

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_dynamic_snn_test_fn(model, time_steps)
    }


def get_trial_generator(input_dim):
    def trial_generator(trial):
        lr = trial.suggest_float("lr", 1e-5, 1, log=True)
        multiplier = trial.suggest_float("multiplier", 1e-5, 1, log=True)
        time_steps = trial.suggest_int("time_steps", 10, 100)
        beta = trial.suggest_float("beta", 0.5, 1.0)

        return get_model('SNN_Pepita', [input_dim, 128, 10], time_steps, beta, False,
                         lr, 'default', multiplier)

    return trial_generator

