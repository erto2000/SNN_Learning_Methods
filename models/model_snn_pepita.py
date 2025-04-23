import torch
from models.SNN import SNNDynamic, get_dynamic_snn_test_fn, accuracy_fn
from utility import get_random_matrix


def compute_grad(h, h_mod, h_prev, use_average=False):
    time_steps, batch_size, dim = h.shape
    dim_prev = h_prev.shape[2]

    if use_average:
        # Average method
        h_norm_mean = h.mean(dim=(0, 1))      # Shape: (dim,)
        h_mod_mean = h_mod.mean(dim=(0, 1))   # Shape: (dim,)
        prev_mean = h_prev.mean(dim=(0, 1))   # Shape: (dim_prev,)

        grad = torch.ger(h_norm_mean - h_mod_mean, prev_mean)

    else:
        # Sum method
        # Compute (h - h_mod) --> Shape: (time_steps, batch_size, dim)
        diff = h - h_mod  # Shape: (time_steps, batch_size, dim)

        # Reshape for batch matrix multiplication
        diff = diff.view(-1, dim)           # Shape: (time_steps * batch_size, dim)
        prev = h_prev.view(-1, dim_prev)    # Shape: (time_steps * batch_size, dim_prev)

        # Perform matrix multiplication (dim, N) @ (N, dim_prev) = (dim, dim_prev)
        grad = diff.T @ prev
        grad /= (batch_size * time_steps)

    return grad

# Backpropagation model
def get_model(name, structure, beta, time_steps=None, output_neuron=False,
              lr=0.1, user_average=True, init_method=None, multiplier=0.005):
    model = SNNDynamic(structure, beta, output_neuron=output_neuron)

    f_proj = get_random_matrix((structure[-1], structure[0]), method=init_method, multiplier=multiplier)

    def optimize_fn(data, targets):
        with torch.no_grad():
            f = f_proj.to(data.device)
            if time_steps:
                data = data.unsqueeze(0).repeat(time_steps, 1, 1)
            else:
                data = data.permute(1, 0, 2)  # shape: (time, batch, input_dim)

            model.reset()
            _ = model.run(data)
            layer_spikes = [layer.get_spk_rec() for layer in model.layers]
            output_activations = layer_spikes[-1].sum(dim=0)  # shape: (batch, output_dim)
            p = torch.softmax(output_activations, dim=1)
            onehot_labels = torch.nn.functional.one_hot(targets, structure[-1]).float()
            e = p - onehot_labels  # shape: (batch, output_dim)

            projected_error = e @ f  # shape: (batch, input_dim)
            modulated_time_series = data + projected_error

            model.reset()
            _ = model.run(modulated_time_series)
            layer_spikes_mod = [layer.get_spk_rec() for layer in model.layers]

            # Compute weight updates dynamically
            prev_h = modulated_time_series
            for i, layer in enumerate(model.layers):
                if i < len(model.layers) - 1:  # Hidden layers
                    grad = compute_grad(layer_spikes[i], layer_spikes_mod[i], prev_h, use_average=user_average)
                else:  # Last layer (uses error signal)
                    prev_h_avg = prev_h.mean(dim=0)  # Average over time
                    grad = e.T @ prev_h_avg
                    grad /= e.shape[0] # Normalize by batch size


                # Apply weight update
                layer.update_weight(lr * grad)

                # Update for next iteration
                prev_h = layer_spikes_mod[i]

            return torch.norm(e).item(), accuracy_fn(layer_spikes[-1], targets)

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

