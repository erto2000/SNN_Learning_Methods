import torch
from models.SNN import SNNPepita, get_snn_test_fn, accuracy_fn


lr = 0.1
f_factor = 0.05


# Backpropagation model
def get_model(name, input_dim, time_steps, beta, spike_grad):
    model = SNNPepita(input_dim=input_dim, hidden_dim=128, output_dim=10, time_steps=time_steps, beta=beta,
                spike_grad=spike_grad)
    f_proj = (torch.rand(10, input_dim) * f_factor)

    def optimize_fn(data, targets):
        with torch.no_grad():
            f = f_proj.to(data.device)

            output_spk = model(data, use_first_dim_as_time=False)
            h_normal, out_normal = model.h_sum, model.out_sum
            p = torch.softmax(out_normal, dim=1)
            onehot_labels = torch.nn.functional.one_hot(targets, 10).float()
            e = p - onehot_labels  # shape: (batch, 10)

            projected_error = e @ f  # shape: (batch, 784)
            modulated_input = data + projected_error

            _ = model(modulated_input, use_first_dim_as_time=False)
            h_modulated = model.h_sum

            delta_w1 = -((h_normal - h_modulated) / time_steps).transpose(0, 1) @ modulated_input / data.shape[0]  # shape: (hidden_dim, input_dim)
            delta_w2 = -e.transpose(0, 1) @ (h_modulated / time_steps) / data.shape[0]  # shape: (output_dim, hidden_dim)

            # Manual weight update (note: biases are not updated here)
            model.fc1.weight.data += lr * delta_w1
            model.fc2.weight.data += lr * delta_w2

            return torch.norm(e).item(), accuracy_fn(output_spk, targets)

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_snn_test_fn(model)
    }
