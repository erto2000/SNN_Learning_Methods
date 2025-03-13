from ANN import ANN, get_ann_accuracy_function
from utility import init_model_weights, initialize_F_proj
import torch
import torch.nn.functional as F


# Backpropagation model for ANN
def model_ann_pepita(input_dim, hidden_dim, output_dim, lr=0.01):
    model = ANN(input_dim, hidden_dim, output_dim)
    init_model_weights(model, init_method='default')
    f_proj = initialize_F_proj((10, 784), init_method='default')

    def optimize_fn(data, targets):
        with torch.no_grad():
            f = f_proj.to(data.device)

            outputs = model(data)
            target_onehot = F.one_hot(targets, num_classes=10).float()

            h = model.hidden_activation
            e = outputs - target_onehot
            proj_err = e @ f

            modulated_input = data + proj_err
            _ = model(modulated_input)
            h_err = model.hidden_activation

            # Manual weight updates
            delta_w1 = (h - h_err).T @ modulated_input
            delta_w2 = e.T @ h_err

            # Apply updates with the chosen learning rate
            model.fc1.weight -= lr * delta_w1
            model.fc2.weight -= lr * delta_w2

            return torch.norm(e).item()

    return {
        'name': 'ANN_PEPITA',
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_ann_accuracy_function(model)
    }
