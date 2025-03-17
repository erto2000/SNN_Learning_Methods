import torch
import torch.nn as nn
import torch.optim as optim
from models.ANN import ANN, get_ann_test_fn, accuracy_fn


# Direct Feedback Alignment (DFA) model for ANN
def model_ann_dfa(name, structure):
    model = ANN(structure)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))
    projection_matrix = torch.randn(structure[-1], structure[1])

    def optimize_fn(data, targets):
        B = projection_matrix.to(data.device)
        optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(data, hold_nonlinear_activations=True)

        target_one_hot = torch.zeros_like(outputs)
        target_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        error = outputs - target_one_hot
        batch_size = data.size(0)

        hidden_activation = model.nonlinear_activations[0]
        grad_fc2_weight = (hidden_activation.t() @ error) / batch_size

        # Compute the elementwise derivative of ReLU on the hidden layer activations.
        dReLU = (hidden_activation > 0).float()
        delta_hidden = (error @ B) * dReLU

        # Compute gradients for fc1 using the input data and delta_hidden.
        grad_fc1_weight = (data.t() @ delta_hidden) / batch_size

        # === Manually assign the computed gradients ===
        with torch.no_grad():
            model.layers[0].weight.grad = grad_fc1_weight.t()
            model.layers[1].weight.grad = grad_fc2_weight.t()

        optimizer.step()

        return loss_fn(outputs, targets).item(), accuracy_fn(outputs, targets)

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_ann_test_fn(model)
    }
