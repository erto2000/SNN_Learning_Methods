import torch.nn as nn
import torch.optim as optim
from ANN import ANN, get_ann_accuracy_function


# Backpropagation model for ANN
def model_ann_backprop(input_dim, hidden_dim, output_dim):
    model = ANN(input_dim, hidden_dim, output_dim, output_softmax=False)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))

    def optimize_fn(data, targets):
        outputs = model(data)
        loss_val = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        return loss_val.item()

    return {
        'name': 'ANN_Backprop',
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_ann_accuracy_function(model)
    }
