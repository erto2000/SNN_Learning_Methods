import torch.nn as nn
import torch.optim as optim
from models.ANN import ANN, get_ann_test_fn, accuracy_fn


# Backpropagation model for ANN
def get_model(name, structure, lr=1e-3):
    model = ANN(structure)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr)

    def optimize_fn(data, targets):
        outputs = model(data)
        loss_val = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        return loss_val.item(), accuracy_fn(outputs, targets)

    return {
        'name': name,
        'model': model,
        'optimize_fn': optimize_fn,
        'test_fn': get_ann_test_fn(model)
    }


def get_trial_generator(structure):
    def trial_generator(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        return get_model('ANN_Backprop', structure, lr)

    return trial_generator
