import torch
import torch.nn as nn


# Define a simple feedforward ANN
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_softmax=True):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_softmax = output_softmax
        self.hidden_activation = None

    def forward(self, x):
        x1 = self.fc1(x)
        a1 = self.relu(x1)
        x2 = self.fc2(a1)
        if self.output_softmax:
            x2 = torch.softmax(x2, dim=1)

        self.hidden_activation = a1
        return x2


def get_ann_accuracy_function(model):
    def test_fn(data, targets):
        with torch.no_grad():
            outputs = model(data)  # Forward pass
            predictions = outputs.argmax(dim=1)  # Get the predicted class
            return (predictions == targets).float().mean().item()  # Compute accuracy

    return test_fn
