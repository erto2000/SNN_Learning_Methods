import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, structure, hidden_activation=nn.ReLU(), output_activation=None):
        """
        Args:
            structure (list): Defines the layer sizes, e.g., [input_dim, hidden1, hidden2, ..., output_dim]
            hidden_activation (nn.Module, optional): Activation function for hidden layers (default: ReLU)
            output_activation (nn.Module, optional): Activation function for output layer (default: None)
        """
        super(ANN, self).__init__()

        self.layers = nn.ModuleList()
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Activation storage (persistent, but only used if requested)
        self.linear_activations = []
        self.nonlinear_activations = []

        # Create layers dynamically
        for i in range(len(structure) - 1):
            self.layers.append(nn.Linear(structure[i], structure[i + 1]))

    def forward(self, x, hold_linear_activations=False, hold_nonlinear_activations=False):
        """
        Args:
            x (torch.Tensor): Input tensor
            hold_linear_activations (bool): If True, stores activations after linear layers
            hold_nonlinear_activations (bool): If True, stores activations after activation functions
        Returns:
            torch.Tensor: Output tensor
        """
        # Clear activations before each forward pass
        self.linear_activations.clear()
        self.nonlinear_activations.clear()

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if hold_linear_activations:
                self.linear_activations.append(x.detach().clone())  # Keep activations on the same device

            if i < len(self.layers) - 1:  # Apply activation for hidden layers
                x = self.hidden_activation(x)
                if hold_nonlinear_activations:
                    self.nonlinear_activations.append(x.detach().clone())

        # Apply output activation if provided
        if self.output_activation:
            x = self.output_activation(x)

        return x


def get_ann_accuracy_function(model):
    def test_fn(data, targets):
        with torch.no_grad():
            outputs = model(data)  # Forward pass
            predictions = outputs.argmax(dim=1)  # Get the predicted class
            return (predictions == targets).float().mean().item()  # Compute accuracy

    return test_fn
