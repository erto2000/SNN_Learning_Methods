import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Neural Network Model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Dataset Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ANN().to(device)
loss = nn.CrossEntropyLoss()


def perturbation_update(model, data, target, device, loss, sigma=0.1):
    model.eval()
    data, target = data.to(device), target.to(device)

    # Save the original parameters
    original_state = {name: param.data.clone() for name, param in model.named_parameters()}

    # Generate random perturbations
    perturbations = {
        name: torch.normal(0, sigma, size=param.size()).to(device)
        for name, param in model.named_parameters()
    }

    def compute_loss(state):
        model.load_state_dict(state, strict=False)
        with torch.no_grad():
            output = model(data)
            return loss(output, target).item()

    # Compute original loss
    loss_orig = compute_loss(original_state)

    # Compute loss for theta + v
    perturbed_pos = {name: original_state[name] + perturbations[name] for name in original_state}
    loss_pos = compute_loss(perturbed_pos)

    # Compute loss for theta - v
    perturbed_neg = {name: original_state[name] - perturbations[name] for name in original_state}
    loss_neg = compute_loss(perturbed_neg)

    # Determine the best perturbation
    if loss_pos < loss_neg and loss_pos < loss_orig:
        best_state = perturbed_pos
    elif loss_neg < loss_orig:
        best_state = perturbed_neg
    else:
        best_state = original_state

    # Update the model with the best state
    model.load_state_dict(best_state)

    return model, loss_orig


# Training Loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        model, loss_orig = perturbation_update(model, data, target, device, loss, sigma=0.01)

        # Accuracy Evaluation
        if batch_idx % 50 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Accuracy: {100 * correct / total:.2f}%")
