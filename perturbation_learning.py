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
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

# Model Initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ANN().to(device)
criterion = nn.CrossEntropyLoss()


# RSO Algorithm with Learning Rate
def rso_update(model, data, target, sigma=0.1):
    """
    1) Store a copy of current parameters (theta).
    2) Sample a single random vector v (same shape as theta).
    3) Compare loss(theta + v) and loss(theta - v).
    4) Move in the better direction.
    """
    model.eval()
    data, target = data.to(device), target.to(device)

    # 1) Current params
    theta = [p.data.clone() for p in model.parameters()]

    # 2) Single random direction (v)
    direction = []
    for p in model.parameters():
        # Choose a fixed scale or scale by p's std
        v = torch.normal(0, sigma, size=p.data.size()).to(device)
        direction.append(v)

    # Evaluate original loss
    output = model(data)
    loss_orig = criterion(output, target)

    # 3) Evaluate (theta + v)
    for p, v in zip(model.parameters(), direction):
        p.data = p.data + v  # Move in positive direction
    output_pos = model(data)
    loss_pos = criterion(output_pos, target)

    # Evaluate (theta - v)
    for p, t, v in zip(model.parameters(), theta, direction):
        p.data = t - v  # revert to original then negative
    output_neg = model(data)
    loss_neg = criterion(output_neg, target)

    # 4) Keep whichever is best
    if loss_pos < loss_neg and loss_pos < loss_orig:
        # Use (theta + v)
        for p, t, v in zip(model.parameters(), theta, direction):
            p.data = t + v
    elif loss_neg < loss_orig:
        # Use (theta - v)
        for p, t, v in zip(model.parameters(), theta, direction):
            p.data = t - v
    else:
        # Revert to original
        for p, t in zip(model.parameters(), theta):
            p.data = t

    return model


# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        model = rso_update(model, data, target, sigma=0.01)

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
