import snntorch as snn
from snntorch import functional as SF
from snntorch import utils
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Parameters
num_epochs = 5
batch_size = 128
beta = 0.9
time_steps = 50
spike_grad = surrogate.fast_sigmoid(slope=25)
data_percentage = 0.1  # Load only 50% of the dataset (set between 0 and 1)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load and Subsample MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Function to subsample dataset
def subsample_dataset(dataset, percentage):
    num_samples = int(len(dataset) * percentage)
    indices = torch.randperm(len(dataset))[:num_samples]  # Randomly select indices
    return torch.utils.data.Subset(dataset, indices)

# Load dataset
full_train_dataset = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
full_test_dataset = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)

train_dataset = subsample_dataset(full_train_dataset, data_percentage)
test_dataset = subsample_dataset(full_test_dataset, data_percentage)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define SNN Model
class SNN(nn.Module):
    def __init__(self, input_dim, time_steps, beta, spike_grad):
        super().__init__()
        self.time_steps = time_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(128, 10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        )

    def forward(self, data):
        spk_rec = []
        utils.reset(self.net)  # Reset hidden states

        for _ in range(self.time_steps):
            spk_out = self.net(data)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)


# Initialize Model
model = SNN(input_dim=28*28, time_steps=time_steps, beta=beta, spike_grad=spike_grad).to(device)
loss_fn = SF.ce_rate_loss()
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999))

# Accuracy Function
def test_fn(model, data, targets):
    with torch.no_grad():
        spk_rec = model(data)
        return SF.accuracy_rate(spk_rec, targets)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        spk_rec = model(images)
        loss = loss_fn(spk_rec, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += test_fn(model, images, labels)

    epoch_loss /= len(train_loader)
    epoch_accuracy = (epoch_accuracy / len(train_loader)) * 100

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Evaluation
model.eval()
accuracy = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        accuracy += test_fn(model, images, labels)

print(f"Test Accuracy: {accuracy / len(test_loader) * 100:.2f}%")
