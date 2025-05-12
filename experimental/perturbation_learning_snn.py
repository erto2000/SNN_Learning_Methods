import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import surrogate
from snntorch import functional as SF
import copy
import snntorch as snn
import torch.nn as nn


# Hyperparameters
input_size = 28 * 28  # MNIST image size (28x28 pixels)
hidden_size = 128     # Number of hidden neurons
output_size = 10      # Number of output classes (digits 0-9)
learning_rate = 0.01
batch_size = 128
epochs = 10
beta = 0.9
time_steps = 50
spike_grad = surrogate.fast_sigmoid(slope=25)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#  Network architecture
class SNN(torch.nn.Module):
    def __init__(self, input_dim, time_steps, beta, spike_grad, linear_layer=nn.Linear):
        super().__init__()

        self.time_steps = time_steps

        self.fc1 = linear_layer(input_dim, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = linear_layer(128, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, data):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        for step in range(self.time_steps):
            cur1 = self.fc1(data)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0)


def perturbation_update(model, data, target, device, loss_fn, sigma=0.1):
    model.eval()
    data, target = data.to(device), target.to(device)

    # Deep copy the original state_dict to ensure all keys (including buffers) are preserved
    original_state = copy.deepcopy(model.state_dict())

    # Generate perturbations only for parameters (exclude buffers)
    perturbations = {
        name: torch.normal(0, sigma, size=param.size()).to(device)
        for name, param in model.named_parameters()
    }

    def compute_loss(state):
        model.load_state_dict(state)
        with torch.no_grad():
            output = model(data)
            return loss_fn(output, target).item()

    # Compute original loss
    loss_orig = compute_loss(original_state)

    # Create perturbed state_dicts
    perturbed_pos = copy.deepcopy(original_state)
    perturbed_neg = copy.deepcopy(original_state)

    for name, perturb in perturbations.items():
        perturbed_pos[name] += perturb
        perturbed_neg[name] -= perturb

    # Compute losses for perturbed states
    loss_pos = compute_loss(perturbed_pos)
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


# Model Initialization
model = SNN(input_size, time_steps, beta, spike_grad).to(device)
loss_fn = SF.ce_rate_loss()


# Training Loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        model, loss_orig = perturbation_update(model, data, target, device, loss_fn, sigma=0.01)

        # Accuracy Evaluation
        if batch_idx % 50 == 0:
            model.eval()
            total = 0
            acc = 0
            with torch.no_grad():
                for test_data, test_target in test_loader:
                    test_data, test_target = test_data.to(device), test_target.to(device)
                    spk_rec = model(test_data)
                    acc += SF.accuracy_rate(spk_rec, test_target) * spk_rec.size(1)
                    total += spk_rec.size(1)

            accuracy = 100 * acc/total
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Accuracy: {accuracy:.2f}%")
