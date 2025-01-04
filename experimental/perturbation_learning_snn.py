import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from core import SNN
from snntorch import surrogate
from snntorch import functional as SF
import copy

# Hyperparameters
input_size = 28 * 28  # MNIST image size (28x28 pixels)
hidden_size = 128     # Number of hidden neurons
output_size = 10      # Number of output classes (digits 0-9)
learning_rate = 0.01
batch_size = 128
epochs = 5
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

# Model Initialization
model = SNN(input_size, time_steps, beta, spike_grad).to(device)
loss_fn = SF.ce_rate_loss()


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


# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
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
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Accuracy: {accuracy:.2f}%")
