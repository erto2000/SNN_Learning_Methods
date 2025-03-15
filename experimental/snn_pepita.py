import snntorch as snn
from snntorch import surrogate, utils
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Parameters
num_epochs = 10
batch_size = 64
beta = 0.9
time_steps = 50
spike_grad = surrogate.fast_sigmoid(slope=25)
data_percentage = 0.1  # Load a fraction of the dataset
lr = 0.00001
f_factor = 0.0005

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Data transforms and subsampling function
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the 28x28 image into a vector of 784
])


def subsample_dataset(dataset, percentage):
    num_samples = int(len(dataset) * percentage)
    indices = torch.randperm(len(dataset))[:num_samples]
    return torch.utils.data.Subset(dataset, indices)


# Load MNIST dataset and subsample
full_train_dataset = torchvision.datasets.MNIST(root="../data", train=True, transform=transform, download=True)
full_test_dataset = torchvision.datasets.MNIST(root="../data", train=False, transform=transform, download=True)

train_dataset = subsample_dataset(full_train_dataset, data_percentage)
test_dataset = subsample_dataset(full_test_dataset, data_percentage)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a new SNN model with explicit layers
class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps, beta, spike_grad):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.time_steps = time_steps

    def forward_pass(self, x):
        # Reset hidden states for the Leaky neurons
        utils.reset(self.lif1)
        utils.reset(self.lif2)

        # Accumulators for hidden and output spikes
        h_sum = 0
        out_sum = 0
        for _ in range(self.time_steps):
            h = self.lif1(self.fc1(x))
            out = self.lif2(self.fc2(h))
            h_sum += h  # accumulate hidden-layer spikes
            out_sum += out  # accumulate output spikes
        return h_sum, out_sum


# Initialize the model and the random projection matrix.
# The projection matrix projects a 10-dimensional error to the input dimension (784).
model = SNN(input_dim=28 * 28, hidden_dim=128, output_dim=10, time_steps=time_steps, beta=beta,
            spike_grad=spike_grad).to(device)
projection = (torch.rand(10, 28 * 28) * f_factor).to(device)


# Helper to generate one-hot encoded labels
def one_hot(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes).float()


# Training loop using the two-pass method
for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # === First Pass (Normal Pass) ===
        # Get hidden activity and output spike counts from normal input
        h_normal, out_normal = model.forward_pass(images)
        # Compute probabilities from output spike counts
        p = torch.softmax(out_normal, dim=1)
        # Create one-hot targets
        onehot_labels = one_hot(labels, 10)
        # Calculate error: difference between softmax output and one-hot label
        e = p - onehot_labels  # shape: (batch, 10)

        # === Error Projection and Modulated Input ===
        # Project error to input dimensionality (from 10 to 784)
        projected_error = e @ projection  # shape: (batch, 784)
        # Create modulated input by adding the projected error to the original input
        modulated_input = images + projected_error

        # === Second Pass (Modulated Pass) ===
        # Run the modulated input through the network to get hidden activity
        h_modulated, _ = model.forward_pass(modulated_input)

        # === Compute Weight Updates Manually ===
        # For the first layer, use the difference between the hidden activations
        delta_w1 = (h_normal - h_modulated).transpose(0, 1) @ modulated_input  # shape: (hidden_dim, input_dim)
        # For the second layer, project the error onto the hidden activation from the modulated pass
        delta_w2 = e.transpose(0, 1) @ h_modulated  # shape: (output_dim, hidden_dim)

        # Manual weight update (note: biases are not updated here)
        model.fc1.weight.data -= lr * delta_w1
        model.fc2.weight.data -= lr * delta_w2

        # Optionally, compute training accuracy using the probabilities from the normal pass
        pred = torch.argmax(p, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Accuracy: {train_accuracy:.2f}%")

# Evaluation on the test set using the normal pass
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, out_normal = model.forward_pass(images)
        p = torch.softmax(out_normal, dim=1)
        pred = torch.argmax(p, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
