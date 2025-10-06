import snntorch as snn
from snntorch import surrogate, utils
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from snntorch.spikegen import rate

# Parameters
num_epochs = 10
batch_size = 128
beta = 0.9
time_steps = 50
data_percentage = 1  # Load a fraction of the dataset
lr = 0.1
f_factor = 0.05

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
    def __init__(self, input_dim, hidden_dim, output_dim, time_steps, beta):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.time_steps = time_steps

    def forward_pass(self, x, use_first_dim_as_time=False):
        # Reset hidden states for the Leaky neurons
        utils.reset(self.lif1)

        # Accumulators for hidden and output spikes
        h_sum = 0
        out_sum = 0

        # Repeat for a number of time steps
        time_steps  = x.shape[0] if use_first_dim_as_time else self.time_steps
        for t in range(time_steps):
            input = x[t] if use_first_dim_as_time else x
            h = self.lif1(self.fc1(input))
            out = self.fc2(h)
            h_sum += h  # accumulate hidden-layer spikes
            out_sum += out  # accumulate output spikes

        return h_sum, out_sum


# Initialize the model and the random projection matrix.
# The projection matrix projects a 10-dimensional error to the input dimension (784).
model = SNN(input_dim=28 * 28, hidden_dim=128, output_dim=10, time_steps=time_steps, beta=beta).to(device)
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
        # images_spike = rate(images, time_steps)
        h_normal, out_normal = model.forward_pass(images, use_first_dim_as_time=False)
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
        # modulated_input_spike = rate(modulated_input, time_steps)
        # modulated_input_spike_count = torch.sum(modulated_input_spike, dim=0) / time_steps
        h_modulated, _ = model.forward_pass(modulated_input, use_first_dim_as_time=False)

        # === Compute Weight Updates Manually ===
        # For the first layer, use the difference between the hidden activations
        delta_w1 = -((h_normal - h_modulated) / time_steps).transpose(0, 1) @ modulated_input / images.shape[0] # shape: (hidden_dim, input_dim)
        # For the second layer, project the error onto the hidden activation from the modulated pass
        delta_w2 = -e.transpose(0, 1) @ (h_modulated / time_steps) / images.shape[0] # shape: (output_dim, hidden_dim)

        # Manual weight update (note: biases are not updated here)
        model.fc1.weight.data += lr * delta_w1
        model.fc2.weight.data += lr * delta_w2

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
