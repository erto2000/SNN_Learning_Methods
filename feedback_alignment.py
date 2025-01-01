import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)


# Define the neural network
class FeedbackAlignmentNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedbackAlignmentNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Random feedback weights for feedback alignment
        self.B = torch.randn(hidden_size, output_size, device=device)  # Random fixed weights

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


# Hyperparameters
input_size = 28 * 28  # MNIST image size (28x28 pixels)
hidden_size = 128  # Number of hidden neurons
output_size = 10  # Number of output classes (digits 0-9)
learning_rate = 0.001
batch_size = 128
epochs = 5

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeedbackAlignmentNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Changed to Adam optimizer

# Lists for storing metrics
iterations = []
losses = []
accuracies = []

# Training loop with Feedback Alignment
for epoch in range(epochs):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # Flatten images and move to device
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute error signal
        labels_one_hot = torch.zeros_like(outputs).scatter_(1, labels.unsqueeze(1), 1.0)
        error = labels_one_hot - outputs

        # Feedback Alignment: Compute hidden layer updates
        hidden_activations = torch.relu(model.hidden(images))
        delta_hidden = error @ model.B.T  # Random feedback weights applied

        # Update weights
        optimizer.zero_grad()
        model.output.weight.grad = -torch.matmul(error.T, hidden_activations) / batch_size
        model.hidden.weight.grad = -torch.matmul(delta_hidden.T, images) / batch_size

        optimizer.step()

        # Calculate accuracy every 50 iterations
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 50 == 0:  # Print accuracy every 50 iterations
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Iteration [{i + 1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

            # Store metrics
            iterations.append(epoch * len(train_loader) + i + 1)
            losses.append(loss.item())
            accuracies.append(accuracy)

    # Print epoch-level progress
    print(f"Epoch [{epoch + 1}/{epochs}] completed.")

# Save the model
torch.save(model.state_dict(), "feedback_alignment_mnist.pth")

# Final Test Accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(iterations, losses, marker='o', label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.grid(True)
plt.legend()
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(iterations, accuracies, marker='o', color='green', label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Iterations')
plt.grid(True)
plt.legend()
plt.show()
