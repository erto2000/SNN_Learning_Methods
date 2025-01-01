import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from FeedbackLinear import FeedbackLinear

# Set random seed for reproducibility
torch.manual_seed(0)

# Hyperparameters
input_size = 28 * 28  # MNIST image size (28x28 pixels)
hidden_size = 128  # Number of hidden neurons
output_size = 10  # Number of output classes (digits 0-9)
learning_rate = 0.01
batch_size = 128
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define Feedback Alignment Network
feedback_model = nn.Sequential(
    FeedbackLinear(input_size, hidden_size),
    nn.ReLU(),
    FeedbackLinear(hidden_size, output_size)
).to(device)


# Define loss and optimizer
optimizer = optim.Adam(feedback_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Lists for storing metrics
iterations = []
losses = []
accuracies = []

# Training
for epoch in range(epochs):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # Prepare input
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = feedback_model(images)
        loss = criterion(outputs, labels)

        # Backward pass (no manual gradients)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Log metrics
        if (i + 1) % 50 == 0:
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            iterations.append(epoch * len(train_loader) + i + 1)
            losses.append(loss.item())
            accuracies.append(accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}] completed.")

# Final Test Accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:  # Use test_loader for final evaluation
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = feedback_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Accuracy: {accuracy:.2f}%")


# Plot Loss and Accuracy in the same window
plt.figure(figsize=(12, 6))  # Set figure size

# Subplot 1 - Loss
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
plt.plot(iterations, losses, marker='o', label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.grid(True)
plt.legend()

# Subplot 2 - Accuracy
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
plt.plot(iterations, accuracies, marker='o', color='green', label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Iterations')
plt.grid(True)
plt.legend()

# Display both plots
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()