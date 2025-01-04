import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from FeedbackLinear import FeedbackLinear  # Ensure this module is correctly implemented


# Hyperparameters
input_size = 28 * 28  # MNIST image size (28x28 pixels)
hidden_size = 128     # Number of hidden neurons
output_size = 10      # Number of output classes (digits 0-9)
learning_rate = 0.01
batch_size = 128
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root="../data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../data", train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define Feedback Alignment Network
feedback_model = nn.Sequential(
    FeedbackLinear(input_size, hidden_size),
    nn.ReLU(),
    FeedbackLinear(hidden_size, output_size)
).to(device)

# Define Normal Neural Network
normal_model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
).to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizers for both models
optimizer_feedback = optim.Adam(feedback_model.parameters(), lr=learning_rate)
optimizer_normal = optim.Adam(normal_model.parameters(), lr=learning_rate)

# Dictionaries to store metrics for both models
metrics = {
    'feedback': {'iterations': [], 'losses': [], 'accuracies': []},
    'normal': {'iterations': [], 'losses': [], 'accuracies': []}
}

# Training Loop
for epoch in range(epochs):
    # Initialize counters for both models
    total_feedback = 0
    correct_feedback = 0
    total_normal = 0
    correct_normal = 0

    for i, (images, labels) in enumerate(train_loader):
        # Prepare input
        images = images.view(-1, input_size).to(device)
        labels = labels.to(device)

        # -------------------- Feedback Alignment Model --------------------
        # Forward pass
        outputs_feedback = feedback_model(images)
        loss_feedback = criterion(outputs_feedback, labels)

        # Backward pass and optimization
        optimizer_feedback.zero_grad()
        loss_feedback.backward()
        optimizer_feedback.step()

        # Calculate accuracy
        _, predicted_feedback = torch.max(outputs_feedback, 1)
        total_feedback += labels.size(0)
        correct_feedback += (predicted_feedback == labels).sum().item()

        # -------------------- Normal Model --------------------
        # Forward pass
        outputs_normal = normal_model(images)
        loss_normal = criterion(outputs_normal, labels)

        # Backward pass and optimization
        optimizer_normal.zero_grad()
        loss_normal.backward()
        optimizer_normal.step()

        # Calculate accuracy
        _, predicted_normal = torch.max(outputs_normal, 1)
        total_normal += labels.size(0)
        correct_normal += (predicted_normal == labels).sum().item()

        # Log metrics every 50 iterations
        if (i + 1) % 50 == 0:
            # Feedback Model Metrics
            accuracy_feedback = 100 * correct_feedback / total_feedback
            metrics['feedback']['iterations'].append(epoch * len(train_loader) + i + 1)
            metrics['feedback']['losses'].append(loss_feedback.item())
            metrics['feedback']['accuracies'].append(accuracy_feedback)

            # Normal Model Metrics
            accuracy_normal = 100 * correct_normal / total_normal
            metrics['normal']['iterations'].append(epoch * len(train_loader) + i + 1)
            metrics['normal']['losses'].append(loss_normal.item())
            metrics['normal']['accuracies'].append(accuracy_normal)

            # Print metrics
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")
            print(f"  Feedback Model - Loss: {loss_feedback.item():.4f}, Accuracy: {accuracy_feedback:.2f}%")
            print(f"  Normal Model   - Loss: {loss_normal.item():.4f}, Accuracy: {accuracy_normal:.2f}%\n")

            # Reset counters after logging
            total_feedback = 0
            correct_feedback = 0
            total_normal = 0
            correct_normal = 0

    print(f"Epoch [{epoch + 1}/{epochs}] completed.\n")

# Function to evaluate model accuracy on test set
def evaluate_model(model, loader):
    correct = 0
    total = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Reset model to training mode
    return 100 * correct / total

# Final Test Accuracy for Feedback Model
test_accuracy_feedback = evaluate_model(feedback_model, test_loader)
print(f"Final Test Accuracy - Feedback Model: {test_accuracy_feedback:.2f}%")

# Final Test Accuracy for Normal Model
test_accuracy_normal = evaluate_model(normal_model, test_loader)
print(f"Final Test Accuracy - Normal Model: {test_accuracy_normal:.2f}%")

# Plot Loss and Accuracy for Both Models
plt.figure(figsize=(14, 6))  # Set figure size

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(metrics['feedback']['iterations'], metrics['feedback']['losses'], marker='o', label='Feedback Loss')
plt.plot(metrics['normal']['iterations'], metrics['normal']['losses'], marker='s', label='Normal Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(metrics['feedback']['iterations'], metrics['feedback']['accuracies'], marker='o', label='Feedback Accuracy')
plt.plot(metrics['normal']['iterations'], metrics['normal']['accuracies'], marker='s', label='Normal Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')
plt.legend()
plt.grid(True)

# Adjust layout and display plots
plt.tight_layout()
plt.show()
