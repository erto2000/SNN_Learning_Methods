import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST with ToTensor transform
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)

# Collect images with three different labels
found_labels = set()
images = []
labels = []

for img, label in mnist_data:
    if label not in found_labels:
        images.append(img)
        labels.append(label)
        found_labels.add(label)
    if len(found_labels) == 3:
        break

# Plot the 3 images
plt.figure(figsize=(9, 3))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
