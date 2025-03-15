from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_loaders(batch_size, data_percentage=1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Load full datasets
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Apply percentage-based selection
    if 0 < data_percentage < 1.0:
        train_size = int(len(train_dataset) * data_percentage)
        test_size = int(len(test_dataset) * data_percentage)

        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    dim = train_dataset[0][0].shape[0]
    return train_loader, test_loader, dim
