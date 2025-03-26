import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataset(
    dataset_name="MNIST",
    data_percentage=1.0,
    flatten=True,
    root="./data",
    download=True,
    **kwargs
):
    """
    Generalized dataset loader.

    Parameters:
        dataset_name (str): Name of the dataset (e.g., 'MNIST', 'FashionMNIST', 'CIFAR10').
        data_percentage (float): Fraction of the data to use (0 < x <= 1).
        flatten (bool): Whether to flatten the image into 1D tensor.
        root (str): Directory to store/load dataset.
        download (bool): Whether to download the dataset if not present.
        **kwargs: Extra arguments passed to the dataset class.

    Returns:
        train_dataset (Subset or full dataset)
        test_dataset (Subset or full dataset)
        input_dim (int): Flattened input dimension (or channel x H x W if not flattened)
    """

    # Basic transform
    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)

    # Get dataset class dynamically
    dataset_class = getattr(datasets, dataset_name)

    # Load datasets
    train_dataset = dataset_class(root=root, train=True, transform=transform, download=download, **kwargs)
    test_dataset = dataset_class(root=root, train=False, transform=transform, download=download, **kwargs)

    # Subset selection
    if 0 < data_percentage < 1.0:
        train_size = int(len(train_dataset) * data_percentage)
        test_size = int(len(test_dataset) * data_percentage)
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), test_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Determine input dimension
    sample_input = train_dataset[0][0]
    input_dim = sample_input.shape[0]

    return train_dataset, test_dataset, input_dim


def get_loaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_loaders_getter(train_dataset, test_dataset, batch_size):
    return lambda: get_loaders(train_dataset, test_dataset, batch_size)
