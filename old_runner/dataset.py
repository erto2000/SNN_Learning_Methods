import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from tslearn.datasets import UCR_UEA_datasets


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_image_dataset(dataset_name, data_percentage=1.0, flatten=True, root="./data", download=True, **kwargs):
    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform_list)

    dataset_class = getattr(datasets, dataset_name)
    train_dataset = dataset_class(root=root, train=True, transform=transform, download=download, **kwargs)
    test_dataset  = dataset_class(root=root, train=False, transform=transform, download=download, **kwargs)

    if 0 < data_percentage < 1.0:
        train_dataset = Subset(train_dataset, np.random.choice(len(train_dataset), int(len(train_dataset) * data_percentage), replace=False))
        test_dataset  = Subset(test_dataset,  np.random.choice(len(test_dataset),  int(len(test_dataset)  * data_percentage), replace=False))

    sample_input = train_dataset[0][0]
    input_dim = sample_input.shape[0]

    # Get number of classes
    if hasattr(train_dataset, 'classes'):
        n_classes = len(train_dataset.classes)
    else:
        targets = train_dataset.dataset.targets if isinstance(train_dataset, Subset) else train_dataset.targets
        n_classes = len(torch.unique(torch.tensor(targets)))

    return train_dataset, test_dataset, input_dim, n_classes



def get_timeseries_dataset(dataset_name, data_percentage=1.0):
    ucr = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset_name)
    y_train = y_train.astype(int) - 1
    y_test  = y_test.astype(int)  - 1

    if 0 < data_percentage < 1.0:
        train_size = int(len(X_train) * data_percentage)
        test_size  = int(len(X_test)  * data_percentage)

        train_indices = np.random.choice(len(X_train), train_size, replace=False)
        test_indices  = np.random.choice(len(X_test),  test_size,  replace=False)

        X_train, y_train = X_train[train_indices], y_train[train_indices]
        X_test,  y_test  = X_test[test_indices],  y_test[test_indices]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset  = TimeSeriesDataset(X_test,  y_test)

    input_dim = (X_train.shape[1], X_train.shape[2])  # (time, dim)
    n_classes = len(np.unique(y_train))

    return train_dataset, test_dataset, input_dim, n_classes


def get_dataset(dataset_name, data_percentage=1.0, flatten=True, root="./data", download=True, **kwargs):
    """
    Generic dataset loader for both image datasets and time series datasets.
    Returns: train_dataset, test_dataset, input_dim, n_classes
    """
    image_datasets = dir(datasets)
    if dataset_name in image_datasets:
        return get_image_dataset(dataset_name, data_percentage, flatten, root, download, **kwargs)
    else:
        return get_timeseries_dataset(dataset_name, data_percentage)



def get_loaders(train_dataset, test_dataset, batch_size, shuffle_train=True, shuffle_test=False):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=shuffle_test)
    return train_loader, test_loader


def get_loaders_getter(train_dataset, test_dataset, batch_size):
    return lambda: get_loaders(train_dataset, test_dataset, batch_size)
