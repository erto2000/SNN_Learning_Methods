import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


class OneHiddenLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super(OneHiddenLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
        out = torch.softmax(logits, dim=1)
        return h, out


def init_model_weights(model, init_method="default"):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_method == "he_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == "default":
                m.reset_parameters()
            else:
                raise ValueError("Unknown initialization method: " + init_method)


def initialize_F_proj(device, shape, init_method="default", factor=0.05):
    if init_method == "he_uniform":
        F_proj = torch.empty(*shape, device=device)
        nn.init.kaiming_uniform_(F_proj, nonlinearity='relu')
        F_proj = F_proj * factor
    elif init_method == "default":
        F_proj = torch.randn(*shape, device=device) * factor
    else:
        raise ValueError("Unknown initialization method: " + init_method)
    return F_proj


def train_mnist_two_forward_passes(epochs=5, batch_size=64, lr=0.01, init_method="he_uniform", factor=0.05, modulation_count=1):
    dataset_size = 100

    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root="../data", train=True, download=True, transform=transform)
    # train_dataset = Subset(train_dataset, list(range(dataset_size)))
    test_dataset = torchvision.datasets.MNIST(root="../data", train=False, download=True, transform=transform)
    # test_dataset = Subset(test_dataset, list(range(100)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneHiddenLayerNet().to(device)
    init_model_weights(model, init_method=init_method)

    f = initialize_F_proj(device, (10, 784), init_method=init_method, factor=factor)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                target_onehot = F.one_hot(target, num_classes=10).float()

                modulated_input = data.clone()

                h_prev, out_prev = model(modulated_input)

                for step in range(modulation_count):
                    preds = out_prev.argmax(dim=1)
                    correct = (preds == target).sum().item()
                    total = target.size(0)
                    # print(f"{epoch}.{batch_idx}.{step}={correct/total}")


                    error = out_prev - target_onehot
                    proj_err = error @ f
                    modulated_input = modulated_input + proj_err

                    h_curr, out_curr = model(modulated_input)

                    delta_w1 = (h_prev - h_curr).T @ modulated_input / data.shape[0] / modulation_count
                    delta_w2 = error.T @ h_curr / data.shape[0] / modulation_count

                    model.fc1.weight -= lr * delta_w1
                    model.fc2.weight -= lr * delta_w2

                    h_prev, out_prev = h_curr, out_curr

        train_acc = evaluate_accuracy(model, train_loader, device)
        test_acc = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")




def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)

            _, out = model(data)

            # Predicted labels
            preds = out.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    train_mnist_two_forward_passes(epochs=10, batch_size=128, lr=0.1,
                                   init_method="he_uniform", factor=0.05, modulation_count=10)
