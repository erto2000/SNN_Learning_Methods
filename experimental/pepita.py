import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ------------------------------------------------
# 1. Define a 1-hidden-layer MLP with no biases
#    and softmax on the output.
# ------------------------------------------------
class OneHiddenLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=1024, output_size=10):
        super(OneHiddenLayerNet, self).__init__()
        # No bias in these Linear layers
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        """
        Returns:
          h: hidden-layer activation (ReLU)
          out: softmax probabilities (batch_size x 10)
        """
        h = torch.relu(self.fc1(x))
        logits = self.fc2(h)
        out = torch.softmax(logits, dim=1)
        return h, out


def init_model_weights(model, init_method="default"):
    """
    Initialize model weights based on the chosen method.
    By default (init_method="default"), we use the built-in reset_parameters()
    so that the behavior is identical to before.
    For 'he_uniform', we apply Kaiming He uniform initialization.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_method == "he_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_method == "default":
                m.reset_parameters()
            else:
                raise ValueError("Unknown initialization method: " + init_method)


def initialize_F_proj(device, shape, init_method="default"):
    """
    Initialize the random projection matrix F_proj.
    For 'default', F_proj is initialized as before.
    For 'he_uniform', we use Kaiming He uniform initialization.
    """
    if init_method == "he_uniform":
        F_proj = torch.empty(*shape, device=device)
        nn.init.kaiming_uniform_(F_proj, nonlinearity='relu')
        F_proj = F_proj * 0.005
    elif init_method == "default":
        # Keep same behavior as original: Gaussian with scaling 0.005
        F_proj = torch.randn(*shape, device=device) * 0.005
    else:
        raise ValueError("Unknown initialization method: " + init_method)
    return F_proj


def train_mnist_two_forward_passes(epochs=5, batch_size=64, lr=0.01, init_method="default"):
    # ------------------------------------------------
    # 2. Prepare MNIST with [0,1] normalization only
    # ------------------------------------------------
    transform = transforms.ToTensor()  # ToTensor() already scales pixels to [0,1]

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # ------------------------------------------------
    # 3. Create the Model and Random Projection Matrix
    # ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneHiddenLayerNet().to(device)

    # Initialize model weights with the desired method
    init_model_weights(model, init_method=init_method)

    # Initialize the random projection matrix F_proj based on init_method.
    # F_proj should be of shape [10, 784]
    F_proj = initialize_F_proj(device, (10, 784), init_method=init_method)

    # ------------------------------------------------
    # 4. Training Loop
    # ------------------------------------------------
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch.no_grad():
                # Move to device
                data, target = data.to(device), target.to(device)
                # Flatten images: (batch_size, 1, 28, 28) -> (batch_size, 784)
                data = data.view(data.size(0), -1)

                # Convert target to one-hot for direct difference
                target_onehot = F.one_hot(target, num_classes=10).float()

                # -----------------------------
                # FIRST FORWARD PASS
                # -----------------------------
                h, out = model(data)  # out: [batch_size, 10] (softmax)
                e = out - target_onehot  # error: [batch_size, 10]

                # -----------------------------
                # PROJECT ERROR -> INPUT SPACE
                # -----------------------------
                proj_err = e @ F_proj  # [batch_size, 784]
                modulated_input = data + proj_err

                # -----------------------------
                # SECOND FORWARD PASS
                # -----------------------------
                h_err, out_err = model(modulated_input)

                # -----------------------------
                # MANUAL WEIGHT UPDATES using @
                # -----------------------------
                delta_w1 = (h - h_err).T @ modulated_input  # [hidden_size, 784]
                delta_w2 = e.T @ h_err  # [10, hidden_size]

                # Apply updates with the chosen learning rate
                model.fc1.weight -= lr * delta_w1
                model.fc2.weight -= lr * delta_w2

        # -----------------------------
        # Compute Training & Test Accuracy
        # -----------------------------
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

            # Single forward pass (no error projection for testing)
            _, out = model(data)

            # Predicted labels
            preds = out.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


if __name__ == "__main__":
    # You can change init_method to "he_uniform" to use Kaiming He uniform initialization.
    train_mnist_two_forward_passes(epochs=10, batch_size=64, lr=0.01, init_method="he_uniform")
