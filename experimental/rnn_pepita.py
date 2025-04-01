import torch
import torch.nn as nn
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
input_size = 1
hidden_size = 32
batch_size = 64
num_epochs = 10
beta = 0.1
learning_rate = 0.01
f_factor = 0.01

# Load data
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("ECG5000")
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Convert to tensors and reshape to (seq_len, input_size)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


# === RNN ===
class CustomRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, beta=0.1):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size, device=device))
        # nn.init.xavier_uniform_(self.W_ih)
        self.W_hh = torch.eye(hidden_size, device=device) * beta  # fixed recurrent

    def forward(self, x_t, h_prev):
        return torch.tanh(x_t @ self.W_ih.T + h_prev @ self.W_hh.T)


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.1):
        super(CustomRNN, self).__init__()
        self.rnn_cell = CustomRNNCell(input_size, hidden_size, beta)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, xt, h_prev):
        h_new = self.rnn_cell(xt, h_prev)
        output = self.readout(h_new)
        return h_new, output


# === Custom Training Logic ===
num_classes = len(le.classes_)
rnn = CustomRNN(input_size, hidden_size, num_classes, beta).to(device)
projection_matrix = torch.randn(num_classes, input_size, device=device) * f_factor

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0

    for data, target in train_loader:
        with torch.no_grad():
            feedback = torch.zeros((data.size(0), input_size), device=device)  # (batch_size, input_size)
            output = None

            prev_hidden_state = None
            hidden_state = torch.zeros(data.size(0), hidden_size, device=device)
            delta_rnn = torch.zeros_like(rnn.rnn_cell.W_ih, device=device)
            delta_ro = torch.zeros_like(rnn.readout.weight, device=device)  # no bias
            for t in range(data.size(1)):
                x_t = data[:, t, :]  # (batch_size, input_size)
                x_t += feedback

                # calculate error
                hidden_state, output = rnn(x_t, hidden_state)
                one_hot = nn.functional.one_hot(target, num_classes=num_classes).float()
                soft_output = torch.softmax(output, dim=1)
                error = soft_output - one_hot
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()

                # print(f"Accuracy at time step {t}: {100.0 * (output.argmax(dim=1) == target).float().mean():.2f}%")

                # === Manual Updates ===
                if t != 0:
                    # Update RNN weights
                    delta_rnn += (hidden_state - prev_hidden_state).T @ x_t
                    # rnn.rnn_cell.W_ih -= (learning_rate / data.size(0) / data.size(1)) * delta_rnn

                    # Update readout weights
                    delta_ro += error.T @ hidden_state
                    # rnn.readout.weight -= (learning_rate / data.size(0) / data.size(1)) * delta_ro  # no bias
                    # print(f"delta_rnn_norm: {torch.norm(delta_rnn):.4f}, delta_ro_norm: {torch.norm(delta_ro):.4f}")

                # === Project error ===
                feedback = error @ projection_matrix  # (batch_size, input_size)
                prev_hidden_state = hidden_state

            rnn.rnn_cell.W_ih -= (learning_rate / data.size(0) / data.size(1)) * delta_rnn
            rnn.readout.weight -= (learning_rate / data.size(0) / data.size(1)) * delta_ro  # no bias

            # Prediction from final hidden state
            prediction = torch.argmax(output, dim=1)
            correct += (prediction == target).sum().item()
            # print(f"Batch Accuracy: {100.0 * (prediction == target).sum().item() / data.size(0):.2f}%")

    acc = 100.0 * correct / len(train_dataset)
    # print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_dataset):.4f}, Train Accuracy = {acc:.2f}%")

    # === Evaluate on test data ===
    rnn.eval()
    correct_test = 0
    for data, target in test_loader:
        with torch.no_grad():
            outputs = []

            hidden_state = torch.zeros(data.size(0), hidden_size, device=device)
            for t in range(data.size(1)):
                x_t = data[:, t, :]
                hidden_state, output = rnn(x_t, hidden_state)
                outputs.append(output)

            # Average outputs over time steps
            avg_output = torch.stack(outputs).mean(dim=0)  # (batch_size, num_classes)
            prediction = torch.argmax(avg_output, dim=1)
            correct_test += (prediction == target).sum().item()

    test_acc = 100.0 * correct_test / len(test_dataset)
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_dataset):.4f}, "
          f"Train Accuracy = {acc:.2f}%, Test Accuracy = {test_acc:.2f}%")

