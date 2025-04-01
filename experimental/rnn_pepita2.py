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
beta_low, beta_high = 0.01, 1
learning_rate = 0.01
# f_factor = 0.1

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
    def __init__(self, input_size, hidden_size, beta_low=0.01, beta_high=1):
        super(CustomRNNCell, self).__init__()
        self.input_size = input_size
        self.W_ih = nn.Parameter(torch.randn(hidden_size, input_size, device=device))
        # nn.init.xavier_uniform_(self.W_ih)
        beta_values = torch.logspace(beta_low, beta_high, steps=hidden_size).to(device)
        self.W_hh = torch.diag(beta_values)  # fixed recurrent

    def forward(self, x_t, h_prev):
        return torch.tanh(x_t @ self.W_ih.T + h_prev @ self.W_hh.T)


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta_low=0.01, beta_high=1):
        super(CustomRNN, self).__init__()
        self.rnn_cell = CustomRNNCell(input_size, hidden_size, beta_low, beta_high)
        self.prediction_readout = nn.Linear(hidden_size, input_size)
        self.classification_readout = nn.Linear(hidden_size, output_size)

    def forward(self, xt, h_prev):
        h_new = self.rnn_cell(xt, h_prev)
        prediction_output = self.prediction_readout(h_new)
        classification_output = self.classification_readout(h_new)
        return h_new, prediction_output, classification_output


# === Custom Training Logic ===
num_classes = len(le.classes_)
rnn = CustomRNN(input_size, hidden_size, num_classes, beta_low, beta_high).to(device)
# projection_matrix = torch.eye(input_size, device=device) * f_factor  # Identity matrix for projection

for epoch in range(num_epochs):
    total_classification_loss = 0
    total_prediction_loss = 0
    correct = 0

    for data, target in train_loader:
        with torch.no_grad():
            feedback = torch.zeros((data.size(0), input_size), device=device)  # (batch_size, input_size)
            prediction_output = None

            prev_hidden_state = None
            hidden_state = torch.zeros(data.size(0), hidden_size, device=device)
            delta_rnn = torch.zeros_like(rnn.rnn_cell.W_ih, device=device)
            delta_prediction_ro = torch.zeros_like(rnn.prediction_readout.weight, device=device)
            delta_classification_ro = torch.zeros_like(rnn.classification_readout.weight, device=device)
            for t in range(data.size(1)):
                x_t = data[:, t, :]  # (batch_size, input_size)
                x_t += feedback

                # calculate error
                hidden_state, prediction_output, classification_output = rnn(x_t, hidden_state)
                if t == data.size(1) - 1:
                    classification_loss = nn.CrossEntropyLoss()(classification_output, target)
                    total_classification_loss += classification_loss.item()
                else:
                    prediction_loss = nn.MSELoss()(prediction_output, data[:, t+1, :])
                    total_prediction_loss += prediction_loss.item()

                # === Manual Updates ===
                if t != 0:
                    if t == data.size(1) - 1:
                        # === Classification error ===
                        one_hot = nn.functional.one_hot(target, num_classes=num_classes).float()
                        soft_classification_output = torch.softmax(classification_output, dim=1)
                        classification_error = soft_classification_output - one_hot
                        delta_classification_ro += classification_error.T @ hidden_state
                    else:
                        # === Prediction error ===
                        prediction_error = prediction_output - data[:, t, :]
                        delta_prediction_ro += prediction_error.T @ hidden_state

                    # === RNN error ===
                    delta_rnn += (hidden_state - prev_hidden_state).T @ x_t

                # # === Project error ===
                # if t != data.size(1) - 1:
                #     prediction_error = prediction_output - data[:, t+1, :]
                #     feedback = prediction_error @ projection_matrix  # (batch_size, input_size)

                prev_hidden_state = hidden_state

            rnn.rnn_cell.W_ih -= (learning_rate / data.size(0) / data.size(1)) * delta_rnn
            # rnn.prediction_readout.weight -= (learning_rate / data.size(0) / data.size(1)) * delta_prediction_ro
            rnn.classification_readout.weight -= (learning_rate / data.size(0)) * delta_classification_ro

            # Prediction from final hidden state
            prediction = torch.argmax(classification_output, dim=1)
            correct += (prediction == target).sum().item()

    # print(f"Epoch {epoch+1}: "
    #       f"Prediction Loss = {total_prediction_loss/len(train_dataset):.4f}, "
    #       f"Classification Loss = {total_classification_loss/len(train_dataset):.4f}, "
    #       f"Train Accuracy = {100.0 * correct / len(train_dataset):.2f}%")

    # === Evaluate on test data ===
    rnn.eval()
    correct_test = 0
    for data, target in test_loader:
        with torch.no_grad():
            hidden_state = torch.zeros(data.size(0), hidden_size, device=device)
            classification_output = None
            for t in range(data.size(1)):
                x_t = data[:, t, :]
                hidden_state, prediction_output, classification_output = rnn(x_t, hidden_state)

            # Average outputs over time steps
            prediction = torch.argmax(classification_output, dim=1)
            correct_test += (prediction == target).sum().item()

    print(f"Epoch {epoch+1}: "
          f"Prediction Loss = {total_prediction_loss/len(train_dataset):.4f}, "
          f"Classification Loss = {total_classification_loss/len(train_dataset):.4f}, "
          f"Train Accuracy = {100.0 * correct / len(train_dataset):.2f}%, "
          f"Test Accuracy = {100.0 * correct_test / len(test_dataset):.2f}%")

