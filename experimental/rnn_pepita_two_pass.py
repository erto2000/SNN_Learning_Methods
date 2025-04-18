import torch
import torch.nn as nn
import torch.nn.functional as F
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Hyperparameters
# ----------------------------
batch_size  = 64
hidden_size = 32
lr          = 0.01
epochs      = 20
init_method = "default"   # "default" or "he_uniform"
factor      = 0.05           # scaling for random projections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 1. Load & preprocess ECG5000
# ----------------------------
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("ECG5000")

# encode labels 0…K‑1
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test  = torch.tensor(X_test,  dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test  = torch.tensor(y_test,  dtype=torch.long).to(device)

# ensure shape is [N, seq_len, input_size] by collapsing any trailing dims
# works for shape [N, seq_len], [N, seq_len, 1], [N, seq_len, 1, 1], etc.
X_train = X_train.view(X_train.size(0), X_train.size(1), -1)
X_test  = X_test.view(X_test.size(0),  X_test.size(1),  -1)

input_size  = X_train.size(2)
output_size = len(le.classes_)

# DataLoaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test,  y_test),
    batch_size=batch_size, shuffle=False
)

# ------------------------------------------------
# 2. Define PepITA RNN
# ------------------------------------------------
class PepitaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn_cell   = nn.RNNCell(input_size, hidden_size, bias=False)
        self.fc_out     = nn.Linear(hidden_size, output_size, bias=False)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        for t in range(x.size(1)):
            h = torch.relu(self.rnn_cell(x[:, t, :], h))
        return torch.softmax(self.fc_out(h), dim=1)

def init_model_weights(model, init_method="default"):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_method == "he_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            else:
                m.reset_parameters()
        elif isinstance(m, nn.RNNCell):
            if init_method == "he_uniform":
                nn.init.kaiming_uniform_(m.weight_ih, nonlinearity='relu')
                nn.init.kaiming_uniform_(m.weight_hh, nonlinearity='relu')
            else:
                m.reset_parameters()

def initialize_F_proj(device, shape, init_method="default", factor=0.05):
    F = torch.empty(*shape, device=device)
    if init_method == "he_uniform":
        nn.init.kaiming_uniform_(F, nonlinearity='relu')
    else:
        F.normal_(0, 1)
    return F * factor

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            out   = model(x_batch)
            pred  = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total   += y_batch.size(0)
    return 100.0 * correct / total

# ------------------------------------------------
# 3. Training loop with two forward passes (PepITA)
# ------------------------------------------------
def train_pepita_rnn(model, F_x, F_h, train_loader, test_loader, epochs, lr, device):
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            # x_batch: [batch, seq_len, input_size]
            batch_size, seq_len, _ = x_batch.shape

            # one-hot targets
            target_1hot = F.one_hot(y_batch, num_classes=model.output_size).float()

            # ---- FIRST PASS ----
            h = torch.zeros(batch_size, model.hidden_size, device=device)
            h1_seq = []
            for t in range(seq_len):
                x_t = x_batch[:, t, :]
                h   = torch.relu(model.rnn_cell(x_t, h))
                h1_seq.append(h)
            out1 = torch.softmax(model.fc_out(h), dim=1)
            e    = out1 - target_1hot                      # [batch, output_size]

            # ---- PROJECT ERROR ----
            delta_X = e @ F_x                              # [batch, input_size]
            delta_H = e @ F_h                              # [batch, hidden_size]

            # ---- SECOND PASS ----
            h = torch.zeros(batch_size, model.hidden_size, device=device)
            h2_seq = []
            for t in range(seq_len):
                x_t = x_batch[:, t, :]
                x2  = x_t + delta_X
                h   = torch.relu(model.rnn_cell(x2, h + delta_H))
                h2_seq.append(h)

            # ---- WEIGHT UPDATES ----
            # output weights
            delta_W_ho = - e.t() @ h2_seq[-1] / batch_size  # [output_size, hidden_size]

            # input-to-hidden & hidden-to-hidden
            delta_W_ih = torch.zeros_like(model.rnn_cell.weight_ih)
            delta_W_hh = torch.zeros_like(model.rnn_cell.weight_hh)
            for t in range(seq_len):
                h1_t    = h1_seq[t]
                h2_t    = h2_seq[t]
                prev_h1 = h1_seq[t-1] if t > 0 else torch.zeros_like(h1_seq[0])
                x_t     = x_batch[:, t, :]
                delta_W_ih += - (h1_t - h2_t).t() @ (x_t + delta_X)
                delta_W_hh += - (h1_t - h2_t).t() @ (prev_h1 + delta_H)
            delta_W_ih /= (batch_size * seq_len)
            delta_W_hh /= (batch_size * seq_len)

            # apply updates
            model.rnn_cell.weight_ih.data += lr * delta_W_ih
            model.rnn_cell.weight_hh.data += lr * delta_W_hh
            model.fc_out.weight.data     += lr * delta_W_ho

        # end of epoch — report
        train_acc = evaluate_accuracy(model, train_loader, device)
        test_acc  = evaluate_accuracy(model, test_loader,  device)
        print(f"Epoch {epoch:2d}/{epochs} — "
              f"Train Acc: {train_acc:5.2f}% | Test Acc: {test_acc:5.2f}%")

# ------------------------------------------------
# 4. Instantiate & run
# ------------------------------------------------
if __name__ == "__main__":
    model = PepitaRNN(input_size, hidden_size, output_size).to(device)
    init_model_weights(model, init_method=init_method)

    # random feedback/projection matrices
    F_x = initialize_F_proj(device, (output_size, input_size),
                            init_method=init_method, factor=factor)
    F_h = initialize_F_proj(device, (output_size, hidden_size),
                            init_method=init_method, factor=factor)

    train_pepita_rnn(model, F_x, F_h,
                     train_loader, test_loader,
                     epochs, lr, device)
