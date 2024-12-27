# imports
from snntorch import surrogate
from snntorch import functional as SF
import torch

from core import get_loaders, SNN
from trainer import Trainer
from plot import plot_results


# neuron and simulation parameters
beta = 0.9
time_steps = 50
batch_size = 128
num_epochs = 1

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load data
train_loader, test_loader, input_dim = get_loaders(batch_size)


# Define alternative models and training configurations
def build_model1():
    model = SNN(input_dim, time_steps, beta, surrogate.fast_sigmoid(slope=25))
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))

    def optimize_fn(model, spk_rec, targets):
        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    return {
        'name': 'Model_Fast_Sigmoid',
        'model': model,
        'optimize_fn': optimize_fn,
    }


def build_model2():
    model = SNN(input_dim, time_steps, beta, surrogate.atan())
    loss_fn = SF.mse_count_loss()
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)

    def optimize_fn(model, spk_rec, targets):
        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    return {
        'name': 'Model_Atan',
        'model': model,
        'optimize_fn': optimize_fn,
    }


# Build configurations
configs = [
    build_model1(),
    build_model2(),
]

# Train multiple models in parallel
trainer = Trainer(configs, device)
trainer.train(train_loader, test_loader, num_epochs)

# Plot results
plot_results(trainer)
