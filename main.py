import torch
from snntorch import surrogate
from core import get_loaders
from trainer import Trainer
from plot import plot_results
from model_backprop import model_backprop
from model_perturbation import model_perturbation
from model_random_feedback import model_random_feedback


# neuron and simulation parameters
beta = 0.9
time_steps = 50
batch_size = 128
num_epochs = 1
spike_grad = surrogate.fast_sigmoid(slope=25)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load data
train_loader, test_loader, input_dim = get_loaders(batch_size)


# Configurations
configs = [
    model_backprop(input_dim, time_steps, beta, spike_grad),
    # model_perturbation(input_dim, time_steps, beta, spike_grad),
    model_random_feedback(input_dim, time_steps, beta, spike_grad, device),
]

# Train multiple models in parallel
trainer = Trainer(configs, device)
trainer.train(train_loader, test_loader, num_epochs)

# Plot results
plot_results(trainer)
