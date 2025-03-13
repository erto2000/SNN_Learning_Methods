import torch
from snntorch import surrogate
from core import get_loaders
from trainer import Trainer
from plot import plot_results
from model_snn_backprop import model_snn_backprop
from model_snn_perturbation import model_snn_perturbation
from model_snn_random_feedback import model_snn_random_feedback
from model_ann_backprop import model_ann_backprop
from model_ann_pepita import model_ann_pepita
from model_ann_dfa import model_ann_dfa


# neuron and simulation parameters
num_epochs = 5
batch_size = 128
beta = 0.9
time_steps = 50
spike_grad = surrogate.fast_sigmoid(slope=25)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load data
train_loader, test_loader, input_dim = get_loaders(batch_size)

# Configurations
configs = [
    # model_snn_backprop(input_dim, time_steps, beta, spike_grad),
    # model_snn_perturbation(input_dim, time_steps, beta, spike_grad),
    # model_snn_random_feedback(input_dim, time_steps, beta, spike_grad),

    model_ann_backprop(input_dim, 1024, 10),
    model_ann_pepita(input_dim, 1024, 10, lr=0.01),
    model_ann_dfa(input_dim, 1024, 10),
]

# Train multiple models in parallel
trainer = Trainer(configs, device)
trainer.train(train_loader, test_loader, num_epochs, test_interval=50)

# Plot results
plot_results(trainer)
