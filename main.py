import torch
from snntorch import surrogate
from core import get_loaders
from trainer import Trainer
from plot import plot_results
from models.model_snn_backprop import model_snn_backprop
from models.model_snn_random_feedback import model_snn_random_feedback
from models.model_snn_perturbation import model_snn_perturbation
from models.model_ann_backprop import model_ann_backprop
# from models.model_ann_pepita import model_ann_pepita
from models.model_ann_dfa import model_ann_dfa


# neuron and simulation parameters
dataset_fraction = 1
num_epochs = 5
batch_size = 128
beta = 0.9
time_steps = 50
spike_grad = surrogate.fast_sigmoid(slope=25)

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load data
train_loader, test_loader, input_dim = get_loaders(batch_size, dataset_fraction)

# Configurations
configs = [
    model_snn_backprop('SNN_Backprop', input_dim, time_steps, beta, spike_grad),
    # model_snn_perturbation('SNN_Perturbation', input_dim, time_steps, beta, spike_grad),
    # model_snn_random_feedback('SNN_Random_Feedback', input_dim, time_steps, beta, spike_grad),

    # model_ann_backprop('ANN_Backprop', [input_dim, 1024, 10]),
    # model_ann_dfa('ANN_DFA', [input_dim, 1024, 10]),
    # model_ann_pepita('ANN_PEPITA', [input_dim, 1024, 10], lr=0.01),
]

# Train multiple models in parallel
trainer = Trainer(configs, device)
trainer.train(train_loader, test_loader, num_epochs, test_interval=50)

# Plot results
plot_results(trainer)
