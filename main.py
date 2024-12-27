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


# Backpropagation model
def backprop_model():
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


# Perturbation learning model
def perturbation_model():
    model = SNN(input_dim, time_steps, beta, surrogate.fast_sigmoid(slope=25))
    loss_fn = SF.ce_rate_loss()

    perturbation_scale = 0.01

    def optimize_fn(model, spk_rec, targets):
        with torch.no_grad():
            baseline_loss = loss_fn(spk_rec, targets)

            perturbations = []
            for param in model.parameters():
                perturbation = torch.randn_like(param) * perturbation_scale
                param.add_(perturbation)
                perturbations.append(perturbation)

            perturbed_loss = loss_fn(spk_rec, targets)
            delta_loss = perturbed_loss - baseline_loss

            for param, perturbation in zip(model.parameters(), perturbations):
                param.add_(-perturbation * delta_loss / perturbation_scale * 0.1)  # Apply update manually

    return {
        'name': 'Model_Perturbation',
        'model': model,
        'optimize_fn': optimize_fn,
    }


# Random feedback model
def random_feedback_model():
    model = SNN(input_dim, time_steps, beta, surrogate.fast_sigmoid(slope=25))
    loss_fn = SF.ce_rate_loss()
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999))

    # Generate random feedback weights
    feedback_weights = []
    for param in model.parameters():
        feedback_weights.append(torch.randn_like(param).to(device))

    def optimize_fn(model, spk_rec, targets):
        loss_val = loss_fn(spk_rec, targets)
        optimizer.zero_grad()

        # Compute gradients using random feedback weights
        loss_val.backward()
        with torch.no_grad():
            for param, feedback in zip(model.parameters(), feedback_weights):
                if param.grad is not None:
                    param.grad = feedback * param.grad  # Apply random feedback weights

        optimizer.step()

    return {
        'name': 'Model_RF_Backprop',
        'model': model,
        'optimize_fn': optimize_fn,
    }


# Build configurations
configs = [
    backprop_model(),
    perturbation_model(),
    random_feedback_model(),
]

# Train multiple models in parallel
trainer = Trainer(configs, device)
trainer.train(train_loader, test_loader, num_epochs)

# Plot results
plot_results(trainer)
