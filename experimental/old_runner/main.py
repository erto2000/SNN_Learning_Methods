# Libraries
import torch
import optuna
import os
from snntorch import surrogate

# General imports
from dataset import get_dataset, get_loaders, get_loaders_getter
from trainer import Trainer
from plot import plot_results
from objective import create_objective

# Model imports
import models.model_ann_backprop
import models.model_ann_dfa
import models.model_ann_pepita
import models.model_snn_backprop
import models.model_snn_pepita
import models.model_snn_perturbation
import models.model_snn_random_feedback


# Dataset parameters
dataset_fraction = 1
batch_size = 128

# SNN parameters
beta = 0.9
time_steps = 50
spike_grad = surrogate.fast_sigmoid(slope=25)

# Training parameters
training_epochs = 10
info_interval = 50

# Optuna parameters
optuna_study_name = 'optimization_test'
optuna_epochs = 10
optuna_trials = 50

# Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dataset
train_dataset, test_dataset, input_dim = get_dataset("MNIST", dataset_fraction, flatten=True)
train_loader, test_loader = get_loaders(train_dataset, test_dataset, batch_size)

# Training models
configs = [
    models.model_snn_backprop.get_model('SNN_Backprop', input_dim, time_steps, beta, spike_grad),
    models.model_snn_perturbation.get_model('SNN_Perturbation', input_dim, time_steps, beta, spike_grad),
    models.model_snn_random_feedback.get_model('SNN_Random_Feedback', input_dim, time_steps, beta, spike_grad),
    models.model_snn_pepita.get_model('SNN_PEPITA', [input_dim, 128, 10], beta, time_steps=time_steps,
                                          output_neuron=False, lr=0.0058, init_method='gaussian', multiplier=0.076),

    models.model_ann_backprop.get_model('ANN_Backprop', [input_dim, 128, 10]),
    models.model_ann_dfa.get_model('ANN_DFA', [input_dim, 128, 10]),
    models.model_ann_pepita.get_model('ANN_PEPITA', [input_dim, 128, 10], lr=0.01),
]


# Training function
def training():
    trainer = Trainer(configs, device)
    trainer.train(train_loader, test_loader, training_epochs, info_interval=info_interval)
    plot_results(trainer)


# Optuna function
def optuna_run(trail_generator):
    study = optuna.create_study(study_name=optuna_study_name, direction='maximize')
    loaders_getter = get_loaders_getter(train_dataset, test_dataset, batch_size)
    objective = create_objective(trail_generator, loaders_getter, optuna_epochs, device)
    study.optimize(objective, n_trials=optuna_trials)

    print("Best Trial:")
    print(f"  Accuracy: {study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    os.makedirs('optuna_results', exist_ok=True)
    optuna.visualization.plot_optimization_history(study).write_html(f'optuna_results/{optuna_study_name}.html')


# Training run
training()

# Optuna run
# trail_generator = models.model_ann_pepita.get_trial_generator(input_dim, 10)
# optuna_run(trail_generator)
