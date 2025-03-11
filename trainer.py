import torch
import time
from core import batch_accuracy


class Trainer:
    def __init__(self, configs, device):
        self.device = device
        self.names = []
        self.models = []
        self.optimize_fns = []
        self.metrics = {}
        self.iteration = 0  # Global iteration counter
        self.accumulated_times = {}  # Store accumulated time per model

        for config in configs:
            self.names.append(config['name'])
            self.models.append(config['model'].to(device))
            self.optimize_fns.append(config['optimize_fn'])
            self.metrics[config['name']] = {
                'iterations': [],
                'losses': [],
                'accuracies': [],
                'times': []  # Store cumulative time
            }
            self.accumulated_times[config['name']] = 0.0  # Initialize time accumulator

    def train(self, train_loader, test_loader, num_epochs, test_interval=50):
        for epoch in range(num_epochs):
            print()
            for i, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                for m, model in enumerate(self.models):
                    model.train()

                    start_time = time.time()  # Start timing
                    loss = self.optimize_fns[m](data, targets)
                    elapsed_time = time.time() - start_time  # Compute time for this iteration

                    self.accumulated_times[self.names[m]] += elapsed_time  # Accumulate time

                    # Evaluate test accuracy periodically
                    if self.iteration % test_interval == 0:
                        with torch.no_grad():
                            model.eval()
                            test_acc = batch_accuracy(test_loader, self.device, model)
                            print(f"Model {self.names[m]}, Iteration {self.iteration}, "
                                  f"Loss: {loss:.4f}, Test Acc: {test_acc * 100:.2f}%, "
                                  f"Time: {self.accumulated_times[self.names[m]]:.4f}s (Accumulated)")

                            self.metrics[self.names[m]]['iterations'].append(self.iteration)
                            self.metrics[self.names[m]]['losses'].append(loss)
                            self.metrics[self.names[m]]['accuracies'].append(test_acc)
                            self.metrics[self.names[m]]['times'].append(self.accumulated_times[self.names[m]])  # Store accumulated time

                self.iteration += 1  # Increase global iteration counter
