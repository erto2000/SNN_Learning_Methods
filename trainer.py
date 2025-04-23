import torch
import time

class Trainer:
    def __init__(self, configs, device):
        self.device = device
        self.names = []
        self.models = []
        self.optimize_fns = []
        self.test_fns = []
        self.metrics = {}
        self.global_iteration = 0
        self.accumulated_times = {}  # Store accumulated time per model

        for config in configs:
            self.names.append(config['name'])
            self.models.append(config['model'].to(device))
            self.optimize_fns.append(config['optimize_fn'])
            self.test_fns.append(config['test_fn'])
            self.metrics[config['name']] = {
                'iterations': [],
                'losses': [],
                'train_accuracies': [],  # Track training accuracy
                'test_accuracies': [],
                'times': []  # Store cumulative time
            }
            self.accumulated_times[config['name']] = 0.0  # Initialize time accumulator

    def train(self, train_loader, test_loader, num_epochs, info_interval=50):
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}\n")
            iteration = 0
            for i, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                losses = []
                train_accuracies = []

                for m, model in enumerate(self.models):
                    model.train()
                    start_time = time.time()
                    loss, train_accuracy = self.optimize_fns[m](data, targets)  # Now returns loss & accuracy
                    elapsed_time = time.time() - start_time
                    losses.append(loss)
                    train_accuracies.append(train_accuracy)
                    self.accumulated_times[self.names[m]] += elapsed_time

                # Run test evaluation if it's the correct iteration.
                if iteration % info_interval == 0:
                    self.print_info(test_loader, self.global_iteration, epoch, losses, train_accuracies)
                iteration += 1
                self.global_iteration += 1

    def print_info(self, test_loader, iteration, epoch, losses, train_accuracies):
        header = f"{'Model':<25}{'Epoch':<8}{'Iter':<8}{'Loss':<10}{'Train Acc (%)':<15}{'Test Acc (%)':<15}{'Time (s)':<10}"
        print(header)
        print("-" * len(header))

        # Dictionary to accumulate test results per model.
        test_results = {name: {"correct": 0, "total": 0} for name in self.names}

        with torch.no_grad():
            # Process each test batch one by one.
            for test_data, test_targets in test_loader:
                test_data = test_data.to(self.device)
                test_targets = test_targets.to(self.device)
                for m, model in enumerate(self.models):
                    # Run test function for current model and batch.
                    batch_acc = self.test_fns[m](test_data, test_targets)
                    batch_size = test_data.size(0)
                    test_results[self.names[m]]["correct"] += batch_acc * batch_size
                    test_results[self.names[m]]["total"] += batch_size

        # Save metrics and print the results for each model.
        for m, name in enumerate(self.names):
            total_correct = test_results[name]["correct"]
            total_samples = test_results[name]["total"]
            test_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            self.metrics[name]['iterations'].append(iteration)
            self.metrics[name]['losses'].append(losses[m])
            self.metrics[name]['train_accuracies'].append(train_accuracies[m])  # Save train accuracy
            self.metrics[name]['test_accuracies'].append(test_accuracy)  # Save test accuracy
            self.metrics[name]['times'].append(self.accumulated_times[name])

            print(f"{name:<25}{epoch + 1:<8}{iteration:<8}{losses[m]:<10.4f}{train_accuracies[m] * 100:<15.2f}{test_accuracy * 100:<15.2f}{self.accumulated_times[name]:<10.4f}")
        print("\n")
