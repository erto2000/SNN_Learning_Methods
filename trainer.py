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
        self.iteration = 0  # Global iteration counter
        self.accumulated_times = {}  # Store accumulated time per model

        for config in configs:
            self.names.append(config['name'])
            self.models.append(config['model'].to(device))
            self.optimize_fns.append(config['optimize_fn'])
            self.test_fns.append(config['test_fn'])
            self.metrics[config['name']] = {
                'iterations': [],
                'losses': [],
                'accuracies': [],
                'times': []  # Store cumulative time
            }
            self.accumulated_times[config['name']] = 0.0  # Initialize time accumulator

    def train(self, train_loader, test_loader, num_epochs, test_interval=50):
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}\n")
            for i, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Store losses per model for this iteration.
                losses = []
                for m, model in enumerate(self.models):
                    model.train()
                    start_time = time.time()
                    loss = self.optimize_fns[m](data, targets)
                    losses.append(loss)
                    elapsed_time = time.time() - start_time
                    self.accumulated_times[self.names[m]] += elapsed_time

                # Run test evaluation if it's the correct iteration.
                if self.iteration % test_interval == 0:
                    self.test(test_loader, self.iteration, epoch, losses)

                self.iteration += 1  # Increase the global iteration counter

    def test(self, test_loader, iteration, epoch, losses):
        header = f"{'Model':<15}{'Epoch':<8}{'Iter':<8}{'Loss':<10}{'Acc (%)':<10}{'Time (s)':<10}"
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
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            self.metrics[name]['iterations'].append(iteration)
            self.metrics[name]['losses'].append(losses[m])
            self.metrics[name]['accuracies'].append(accuracy)
            self.metrics[name]['times'].append(self.accumulated_times[name])

            print(f"{name:<15}{epoch + 1:<8}{iteration:<8}{losses[m]:<10.4f}{accuracy * 100:<10.2f}{self.accumulated_times[name]:<10.4f}")
        print("\n")
