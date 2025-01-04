import torch
from core import batch_accuracy


class Trainer:
    def __init__(self, configs, device):
        self.device = device
        self.names = []
        self.models = []
        self.optimize_fns = []
        self.metrics = {}

        for config in configs:
            self.names.append(config['name'])
            self.models.append(config['model'].to(device))
            self.optimize_fns.append(config['optimize_fn'])
            self.metrics[config['name']] = {'iterations': [], 'losses': [], 'accuracies': []}

    def train(self, train_loader, test_loader, num_epochs, test_interval=50):
        for epoch in range(num_epochs):
            print()
            counter = 0
            for i, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                for m, model in enumerate(self.models):
                    # Forward pass
                    model.train()

                    loss = self.optimize_fns[m](data, targets)

                    # Evaluate test accuracy periodically
                    if counter % test_interval == 0:
                        with torch.no_grad():
                            model.eval()
                            test_acc = batch_accuracy(test_loader, self.device, model)
                            print(f"Model {self.names[m]}, Iteration E{epoch+1}-{counter}/{len(train_loader)}, "
                                  f"Loss: {loss:.4f}, Test Acc: {test_acc * 100:.2f}%")
                            self.metrics[self.names[m]]['iterations'].append(epoch * len(train_loader) + i + 1)
                            self.metrics[self.names[m]]['losses'].append(loss)
                            self.metrics[self.names[m]]['accuracies'].append(test_acc)
                counter += 1