import torch
from core import batch_accuracy


class Trainer:
    def __init__(self, configs, device):
        self.device = device
        self.names = []
        self.models = []
        self.optimize_fns = []
        self.test_acc_hist = []

        for config in configs:
            self.names.append(config['name'])
            self.models.append(config['model'].to(device))
            self.optimize_fns.append(config['optimize_fn'])
            self.test_acc_hist.append([])

    def train(self, train_loader, test_loader, num_epochs, test_interval=50):
        counter = 0
        for epoch in range(num_epochs):
            for data, targets in iter(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                for i, model in enumerate(self.models):
                    # Forward pass
                    model.train()
                    spk_rec = model(data)

                    self.optimize_fns[i](spk_rec, targets)

                    # Evaluate test accuracy periodically
                    if counter % test_interval == 0:
                        with torch.no_grad():
                            model.eval()
                            test_acc = batch_accuracy(test_loader, self.device, model)
                            print(f"Model {self.names[i]}, Iteration {counter}/{len(train_loader)}, Test Acc: {test_acc * 100:.2f}%\n")
                            self.test_acc_hist[i].append(test_acc.item())
                counter += 1