import torch


def create_objective(trial_generator, loaders_getter, epoch, device):
    def objective(trial):
        config = trial_generator(trial)

        train_loader, test_loader = loaders_getter()

        # Training loop
        for _ in range(epoch):
            for i, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                config['model'].to(device).train()
                _, _ = config['optimize_fn'](data, targets)

        # Validation
        with torch.no_grad():
            # Process each test batch one by one.
            total_accuracy = 0
            total_samples = 0
            for test_data, test_targets in test_loader:
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)
                batch_acc = config['test_fn'](test_data, test_targets)
                batch_size = test_data.size(0)
                total_accuracy += batch_acc * batch_size
                total_samples += batch_size

        accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0
        return accuracy

    return objective
