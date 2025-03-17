import matplotlib.pyplot as plt


def plot_results(trainer):
    plt.figure(figsize=(14, 10))  # Adjust figure size for better readability

    # Plot Loss
    plt.subplot(2, 2, 1)  # First plot (Row 1, Column 1)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['losses'], marker='o', label=f'{name} Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss for All Methods')
    plt.legend()
    plt.grid(True)

    # Plot Training Accuracy
    plt.subplot(2, 2, 2)  # Second plot (Row 1, Column 2)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['train_accuracies'], marker='o', label=f'{name} Train Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy for All Methods')
    plt.legend()
    plt.grid(True)

    # Plot Test Accuracy
    plt.subplot(2, 2, 3)  # Third plot (Row 2, Column 1)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['test_accuracies'], marker='o', label=f'{name} Test Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy for All Methods')
    plt.legend()
    plt.grid(True)

    # Plot Time vs. Test Accuracy
    plt.subplot(2, 2, 4)  # Fourth plot (Row 2, Column 2)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['times'], metrics['test_accuracies'], marker='o', label=f'{name} Time vs Test Acc')
    plt.xlabel('Accumulated Time (s)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Time vs. Test Accuracy')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
