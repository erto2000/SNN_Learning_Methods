import matplotlib.pyplot as plt


def plot_results(trainer):
    plt.figure(figsize=(14, 6))  # Set figure size

    # Plot Loss
    plt.subplot(1, 2, 1)  # Create subplot (1 row, 2 columns, first plot)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['losses'], marker='o', label=f'{name} Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss for All Methods')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)  # Create subplot (1 row, 2 columns, second plot)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['accuracies'], marker='o', label=f'{name} Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for All Methods')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
