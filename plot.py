import matplotlib.pyplot as plt


def plot_results(trainer):
    plt.figure(figsize=(21, 6))  # Increase figure size to accommodate 3 plots

    # Plot Loss
    plt.subplot(1, 3, 1)  # Create subplot (1 row, 3 columns, first plot)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['losses'], marker='o', label=f'{name} Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss for All Methods')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)  # Create subplot (1 row, 3 columns, second plot)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['iterations'], metrics['accuracies'], marker='o', label=f'{name} Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for All Methods')
    plt.legend()
    plt.grid(True)

    # Plot Time vs. Accuracy
    plt.subplot(1, 3, 3)  # Create subplot (1 row, 3 columns, third plot)
    for name in trainer.names:
        metrics = trainer.metrics[name]
        plt.plot(metrics['times'], metrics['accuracies'], marker='o', label=f'{name} Time vs Accuracy')
    plt.xlabel('Accumulated Time (s)')
    plt.ylabel('Accuracy (%)')
    plt.title('Time vs. Accuracy')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
