import matplotlib.pyplot as plt


# Plot test accuracy for all models on the same graph
def plot_results(trainer):
    plt.figure(figsize=(8, 5))

    for i in range(len(trainer.test_acc_hist)):
        plt.plot(trainer.test_acc_hist[i], label=trainer.names[i])

    plt.title("Test Set Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)
