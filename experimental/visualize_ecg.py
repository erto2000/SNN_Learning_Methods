import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.datasets import UCR_UEA_datasets

def plot_one_sample_per_class(random_seed: int = 0):
    # 1. Load the ECG5000 dataset
    loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = loader.load_dataset("ECG5000")

    # 2. Squeeze singleton dimension if present (n_samples, ts_length, 1 → n_samples, ts_length)
    if X_train.ndim == 3 and X_train.shape[2] == 1:
        X_train = X_train.squeeze(-1)
        X_test  = X_test.squeeze(-1)

    # 3. Combine train + test for unified indexing
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # 4. Identify unique classes
    classes = np.unique(y)
    print("Found classes:", classes)

    # 5. Pick one sample index per class (first occurrence)
    #    To pick randomly instead, uncomment the np.random.seed / np.random.choice lines.
    # np.random.seed(random_seed)
    # sample_indices = [np.random.choice(np.where(y == cls)[0]) for cls in classes]
    sample_indices = [np.where(y == cls)[0][0] for cls in classes]
    for cls, idx in zip(classes, sample_indices):
        print(f"Class {int(cls)} → sample index {idx}")

    # 6. Overlay plot: one waveform per class
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for cls, idx in zip(classes, sample_indices):
        plt.plot(X[idx], label=f"Class {int(cls)}", linewidth=1.5)
    plt.title("ECG5000: One Sample per Class (Overlaid)")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 7. Separate subplots: one waveform per class
    fig, axs = plt.subplots(len(classes), 1, figsize=(10, 12), sharex=True)
    for ax, cls, idx in zip(axs, classes, sample_indices):
        ax.plot(X[idx], color=f"C{int(cls)}", linewidth=1.5)
        ax.set_title(f"Class {int(cls)} Sample (index {idx})")
        ax.set_ylabel("Amplitude")
    axs[-1].set_xlabel("Time Index")
    plt.suptitle("ECG5000: One Representative Waveform per Class", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_one_sample_per_class()