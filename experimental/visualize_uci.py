#!/usr/bin/env python3
"""
Full end-to-end script to download, load, explore, and visualize
the UCI HAR (Human Activity Recognition Using Smartphones) dataset.
"""

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
DATA_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
ZIP_PATH   = "../data/UCI_HAR.zip"
DATA_DIR   = "../data/UCI_HAR_Dataset"
WINDOW_LEN = 128
CHANNELS   = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x","body_gyro_y","body_gyro_z",
    "total_acc_x","total_acc_y","total_acc_z"
]
CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]
# ────────────────────────────────────────────────────────────────────────────────

def download_and_extract():
    """Download and unzip the dataset if not already present."""
    if not os.path.exists(ZIP_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    if not os.path.exists(DATA_DIR):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(".")
        os.rename("UCI HAR Dataset", DATA_DIR)
    print(f"Dataset ready at ./{DATA_DIR}")

def load_split(split="train"):
    """
    Load X and y for a given split.
    Returns:
      X : np.ndarray, shape = [N, WINDOW_LEN, len(CHANNELS)]
      y : np.ndarray, shape = [N]
    """
    folder = os.path.join(DATA_DIR, split, "Inertial Signals")
    # Load each channel: array shape [N, WINDOW_LEN]
    arrays = []
    for ch in CHANNELS:
        path = os.path.join(folder, f"{ch}_{split}.txt")
        arr  = np.loadtxt(path)         # [N, WINDOW_LEN]
        arrays.append(arr[..., np.newaxis])  # → [N, WINDOW_LEN, 1]
    # Concatenate channels → [N, WINDOW_LEN, C]
    X = np.concatenate(arrays, axis=2)
    # Load labels (1..6) → zero-indexed
    y_path = os.path.join(DATA_DIR, split, f"y_{split}.txt")
    y      = np.loadtxt(y_path).astype(int) - 1
    return X, y

def print_basic_stats(X_train, y_train):
    """Print dataset shapes and class distribution."""
    print("Training set shape:", X_train.shape)
    print("Test set shape:    ", X_test.shape)
    dist = pd.Series(y_train).value_counts().sort_index()
    print("\nClass distribution (train):")
    for i, cnt in dist.items():
        print(f"  {i} ({CLASS_NAMES[i]}): {cnt} windows")
    print()

def plot_class_distribution(y_train):
    """Bar plot of the training set class counts."""
    dist = pd.Series(y_train).value_counts().sort_index()
    plt.figure(figsize=(6,4))
    dist.plot.bar(color="skyblue")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.xlabel("Activity")
    plt.ylabel("Number of windows")
    plt.title("Training Set Class Distribution")
    plt.tight_layout()
    plt.show()

def plot_sample_windows(X_train, y_train):
    """Plot one window per class, all channels over time."""
    fig, axes = plt.subplots(len(CLASS_NAMES), 1, figsize=(10, 12), sharex=True)
    for cls in range(len(CLASS_NAMES)):
        idx    = np.where(y_train == cls)[0][0]
        sample = X_train[idx]  # [WINDOW_LEN, C]
        ax     = axes[cls]
        for c in range(X_train.shape[2]):
            ax.plot(sample[:, c], label=CHANNELS[c].split("_")[-2] + "_" + CHANNELS[c].split("_")[-1], alpha=0.7)
        ax.set_ylabel(f"{CLASS_NAMES[cls]}")
        if cls == 0:
            ax.legend(loc="upper right", fontsize="small")
    axes[-1].set_xlabel("Time step")
    plt.suptitle("One 128-step Window per Activity Class")
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.show()

def plot_channel_stats(X_train):
    """Plot mean ± std for each channel over the entire training set."""
    N, T, C = X_train.shape
    flat = X_train.reshape(-1, C)
    means = flat.mean(axis=0)
    stds  = flat.std(axis=0)

    plt.figure(figsize=(6,4))
    plt.errorbar(range(C), means, yerr=stds, fmt="o", capsize=5)
    plt.xticks(range(C), [ch.split("_")[-2] + "_" + ch.split("_")[-1] for ch in CHANNELS], rotation=45, ha="right")
    plt.ylabel("Magnitude")
    plt.title("Channel-wise Mean ± Std over Training Set")
    plt.tight_layout()
    plt.show()

def plot_tsne(X_train, y_train):
    """Compute PCA → t-SNE and plot a 2D embedding colored by class."""
    N, T, C = X_train.shape
    flat = X_train.reshape(N, -1)  # [N, T*C]

    print("Computing PCA...")
    pca = PCA(n_components=50, random_state=42)
    pca_proj = pca.fit_transform(flat)

    print("Computing t-SNE (this may take a few minutes)...")
    tsne = TSNE(n_components=2, init="pca", random_state=42)
    Z    = tsne.fit_transform(pca_proj)

    plt.figure(figsize=(6,6))
    scatter = plt.scatter(Z[:,0], Z[:,1], c=y_train, cmap="tab10", s=5)
    plt.legend(*scatter.legend_elements(), title="Class", labels=CLASS_NAMES, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE of HAR Windows")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Download & extract
    download_and_extract()

    # 2. Load data
    X_train, y_train = load_split("train")
    X_test,  y_test  = load_split("test")

    # 3. Basic stats
    print_basic_stats(X_train, y_train)

    # 4. Visualizations
    plot_class_distribution(y_train)
    plot_sample_windows(X_train, y_train)
    plot_channel_stats(X_train)
    plot_tsne(X_train, y_train)
