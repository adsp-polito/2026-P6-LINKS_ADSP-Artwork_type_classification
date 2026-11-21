import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(logits, labels, classes):
    """logits: (N, C) numpy array, labels: (N,) class indices."""
    preds = logits.argmax(1)
    ids = list(range(len(classes)))
    f1 = f1_score(labels, preds, labels=ids, average=None, zero_division=0)
    return {
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_per_class": {c: float(f) for c, f in zip(classes, f1)},
        "support": {c: int((labels == i).sum()) for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(labels, preds, labels=ids).tolist(),
    }


def plot_confusion_matrix(cm, classes, path, title=""):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, 0.6 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] == 0:
                continue  # skip zero counts
            # relative value (row-normalized) on top, absolute count below
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")

    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
