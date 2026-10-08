"""Evaluate a saved checkpoint: classification report and confusion matrix."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from data import get_loaders
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", default="outputs/best.pt")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu")
    _, val_dl, classes = get_loaders(a.data, ck["img_size"])
    model = build_model(ck["arch"], len(classes)); model.load_state_dict(ck["model"]); model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in val_dl:
            ps += model(x).argmax(1).tolist(); ys += y.tolist()
    print(classification_report(ys, ps, target_names=classes, digits=3))
    cm = confusion_matrix(ys, ps)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right"); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
    Path(a.out).mkdir(exist_ok=True); fig.savefig(Path(a.out) / "confusion_matrix.png", dpi=130)


if __name__ == "__main__":
    main()
