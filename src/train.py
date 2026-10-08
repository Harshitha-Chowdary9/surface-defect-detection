"""Train a defect classifier and save the best checkpoint and curves."""
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data import get_loaders
from model import build_model


def run_epoch(model, loader, device, opt=None, loss_fn=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per class")
    ap.add_argument("--model", choices=["cnn", "resnet50"], default="cnn")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, classes = get_loaders(a.data, a.img_size, a.batch_size)
    model = build_model(a.model, len(classes), a.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out = Path(a.out); out.mkdir(exist_ok=True)
    hist, best = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}, 0.0
    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, opt, loss_fn)
        vl, va = run_epoch(model, val_dl, device, None, loss_fn)
        sched.step()
        for k, v in zip(hist, (ta, va, tl, vl)):
            hist[k].append(v)
        print(f"epoch {ep:02d} train_loss={tl:.3f} acc={ta:.3f} | val_loss={vl:.3f} acc={va:.3f}")
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "classes": classes, "arch": a.model,
                        "img_size": a.img_size}, out / "best.pt")
    json.dump({"best_val_acc": best, "classes": classes, "history": hist}, open(out / "history.json", "w"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "curves.png", dpi=130)
    print(f"best val acc: {best:.3f}")


if __name__ == "__main__":
    main()
