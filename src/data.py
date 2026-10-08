"""Dataset loading: any folder with one sub-folder per class (torchvision ImageFolder)."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_loaders(root: str, img_size: int = 128, batch_size: int = 32, val_frac: float = 0.2,
                seed: int = 0, num_workers: int = 0):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.Grayscale(3),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    full_train = datasets.ImageFolder(root, transform=train_tf)
    full_eval = datasets.ImageFolder(root, transform=eval_tf)
    idx = list(range(len(full_train)))
    tr, va = train_test_split(idx, test_size=val_frac, stratify=full_train.targets, random_state=seed)
    mk = lambda ds, ix, sh: DataLoader(Subset(ds, ix), batch_size=batch_size, shuffle=sh,
                                       num_workers=num_workers)
    return mk(full_train, tr, True), mk(full_eval, va, False), full_train.classes
