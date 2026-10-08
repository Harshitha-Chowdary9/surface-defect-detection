"""Classify a single image: python src/predict.py image.png"""
import argparse
import torch
from PIL import Image
from torchvision import transforms
from data import MEAN, STD
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--checkpoint", default="outputs/best.pt")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu")
    model = build_model(ck["arch"], len(ck["classes"])); model.load_state_dict(ck["model"]); model.eval()
    tf = transforms.Compose([transforms.Resize((ck["img_size"],) * 2), transforms.Grayscale(3),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    with torch.no_grad():
        p = torch.softmax(model(tf(Image.open(a.image).convert("RGB")).unsqueeze(0)), 1)[0]
    for i in p.argsort(descending=True)[:3]:
        print(f"{ck['classes'][i]:<18}{p[i]:.3f}")


if __name__ == "__main__":
    main()
