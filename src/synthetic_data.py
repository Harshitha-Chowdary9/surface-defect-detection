"""Generate a small synthetic metal-surface dataset in ImageFolder layout.

Lets you run the full pipeline without downloading anything. Six classes mimic
the NEU surface defect categories (crazing, inclusion, patches, pitted surface,
rolled-in scale, scratches) with simple procedural textures. For real results use
the NEU-DET dataset instead (see README).
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

CLASSES = ["crazing", "inclusion", "patches", "pitted", "rolled_in_scale", "scratches"]


def base_texture(rng, size):
    noise = rng.normal(128, 12, (size, size))
    img = Image.fromarray(np.clip(noise, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(img).astype(float)


def add_defect(arr, cls, rng):
    s = arr.shape[0]
    if cls == "crazing":  # fine crossing cracks
        for _ in range(25):
            x, y = rng.integers(0, s, 2); n = rng.integers(10, 40)
            dx, dy = rng.integers(-1, 2, 2)
            for k in range(n):
                arr[min(max(y + k * dy, 0), s - 1), min(max(x + k * dx, 0), s - 1)] -= 50
    elif cls == "inclusion":  # dark blobs
        for _ in range(rng.integers(3, 8)):
            x, y, r = rng.integers(10, s - 10), rng.integers(10, s - 10), rng.integers(3, 8)
            yy, xx = np.ogrid[:s, :s]; arr[(yy - y) ** 2 + (xx - x) ** 2 < r * r] -= 70
    elif cls == "patches":  # large bright/dark region
        x, y = rng.integers(0, s // 2, 2); w, h = rng.integers(s // 4, s // 2, 2)
        arr[y:y + h, x:x + w] += rng.choice([-45, 45])
    elif cls == "pitted":  # many small pits
        for _ in range(rng.integers(60, 120)):
            arr[rng.integers(0, s), rng.integers(0, s)] -= 90
    elif cls == "rolled_in_scale":  # wavy bands
        yy = np.arange(s)[:, None]; arr += 25 * np.sin(yy / rng.uniform(3, 6) + rng.uniform(0, 6))
    elif cls == "scratches":  # long straight lines
        for _ in range(rng.integers(2, 5)):
            x = rng.integers(0, s); arr[:, x:x + 2] -= 55
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for cls in CLASSES:
        d = Path(a.out) / cls; d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_class):
            arr = add_defect(base_texture(rng, a.size), cls, rng)
            Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(d / f"{cls}_{i:04d}.png")
    print("wrote", len(CLASSES) * a.per_class, "images to", a.out)


if __name__ == "__main__":
    main()
