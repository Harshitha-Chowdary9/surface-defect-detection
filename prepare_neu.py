"""Copy an extracted official NEU-CLS folder into ImageFolder layout.

Usage: python prepare_neu.py /path/to/extracted/NEU-CLS --out data/NEU
The dataset is not bundled. Download NEU-CLS from the authors' page linked in README.
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil

CLASSES = {
    "Cr": "crazing", "In": "inclusion", "Pa": "patches",
    "PS": "pitted_surface", "RS": "rolled_in_scale", "Sc": "scratches",
}


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    files = sorted(source.glob("*.bmp"))
    counts = Counter(path.name.split("_")[0] for path in files)
    if counts != Counter({prefix: 300 for prefix in CLASSES}):
        raise ValueError(f"Expected 300 BMP images for each NEU-CLS prefix; found {dict(counts)}")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output must be empty: {destination}")
    for path in files:
        target = destination / CLASSES[path.name.split("_")[0]] / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return {name: counts[prefix] for prefix, name in CLASSES.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Extracted NEU-CLS directory containing Cr_1.bmp etc.")
    parser.add_argument("--out", default="data/NEU")
    args = parser.parse_args()
    print(prepare(args.source, args.out))
