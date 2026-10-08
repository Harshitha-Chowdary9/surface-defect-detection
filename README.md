# Surface Defect Detection (CNN, PyTorch)

Image classification of metal surface defects with a small CNN trained from scratch, plus a
ResNet50 transfer-learning option. Built around the same problem area as published work on
blade surface defect detection: telling defect types apart from grayscale surface images.

## Features

- Works with any folder of images laid out one sub-folder per class
- Grayscale-aware preprocessing, flip augmentation, stratified train/validation split
- Two architectures: `cnn` (4-block CNN with batch norm) and `resnet50` (optionally pretrained)
- Training curves, per-class report, confusion matrix, single-image prediction
- Built-in synthetic dataset generator so the pipeline runs with no downloads

## Structure

```
src/data.py            dataset loading and transforms
src/model.py           SmallCNN and ResNet50 builders
src/train.py           training loop, saves outputs/best.pt and curves.png
src/evaluate.py        classification report + confusion matrix
src/predict.py         classify one image
src/synthetic_data.py  procedural 6-class surface dataset for smoke testing
tests/test_smoke.py    forward-pass shape test
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Quick run on the synthetic data:

```bash
python src/synthetic_data.py --per-class 100
python src/train.py --data data/synthetic --epochs 5 --img-size 96
python src/evaluate.py --data data/synthetic
python src/predict.py data/synthetic/scratches/scratches_0001.png
```

Real data: download the NEU surface defect dataset (NEU-CLS: 6 classes, 1,800 grayscale
200x200 images - crazing, inclusion, patches, pitted surface, rolled-in scale, scratches),
arrange it as `data/NEU/<class_name>/*.bmp`, then:

```bash
python src/train.py --data data/NEU --model cnn --epochs 30 --img-size 128
python src/train.py --data data/NEU --model resnet50 --pretrained --epochs 15 --img-size 224 --lr 1e-4
python src/evaluate.py --data data/NEU
```

## Results

On the synthetic data the CNN reaches 100% validation accuracy within 5 epochs on CPU. That data is
deliberately easy and only proves the pipeline works end to end. **No claims are made for real
data until you train on NEU and fill in the numbers here.**

| Dataset | Model | Val accuracy |
|---------|-------|--------------|
| Synthetic | cnn | 1.00 |
| NEU-CLS | cnn | _to fill in_ |
| NEU-CLS | resnet50 (pretrained) | _to fill in_ |

## Ideas for extension

- Compare CNN vs ResNet50 vs Xception
- Grad-CAM to visualise what the model looks at
- Detection (NEU-DET) with bounding boxes
