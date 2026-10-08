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

Real data: download NEU-CLS from the [authors' official dataset page](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm), using its NEU-CLS Google Drive link (no Kaggle login was needed for this run). The NEU surface defect dataset (NEU-CLS: 6 classes, 1,800 grayscale
200x200 images - crazing, inclusion, patches, pitted surface, rolled-in scale, scratches),
extract the archive, then use the checked preparation helper:

```bash
python prepare_neu.py /path/to/extracted/NEU-CLS --out data/NEU
python -m unittest test_prepare_neu
```

This copies the flat `Cr_*.bmp`, `In_*.bmp`, `Pa_*.bmp`, `PS_*.bmp`,
`RS_*.bmp`, and `Sc_*.bmp` files into `data/NEU/<class_name>/*.bmp`, verifies
300 images per class, and refuses to overwrite a nonempty output folder. Then:

```bash
python src/train.py --data data/NEU --model cnn --epochs 30 --img-size 128
python src/train.py --data data/NEU --model resnet50 --pretrained --epochs 15 --img-size 224 --lr 1e-4
python src/evaluate.py --data data/NEU
```

## Results

On the synthetic data the CNN reaches 100% validation accuracy within 5 epochs on CPU. That data is
deliberately easy and only proves the pipeline works end to end. The real NEU-CLS baseline below is a separate run; synthetic scores are not real-data performance.

| Dataset | Model | Val accuracy |
|---------|-------|--------------|
| Synthetic | cnn | 1.00 |
| NEU-CLS | cnn (2 epochs, 64x64) | 0.9333 (336/360) |
| NEU-CLS | resnet50 (pretrained) | _to fill in_ |

## Ideas for extension

- Compare CNN vs ResNet50 vs Xception
- Grad-CAM to visualise what the model looks at
- Detection (NEU-DET) with bounding boxes

## Real NEU-CLS baseline (2026-10-08)

A small CPU run on all 1,800 official grayscale 200x200 images, resized to 64x64.
The existing stratified split uses seed 0: 1,440 training images and 360 validation
images (60 per class). Architecture: `cnn`; batch size 32; Adam learning rate 0.001;
2 epochs; no pretrained weights. The best validation checkpoint was epoch 2.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python src/train.py --data data/NEU --model cnn --epochs 2 --img-size 64 --out outputs/neu-cnn-64
OMP_NUM_THREADS=2 python src/evaluate.py --data data/NEU --checkpoint outputs/neu-cnn-64/best.pt --out outputs/neu-cnn-64
```

- Validation accuracy: **93.33% (336/360)**; macro F1: **0.9332**.
- Weakest recall: pitted surface, 50/60 (83.33%); inclusion/pitted-surface confusion remains.
- Full metrics, training history, validation filenames, package versions and archive
  SHA-256: [neu-benchmark.json](neu-benchmark.json).
- The archive has one byte-identical pair, `Pa_101.bmp` and `Pa_105.bmp`.
  Both are in training, so that pair does not cross this train/validation split.
- This is **validation used to select the checkpoint, not an independent test**.
  One split and one seed do not establish generalization. A grouped/deduplicated
  split, separate test set, multiple seeds and ResNet50 comparison are future work.
- NEU-CLS contains steel-strip defects, not blade images. This is not a reproduction
  of the published blade-defect paper and is not a production inspection model.
- Two epochs are an initial baseline, not a convergence claim. Exact scores can vary
  with dependency versions/hardware. Two longer local attempts were interrupted;
  their partial outputs are not included in this reported run.
- Dataset images and checkpoint weights are not bundled or rehosted. Download from
  the authors and follow their terms and citation request.

Confusion matrix (rows = true, columns = predicted), class order:
`crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches`.

```text
55  0  0  0  5  0
 0 54  0  6  0  0
 0  0 60  0  0  0
 0  5  0 50  3  2
 0  0  0  0 60  0
 0  3  0  0  0 57
```

Dataset citation: K. Song and Y. Yan, "A noise robust method based on completed
local binary patterns for hot-rolled steel strip surface defects," Applied Surface
Science, vol. 285, pp. 858-864, November 2013. Source: the authors' dataset page above.
