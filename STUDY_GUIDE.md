# Study guide - interview questions this project prepares you for

Read this alongside the code. Every answer points at the file where you can
see the idea working.

## Data and splits

**Q: How do you keep the class balance fair in the split?**
`src/data.py` uses `train_test_split(..., stratify=targets)`: every class
keeps the same 80/20 proportion, so a 6-class problem cannot end up with a
validation set missing a class by chance. Seed is fixed (0), so the split
is reproducible.

**Q: Why two different transform pipelines?**
Training gets augmentation (random horizontal + vertical flips) on top of
resize/grayscale/normalize; evaluation gets none. Augmentation is a
training-only regularizer - evaluating on flipped images would add noise to
the metric without adding information.

**Q: Why augment with flips for this data?**
A scratch or patch is still the same defect mirrored, and surface images
have no canonical orientation. Flips roughly double effective data variety
for free. (For digits or text this would be wrong - augmentation must match
the symmetries of the problem.)

**Q: Why convert grayscale images to 3 channels?**
`transforms.Grayscale(3)` keeps one intensity but replicates it to three
channels so the same code path feeds both the from-scratch CNN and a
pretrained ResNet50, which expects 3-channel ImageNet-style input.

## Models

**Q: Describe your from-scratch CNN.**
`SmallCNN` in `src/model.py`: four conv blocks (conv 3x3 -> batch norm ->
ReLU -> max pool, channels 32/64/128/128), then global average pooling,
dropout 0.3, and a linear head. Global average pooling instead of a big
flatten keeps the parameter count small and makes the head independent of
input size.

**Q: When would you pick ResNet50 transfer learning over the small CNN?**
`build_model("resnet50", pretrained=True)` loads ImageNet weights and
replaces only the final `fc` layer. Pretrained features (edges, textures)
transfer well to surface images, so transfer learning wins when data is
scarce - 1,800 images here. Training a deep net from scratch on 1,800
images would overfit; the small CNN works precisely because it is small.

## Evaluation honesty (this repo's README is the example to copy)

**Q: Your headline number is 93.3% - what caveats go with it?**
The README lists them: it is validation accuracy used to select the
checkpoint, not an independent test set; one split and one seed; a 2-epoch
CPU baseline, not convergence; NEU-CLS is steel-strip defects, not a
reproduction of the blade-defect paper. Knowing what a number does NOT
claim is what separates a careful engineer from a leaderboard screenshot.

**Q: Where does the model still fail?**
Per the confusion matrix in the README: pitted surface has the weakest
recall (50/60), confused mostly with inclusion (and vice versa - 6
inclusions called pitted). Saying which classes confuse and having a
hypothesis why (similar local texture) beats quoting one accuracy number.

**Q: Why report the archive SHA-256 and a duplicate-image note?**
`neu-benchmark.json` records the dataset archive hash and the run's package
versions, so the result is reproducible and auditable. The README even
notes a byte-identical image pair (`Pa_101`/`Pa_105`) and confirms both
landed in training, so no leakage through that pair. Checking your data is
part of the job.

## Engineering

**Q: How does the repo run with no dataset download?**
`src/synthetic_data.py` procedurally generates a 6-class surface-like
dataset, and `tests/test_smoke.py` checks forward-pass shapes. The pipeline
is provably end-to-end before anyone touches real data - and the README is
explicit that synthetic 100% accuracy only proves plumbing, not skill.

**Q: What checkpointing strategy does training use?**
`src/train.py` tracks validation accuracy and saves the best checkpoint to
`outputs/best.pt` with curves in `curves.png`; evaluation and single-image
prediction (`src/evaluate.py`, `src/predict.py`) load that checkpoint. Best-
on-validation, not last-epoch, is what you ship.
