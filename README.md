# Skin Lesions HM10000


Binary (benign / malignant) skin lesion classification using the HAM10000 dataset
([`preetsojitra/skin-lesion-HM10000-binary-classificationt`](https://huggingface.co/datasets/preetsojitra/skin-lesion-HM10000-binary-classificationt) on Hugging Face).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data pipeline

```bash
python fetch_data.py   # downloads ~3 GB into data/raw
python quality.py      # cleans data and writes the splits to data/processed
```

`quality.py` removes corrupt images, label/diagnosis mismatches, exact duplicates and
near-duplicates. It then splits the HF train split 75/25, **grouped by lesion**, so the same
lesion never shows up in both train and test. The HF test split is kept as a separate holdout.

```
data/processed/
  train/{benign,malignant}/*.jpg   train.csv
  test/{benign,malignant}/*.jpg    test.csv
  holdout/{benign,malignant}/*.jpg holdout.csv
  quality_report.json
```

## Training (ResNet18 / EfficientNet fine-tuning)

```bash
python train.py                                       # ResNet18 (default); auto-selects CUDA / Apple MPS / CPU
python train.py --arch efficientnet_b0                # EfficientNet-B0
python train.py --arch efficientnet_b3 --img-size 300 # larger EfficientNet at its native resolution
```

The script carves a lesion-grouped validation set out of train for early stopping, checkpoint
selection and threshold tuning. It trains the head first, then fine-tunes the whole network
with class-weighted loss. It reports validation metrics only. The checkpoint (with its tuned
threshold) and metrics go to `models/`.

## Final benchmark (run once, at the end)

`train.py` never reads the test or holdout sets. They're reserved as the final benchmark.
Make every decision (architecture, hyperparameters, augmentation) from validation results,
then score the chosen model once:

```bash
python evaluate.py models/resnet18_best.pt   # AUC, balanced accuracy, sensitivity, specificity
```

Results are appended to `models/final_benchmark.json`. Scoring a split a second time needs
`--rerun`, so any repeat run is deliberate and recorded.

## Visualising the network

During training, `train.py` saves figures that follow an image left to right through the CNN:
the input and its Grad-CAM heatmap, then the most active feature maps of each conv stage (with
their channels×height×width), and finally the output probabilities.

```bash
python train.py                    # fixed validation images, one figure per epoch + evolution.gif
python train.py --viz-steps 50     # also the training image going through the net, every 50 steps
python train.py --viz-live         # live window: every batch's image going through each layer
python visualize.py models/resnet18_best.pt some_image.jpg   # one image through a trained model
```

Output goes to `models/viz/<arch>/`. `epoch_00` is the untrained ImageNet network; comparing it
with later epochs shows the deeper filters becoming specific to lesions.
