"""Fine-tune an ImageNet-pretrained ResNet18 or EfficientNet on the cleaned HAM10000 splits (benign vs malignant).

Expects the output of quality.py (data/processed/train.csv + image folders).
A validation set is carved out of train (grouped by group_id, so no lesion leaks into it)
and used for early stopping, model selection and threshold tuning.

This script never reads the test or holdout sets. They are the final benchmark: run
evaluate.py on them once, after all training and tuning decisions are made.

Training runs in two phases:
  1. head only  - backbone frozen, only the new classifier layer learns (--head-epochs)
  2. full       - every layer unfrozen at a lower learning rate, with cosine decay (--epochs)

Feature-map visualisation (see visualize.py) is saved to models/viz/<arch>/ while training:
  sample_<k>/epoch_XX.png  fixed validation images through every conv stage, once per epoch
  sample_<k>/evolution.gif the same image across epochs, showing what the filters learn
  steps/step_XXXXX.png     (--viz-steps N) the training image currently going through the network
Pass --viz-live to watch every training step in a live window (the first image of each batch
going through every layer, updated in place), or --no-viz to turn visualisation off.

Usage:
    python train.py                              # ResNet18 (default)
    python train.py --arch efficientnet_b0       # EfficientNet-B0
    python train.py --arch efficientnet_b3 --img-size 300
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from visualize import FlowVisualizer, init_backend, make_gif

ROOT = Path(__file__).resolve().parent
CLASSES = ["benign", "malignant"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1),
}
HEAD_PREFIX = ("fc.", "classifier.")  # ResNet head / EfficientNet head


class LesionDataset(Dataset):
    def __init__(self, df, root, transform):
        self.paths = [root / p for p in df["filepath"]]
        self.labels = df["label"].map(CLASSES.index).to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(size):
    # dermoscopy images have no canonical orientation, so flips and rotations are label-preserving
    train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    evaluate = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, evaluate


def split_validation(train_df, val_fraction, seed):
    sgkf = StratifiedGroupKFold(n_splits=round(1 / val_fraction), shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf.split(train_df, train_df["label"], groups=train_df["group_id"]))
    return train_df.iloc[tr_idx], train_df.iloc[val_idx]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(arch):
    ctor, weights = ARCHS[arch]
    model = ctor(weights=weights)
    if arch.startswith("resnet"):
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASSES)))
    else:
        # EfficientNet ships with its own dropout in classifier[0], so only swap the final linear layer
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model


def set_backbone_trainable(model, trainable):
    for name, p in model.named_parameters():
        if not name.startswith(HEAD_PREFIX):
            p.requires_grad = trainable


def run_epoch(model, loader, criterion, device, optimizer=None, desc="", on_step=None):
    """Train when given an optimizer, otherwise evaluate. Returns (loss, malignant probabilities, labels).

    on_step(step, image, label, loss) is called after every training step with the first image of the batch.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, probs, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(tqdm(loader, desc=desc, leave=False), start=1):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if on_step is not None:
                    on_step(step, x[0], int(y[0]), loss.item())
            total_loss += loss.item() * len(y)
            probs.append(torch.softmax(logits, 1)[:, 1].detach().cpu())
            labels.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(probs).numpy(), torch.cat(labels).numpy()


def metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "sensitivity": float(tp / max(tp + fn, 1)),  # malignant recall: the clinically important number
        "specificity": float(tn / max(tn + fp, 1)),
        "threshold": float(threshold),
    }


def best_threshold(probs, labels):
    """Pick the threshold that maximises Youden's J (sensitivity + specificity - 1) on validation."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [(m := metrics(probs, labels, t))["sensitivity"] + m["specificity"] for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--arch", choices=ARCHS, default="resnet18")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience on validation AUC")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viz", action="store_true", help="disable feature-map visualisation")
    parser.add_argument("--viz-samples", type=int, default=4, help="fixed validation images rendered each epoch")
    parser.add_argument("--viz-channels", type=int, default=6, help="feature maps shown per conv stage")
    parser.add_argument("--viz-steps", type=int, default=0,
                        help="also render the current training image every N steps (0 = off)")
    parser.add_argument("--viz-live", action="store_true",
                        help="live window showing the image currently going through the network")
    parser.add_argument("--viz-live-every", type=int, default=1,
                        help="update the live window every N training steps (1 = every batch)")
    args = parser.parse_args()

    if not (args.data / "train.csv").exists():
        raise SystemExit(f"{args.data / 'train.csv'} not found. Run `python quality.py` first.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    # only train.csv is read here, because test/holdout are reserved for the final benchmark (evaluate.py)
    train_df, val_df = split_validation(pd.read_csv(args.data / "train.csv"), args.val_fraction, args.seed)
    print(f"train {len(train_df)} | val {len(val_df)}  (test/holdout untouched until evaluate.py)")

    train_tf, eval_tf = build_transforms(args.img_size)
    loader = lambda df, tf, shuffle: DataLoader(  # noqa: E731
        LesionDataset(df, args.data, tf), batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, persistent_workers=args.workers > 0)
    train_dl = loader(train_df, train_tf, True)
    val_dl = loader(val_df, eval_tf, False)

    # roughly 80:20 benign:malignant, so weight the loss by inverse class frequency
    counts = np.bincount(train_dl.dataset.labels, minlength=len(CLASSES))
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(args.arch).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / f"{args.arch}_best.pt"
    history, best_auc, stale = [], -1.0, 0

    viz, viz_dir, viz_samples, viz_frames, global_step = None, args.out / "viz" / args.arch, [], {}, 0
    if not args.no_viz:
        init_backend(args.viz_live)
        viz = FlowVisualizer(model, args.arch, CLASSES, channels=args.viz_channels, live=args.viz_live)
        # an even benign/malignant mix of validation images, fixed for the whole run so epochs are comparable
        per_class = max(1, args.viz_samples // len(CLASSES))
        picks = pd.concat([val_df[val_df["label"] == c].sample(min(per_class, (val_df["label"] == c).sum()),
                                                               random_state=args.seed) for c in CLASSES])
        viz_set = LesionDataset(picks, args.data, eval_tf)
        viz_samples = [(i, *viz_set[i], picks.iloc[i]["image_id"]) for i in range(len(viz_set))]

    def render_samples(tag, title):
        for k, x, y, image_id in viz_samples:
            path = viz_dir / f"sample_{k}" / f"{tag}.png"
            viz.render(x.to(device), y, f"{args.arch} | {title} | {image_id}", path)
            viz_frames.setdefault(k, []).append(path)

    where = {"phase": "", "epoch": 0, "n_epochs": 0}

    def on_step(step, x, y, loss):
        nonlocal global_step
        global_step += 1
        save = bool(args.viz_steps) and global_step % args.viz_steps == 0
        show = args.viz_live and global_step % args.viz_live_every == 0
        if save or show:
            title = (f"{args.arch} | {where['phase']} epoch {where['epoch']}/{where['n_epochs']} | "
                     f"step {step}/{len(train_dl)} | batch loss {loss:.3f}")
            viz.render(x, y, title, viz_dir / "steps" / f"step_{global_step:05d}.png" if save else None)

    if viz is not None:
        render_samples("epoch_00", "before training (ImageNet weights)")

    phases = [("head", args.head_epochs, False, args.lr_head), ("full", args.epochs, True, args.lr)]
    for phase, n_epochs, full, lr in phases:
        if n_epochs <= 0:
            continue
        set_backbone_trainable(model, full)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        stale = 0
        for epoch in range(1, n_epochs + 1):
            start = time.time()
            where.update(phase=phase, epoch=epoch, n_epochs=n_epochs)
            tr_loss, _, _ = run_epoch(model, train_dl, criterion, device, optimizer, f"{phase} {epoch} train",
                                      on_step=on_step if viz is not None else None)
            val_loss, val_probs, val_labels = run_epoch(model, val_dl, criterion, device, desc=f"{phase} {epoch} val")
            scheduler.step()
            m = metrics(val_probs, val_labels)
            history.append({"phase": phase, "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, **m})
            print(f"[{phase} {epoch}/{n_epochs}] train_loss {tr_loss:.4f} val_loss {val_loss:.4f} "
                  f"val_auc {m['auc']:.4f} bal_acc {m['balanced_acc']:.4f} sens {m['sensitivity']:.3f} "
                  f"({time.time() - start:.0f}s)")
            if viz is not None:
                render_samples(f"epoch_{len(history):02d}", f"{phase} epoch {epoch} | val AUC {m['auc']:.3f}")
            if m["auc"] > best_auc:
                best_auc, stale = m["auc"], 0
                torch.save({"arch": args.arch, "img_size": args.img_size, "classes": CLASSES,
                            "state_dict": model.state_dict()}, ckpt_path)
            else:
                stale += 1
                if full and stale >= args.patience:
                    print(f"Early stopping: val AUC has not improved for {args.patience} epochs")
                    break

    # tune the decision threshold on validation and store it in the checkpoint for evaluate.py
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, val_probs, val_labels = run_epoch(model, val_dl, criterion, device, desc="val")
    threshold = best_threshold(val_probs, val_labels)
    ckpt["threshold"] = threshold
    torch.save(ckpt, ckpt_path)
    results = {"best_val_auc": best_auc, "threshold": threshold, "val": metrics(val_probs, val_labels, threshold)}

    with open(args.out / f"{args.arch}_results.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "history": history, "results": results}, f,
                  indent=2)
    r = results["val"]
    print(f"\nValidation (threshold {threshold:.2f}, tuned on validation): AUC {r['auc']:.4f}  "
          f"bal_acc {r['balanced_acc']:.4f}  sens {r['sensitivity']:.3f}  spec {r['specificity']:.3f}")
    print(f"Best checkpoint: {ckpt_path}")
    print("Test/holdout not evaluated. When development is finished, run: "
          f"python evaluate.py {ckpt_path}")

    if viz is not None:
        for k, frames in viz_frames.items():
            make_gif(frames, viz_dir / f"sample_{k}" / "evolution.gif")
        print(f"Feature-map visualisations: {viz_dir}")


if __name__ == "__main__":
    main()
