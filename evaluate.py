"""Final benchmark: score a trained checkpoint on the test and holdout sets.

Run this once, after every training and tuning decision (architecture, epochs, learning
rate, augmentation, threshold) has been made using the validation set only. If test results
feed back into further tuning, the test set stops being an unbiased estimate.

The decision threshold is the one train.py tuned on validation and stored in the checkpoint.
Nothing here is fitted to the test data.

Every run is appended to models/final_benchmark.json. The script refuses to score a split
that has already been benchmarked unless --rerun is passed, so repeated peeking is deliberate
and leaves a record.

Usage:
    python evaluate.py models/resnet18_best.pt
    python evaluate.py models/resnet18_best.pt --splits test      # only the 75/25 test split
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from train import CLASSES, LesionDataset, build_model, build_transforms, metrics, pick_device, run_epoch

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "models" / "final_benchmark.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--splits", nargs="+", choices=["test", "holdout"], default=["test", "holdout"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rerun", action="store_true", help="allow scoring a split that was already benchmarked")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "threshold" not in ckpt:
        raise SystemExit("Checkpoint has no validation-tuned threshold. Re-train it with the current train.py.")
    ckpt_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:16]

    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    already = {e["split"] for e in log}
    repeats = [s for s in args.splits if s in already]
    if repeats and not args.rerun:
        raise SystemExit(f"Already benchmarked: {', '.join(repeats)} (see {LOG_PATH}). The final benchmark is meant "
                         "to run once. Pass --rerun if you really mean to score it again.")

    device = pick_device()
    model = build_model(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    _, eval_tf = build_transforms(ckpt["img_size"])
    criterion = nn.CrossEntropyLoss()

    print(f"Final benchmark: {ckpt['arch']} ({args.checkpoint.name}), threshold {ckpt['threshold']:.2f} "
          "from validation")
    for split in args.splits:
        df = pd.read_csv(args.data / f"{split}.csv")
        loader = DataLoader(LesionDataset(df, args.data, eval_tf), batch_size=args.batch_size,
                            num_workers=args.workers)
        _, probs, labels = run_epoch(model, loader, criterion, device, desc=split)
        m = metrics(probs, labels, ckpt["threshold"])
        print(f"  {split:<8} n={len(df):<5} AUC {m['auc']:.4f}  bal_acc {m['balanced_acc']:.4f}  "
              f"sens {m['sensitivity']:.3f}  spec {m['specificity']:.3f}")
        log.append({"time": datetime.now().isoformat(timespec="seconds"), "split": split,
                    "checkpoint": str(args.checkpoint), "checkpoint_sha256": ckpt_hash, "arch": ckpt["arch"],
                    "n": len(df), "classes": CLASSES, **m})

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"Recorded in {LOG_PATH}")


if __name__ == "__main__":
    main()
