"""Clean the raw HAM10000 dataset and split it into train/test sets without leakage.

HAM10000 contains several images of the same lesion, so a random split puts
near-identical images on both sides. Here the split is grouped by lesion
(plus any perceptual near-duplicates across lesions) and stratified on label.

Pipeline (counts for each step go to quality_report.json):
  1. integrity           drop undecodable images / missing label or lesion_id
  2. label consistency   drop rows where label disagrees with dx, and lesions with mixed labels
  3. exact duplicates    drop byte-identical images (MD5)
  4. near-duplicates     merge lesion groups whose images match by perceptual hash
  5. metadata cleanup    normalise "unknown" values to NaN and add *_missing flags
  6. grouped split       StratifiedGroupKFold -> train / test
  7. leakage checks      assert no lesion, group or image hash is shared across splits

The HF "test" split is kept aside as a holdout. It goes through steps 1-3, and any
image that duplicates train/test is removed from it.

Usage:
    python quality.py [--raw data/raw] [--out data/processed] [--test-size 0.25] [--seed 42]
"""

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from datasets import Image as HFImage
from datasets import load_from_disk
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
MALIGNANT_DX = {"mel", "bcc", "akiec"}
BENIGN_DX = {"nv", "bkl", "df", "vasc"}
PHASH_MAX_DISTANCE = 4
FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png"}


# ---------------------------------------------------------------- loading


def load_split(part, name):
    """Return the metadata DataFrame for one split, plus a hash and decode status per image."""
    df = part.remove_columns("image").to_pandas()
    df["source_split"] = name
    df["source_idx"] = np.arange(len(df))

    raw = part.cast_column("image", HFImage(decode=False))
    md5s, phashes, formats, ok = [], [], [], []
    for row in tqdm(raw, total=len(raw), desc=f"hashing {name}"):
        data = (row["image"] or {}).get("bytes")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            md5s.append(hashlib.md5(data).hexdigest())
            phashes.append(int(str(imagehash.phash(img)), 16))
            formats.append(img.format)
            ok.append(True)
        except Exception:
            md5s.append(None)
            phashes.append(0)
            formats.append(None)
            ok.append(False)

    df["md5"] = md5s
    df["phash"] = np.array(phashes, dtype=np.uint64)
    df["format"] = formats
    df["decode_ok"] = ok
    return df


# ---------------------------------------------------------------- cleaning steps


def drop(df, mask, step, report, split):
    report.setdefault(split, {})[step] = int(mask.sum())
    print(f"  [{split}] {step}: dropped {int(mask.sum())}")
    return df[~mask].copy()


def step_integrity(df, report, split):
    bad = ~df["decode_ok"]
    for col in ("label", "lesion_id"):
        bad |= df[col].isna() | (df[col].astype(str).str.strip() == "")
    return drop(df, bad, "1_integrity", report, split)


def step_label_consistency(df, report, split):
    expected = df["dx"].map(lambda d: "malignant" if d in MALIGNANT_DX else "benign" if d in BENIGN_DX else None)
    bad = expected.isna() | (expected != df["label"])
    df = drop(df, bad, "2a_label_vs_dx", report, split)
    mixed = df.groupby("lesion_id")["label"].transform("nunique") > 1
    return drop(df, mixed, "2b_mixed_label_lesions", report, split)


def step_exact_duplicates(df, report, split):
    # a byte-identical image with conflicting labels is unreliable, so drop every copy of it
    conflicting = df.groupby("md5")["label"].transform("nunique") > 1
    df = drop(df, conflicting, "3a_duplicates_conflicting_label", report, split)
    return drop(df, df.duplicated("md5", keep="first"), "3b_exact_duplicates", report, split)


def hamming_matrix_pairs(a, b, max_dist, same=False):
    """Yield (i, j) index pairs where the phash distance between a[i] and b[j] is <= max_dist."""
    for i in range(len(a)):
        start = i + 1 if same else 0
        if start >= len(b):
            continue
        dist = np.bitwise_count(a[i] ^ b[start:])
        for j in np.nonzero(dist <= max_dist)[0]:
            yield i, start + j


def step_near_duplicate_groups(df, report, split):
    """Union lesion_ids whose images are perceptual near-duplicates. Returns a group id per row."""
    parent = {lid: lid for lid in df["lesion_id"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lesions = df["lesion_id"].to_numpy()
    hashes = df["phash"].to_numpy()
    merges = 0
    for i, j in hamming_matrix_pairs(hashes, hashes, PHASH_MAX_DISTANCE, same=True):
        ri, rj = find(lesions[i]), find(lesions[j])
        if ri != rj:
            parent[rj] = ri
            merges += 1

    df["group_id"] = [find(lid) for lid in lesions]
    report.setdefault(split, {})["4_near_duplicate_lesion_merges"] = merges
    print(f"  [{split}] 4_near_duplicates: merged {merges} lesion pairs into shared groups")
    return df


def step_metadata(df):
    for col in ("sex", "localization"):
        df[col] = df[col].replace({"unknown": np.nan, "": np.nan})
    for col in ("age", "sex", "localization"):
        df[f"{col}_missing"] = df[col].isna()
    return df


# ---------------------------------------------------------------- split + checks


def grouped_split(df, test_size, seed):
    n_splits = max(2, round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["group_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def assert_no_leakage(train, test):
    for col in ("lesion_id", "group_id", "md5"):
        shared = set(train[col]) & set(test[col])
        assert not shared, f"leakage: {len(shared)} {col} values shared between train and test"
    pairs = list(hamming_matrix_pairs(train["phash"].to_numpy(), test["phash"].to_numpy(), PHASH_MAX_DISTANCE))
    assert not pairs, f"leakage: {len(pairs)} near-duplicate image pairs between train and test"
    print("  leakage checks passed: no shared lesion_id / group_id / md5 / near-duplicate")


def remove_holdout_overlap(holdout, seen, report):
    exact = holdout["md5"].isin(set(seen["md5"]))
    near = np.zeros(len(holdout), dtype=bool)
    for i, _ in hamming_matrix_pairs(holdout["phash"].to_numpy(), seen["phash"].to_numpy(), PHASH_MAX_DISTANCE):
        near[i] = True
    return drop(holdout, exact | near, "5_overlap_with_train_test", report, "holdout")


# ---------------------------------------------------------------- output


def write_images(raw_part, rows, out_dir, split_name):
    """Write one image file per row to out_dir/<split>/<label>/ and set its filepath column."""
    rows = rows.sort_values("source_idx")
    rows["image_id"] = rows["lesion_id"] + "_" + rows["source_idx"].astype(str)
    paths = []
    raw = raw_part.cast_column("image", HFImage(decode=False)).select(rows["source_idx"].tolist())
    for (_, row), item in tqdm(zip(rows.iterrows(), raw), total=len(rows), desc=f"writing {split_name}"):
        ext = FORMAT_EXT.get(row["format"], ".jpg")
        rel = Path(split_name) / row["label"] / f"{row['image_id']}{ext}"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / rel).write_bytes(item["image"]["bytes"])
        paths.append(rel.as_posix())
    rows["filepath"] = paths
    return rows


def save_csv(rows, out_dir, name):
    cols = ["filepath", "image_id", "lesion_id", "group_id", "label", "dx", "dx_type", "age", "sex",
            "localization", "dataset", "age_missing", "sex_missing", "localization_missing", "md5", "source_split",
            "source_idx"]
    rows[[c for c in cols if c in rows]].to_csv(out_dir / f"{name}.csv", index=False)


def balance(df):
    counts = df["label"].value_counts()
    return {"images": int(len(df)), "lesions": int(df["lesion_id"].nunique()),
            **{k: int(v) for k, v in counts.items()},
            "malignant_ratio": round(float(counts.get("malignant", 0) / max(len(df), 1)), 4)}


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"{args.raw} not found. Run `python fetch_data.py` first.")
    ds = load_from_disk(str(args.raw))
    report = {"config": {"test_size": args.test_size, "seed": args.seed, "phash_max_distance": PHASH_MAX_DISTANCE}}

    # main pool (HF train split) -> train/test
    print("Cleaning HF train split")
    pool = load_split(ds["train"], "train")
    report["input_rows"] = {"train": len(pool)}
    pool = step_integrity(pool, report, "pool")
    pool = step_label_consistency(pool, report, "pool")
    pool = step_exact_duplicates(pool, report, "pool")
    pool = step_near_duplicate_groups(pool, report, "pool")
    pool = step_metadata(pool)

    train, test = grouped_split(pool, args.test_size, args.seed)
    assert_no_leakage(train, test)

    # holdout (HF test split) -> cleaned, deduplicated against train/test
    holdout = None
    if "test" in ds:
        print("Cleaning HF test split (holdout)")
        holdout = load_split(ds["test"], "test")
        report["input_rows"]["test"] = len(holdout)
        holdout = step_integrity(holdout, report, "holdout")
        holdout = step_label_consistency(holdout, report, "holdout")
        holdout = step_exact_duplicates(holdout, report, "holdout")
        holdout = remove_holdout_overlap(holdout, pool, report)
        holdout["group_id"] = holdout["lesion_id"]
        holdout = step_metadata(holdout)

    # write
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    outputs = {"train": (ds["train"], train), "test": (ds["train"], test)}
    if holdout is not None:
        outputs["holdout"] = (ds["test"], holdout)
    report["splits"] = {}
    for name, (raw_part, rows) in outputs.items():
        rows = write_images(raw_part, rows, args.out, name)
        save_csv(rows, args.out, name)
        report["splits"][name] = balance(rows)

    with open(args.out / "quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSplit summary")
    for name, stats in report["splits"].items():
        print(f"  {name:<8} {stats}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
