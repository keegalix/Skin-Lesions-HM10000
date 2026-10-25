"""Download the HAM10000 binary skin-lesion dataset from Hugging Face and store it locally.

The parquet files are downloaded with resumable HTTP range requests, a hard read timeout
and automatic retries. Completed files (checked against the size the Hub reports) are
skipped and partial ones resume, so a flaky connection only slows the download down and
re-running the script picks up where it stopped. They are then saved as an Arrow dataset
in data/raw, which quality.py reads.

Usage:
    python fetch_data.py                # downloads to data/raw (skips if it already exists)
    python fetch_data.py --force        # re-download and overwrite
"""

import argparse
import fcntl
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ID = "preetsojitra/skin-lesion-HM10000-binary-classificationt"
ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "raw"
DOWNLOAD_DIR = ROOT / "data" / "download"
CACHE_DIR = ROOT / "data" / "hf_cache"
LOCK_FILE = ROOT / "data" / ".fetch.lock"

# keep every Hugging Face cache inside the repo, and be patient with slow connections.
# These must be set before huggingface_hub / datasets are imported.
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from datasets import load_dataset, load_from_disk  # noqa: E402
from huggingface_hub import HfApi, hf_hub_url  # noqa: E402
from tqdm import tqdm  # noqa: E402

READ_TIMEOUT = 60  # seconds without receiving any data before a connection is treated as dead
CHUNK = 1 << 18
# a connection that has degraded to a trickle is dropped and reopened, because a fresh one is usually far faster
MIN_SPEED_WINDOW = 30  # seconds
MIN_SPEED_BYTES = 1 << 20  # bytes expected per window


def summarise(ds):
    for split, part in ds.items():
        counts = Counter(part["label"])
        print(f"  {split:<6} {part.num_rows:>6} rows  {dict(counts)}")


def download_file(url, dest, size, position):
    """Download url to dest, resuming from a .part file with HTTP range requests.

    Each read has a hard timeout, so a connection that goes quiet raises an error and is retried
    instead of hanging forever. A connection that slows to a trickle is also dropped and retried.
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    with tqdm(total=size, initial=have, unit="B", unit_scale=True, desc=dest.name, position=position,
              leave=True, mininterval=2) as bar:
        if have < size:
            req = Request(url, headers={"Range": f"bytes={have}-", "User-Agent": "fetch_data.py"})
            with urlopen(req, timeout=READ_TIMEOUT) as resp, open(part, "ab") as f:
                if have and resp.status != 206:  # server ignored the range, so start over
                    f.truncate(0)
                    bar.reset()
                window_start, window_bytes = time.monotonic(), 0
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
                    window_bytes += len(chunk)
                    if time.monotonic() - window_start >= MIN_SPEED_WINDOW:
                        if window_bytes < MIN_SPEED_BYTES:
                            raise TimeoutError(f"connection slowed to {window_bytes / MIN_SPEED_WINDOW / 1e3:.0f} kB/s")
                        window_start, window_bytes = time.monotonic(), 0
    if part.stat().st_size != size:
        raise IOError(f"{dest.name}: got {part.stat().st_size} of {size} bytes")
    part.rename(dest)


def download_parquet(retries, workers):
    """Download the repo's parquet files to DOWNLOAD_DIR. Complete files are skipped, partial ones resume."""
    for attempt in range(1, retries + 1):
        try:
            info = HfApi().dataset_info(REPO_ID, files_metadata=True)
            break
        except Exception as e:  # noqa: BLE001 - offline / flaky network: wait and try again
            if attempt == retries:
                raise
            print(f"Could not reach the Hugging Face Hub ({type(e).__name__}), retrying in 15s ...")
            time.sleep(15)
    files =[s for s in info.siblings if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet")]
    todo = [s for s in files
            if not ((p := DOWNLOAD_DIR / s.rfilename).exists() and p.stat().st_size == s.size)]
    done = len(files) - len(todo)
    remaining = sum(s.size for s in todo) - sum(
        (DOWNLOAD_DIR / (s.rfilename + ".part")).stat().st_size
        for s in todo if (DOWNLOAD_DIR / (s.rfilename + ".part")).exists())
    print(f"{done}/{len(files)} files already complete, {remaining / 1e6:,.0f} MB to download")

    for attempt in range(1, retries + 1):
        pending = [s for s in todo if not (DOWNLOAD_DIR / s.rfilename).exists()]
        if not pending:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_file, hf_hub_url(REPO_ID, s.rfilename, repo_type="dataset",
                                                             revision=info.sha),
                                   DOWNLOAD_DIR / s.rfilename, s.size, i): s
                       for i, s in enumerate(pending)}
            errors = [(futures[f].rfilename, f.exception()) for f in as_completed(futures) if f.exception()]
        if not errors:
            return
        if attempt == retries:
            raise RuntimeError(f"giving up after {retries} attempts: {errors}")
        wait = min(30, 5 * attempt)
        for name, e in errors:
            print(f"\nDownload interrupted: {name} ({type(e).__name__}: {e})")
        print(f"Retry {attempt}/{retries - 1} in {wait}s - progress is kept, downloads resume where they stopped.")
        time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to save the dataset (Arrow format)")
    parser.add_argument("--force", action="store_true", help="re-download even if --out already exists")
    parser.add_argument("--retries", type=int, default=200, help="download attempts before giving up")
    parser.add_argument("--workers", type=int, default=2, help="files downloaded in parallel")
    parser.add_argument("--keep-cache", action="store_true", help="keep the downloaded parquet files (≈3 GB)")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"Dataset already present at {args.out} (use --force to re-download)")
        summarise(load_from_disk(str(args.out)))
        return
    # two copies appending to the same .part files would corrupt them, so only one may run at a time
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another fetch_data.py is already downloading. Wait for it or stop it first.")

    if args.force:
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)

    print(f"Downloading {REPO_ID} ...")
    download_parquet(args.retries, args.workers)

    parquet = DOWNLOAD_DIR / "data"
    ds = load_dataset("parquet", cache_dir=str(CACHE_DIR), data_files={
        "train": str(parquet / "train-*.parquet"),
        "test": str(parquet / "test-*.parquet"),
    })

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"Saved to {args.out}")
    summarise(ds)

    if not args.keep_cache:
        # load_from_disk only needs args.out, so the parquet download and Arrow cache are redundant
        del ds
        shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
