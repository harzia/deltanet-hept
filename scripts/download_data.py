"""Download + extract the preprocessed HEPT datasets from Zenodo.

Non-interactive replacement for the datasets' built-in `download()` prompts.
Places files exactly where the `Tracking`/`Pileup` loaders expect them:

    data/tracking/processed/tracking-6k/data-6k.pt
    data/tracking/processed/tracking-60k/data-60k.pt
    data/pileup/processed/data.pt

Zips are small (~0.4-2.5 GB); extracted tracking files are ~60-66 GB (disk only,
not RAM). See the memory note in the README before loading them for training.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SPECS = {
    "pileup": dict(
        url="https://zenodo.org/records/10694703/files/pileup-10k-processed.zip",
        extract_to="data/pileup/processed",
    ),
    "tracking-6k": dict(
        url="https://zenodo.org/records/10694703/files/tracking-6k-processed.zip",
        extract_to="data/tracking/processed",
    ),
    "tracking-60k": dict(
        url="https://zenodo.org/records/10694703/files/tracking-60k-processed.zip",
        extract_to="data/tracking/processed",
    ),
}


def _progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        sys.stdout.write(f"\r  downloading... {pct:3d}%")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPECS))
    ap.add_argument("--data_root", default=str(REPO))
    ap.add_argument("--keep_zip", action="store_true")
    args = ap.parse_args()

    spec = SPECS[args.dataset]
    root = Path(args.data_root)
    zips = root / "data" / "_zips"
    zips.mkdir(parents=True, exist_ok=True)
    zip_path = zips / f"{args.dataset}-processed.zip"
    extract_to = root / spec["extract_to"]
    extract_to.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"[{args.dataset}] fetching {spec['url']}")
        urllib.request.urlretrieve(spec["url"], zip_path, _progress)
        print()
    else:
        print(f"[{args.dataset}] zip already present: {zip_path}")

    print(f"[{args.dataset}] extracting to {extract_to} (disk only)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    if not args.keep_zip:
        zip_path.unlink()
    print(f"[{args.dataset}] done.")


if __name__ == "__main__":
    main()
