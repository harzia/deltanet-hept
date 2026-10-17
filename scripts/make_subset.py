"""Create a small REAL subset of a processed HEPT dataset for fast experiments.

WARNING (memory): PyG's InMemoryDataset stores every event in one collated
tensor, so producing a subset requires loading the *entire* source file into RAM
once (tracking-6k/-60k are ~66/60 GB; pileup ~11 GB). Only run this on a machine
with enough free RAM. It is a one-time operation; afterwards the small subset
loads cheaply for iteration. This is why smoke tests default to synthetic
tracking data (see make_synthetic_tracking.py) instead.

Example:
    python scripts/make_subset.py --dataset tracking-6k --n_events 60 \
        --out_root data_subset
    # then set  data_dir: data_subset  in a config
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from torch_geometric.data import InMemoryDataset  # noqa: E402
from datasets import Tracking, Pileup  # noqa: E402


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tracking-6k", "tracking-60k", "pileup"])
    ap.add_argument("--src_data_dir", default=str(REPO / "data"))
    ap.add_argument("--out_root", default=str(REPO / "data_subset"))
    ap.add_argument("--n_events", type=int, default=60)
    ap.add_argument("--confirm", action="store_true",
                    help="required acknowledgement that the full source file will be loaded into RAM")
    args = ap.parse_args()

    if not args.confirm:
        print("Refusing to run without --confirm (this loads the FULL source file into RAM). "
              "Re-run with --confirm once you have verified free memory.")
        sys.exit(2)

    src = Path(args.src_data_dir)
    if args.dataset.startswith("tracking"):
        ds = Tracking(src / "tracking", dataset_name=args.dataset)
        out = Path(args.out_root) / "tracking" / "processed" / args.dataset
        fname = f"data-{args.dataset.split('-')[-1]}.pt"
    else:
        ds = Pileup(src / "pileup")
        out = Path(args.out_root) / "pileup" / "processed"
        fname = "data.pt"

    n = min(args.n_events, len(ds))
    print(f"loaded {args.dataset}: {len(ds)} events -> taking first {n}")
    sub = [ds.get(i) for i in range(n)]
    data, slices = InMemoryDataset.collate(sub)
    idx = list(range(n))
    n_tr = max(1, int(n * 0.8))
    n_va = max(1, (n - n_tr) // 2)
    idx_split = {"train": idx[:n_tr], "valid": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    out.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices, idx_split), out / fname)
    for name in ("pre_filter.pt", "pre_transform.pt"):
        torch.save(None, out / name)
    print(f"wrote {out/fname}")


if __name__ == "__main__":
    main()
