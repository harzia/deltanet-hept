"""Convert a monolithic HEPT `InMemoryDataset` into per-event shards for lazy,
low-RAM loading. Behaviour of the resulting dataset is identical (see
src/datasets/sharded.py and tests/test_sharded_equivalence.py).

SAFETY: this only READS the original processed `.pt` and WRITES new shard files
to a SEPARATE output root. It never deletes or modifies the source dataset.

MEMORY: the one-time conversion loads the full source file into RAM once
(tracking ~60-66 GB, pileup ~11 GB). Run it on a box with enough free RAM; the
per-event shards it writes are tiny and load lazily afterwards. Each event is
`.clone()`d before saving so a shard contains only that event's bytes (not a
view into the giant storage).

Example:
    python scripts/shard_dataset.py --dataset tracking-6k \
        --src_data_dir data --out_root data_sharded
    # then run with:  data_dir: data_sharded   use_sharded: true
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from datasets import Tracking, Pileup  # noqa: E402
from datasets.sharded import shard_dir_for  # noqa: E402


def load_source(dataset, src_data_dir):
    """Load the ORIGINAL in-memory dataset (transform=None so get() is raw)."""
    src = Path(src_data_dir)
    if dataset.startswith("tracking"):
        return Tracking(src / "tracking", dataset_name=dataset, transform=None)
    return Pileup(src / "pileup", transform=None)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tracking-6k", "tracking-60k", "pileup"])
    ap.add_argument("--src_data_dir", default=str(REPO / "data"))
    ap.add_argument("--out_root", default=str(REPO / "data_sharded"))
    ap.add_argument("--limit", type=int, default=None, help="only shard the first N events (debug)")
    args = ap.parse_args()

    print(f"[shard] loading source {args.dataset} (this loads the full file into RAM once)...")
    ds = load_source(args.dataset, args.src_data_dir)
    n_total = len(ds)
    n = min(args.limit, n_total) if args.limit else n_total
    print(f"[shard] {n_total} events; sharding {n}")

    task = "tracking" if args.dataset.startswith("tracking") else "pileup"
    out_root = Path(args.out_root) / task
    shard_dir = shard_dir_for(out_root, args.dataset)
    shard_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        ev = ds.get(i).clone()  # pre-transform, compacted copy (no giant-storage view)
        torch.save(ev, shard_dir / f"evt_{i:06d}.pt")
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"[shard]   {i + 1}/{n}")

    # meta reproduces exactly what the in-memory dataset exposed.
    meta = {
        "num_events": n,
        "idx_split": ds.idx_split if not args.limit else _clip_split(ds.idx_split, n),
        "x_dim": int(ds.x_dim),
        "coords_dim": int(ds.coords_dim),
        "dataset_name": args.dataset,
    }
    torch.save(meta, shard_dir / "meta.pt")
    print(f"[shard] wrote {n} shards + meta.pt to {shard_dir}")
    print(f"[shard] source dataset left untouched at {args.src_data_dir}")


def _clip_split(idx_split, n):
    out = {}
    for k, v in idx_split.items():
        v = v.tolist() if torch.is_tensor(v) else list(v)
        out[k] = [i for i in v if i < n]
    return out


if __name__ == "__main__":
    main()
