"""Prove the sharded (lazy, low-RAM) dataset behaves IDENTICALLY to the
monolithic in-memory dataset: same splits, same dims, same per-event tensors.

Runs on the tiny synthetic tracking dataset (no big load). Creates shards in a
scratch dir; never touches the source.

Run:  .venv/bin/python tests/test_sharded_equivalence.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from utils.get_data import get_dataset  # noqa: E402


def _same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    return a == b


def main():
    synth = REPO / "data" / "tracking_smoke" / "tracking" / "processed" / "tracking-6k" / "data-6k.pt"
    if not synth.exists():
        print("generating synthetic tracking data first...")
        subprocess.run([sys.executable, str(REPO / "scripts" / "make_synthetic_tracking.py")], check=True)

    with tempfile.TemporaryDirectory(prefix="shard_eq_") as tmp:
        compare_shards(Path(tmp))


def compare_shards(tmp):
    print(f"sharding into {tmp}")
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "shard_dataset.py"),
         "--dataset", "tracking-6k",
         "--src_data_dir", str(REPO / "data" / "tracking_smoke"),
         "--out_root", str(tmp)],
        check=True,
    )

    inmem = get_dataset("tracking-6k", str(REPO / "data" / "tracking_smoke" / "tracking"), use_sharded=False)
    shard = get_dataset("tracking-6k", str(tmp / "tracking"), use_sharded=True)

    assert len(inmem) == len(shard), (len(inmem), len(shard))
    assert inmem.x_dim == shard.x_dim, (inmem.x_dim, shard.x_dim)
    assert inmem.coords_dim == shard.coords_dim
    for k in ("train", "valid", "test"):
        a = inmem.idx_split[k]; b = shard.idx_split[k]
        a = a.tolist() if torch.is_tensor(a) else list(a)
        b = b.tolist() if torch.is_tensor(b) else list(b)
        assert a == b, f"split {k} differs: {a} vs {b}"
    print(f"PASS: len={len(inmem)}, x_dim={inmem.x_dim}, coords_dim={inmem.coords_dim}, splits match")

    mismatches = 0
    for i in range(len(inmem)):
        d1, d2 = inmem[i], shard[i]
        keys1, keys2 = set(d1.keys() if callable(d1.keys) else d1.keys), set(d2.keys() if callable(d2.keys) else d2.keys)
        assert keys1 == keys2, f"event {i} key mismatch: {keys1 ^ keys2}"
        for key in keys1:
            if not _same(d1[key], d2[key]):
                print(f"  MISMATCH event {i} key {key}")
                mismatches += 1
    assert mismatches == 0, f"{mismatches} tensor mismatches"
    print(f"PASS: all {len(inmem)} events identical across every key")
    print("\nSHARDED == IN-MEMORY")


if __name__ == "__main__":
    main()
