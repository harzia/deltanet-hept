"""Lazy, on-disk sharded datasets — a drop-in replacement for the monolithic
`InMemoryDataset` loaders that avoids loading the full 60+ GB file into RAM.

Design invariant: **behaviour is identical** to the in-memory dataset. Each
shard `evt_{i}.pt` stores exactly the *pre-transform* `Data` object that
`InMemoryDataset.get(i)` returns; this class returns it from `get(i)` and the
PyG base `Dataset.__getitem__` applies the SAME transform. `idx_split`,
`x_dim`, `coords_dim`, `dataset_name`, and event order all come from a `meta.pt`
written at shard time from the original in-memory dataset, so splits and per-item
tensors match bit-for-bit (verified in tests/test_sharded_equivalence.py).

The only difference is *where the bytes live*: `get(i)` reads one small file
from disk instead of slicing a fully-resident tensor. RAM then scales with the
DataLoader batch, not the dataset. A normal PyG `DataLoader` works unchanged
(the base class handles int/list/slice indexing and subsetting).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Dataset


def shard_dir_for(root: str, dataset_name: str) -> Path:
    """Canonical location of the shards for a (root, dataset_name)."""
    return Path(root) / "processed" / dataset_name / "shards"


class ShardedPointDataset(Dataset):
    """PyG Dataset backed by one file per event. Same external API as the
    original `Tracking`/`Pileup` (len, get, indexing, transform, idx_split,
    x_dim, coords_dim, dataset_name)."""

    def __init__(self, root, dataset_name, transform=None):
        self._shard_dir = shard_dir_for(root, dataset_name)
        meta_path = self._shard_dir / "meta.pt"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No shards found at {self._shard_dir}. Create them with "
                f"`python scripts/shard_dataset.py --dataset {dataset_name} ...`"
            )
        meta = torch.load(meta_path, weights_only=False)
        self.idx_split = meta["idx_split"]
        self.x_dim = meta["x_dim"]
        self.coords_dim = meta["coords_dim"]
        self.dataset_name = meta["dataset_name"]
        self._num_events = int(meta["num_events"])
        # NOTE: we deliberately do NOT define download()/process(), so PyG's
        # has_download/has_process are False and __init__ performs no I/O beyond
        # setting attributes.
        super().__init__(str(root), transform=transform)

    # abstract members required by the base class
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["shards/meta.pt"]

    def len(self):
        return self._num_events

    def get(self, idx):
        # returns the pre-transform Data; base __getitem__ applies self.transform
        return torch.load(self._shard_dir / f"evt_{int(idx):06d}.pt", weights_only=False)
