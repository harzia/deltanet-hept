"""Generate small deterministic pileup fixtures; no downloads or ROOT needed."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(Path(__file__).resolve().parents[1] / "data" / "pileup_smoke"))
    ap.add_argument("--n_events", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.n_events < 6:
        ap.error("--n_events must be at least 6")
    g = torch.Generator().manual_seed(args.seed)
    events = []
    for i in range(args.n_events):
        n = 47 + i % 9  # nonmultiples of the attention block size
        pos = torch.randn(n, 2, generator=g)
        x = torch.randn(n, 8, generator=g)
        x[:, -1] = torch.randint(0, 7, (n,), generator=g).float()
        y = (torch.arange(n) % 4 == 0).long()
        x[:, 2] += y * 0.5
        events.append(Data(x=x, pos=pos, y=y, is_neu=torch.ones(n, dtype=torch.bool),
                           edge_index=torch.empty(2, 0, dtype=torch.long)))
    data, slices = InMemoryDataset.collate(events)
    n_train = int(args.n_events * 0.6)
    n_valid = (args.n_events - n_train) // 2
    ids = np.arange(args.n_events)
    split = dict(train=ids[:n_train], valid=ids[n_train:n_train+n_valid], test=ids[n_train+n_valid:])
    out = Path(args.out_root) / "pileup" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices, split), out / "data.pt")
    for name in ("pre_filter.pt", "pre_transform.pt"):
        torch.save(None, out / name)
    print(f"Wrote {args.n_events} synthetic pileup events to {out}")


if __name__ == "__main__":
    main()
