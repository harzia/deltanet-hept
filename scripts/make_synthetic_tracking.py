"""Generate a TINY synthetic dataset matching the HEPT `Tracking` schema.

Purpose: smoke-test the full tracking pipeline (DeltaNet forward/backward,
InfoNCE loss, accuracy@k metrics) on this CPU dev box WITHOUT loading the real
66 GB `data-6k.pt` into memory. The real dataset is used for actual training on
GPU; for a real small *subset* of the true data see `scripts/make_subset.py`.

Output: <out_root>/tracking/processed/tracking-6k/data-6k.pt  containing
(data, slices, idx_split), which the unmodified `Tracking` loader reads. Point
`data_dir: <out_root>` in the smoke config.

The generator mimics the real structure: events of clustered "hits" (a few
particles + noise), each particle a compact blob in (eta, phi) so the radius
graph links its hits, cluster sizes <= 20 (metric requires k <= 19).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from datasets.tracking import gen_point_pairs  # noqa: E402
from torch_geometric.nn import knn_graph  # noqa: E402
from torch_geometric.utils import to_undirected  # noqa: E402

N_FEATURES = 14  # matches Tracking.feature_names


def make_event(evtid: int, rng: np.random.Generator) -> Data:
    n_particles = int(rng.integers(5, 9))
    etas, phis, pids, pts, recon, layers = [], [], [], [], [], []
    # Keep all clusters inside a compact (eta, phi) region so the radius graph
    # (r=1.0 in gen_point_pairs) links across clusters -> every hit gets both
    # same-cluster (positive) and cross-cluster (negative) pairs, which the
    # HEPT InfoNCE loss implicitly assumes.
    for pid in range(1, n_particles + 1):
        n_hits = int(rng.integers(3, 12))  # cluster size <= ~11 (< 20)
        c_eta = rng.uniform(-0.6, 0.6)
        c_phi = rng.uniform(-0.6, 0.6)
        pt = float(rng.uniform(0.3, 5.0))  # some above the 0.9 threshold
        for _ in range(n_hits):
            etas.append(c_eta + rng.normal(0, 0.05))
            phis.append(c_phi + rng.normal(0, 0.05))
            pids.append(pid)
            pts.append(pt)
            recon.append(1)
            layers.append(int(rng.integers(0, 10)))
    # a few noise hits (particle_id == 0), also in the compact region
    for _ in range(int(rng.integers(3, 10))):
        etas.append(rng.uniform(-0.7, 0.7))
        phis.append(rng.uniform(-0.7, 0.7))
        pids.append(0)
        pts.append(float(rng.uniform(0.1, 0.9)))
        recon.append(0)
        layers.append(int(rng.integers(0, 10)))

    n = len(etas)
    perm = rng.permutation(n)
    pos = torch.tensor(np.stack([etas, phis], axis=1)[perm], dtype=torch.float)
    x = torch.randn(n, N_FEATURES) * 0.1
    data = Data(
        x=x,
        pos=pos,
        particle_id=torch.tensor(np.array(pids)[perm], dtype=torch.long),
        pt=torch.tensor(np.array(pts)[perm], dtype=torch.float),
        reconstructable=torch.tensor(np.array(recon)[perm], dtype=torch.long),
        layer=torch.tensor(np.array(layers)[perm], dtype=torch.long),
        evtid=torch.tensor([evtid], dtype=torch.long),
        s=torch.tensor([0], dtype=torch.long),
    )
    data.point_pairs_index_rad = gen_point_pairs(data, k=256)
    data.knn_edge_index_k60 = to_undirected(knn_graph(data.pos, k=min(60, n - 1), loop=True))
    return data


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(REPO / "data" / "tracking_smoke"))
    ap.add_argument("--n_events", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.n_events < 10:
        ap.error("--n_events must be at least 10 so all tracking splits are nonempty")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data_list = []
    for i in range(args.n_events):
        # keep evtid < 29000 for train, >= 29000 for eval per Tracking split logic,
        # but Tracking recomputes split by evtid rank, so just make them distinct.
        d = make_event(evtid=1000 + i, rng=rng)
        data_list.append(d)

    data, slices = InMemoryDataset.collate(data_list)
    n = len(data_list)
    n_train = max(1, int(n * 0.8))
    idx = np.arange(n)
    idx_split = {
        "train": idx[:n_train].tolist(),
        "valid": idx[n_train: n_train + max(1, (n - n_train) // 2)].tolist(),
        "test": idx[n_train + max(1, (n - n_train) // 2):].tolist(),
    }

    out = Path(args.out_root) / "tracking" / "processed" / "tracking-6k"
    out.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices, idx_split), out / "data-6k.pt")
    # placeholder pre_* files so PyG doesn't complain
    for name in ("pre_filter.pt", "pre_transform.pt"):
        torch.save(None, out / name)
    total_pts = sum(int(d.num_nodes) for d in data_list)
    print(f"wrote {out/'data-6k.pt'}  ({n} events, {total_pts} hits total)")


if __name__ == "__main__":
    main()
