# DeltaNet on the HEPT backbone

An **apples-to-apples** study that drops the [DeltaNet](https://arxiv.org/abs/2406.06484)
linear-attention layer into the [HEPT](https://arxiv.org/abs/2402.12535)
point-transformer backbone for high-energy-physics point clouds, keeping
*everything else identical* (feature encoder, residual blocks, feed-forward,
layer concatenation, task heads, losses, and preprocessing). The only thing that
changes between HEPT and DeltaNet is the **token-mixing mechanism**.

The motivating research question: linear transformers are **not
permutation-equivariant** — they are sequence models — but a point cloud has no
intrinsic order. How do we inject *locality* so a linear transformer can compete
with HEPT's LSH-based local attention? We start with the simplest lever —
**sorting the points** by a geometric/kinematic key — and build toward
space-filling curves and local grouping (see [`docs/LOCALITY.md`](docs/LOCALITY.md)).

---

## What changed vs. the original HEPT code

This repo vendors the HEPT backbone (`src/`) essentially verbatim and adds:

| File | Purpose |
|------|---------|
| `src/models/attention/deltanet.py` | **New.** DeltaNet token-mixer (pure-PyTorch chunkwise + recurrent), point-ordering, optional `fla` GPU kernel. |
| `src/models/baselines/transformer.py` | 3 small edits: register `deltanet` in `Attn`, keep the flat point layout for it in `prepare_input`, thread `pt` through for pt/kt sort keys. |
| `src/models/attention/__init__.py` | Optional imports so DeltaNet+HEPT work without the other baselines' heavy deps. |
| `src/run.py` | **New.** Unified train/test runner (both tasks), structured logging, no `nni`. |
| `src/utils/run_logger.py` | **New.** Human `run.log` + machine `metrics.jsonl` + `config.yaml` dump. |
| `configs/` | DeltaNet + HEPT configs for tracking-6k/60k and pileup, plus tiny smoke configs. |
| `scripts/`, `tests/` | Smoke test, sort-key sweep, subset makers, correctness tests. |

The DeltaNet layer has the **same I/O contract** as `HEPTAttention`
(`forward(q, k, v, **kwargs) -> [N, h_dim]`), so the swap is genuinely local.

---

## Tasks & data

Two HEPT benchmarks, same backbone:

* **tracking** — InfoNCE metric learning on tracker hits; metric `accuracy@0.9`.
* **pileup** — per-node binary classification; metric `auc` (average precision).

Datasets are the preprocessed releases from the HEPT paper
([Zenodo record 10694703](https://zenodo.org/records/10694703)). Download them
fresh:

```bash
python scripts/download_data.py --dataset pileup        # ~1.8 GB zip -> 11 GB
python scripts/download_data.py --dataset tracking-6k   # ~2.5 GB zip -> 66 GB
python scripts/download_data.py --dataset tracking-60k  # ~0.4 GB zip -> 60 GB
```

> **Memory note.** PyG's `InMemoryDataset` loads the *entire* processed `.pt`
> into RAM. The tracking files are ~60–66 GB. Do real training on a box with
> enough RAM/GPU. For quick local iteration use `scripts/make_subset.py`
> (a one-time full load to carve a small subset) or the synthetic smoke data.

### Low-RAM streaming (sharded dataset)

To avoid the full in-RAM load entirely, re-shard the monolithic file into one
small file per event and load lazily (RAM then scales with the DataLoader batch,
not the dataset). Behaviour is **identical** — same splits, dims, and per-event
tensors (verified by `tests/test_sharded_equivalence.py`, and training
reproduces the same numbers bit-for-bit).

```bash
# one-time conversion (loads the source once; source is NOT modified/deleted):
python scripts/shard_dataset.py --dataset tracking-6k --src_data_dir data --out_root data_sharded
# then train against the shards:
cd src && python run.py --config ../configs/tracking/deltanet_tracking-6k.yaml \
                        -o data_dir=../data_sharded -o use_sharded=true
```

Set `use_sharded: true` + `data_dir: data_sharded` in a config to make it the
default. With shards you can also safely use `num_workers>0` in the DataLoader.

Preprocessing is **unchanged** from HEPT (same transforms, same features).

---

## Install

The dev box uses a venv layered on the base conda env (torch 1.13 + pyg 2.3):

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install einops "torchmetrics==0.11.4" fvcore
```

For GPU training with the fast `fla` Triton kernels, additionally
`pip install flash-linear-attention` and set `deltanet_impl: auto` (or `fla`).

---

## Quick start

```bash
# 1. Verify the DeltaNet math + run tiny end-to-end smoke tests (CPU-safe):
bash scripts/smoke_test.sh

# 2. Train on a real benchmark (GPU box):
cd src
python run.py --config ../configs/tracking/deltanet_tracking-6k.yaml
python run.py --config ../configs/pileup/deltanet_pileup.yaml

# 3. Evaluate a checkpoint:
python run.py --config ../configs/tracking/deltanet_tracking-6k.yaml \
              --mode test --ckpt ../logs/<run>/best_model.pt

# 4. HEPT baseline (identical backbone) for comparison:
python run.py --config ../configs/tracking/hept_tracking-6k.yaml

# 5. Sweep the locality order (sort key x direction):
bash scripts/sweep_sortings.sh configs/tracking/deltanet_tracking-6k.yaml
```

Any config value can be overridden on the CLI:
`-o model_kwargs.deltanet_sort_key=phi -o num_epochs=50 -o max_events=100`.

---

## DeltaNet knobs (`model_kwargs`)

| key | values | meaning |
|-----|--------|---------|
| `deltanet_sort_key` | `eta` `phi` `r` `pt` `kt` `original` | the 1-D order the delta rule scans (locality lever) |
| `deltanet_direction` | `bi` `uni` | bidirectional (forward+backward, non-causal) or single causal scan |
| `chunk_size` | int | chunkwise kernel block; also an implicit local window |
| `deltanet_qk_norm` | `l2` `none` | L2-normalize q,k (stability of the delta update) |
| `deltanet_qk_activation` | `silu` `relu` `identity` | q,k feature map |
| `deltanet_use_beta` | bool | learn per-point write strength `beta` (else 1) |
| `deltanet_allow_neg_eigval` | bool | scale `beta` by 2 (allows negative eigenvalues) |
| `deltanet_impl` | `auto` `torch` `fla` | kernel backend; `auto` uses `fla` on CUDA if installed |

> On the tracking task, `pt`/`kt` sort keys feed a weak pt signal into ordering,
> which HEPT does not use — a slightly different setup, flagged in the logs.
> Default is `eta` (pure geometry).

---

## Logs

Every run writes `logs/<stamp>_<model>_<dataset>_<seed>_<note>/`:

* `run.log` — human-readable, timestamped.
* `metrics.jsonl` — one JSON record per event (`kind ∈ {meta,epoch,final}`); load with `pandas.read_json(path, lines=True)`.
* `config.yaml` — exact resolved config.
* `best_model.pt` — best checkpoint by the main metric.

See [`AGENTS.md`](AGENTS.md) for the repo map and conventions (for coding agents),
and [`DEVLOG.md`](DEVLOG.md) for the running development log.

---

## References

* Miao et al., *LSH-Based Efficient Point Transformer (HEPT)*, ICML 2024. https://arxiv.org/abs/2402.12535
* Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length (DeltaNet)*, 2024. https://arxiv.org/abs/2406.06484
* `flash-linear-attention`: https://github.com/fla-org/flash-linear-attention
