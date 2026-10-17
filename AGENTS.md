# Clarified target: Gaussian-kernel DeltaNet

The requested model is `trans_deltanet` with `deltanet_kernel: hept`, or CLI
`--kernel deltanet-hept`. Read docs/HEPT_KERNEL_DELTANET.md. It uses the delta
rule with coordinate-augmented Gaussian similarities; the exact dual backend
is a small-event reference, the RFF backend is scalable and approximate, and
`hash` uses exact Gaussian delta updates in independent HEPT-selected buckets.
Read docs/HASH_KERNEL_DELTANET.md for ordering, merging and resource controls.
All three require the PyTorch path. Do not replace this with HEPT LSH attention.
Run unit tests and smoke checks. No real-data improvement is claimed.

# Current handoff notes (2026-10-01)

This recovered project now has a repository-root `train.py` with `--kernel
hept|deltanet`. Read the current README and docs/CHANGES_AND_CONCERNS.md first.
The environment mentioned below belongs to the source author's dev box; create
an environment using the current README. Run `bash scripts/smoke_test.sh` after
changes; it is now offline and stops on failure. HEPT correctness tests compare
both outputs and gradients to direct Gaussian math. Both supported mixers still
require batch_size=1. Concrete padding, metric, loss and dependency fixes were
made under the user's request to address concerns; see the audit before changing
benchmark behavior further.

---

# AGENTS.md — orientation for coding agents (Claude, Codex, …)

Read this before editing. Human docs are in `README.md`; the research agenda is
in `docs/LOCALITY.md`; the running log is `DEVLOG.md`.

## What this repo is

Drop-in swap of **DeltaNet** (linear attention, arXiv 2406.06484) for the
attention layer of the **HEPT** point-transformer backbone (arXiv 2402.12535),
to compare token-mixers apples-to-apples on two HEP tasks (tracking, pileup).
**Only the token-mixer changes**; keep the backbone, losses, metrics, and
preprocessing identical unless a task explicitly needs otherwise.

## Repo map

```
src/
  models/baselines/transformer.py   # HEPT backbone. Attn.__init__ dispatches the mixer.
  models/attention/deltanet.py      # DeltaNetAttention + delta-rule math + point ordering. <- main file
  models/attention/hept.py          # HEPT LSH attention (baseline, unchanged)
  models/attention/__init__.py      # optional imports (missing baseline deps -> None)
  datasets/{tracking,pileup}.py     # PyG InMemoryDataset loaders (unchanged)
  datasets/sharded.py               # lazy per-event Dataset (low-RAM; identical behaviour)
  utils/{losses,metrics,utils}.py   # InfoNCE/focal, accuracy@k, optim/sched (unchanged)
  utils/run_logger.py               # structured logging
  utils/get_model.py                # builds Transformer; FLOP counting is guarded
  run.py                            # unified train/test runner (both tasks)
configs/{tracking,pileup}/          # *_smoke.yaml (CPU) and real configs
scripts/                            # download_data, make_synthetic_tracking, make_subset, smoke_test, sweep_sortings
tests/test_deltanet_correctness.py  # chunkwise == recurrence, sort roundtrip, layer shape
data/                               # datasets (gitignored); tracking_smoke/ is synthetic
logs/                               # per-run output (gitignored)
```

## Environment

Use the project venv (base conda torch 1.13 + pyg 2.3, plus einops/torchmetrics/fvcore):

```bash
.venv/bin/python ...
```

No GPU on the dev box — everything must run on CPU. Training happens later on a
GPU box as a job. `deltanet_impl: auto` uses the `fla` Triton kernel only when
CUDA + `flash-linear-attention` are present; otherwise the pure-PyTorch path.

## Run things

```bash
bash scripts/smoke_test.sh                                   # fast CPU sanity (do this after changes)
cd src && ../.venv/bin/python run.py --config ../configs/tracking/deltanet_smoke.yaml
# overrides: -o model_kwargs.deltanet_sort_key=phi -o num_epochs=50 -o max_events=100
# test mode: --mode test --ckpt ../logs/<run>/best_model.pt
```

## The DeltaNet ↔ HEPT contract (do not break)

`DeltaNetAttention.forward(query, key, value, **kwargs)`:
* inputs `q,k,v`: `[N, num_heads*h_dim]` (single point cloud, `batch.max()==0`);
* returns `[N, h_dim]` (matches `HEPTAttention.out_linear`);
* uses `kwargs["coords"]` (`[:,0]=eta, [:,1]=phi`) and optional `kwargs["pt"]`
  only to compute the scan order; ignores `pe`/`w_rpe`.

`transformer.py` edits are intentionally minimal: the `deltanet` branch in
`Attn.__init__`, the flat-layout branch in `prepare_input`, and `pt` threading
in `Transformer.forward`. Keep them minimal.

## Logging contract (for both humans and agents)

Each run dir under `logs/` has:
* `metrics.jsonl` — one JSON/line, `kind ∈ {meta, epoch, final}`. Parse with
  `pandas.read_json(p, lines=True)`. `epoch` records carry the metric dict flat;
  `final` carries `best_epoch`, `best_valid`, `best_test`, `best_test_full`.
* `run.log` (human), `config.yaml` (resolved config), `best_model.pt`.

When adding metrics, log through `RunLogger.epoch(...)` / `.event(...)` so both
streams stay in sync. Don't `print()` results only.

## Conventions / gotchas

* **Memory:** never load a full tracking `.pt` (~66 GB) casually. Smoke tests use
  synthetic tracking data + real pileup capped by `max_events`. Use
  `scripts/make_subset.py` (opt-in, one-time full load) for real small subsets.
  For real low-RAM training, re-shard with `scripts/shard_dataset.py` and set
  `use_sharded: true` — behaviour is identical (see `datasets/sharded.py` and
  `tests/test_sharded_equivalence.py`); the source `.pt` is never modified.
* **Apples-to-apples:** DeltaNet and HEPT configs share `num_heads/h_dim/n_layers`.
  For matched-FLOPs comparisons, tune those (HEPT has extra `w_rpe`; DeltaNet has
  a tiny `b_proj`), and report params from the `meta` log record.
* **Correctness first:** any change to the delta-rule math must keep
  `tests/test_deltanet_correctness.py` green (chunkwise == O(L) recurrence).
* **Locality work** goes through `_sort_key_values` in `deltanet.py` — add new
  orderings (Morton/Hilbert/learned) there; the rest is untouched. See
  `docs/LOCALITY.md`.
* Optional baselines (performer/reformer/…) are `None` unless their libs are
  installed; don't assume they import.
```
