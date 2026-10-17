# DeltaNet with the HEPT Gaussian kernel

**Corrected implementation:** DeltaNet now has an option to use HEPT's
geometry-aware Gaussian kernel inside its error-correcting delta rule. The
first delivered archive only switched mixers; it did not implement this.

```bash
# Install using the environment instructions below, then train:
python train.py --config configs/tracking/deltanet_hept_tracking-6k.yaml --kernel deltanet-hept
python train.py --config configs/tracking/deltanet_hept_tracking-60k.yaml --kernel deltanet-hept
python train.py --config configs/pileup/deltanet_hept_pileup.yaml --kernel deltanet-hept

# Exact Gaussian updates inside HEPT-selected hash buckets:
python train.py --config configs/tracking/deltanet_hept_hash_tracking-60k.yaml --kernel deltanet-hept
```

| Selection | Mixer and similarity |
| --- | --- |
| `--kernel deltanet` | Original DeltaNet delta rule with the original query/key feature map |
| `--kernel deltanet-hept` | DeltaNet delta rule with HEPT's augmented Gaussian kernel, approximated by finite features by default |
| `--kernel hept` | Original HEPT normalized LSH attention, retained as a comparison |

Read [docs/HEPT_KERNEL_DELTANET.md](docs/HEPT_KERNEL_DELTANET.md) for the exact
recurrence, the scalable approximation and performance experiments. This is
an experimental combination, not a claim of the paper's benchmark performance.

The new `deltanet_hept_backend: hash` uses HEPT hash-selected blocks with exact
Gaussian DeltaNet updates in independent bucket memories. It avoids RFF and
full-event dense matrices. Dedicated configs cover tracking-6k, tracking-60k
and pileup; see [docs/HASH_KERNEL_DELTANET.md](docs/HASH_KERNEL_DELTANET.md) for
commands, causal ordering, table merging, resource controls and limitations.

Complete source recovered from:
https://j-jepa-vol.nrp-nautilus.io/linear-transformers/deltanet-hept/

This project trains either **HEPT** or **DeltaNet** on the same point-transformer
backbone for tracking and pileup. The linked directory already contained HEPT;
this version makes the selection explicit, fixes correctness and installation
issues, and adds independently checked HEPT tests.

The paper is [Miao et al., ICML 2024](https://arxiv.org/abs/2402.12535).
The paper authors' repository is https://github.com/Graph-COM/HEPT.
That is the upstream HEPT repository; the origin of the custom DeltaNet project
is not established. You do not need either repository to use this archive.

## Install

Recommended: Linux, Python 3.10-3.12, a new virtual environment, PyTorch 2.5.1,
PyG 2.6.1. The CPU stack below was tested with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --only-binary=:all: torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
python -m pip install -r requirements.txt
```

For an NVIDIA GPU with a compatible driver, use the CUDA 12.1 wheels instead
of the two CPU installation commands:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install --only-binary=:all: torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
python -m pip install -r requirements.txt
```

Use matching PyTorch and PyG extension builds; see the official
[PyTorch version guide](https://pytorch.org/get-started/previous-versions/)
and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html).
The CUDA stack has not been executed in this workspace. `environment.hept.yml`
is the original historical environment, retained for reference.

HEPT runs directly in PyTorch on CPU or CUDA; it does **not** require
`flash-linear-attention`. For reproducible DeltaNet checks use
`-o model_kwargs.deltanet_impl=torch`. The inherited optional FLA adapter is
untested here and its API may depend on the installed FLA version. Installing
an arbitrary newer FLA version may require a different PyTorch stack.
`uproot` and `awkward` are optional for raw ROOT preprocessing, not for
preprocessed data or the offline smoke tests.

## Try it without downloading benchmark data

```bash
bash scripts/smoke_test.sh
```

This checks the math, generates small synthetic tracking and pileup fixtures,
compares sharded and in-memory data, and trains standard DeltaNet, Gaussian-kernel DeltaNet and HEPT on both tasks.
Every failure stops the script. If you use a different environment, set
`PYTHON=/path/to/python bash scripts/smoke_test.sh`.
Synthetic results validate execution; they are not physics benchmark scores.

You can also run an individual experiment:

```bash
python scripts/make_synthetic_tracking.py
python train.py --config configs/tracking/hept_smoke.yaml --kernel hept -o device=cpu
python train.py --config configs/tracking/deltanet_smoke.yaml --kernel deltanet -o device=cpu
```

## Train on the benchmark data

The archive includes all source, configs, scripts, tests and attribution. Large
datasets, checkpoints, training logs and Python caches are excluded.
Download data separately, or place your existing preprocessed files at:

| Dataset | Expected file |
| --- | --- |
| tracking-6k | `data/tracking/processed/tracking-6k/data-6k.pt` |
| tracking-60k | `data/tracking/processed/tracking-60k/data-60k.pt` |
| pileup | `data/pileup/processed/data.pt` |

```bash
python scripts/download_data.py --dataset tracking-6k
python train.py --config configs/tracking/hept_tracking-6k.yaml --kernel hept
python train.py --config configs/tracking/hept_tracking-60k.yaml --kernel hept
python train.py --config configs/pileup/hept_pileup.yaml --kernel hept
```

Tracking datasets require roughly 60-66 GB after extraction and the default
loader holds the full processed dataset in RAM. `max_events` caps the split
iteration **after loading**; it does not prevent that allocation. The pileup
processed dataset is about 11 GB. Sizes are estimates inherited from the source.

For lazy per-event loading after a one-time conversion:

```bash
python scripts/shard_dataset.py --dataset tracking-6k --src_data_dir data --out_root data_sharded
python train.py --config configs/tracking/hept_tracking-6k.yaml --kernel hept -o data_dir=data_sharded -o use_sharded=true -o num_workers=2
```

The conversion itself loads the monolithic file once and therefore still needs
sufficient RAM. Shards preserve the original splits and transforms.
Only load trusted `.pt` datasets: legacy PyG objects require pickle deserialization.

## Select the kernel

Use `--kernel deltanet-hept` for the requested combination, `--kernel hept` for the HEPT comparison, or `--kernel deltanet` for the original DeltaNet. Without the flag, `model_name` in the
YAML selects the mixer (`trans_hept` / `trans_deltanet`). `train.py` works from
the repository root; the original `src/run.py` entry point also remains available.
Relative `data_dir` is resolved from the repository root.

The kernel flag preserves the configured backbone, loss, dataset and optimizer.
When switching a DeltaNet config to HEPT, missing HEPT settings get defaults:
block size 100, 3 hash tables, 15 regions for tracking-6k / 150 for tracking-60k / 140 for pileup,
10 RPE weights per distance, and `pe_type: none`. Dedicated HEPT configs are
preferable for benchmark runs. Override any setting with repeatable `-o` flags:

```bash
python train.py --config configs/tracking/deltanet_tracking-6k.yaml --kernel hept -o model_kwargs.block_size=100 -o model_kwargs.n_hashes=3 -o model_kwargs.num_regions=150
```

| HEPT option under `model_kwargs` | Meaning |
| --- | --- |
| `block_size` | Number of points per sparse attention block |
| `n_hashes` | Number of OR hash tables |
| `num_regions` | Target number of coordinate regions for AND construction |
| `num_w_per_dist` | Original positive coordinate-weight parameterization |
| `num_heads`, `h_dim`, `n_layers` | Backbone dimensions; keep matched when comparing mixers |
| `use_ckpt` | Optional activation checkpointing |

Both mixers currently require **`batch_size: 1`**. A forward handles one event;
multiple events in a PyG batch are rejected explicitly to prevent event mixing.

HEPT uses the paper's Gaussian kernel:

```text
q_hat = [q, sqrt(2 * omega) * coordinates]
k_hat = [k, sqrt(2 * omega) * coordinates]
A_ij = exp(-0.5 * ||q_hat_i - k_hat_j||^2)
output = (sum_t A_t V) / (sum_t A_t 1)
```

`omega` is positive and learned through the original `w_rpe` parameterization.
The original transforms augment geometric eta/phi with other coordinate
features; these are preserved. LSH uses coordinate regions to align separately
sorted query and key blocks. Repeated pairs in multiple tables contribute to
each table, as in the authors' implementation. Normalization uses total kernel
mass across tables, not an unweighted average of table outputs.

`--kernel hept` selects the HEPT sparse attention mixer. The new
`--kernel deltanet-hept` option instead uses the Gaussian kernel in DeltaNet's
delta-rule recurrence; see the linked derivation for its exact and approximate paths.
Attention avoids a dense N-by-N matrix, with block attention cost proportional
to `n_hashes * N * block_size` and sorting cost proportional to `N log N` for
fixed head dimensions. Tracking evaluation still takes quadratic time; its
distance matrix is now processed in chunks to bound peak memory.

## Evaluate and inspect logs

```bash
python train.py --config configs/tracking/hept_tracking-6k.yaml --kernel hept --mode test --ckpt logs/YOUR_RUN/best_model.pt
```

Use the same backbone and kernel settings as the checkpoint's `config.yaml`.
Each run records `config.yaml`, `run.log`, `metrics.jsonl`, and `best_model.pt`.
Checkpoint selection uses validation only. The inherited runner also reports
test metrics each epoch; avoid tuning hyperparameters against those test results.
The `auc` field for pileup is **average precision**, not ROC AUC; `roc` reports
ROC AUC separately. Tracking `accuracy@0.9` is the inherited neighborhood
precision metric on reconstructable points above the pt threshold, not complete
track reconstruction efficiency.

## Review and upload to GitHub

See [docs/CHANGES_AND_CONCERNS.md](docs/CHANGES_AND_CONCERNS.md) for exact fixes,
remaining limitations and validation. [docs/SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json)
records hashes of all 83 retrieved source/config/document/test files before edits.
Original docs and legacy baseline trainers are retained. Use `train.py` for the
supported HEPT/DeltaNet workflow; legacy trainers may need extra dependencies.

To publish the extracted folder into your own new, empty GitHub repository:

```bash
git init
git add .
git commit -m "Add HEPT and DeltaNet training code"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Retain `LICENSE` and the upstream attribution. No repository is created or
published automatically.
