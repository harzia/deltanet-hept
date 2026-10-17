# Development log

Reverse-chronological. Each entry: what changed, why, and current state.

## 2026-07-23 — low-RAM sharded dataset

**Why.** `InMemoryDataset` deserializes the full 60–66 GB `.pt` into RAM at
construction; a normal DataLoader over it saves nothing (it only slices already-
resident memory), and `num_workers>0` multiplies it. Fix = make the *storage*
lazy, not the loader.

**Done.**
- `datasets/sharded.py::ShardedPointDataset` — PyG `Dataset` (not InMemory) with
  one file per event; `get(i)` reads a single shard from disk. Same external API
  (len/get/indexing/transform/idx_split/x_dim/coords_dim), so `run.py` and the
  PyG DataLoader are unchanged.
- `scripts/shard_dataset.py` — one-time conversion. READS the source only,
  WRITES shards to a separate `--out_root`; never deletes/modifies the source.
  Each event `.clone()`d so shards are compact (KB), not views into the 60 GB
  storage.
- Wired `use_sharded` config flag through `get_data.get_dataset` + `run.py`.
- **Verified identical:** `tests/test_sharded_equivalence.py` asserts same len,
  dims, splits, and bit-identical tensors for every key of every event; a full
  training run via the sharded path reproduced the in-memory result exactly
  (valid acc@0.9=0.1699, test=0.12933097779750824).

**Note.** The one-time shard conversion still loads the source once (needs RAM
for that step); afterwards training runs at RAM ≈ batch size and can use
`num_workers>0`.

## 2026-07-23 — initial build

**Goal.** Implement DeltaNet (arXiv 2406.06484) on the HEPT backbone
(arXiv 2402.12535) as an apples-to-apples token-mixer swap; train+test scripts;
download data; keep it memory-safe on a CPU dev box (GPU training later).

**Done.**
- Vendored the HEPT `src/` backbone into this repo.
- Built `DeltaNetAttention` (pure-PyTorch chunkwise + recurrent delta rule,
  faithful to `fla`; optional `fla` Triton path on CUDA). Locality via
  configurable point ordering (`eta/phi/r/pt/kt/original`) and uni/bi scan.
- **Verified** chunkwise == O(L) recurrence to ~1e-7 (`tests/test_deltanet_correctness.py`).
- Wired DeltaNet into `transformer.py` with 3 minimal edits (dispatch, flat
  layout in `prepare_input`, `pt` threading). Kept everything else identical.
- Unified `run.py` train/test runner for both tasks; structured logging
  (`run.log` + `metrics.jsonl` + `config.yaml` + `best_model.pt`); dropped `nni`.
- Data: downloaded all three preprocessed sets fresh from Zenodo
  (`scripts/download_data.py`). pileup extracted (11 GB); tracking-6k/60k
  extracting (~66/60 GB, disk only).
- Memory-safe smoke path: synthetic tracking generator
  (`scripts/make_synthetic_tracking.py`) + real pileup capped by `max_events`.
  Opt-in real-subset maker (`scripts/make_subset.py`).
- Configs: DeltaNet + HEPT for tracking-6k/60k and pileup, plus smoke configs.
- Docs: `README.md` (humans), `AGENTS.md` (agents), `docs/LOCALITY.md`
  (permutation-equivariance / locality literature review + research agenda).

**Smoke results (CPU, tiny models — sanity only, not benchmarks).**
- DeltaNet tracking (synthetic, 30 events, 2 epochs): loss 5.9 → 4.4, runs clean.
- HEPT tracking (same synthetic): loss 5.0 → 3.9 (backbone intact).
- DeltaNet pileup (real, 4 events, 1 epoch): trains, metrics computed, RAM fine.
- `test` mode reproduces the training checkpoint's test metric.

**Environment.** venv on base conda (torch 1.13.1 + pyg 2.3.1) + einops,
torchmetrics 0.11.4, fvcore. No GPU here (fvcore FLOP count is guarded/skipped
because DeltaNet's data-dependent control flow isn't traceable — cosmetic only).

**Next.**
- Run the sort-key × direction sweep on a real subset; compare to HEPT baseline
  at matched params/FLOPs.
- Implement Tier-1 locality: Morton/Hilbert serialization in `_sort_key_values`
  (see `docs/LOCALITY.md`).
- Validate the `fla` GPU path on the training box.


## 2026-10-01: recoverable HEPT/DeltaNet training handoff

Recovered the full exposed source tree; added a root training entry point and
explicit kernel selection, repaired padding and mass normalization, tracking
metric memory/self exclusion, missing InfoNCE groups, loading/seed handling and
smoke-test failure propagation. Added offline pileup fixtures, exact reference
output/gradient tests and CPU CI. See docs/CHANGES_AND_CONCERNS.md for validation
and remaining limits. Existing benchmark results are not claimed unchanged.


## 2026-10-01: clarified Gaussian-kernel DeltaNet

The intended target is DeltaNet using the HEPT Gaussian kernel. Added an exact
dual Gaussian delta-rule reference and a unit-norm Fourier-feature approximation
inside the original chunkwise state update. Added learned HEPT coordinate weights
to this path, configs and `--kernel deltanet-hept`. Added six mathematical and
backbone tests; 20 unit tests now pass. See docs/HEPT_KERNEL_DELTANET.md for the
approximation limits and experiments. Also corrected pure HEPT tracking-6k region
default from 150 to 15 based on Appendix C.2; not a verified optimum.

## 2026-10-01: hash-selected Gaussian DeltaNet

Added `deltanet_hept_backend: hash`: HEPT E2LSH/geographic selection with
exact Gaussian delta writes in independent key-bucket memories. Causal reads
compare original event ranks because query/key hash permutations differ.
Forward/reverse outputs are summed and hash tables are averaged. Grouped
processing and optional work-group activation checkpointing limit temporary
matrix allocations. Fixed projections/regions are stored in checkpoints.
Dedicated configs cover both tracking sizes and pileup, plus offline fixtures.
28 unit tests pass, including independent recurrence/gradient checks, HEPT
membership, single-bucket equivalence to full exact, checkpoint recomputation,
and a 60k synthetic forward with bounded pairwise matrix dimensions. No
full-dataset accuracy or GPU benchmark is claimed. See docs/HASH_KERNEL_DELTANET.md.
The expanded eight-path smoke suite completed. Strict checkpoint evaluation
reproduced all 10 tracking and all four pileup metrics within 1e-7.
