# Changes, audit notes and validation

**Clarification update:** the requested Gaussian-kernel DeltaNet is now implemented.
Read HEPT_KERNEL_DELTANET.md for the new exact and RFF paths. The first archive
only selected the existing mixers. The notes below describe that initial audit;
the current package also adds learned coordinate weights to the Gaussian DeltaNet
path, new configs and six additional unit tests (20 total). Pure HEPT tracking-6k
regions are corrected to 15 from the paper's 10/15/20 tuning range.


## What was recovered

The provided URL is an Nginx directory listing, not a GitHub repository.
All 83 exposed source, script, test, config, license and documentation files
were retrieved, including legacy baselines and trainers. The source manifest
records their original SHA-256 hashes. Runtime caches, data and logs were not
included. The recovered `hept.py` was byte-identical to the corresponding
`Graph-COM/HEPT/main/src/models/attention/hept.py` file when inspected.
The original project already had HEPT configs and the `trans_hept` dispatch.

## Supported changes

| Area | Change and reason |
| --- | --- |
| Kernel selection | Added root `train.py` and `--kernel hept\|deltanet`; missing HEPT knobs receive explicit defaults. Existing YAML selection remains supported. |
| HEPT padding | Dummy keys are masked in both the Gaussian numerator and denominator. Previously their zero values still increased the denominator, shrinking real outputs. |
| Hashing and regions | Dummy points no longer determine the hash range or coordinate quantiles. Region partitions are computed on real points before padding. |
| Numerical stability | Each table computes normalized Gaussian weights and log kernel mass; OR results merge by mass in log space. This preserves the paper's sum-of-numerators / sum-of-denominators rule and avoids all-zero underflow normalization. |
| Autograd | Padding masks use out-of-place operations. Reference tests verify query, key, value and RPE gradients. |
| Activation checkpointing | Uses `use_reentrant=False`. Dictionary forwards no longer permanently disable `use_ckpt`; current region parameters are threaded after device moves. |
| Tracking metric memory | Default query chunks are 1024 points, avoiding the full N-by-N distance allocation. The calculation still takes quadratic time. |
| Tracking metric self-neighbor | Self is excluded by index rather than assuming it is the first nearest neighbor; tiny events can use fewer than 19 available neighbors. |
| InfoNCE grouping | Negative reductions are expanded to the actual point IDs before indexing. The prior compact group array could silently return another point's sum or fail when a group was missing. |
| Loss validation | Gives a clear error when no eligible positive pairs exist; validates positive temperature. Focal loss now honors configured keyword arguments. |
| Loading and seed | Explicit dataset pickle loading supports trusted legacy PyG files across the PyTorch 2.6 default change. Checkpoints use weights-only loading. The configured seed is restored after legacy dataset constructors. |
| Downloads | Declining the loaders' download prompt no longer deletes the existing dataset directory. Also fixed the raw tracking path join. |
| Profiling | FLOP tracing is opt-in (`count_flops: true`); it no longer allocates a synthetic 10,000-point experiment on every startup or permanently changes checkpoint mode. The profiler restores model mode and CPU RNG state. |
| Worker setting | `num_workers` is now passed to the loaders; lazy shards can use it. |
| Dependencies | Complete requirements and tested CPU setup replace reliance on an unseen author's base environment. ROOT imports are lazy. |
| Smoke checks | Synthetic fixtures cover both tasks and mixers, with real exit-code propagation. No benchmark download is needed. Modern PyG keys are handled in the shard equivalence check, and temporary shards are cleaned up. |
| Logging | Run directories include microseconds to avoid collisions between quick runs. |
| Handoff | Added usage, source provenance, regression tests, GitHub ignore rules and CPU CI. |

For the initially recovered comparison, the backbone dimensions, feature transforms, positional-weight parameterization,
normal losses and benchmark splits are preserved. Correcting padding, hashing
range, tie handling, RNG side effects and missing negative groups intentionally
changes affected outputs. Existing state-dict tensor shapes are preserved, but
**do not claim bit-for-bit agreement with old training results**. Retrain and
rerun both mixers for a fair comparison.

## What the tests establish

Local validation used Python 3.12, PyTorch 2.5.1+cpu, PyG 2.6.1,
torch-scatter 2.1.2+pt25cpu and torch-cluster 1.6.3+pt25cpu.

- 14 unit tests passed, including the inherited DeltaNet forward tests.
- HEPT outputs and gradients match independent dense squared-distance math
  for a single full block and a multiset of sparse LSH candidates across tables.
  Cases cover one point, exact block multiples, partial blocks and multiple hashes.
- Constant values are no longer diluted by dummy keys; underflow and fully
  masked block behavior are tested.
- A full HEPT backbone backward pass works with activation checkpointing.
- DeltaNet chunkwise gradients match the recurrence in float64.
- Missing InfoNCE negative groups match a manually computed loss.
- Metric chunking agrees with the same metric computed in a single query chunk.
- All 30 synthetic tracking events, transforms, dimensions and splits agree
  between in-memory loading and lazy shards.
- HEPT and DeltaNet each trained, validated, tested and wrote a checkpoint for
  both tracking and pileup on the offline fixtures.
- Reloading each HEPT checkpoint in test mode reproduced all stored test metrics.
- Switching a DeltaNet YAML to HEPT using only `--kernel hept` also completed training.
- Python compilation, shell syntax and whitespace checks passed.

Synthetic training is a pipeline check, not evidence of convergence or physics
quality. No real dataset was downloaded or trained, and no GPU was available.
CI is supplied but has not been run on GitHub.

## Remaining concerns and experimental limits

1. **One event per forward.** Keep `batch_size: 1`. Efficient packed multi-event
   HEPT would require a separate batching design and batch-isolation tests.
2. **Large data allocation.** In-memory tracking files need tens of GB of RAM.
   `max_events` does not solve that. Sharding solves subsequent loading, but the
   current converter still needs one full load and creates many small files.
3. **Sparse approximation.** HEPT is LSH block attention over a Gaussian kernel;
   a small block size, too few hashes or poorly chosen regions can miss useful
   pairs. Tune on validation and record the settings and parameter counts.
4. **GPU and FLA.** PyTorch HEPT has no custom fused Triton implementation here.
   CUDA and the inherited optional FLA adapter were not validated. Use the
   PyTorch DeltaNet backend for the tested path; FLA signatures and supported
   dtypes can vary with the FLA version.
5. **Ground-truth pt.** On simulated tracking, `pt` may be truth metadata
   unavailable before reconstruction. Opt-in DeltaNet `pt`/`kt` ordering can
   leak that information. Default `eta` avoids this extra input.
6. **Chunk size is not a locality window.** DeltaNet carries state across chunks;
   `chunk_size` controls its computation, not a hard cutoff on context. Geometry
   sorting supplies an ordering; ties can still make output order-dependent.
7. **Periodic phi.** The inherited coordinate differences and region sorting
   do not wrap phi across -pi/pi. Changing this would change the benchmark model,
   so it is documented rather than silently changing the physics preprocessing.
8. **Metric names.** Pileup `auc` is average precision. Tracking `accuracy@0.9`
   measures embedding neighbors, not complete tracking efficiency. Evaluation
   assumes cluster sizes fit the inherited `K=19` budget and filters have eligible
   points. Undefined cases remain explicit errors, not fabricated scores.
9. **Objective conventions.** The inherited `l2_rbf` InfoNCE similarity uses
   unsquared L2 distance (distinct from the squared-distance HEPT attention
   kernel). The inherited focal implementation applies a scalar alpha to all
   examples. These conventions are preserved for comparison.
10. **Benchmark protocol.** Test metrics are reported each epoch, though the best
    checkpoint is selected by validation only. Do not use test feedback to tune.
    The tracking loader's inherited event sorting and split recomputation are
    preserved. Use the saved resolved config for repeatable evaluation.
11. **Legacy baselines.** Original alternative attention modules and the NNI
    trainers remain in the source. Their optional libraries and behavior are
    outside the validated HEPT/DeltaNet workflow. Broad optional-import handling
    in the inherited registry may conceal an unavailable baseline's error.
12. **Precision.** End-to-end tests use float32 or float64. Mixed-precision and
    extreme coordinate-weight overflow are not validated; use float32 initially.

## References and attribution

The later hash-backend update adds local exact Gaussian DeltaNet memories
selected by HEPT hashing. It changes the recurrence through bucket resets and
averages independent table outputs. See [HASH_KERNEL_DELTANET.md](HASH_KERNEL_DELTANET.md)
for the explicit math, resource controls and validation. This does not establish
improved physics accuracy; full-dataset and GPU experiments remain unrun.

- Original directory: https://j-jepa-vol.nrp-nautilus.io/linear-transformers/deltanet-hept/
- HEPT paper: https://arxiv.org/abs/2402.12535, especially Sections 4.1-4.3.
- Paper authors' implementation: https://github.com/Graph-COM/HEPT
- Retain the supplied MIT license and Graph-COM copyright notice.
