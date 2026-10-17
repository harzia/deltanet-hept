# Hash-selected Gaussian DeltaNet

This backend keeps DeltaNet's residual writes and evaluates the HEPT Gaussian
kernel exactly within hash-selected blocks. It uses independent memories for
each key block in each hash table. It approximates full-event Gaussian DeltaNet
by limiting interactions, with no random Fourier features.

## Train

```bash
python train.py --config configs/tracking/deltanet_hept_hash_tracking-6k.yaml --kernel deltanet-hept
python train.py --config configs/tracking/deltanet_hept_hash_tracking-60k.yaml --kernel deltanet-hept
python train.py --config configs/pileup/deltanet_hept_hash_pileup.yaml --kernel deltanet-hept
```

Existing Gaussian DeltaNet configs can also select this backend:

```bash
python train.py --config configs/tracking/deltanet_hept_tracking-60k.yaml --kernel deltanet-hept \
  -o model_kwargs.deltanet_hept_backend=hash \
  -o model_kwargs.block_size=100 \
  -o model_kwargs.n_hashes=3 \
  -o model_kwargs.num_regions=150
```

Use repeatable `-o` flags, one assignment per flag. Gaussian backends require
`deltanet_impl: torch`, `deltanet_qk_activation: identity` and
`deltanet_qk_norm: none`; the kernel CLI sets those. Existing RFF and full exact
configs retain their behavior. The new hash configs explicitly select `hash`.

| Option under `model_kwargs` | Meaning / default |
| --- | --- |
| `deltanet_hept_backend: hash` | Select local exact Gaussian delta updates |
| `block_size: 100` | Maximum keys and queries in each selected block |
| `n_hashes: 3` | Independent hash-table memories, averaged at output |
| `num_regions` | HEPT geographic region construction; configs use 15 / 150 / 140 for tracking-6k / tracking-60k / pileup |
| `deltanet_hash_chunk_buckets: 64` | Buckets per head processed in one work group, one table at a time |
| `deltanet_hash_checkpoint: true` | Recompute work-group kernel matrices in backward to reduce retained activations |
| `deltanet_sort_key: eta` | Original event ordering used for causal updates and reads |
| `deltanet_direction: bi` | Sum forward and reverse memory reads; `uni` uses forward only |

`chunk_size` controls the standard/RFF scan and is not used by `hash`.
`deltanet_kernel_features`, `deltanet_kernel_seed` and
`deltanet_hept_exact_max_points` are also not used by `hash`. Global training
`seed` initializes fixed hash projections/regions and learned parameters;
save and evaluate the complete checkpoint and its config.

## Selection, updates, and reads

The usual learned HEPT coordinate augmentation produces q_hat and k_hat.
The kernel remains `K(x,y) = exp(-0.5 * ||x-y||^2)`.

1. Apply the existing HEPT E2LSH projection, coordinate quantile regions and
   geographic hash shifts. Sort augmented queries and keys separately for
   each table and head, then split each sorted list into fixed-size blocks.
   Query block b reads key block b, as in the existing HEPT implementation.
2. Preserve selected block membership, but sort the keys inside each block by
   the configured event rank. Values and beta follow their keys. Each key block
   starts with empty memory; memory does not carry across blocks or tables.
3. Solve the exact Gaussian delta recurrence for the selected keys. Read that
   memory at each selected query, masking keys by their original event ranks.
4. For bidirectional mode, also solve with reversed key order and use the
   reverse causal read mask. Sum the two outputs. Restore query order and take
   the arithmetic mean across tables before the existing output projection.

For table r and block b, the forward equations are:

```text
u_j^(r,b) = beta_j * (v_j - sum_{l in key_block(r,b), rank(l)<rank(j)}
                                  K(k_hat_j,k_hat_l) * u_l^(r,b))
o_i^(r) = sum_{j in selected_key_block(r,i), rank(j)<=rank(i)}
               K(q_hat_i,k_hat_j) * u_j^(r,b)
```

Reverse updates use descending ranks and reverse reads use `rank(j)>=rank(i)`.
When a point's own key is selected, its write is included in both directions,
matching the inherited full DeltaNet bidirectional convention. Separate
query/key hash sorts can place a point's own key outside its selected block;
the backend does not force an additional self edge.

A triangular mask based on query/key *positions within hash blocks* would be
incorrect: those permutations differ. The implementation uses rank comparisons
for reads and a triangular key-key system only after sorting keys by rank.

Tables have separate residual writes. A key appearing in multiple tables is
an ensemble contribution, not a duplicated edge in one shared recurrence;
there is no edge deduplication across tables. Averaging tables makes duplicating
an identical table leave the output unchanged. HEPT's normalized attention
mass merge does not apply to these unnormalized delta outputs.

## Exactness and resource use

Kernel values and delta updates within each bucket are exact to floating-point
precision. With one bucket containing every real point, each table matches
the full exact backend in both directions, including gradients. With multiple
buckets, it is a different, localized recurrence: there is no global state or
cross-bucket residual prediction. It is also different from a single global
recurrence with arbitrarily masked edges. No equivalence to the HEPT paper's
normalized attention or its benchmark scores is claimed.

Hash/region selection is discrete and runs without gradients, as in HEPT.
Selected Gaussian values still backpropagate to projected q/k, coordinates,
coordinate weights, values and beta. Hash projections and region parameters
are fixed per layer and stored in checkpoints. The original HEPT comparison
uses its own existing geographic-region setup; models do not share hash banks.

There are approximately R*H*N*B pairwise kernel entries instead of H*N*N;
sorting still costs time. For fixed block size B and number of tables R,
kernel work scales linearly with event length N. Exact triangular solves and
their gradients cost more as B increases. Groups bound temporary matrices to
`[heads, chunk_buckets, B, B]`, per table. With H=8, chunk_buckets=64, B=100,
one float32 matrix in a group is about 19.5 MiB; this is not total peak memory.

Checkpointing trades recomputation for lower saved activation memory; disabling
it retains bucket kernel/solve intermediates across the event. Gathered
features, outputs, permutations, other model layers and their gradients still
grow with N. Decreasing work-group size reduces temporary allocations but
increases Python/GPU launch overhead. Increasing block size can recover more
neighbors but increases memory and compute. No fused sparse GPU kernel is
provided; CUDA performance and full-model 60k training memory are unmeasured.

The full-event exact size guard does not apply, but choosing a block size as
large as an event recreates dense costs. Padding is hashed only after real-point
selection, has zero beta, and is excluded from reads; fully empty key blocks
produce zero output. Hash/region buffers must be loaded from checkpoints to
reproduce evaluation. Checkpoints from other backends are incompatible with
strict loading into `hash`; train a new run or explicitly manage partial loading.

## Validation

Tests compare outputs and q/k/v/beta gradients to independent sequential
residual writes with direct squared distances; verify HEPT block membership;
compare padded single-bucket layers to full exact layers and their coordinate
and RPE gradients; check multiple tables, separate query/key ordering, work
grouping, recomputation, activation checkpointing and checkpoint restoration.
A synthetic 60,000-point event checks that every kernel matrix stays at most
16 by 16 in its pairwise dimensions. This is a small-model CPU forward check,
not a full training benchmark. Offline tracking and pileup smoke training
exercise the new backend. Real-data accuracy and GPU performance remain to be
measured.

The complete suite passed 28 tests, and all eight offline task/mixer training
paths completed. Saved hash-backend checkpoints reproduced all 10 tracking
test metrics and all four pileup test metrics within 1e-7 after strict loading.

Implementation: `src/models/attention/hept_delta_hash.py`, dispatched by
`DeltaNetAttention` in `src/models/attention/deltanet.py`.
