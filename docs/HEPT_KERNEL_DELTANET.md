# DeltaNet using the HEPT kernel

## What changed from the first handoff

The original `DeltaNetAttention` used coordinates to sort points, then applied
a delta rule to its activated/L2-normalized query and key vectors. The Gaussian
kernel existed only in the separate HEPT attention module. It was not used by
DeltaNet. This update implements the clarified request.

The new option remains `model_name: trans_deltanet`, with
`model_kwargs.deltanet_kernel: hept`. It does not call HEPT LSH attention.
The backbone's feature encoder, residual/FFN blocks, heads, tasks and transforms
are kept. DeltaNet now receives `w_rpe`, giving it the same positive learned
coordinate-weight parameterization as HEPT.

## Mathematical target

Augment the projected queries and keys with coordinates:

```text
q_hat_t = [q_t, sqrt(2*omega) * coords_t]
k_hat_t = [k_t, sqrt(2*omega) * coords_t]
k_HEPT(x,y) = exp(-0.5 * ||x-y||^2)
```

The delta rule in the Gaussian reproducing kernel Hilbert space (RKHS) is:

```text
S_t = S_(t-1) + beta_t * phi(k_hat_t) *
      (v_t - S_(t-1)^T phi(k_hat_t))^T
o_t = S_t^T phi(q_hat_t)
```

This learns an associative memory with error-correcting writes. It is not the
row-normalized Gaussian weighted average computed by HEPT attention. Keeping
DeltaNet means keeping these residual writes, causal dependence within a scan,
and the configured bidirectional combination. The inherited bidirectional
mode sums independent forward and reverse scan outputs, including the current
point in both scans.

The equivalent exact dual recurrence is:

```text
u_t = beta_t * (v_t - sum_(j<t) k_HEPT(k_hat_t,k_hat_j) * u_j)
o_t = sum_(j<=t) k_HEPT(q_hat_t,k_hat_j) * u_j
```

`hept_delta_kernel.py` implements this via a triangular solve. The output and
query/key/value/beta gradients are tested against an independent sequential
kernel recurrence using direct squared differences.

## Backends

| Setting | Exactness | Cost and use |
| --- | --- | --- |
| `deltanet_hept_backend: exact` | Exact Gaussian kernel in the dual delta rule, to floating-point precision | Quadratic matrices in event length; default limit 1024 points; small-event reference/debugging |
| `deltanet_hept_backend: rff` | Finite random Fourier approximation to the same Gaussian kernel | Finite recurrent state; intended full-event training path |
| `deltanet_hept_backend: hash` | Exact Gaussian values and delta updates within HEPT-selected buckets | Independent local memories; approximately R*H*N*B kernel entries; see [HASH_KERNEL_DELTANET.md](HASH_KERNEL_DELTANET.md) |

For the scalable path, with m random frequencies per head:

```text
W[:,j] ~ Normal(0,I)
phi(x) = [cos(xW), sin(xW)] / sqrt(m)
feature_dim = 2*m
E[phi(x)^T phi(y)] = k_HEPT(x,y)
```

Every realized feature vector has norm one. This preserves the bounded-key
condition of the delta update, with beta in (0,1) by default. Raw augmented
queries/keys are not L2-normalized, and no SiLU/ReLU is applied; doing either
would change the desired Gaussian kernel. The base delta implementation's
feature-dimension query scaling is explicitly canceled for this option so the
read/write interactions approximate k_HEPT itself rather than k_HEPT/sqrt(D).

Frequencies are fixed registered buffers, use a local deterministic generator,
move with the model and are saved in checkpoints. Coordinate weights and the
query/key/value projections are learned. This is a finite-feature approximation
of a Gaussian-kernel DeltaNet, not an exact finite-dimensional representation of
the Gaussian kernel on unrestricted inputs.

## Options and commands

```yaml
model_name: trans_deltanet
model_kwargs:
  deltanet_kernel: hept
  deltanet_hept_backend: rff
  deltanet_kernel_features: 128  # even total dimension: 64 cosine + 64 sine
  deltanet_kernel_seed: 0
  deltanet_qk_activation: identity
  deltanet_qk_norm: none
  deltanet_impl: torch
  num_w_per_dist: 10
  # keep the usual h_dim, num_heads, n_layers, beta, order and direction options
```

The CLI `--kernel deltanet-hept` sets the kernel, identity activation, no
pre-feature normalization, and PyTorch backend. Explicit `-o` overrides run
last; incompatible settings are rejected rather than silently changing the
kernel. Switching back with `--kernel deltanet` sets the standard kernel; use
a standard DeltaNet config to recover its usual activation/normalization too.

```bash
# Offline fixture tests
python scripts/make_synthetic_tracking.py
python train.py --config configs/tracking/deltanet_hept_smoke.yaml --kernel deltanet-hept -o device=cpu

# Exact kernel reference on those small fixture events
python train.py --config configs/tracking/deltanet_hept_smoke.yaml --kernel deltanet-hept -o model_kwargs.deltanet_hept_backend=exact -o device=cpu

# Full benchmark, feature-budget experiment
python train.py --config configs/tracking/deltanet_hept_tracking-60k.yaml --kernel deltanet-hept -o model_kwargs.deltanet_kernel_features=256
```

The exact guard can be changed with `deltanet_hept_exact_max_points`, but full
60k-event dense matrices are impractical. `max_events` limits event count, not
points per event, and does not make the exact backend safe on real full events.

The new kernel path currently requires `deltanet_impl: torch`. The inherited FLA
adapter was not verified for the expanded feature dimension and kernel scaling.
The PyTorch path can run on CUDA but local validation is CPU-only. Float32 is
the recommended initial precision; mixed precision was not validated.

Old standard-DeltaNet checkpoints lack coordinate weights and frequency
buffers. Train a new experiment; a strict load into the new model should fail.
Use the saved config, same feature dimension and backend for kernel checkpoint
evaluation. Standard DeltaNet's model parameter names remain unchanged.

## Performance hypotheses and priorities

The reported old scores were 0.6250 tracking-6k, 0.7080 tracking-60k and 0.3667
pileup. These alone do not establish which mixer was run or why it underperformed.
Inspect the saved model_name, complete model_kwargs, epochs, validation curve,
feature normalization, dataset splits and checkpoint selection first.

1. Run standard DeltaNet and Gaussian-kernel DeltaNet on identical data/splits,
   backbone dimensions, loss, order, direction, seed and training budget.
   The Gaussian model adds learned coordinate weights and a larger memory state;
   report parameter count, state dimension, peak memory and runtime alongside
   validation performance. A secondary linear-feature/no-activation ablation
   separates changing the old activation from introducing geometry.
2. On small real events or consistently constructed subclouds, compare the exact
   Gaussian delta rule with the RFF path using 128, 256, 512 and 1024 features.
   Measure error on the kernel values and on complete delta outputs: good pairwise
   approximation does not by itself guarantee good recurrent outputs.
3. RFF errors can be large relative to very small true Gaussian weights and can
   produce negative approximate off-diagonal similarities. These are properties
   of the approximation, not physical negative Gaussian weights. Do not clamp
   the pairwise approximations inside the recurrent state: that would break the
   finite-feature delta rule. Increase features and inspect errors first.
4. Sweep learning rate on validation: tracking 1e-3 and 1e-2, pileup 1e-4 and
   1e-3. These are the HEPT paper's ranges, not proven best values for this new
   hybrid. Monitor learned coordinate coefficients for over-localization and
   high-variance phase gradients. The exact initialization is inherited from
   HEPT; alternatives should be explicit ablations.
5. Compare the new `hash` backend with exact and RFF on matched small events,
   then evaluate hash-local memories on full events. It evaluates selected
   Gaussian values exactly but resets memory across buckets and tables. The
   HEPT paper motivates LSH over global RFF for strongly local large point clouds;
   this new delta-rule variant still needs its own accuracy measurement.

A separate config issue was found while reviewing the original comparison:
`hept_tracking-6k.yaml` inherited `num_regions: 150`, whereas Appendix C.2 of the
paper tunes 10/15/20 for tracking-6k. This update uses 15 for the **pure HEPT
comparison** and its fallback. Tracking-60k keeps 150 and pileup keeps 140. The
midpoint is a starting setting from the reported range, not a verified optimum.
The new hash backend also uses these region counts; RFF and full exact do not.

## Validation

Twenty unit tests passed after this addition. New tests cover exact Gaussian
kernel recurrence and gradients, unit-norm features, translation invariance,
approximation against exact kernels and complete delta outputs, coordinate/RPE
backpropagation, activation checkpointing, checkpoint buffers and size/config
guards. End-to-end synthetic training and saved-checkpoint evaluation cover
both tasks and both Gaussian-kernel backends. No full dataset or GPU experiment
has been run, so no improvement over the user's scores is claimed.

The later hash-backend addition expands this to 28 unit tests and adds offline
training on both tasks. See [HASH_KERNEL_DELTANET.md](HASH_KERNEL_DELTANET.md)
for the new recurrence, checks and the bounded-matrix 60k synthetic forward test.

## References

- HEPT kernel and approximation analysis: https://arxiv.org/abs/2402.12535
- DeltaNet delta rule: https://arxiv.org/abs/2406.06484
- Finite Fourier features: Rahimi and Recht (2007), discussed in HEPT Section 2.
