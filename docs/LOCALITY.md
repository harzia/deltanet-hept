# Locality in linear transformers for point clouds

## The problem

Softmax attention is **permutation-equivariant**: reorder the tokens and the
output is reordered the same way. That is exactly what a *set* / point cloud
needs — there is no canonical order of hits or particles.

**Linear transformers are not.** DeltaNet, linear attention
(Katharopoulos et al. 2020), Mamba, RWKV, etc. are *sequence* models: they carry
a recurrent state `S` and fold tokens into it one position at a time
(chunk by chunk). Two consequences:

1. **Order matters.** The output for token *i* depends only on tokens that come
   *before* it in the scan (for a causal scan). Swapping token order changes the
   result. So feeding a point cloud requires *choosing* an order.
2. **Recency/locality is implicit in the order.** The delta-rule state is a
   sum of rank-1 updates; without gating it does not decay, but chunk structure
   and finite precision still make *nearby-in-scan* tokens interact more
   directly. With gating (Gated DeltaNet, Mamba2) distant-in-scan tokens
   actively fade. Either way, **"local" == "nearby in the chosen order."**

Therefore the order is not a nuisance to be averaged away — it is the **knob
that decides what "local" means**. HEPT gets locality from LSH buckets that put
points close in (η, φ) into the same attention block. Our job is to get a
comparable locality from a 1-D ordering (plus, later, windowing/gating).

This document is the research agenda; the implementation hook is the sort-key
registry in `src/models/attention/deltanet.py` (`_sort_key_values`).

---

## Tier 0 — baselines (implemented)

* **Single-axis sort:** `eta`, `phi`, `r = sqrt(eta²+phi²)`, `pt`, `kt = pt·r`,
  or `original` (dataset order). Cheap `argsort`; captures locality along one
  axis only.
* **Direction:** `uni` (one causal scan) vs `bi` (forward + backward, summed).
  Bidirectional removes the causal asymmetry a set task does not want — the
  first-in-order point should still "see" later points.
* **Chunk size** is an implicit local window: intra-chunk interactions are
  computed densely (parallel), inter-chunk only through the compressed state.

**Experiments to run first** (use `scripts/sweep_sortings.sh`):
- Which single axis wins on tracking vs pileup? (η is the natural detector axis.)
- How much does `bi` beat `uni`? How sensitive is accuracy to `chunk_size`?
- Does `pt`/`kt` ordering help, and is the pt-leakage caveat worth it?

---

## Tier 1 — space-filling curves (recommended next)

A single axis throws away one spatial dimension. **Space-filling curves** map 2-D
(η, φ) to a 1-D index while *preserving locality*: points close in the plane are
(usually) close in the order.

* **Z-order / Morton code** — bit-interleave discretized (η, φ). Trivial to
  compute, decent locality, some long "jumps" at bit boundaries.
* **Hilbert curve** — better locality preservation than Morton (no large jumps),
  slightly more code. This is what **Point Transformer V3** (Wu et al. 2023) uses
  to *serialize* point clouds for efficient attention — directly analogous to
  what we need here.
* **Multi-curve ensembles** — run several serializations (e.g. Hilbert with
  different origins/rotations, or Morton on shuffled axes) and combine outputs,
  mirroring HEPT's **OR-construction** over multiple hash tables and our
  bidirectional scan. Robustness to any single curve's discontinuities.

Add these as new keys in `_sort_key_values` (return a scalar Morton/Hilbert
index); everything downstream is unchanged.

---

## Tier 2 — local groups / windowed state (the "local groups" idea)

Ordering alone still lets the state accumulate across the whole cloud. To make
locality *hard*:

* **Chunk-aligned regions** — choose the order and `chunk_size` so each chunk is
  a physical (η, φ) tile; restrict interactions to within tiles (reset or mask
  the cross-chunk state at region boundaries). This is a linear-attention analog
  of HEPT's block attention and of window attention (FlatFormer, OctFormer).
* **Gated/decaying state** — Gated DeltaNet / Mamba2 style data-dependent decay
  so contributions fall off with scan distance. Only meaningful *once the order
  is locality-preserving* (Tier 1), otherwise decay is along an arbitrary axis.
* **Overlapping windows / shifted serialization** across layers (à la Swin):
  alternate the curve/origin per layer so information still propagates globally.

---

## Tier 3 — learned / equivariance-aware

* **Learned ordering** — predict a per-point scalar to sort by (a tiny MLP on
  features+coords), trained end-to-end. Risk: non-differentiable `argsort`; use
  soft-sort / optimal-transport relaxations or straight-through.
* **Order canonicalization / averaging** — since sorting breaks permutation
  equivariance, quantify sensitivity to order and optionally average over a set
  of canonical orders (a principled version of the multi-curve ensemble).
* **Hybrid** — keep a cheap global linear-attention path *and* a local windowed
  path; let the model weight them.

---

## What to measure

* Accuracy vs HEPT at **matched params/FLOPs** (tune `h_dim`/`n_layers`; see
  `AGENTS.md`). The point is the token-mixer, not model size.
* **Order sensitivity**: variance of the metric across sort keys / random
  reorderings — a direct measure of the broken equivariance.
* **Locality preservation** of each ordering: correlation between
  distance-in-order and distance-in-(η, φ).
* Throughput/memory of `torch` vs `fla` paths at 6k vs 60k points.

## Key references

* Katharopoulos et al., *Transformers are RNNs* (linear attention), 2020.
* Yang et al., *DeltaNet* (2406.06484), 2024; *Gated DeltaNet*, 2024/25.
* Gu & Dao, *Mamba* / *Mamba-2*, 2023/24 (selective/decaying state).
* Miao et al., *HEPT* (2402.12535), ICML 2024 (LSH OR&AND locality).
* Wu et al., *Point Transformer V3*, 2023 (serialization via Z-order/Hilbert).
* Sun et al., *FlatFormer*; Wang et al., *OctFormer* (window attention on points).
