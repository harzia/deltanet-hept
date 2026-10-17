"""DeltaNet token-mixer for the HEPT backbone.

This module implements the DeltaNet linear-attention layer
(Yang, Wang, Zhang, Kim, Shen, Gu; "Parallelizing Linear Transformers with
the Delta Rule over Sequence Length", https://arxiv.org/abs/2406.06484) as a
*drop-in replacement* for ``HEPTAttention`` inside the HEPT point-transformer
backbone (Miao et al., ICML'24, https://arxiv.org/abs/2402.12535).

Design goals
------------
1. **Same interface as ``HEPTAttention``.** ``forward(query, key, value, **kwargs)``
   consumes ``q/k/v`` of shape ``[N, num_heads * dim_per_head]`` (a single point
   cloud, ``batch.max() == 0``) and returns ``[N, dim_per_head]``. Everything
   around it in ``Attn``/``Transformer`` (pre-norm, residuals, feed-forward,
   layer concatenation, task heads, losses, preprocessing) is untouched, so this
   is an apples-to-apples swap of *only* the token-mixing mechanism.

2. **Permutation is not equivariant -> we impose an order.** DeltaNet is a
   causal linear RNN; a point cloud has no intrinsic order. We therefore *sort*
   the points by a configurable geometric/kinematic key (eta, phi, pt, r, kt,
   ...) to inject locality, scan the delta rule along that order, then invert the
   permutation so the output is back in the original point order. This is the
   "easy" locality mechanism; see ``docs/LOCALITY.md`` for the research agenda.

3. **Runs on CPU (dev) and GPU (training).** The math is a pure-PyTorch
   chunkwise implementation (parallel within chunk, recurrent across chunks).
   A ``fla`` (flash-linear-attention) Triton fast path is used automatically
   when ``impl="auto"`` and CUDA + the ``fla`` package are available; otherwise
   we fall back to the pure-PyTorch path with identical numerics contract.

4. **Verifiable.** ``delta_rule_chunkwise`` is checked against the
   ``delta_rule_recurrence`` brute-force reference in
   ``tests/test_deltanet_correctness.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .hept import prep_qk
from .hept_delta_kernel import GaussianFourierFeatures, kernel_delta_rule
from .hept_delta_hash import HashBucketGaussianDelta


# ---------------------------------------------------------------------------
# Point ordering (locality via sorting)
# ---------------------------------------------------------------------------
# All sort keys are per-point scalars computed from quantities the model already
# has access to. ``coords`` is ``[N, coords_dim]`` where, by the HEPT transforms,
# ``coords[:, 0] == eta`` and ``coords[:, 1] == phi`` for both tracking and
# pileup. ``pt`` (transverse momentum) is optional and threaded through kwargs.
#
# NOTE on leakage: for the *tracking* task ``pt`` is also used to define the
# evaluation point filter. Ordering the sequence by ``pt``/``kt`` feeds a weak
# signal about pt into the model. In simulated tracking, true pt may be ground-truth metadata
# unavailable before reconstruction. Using it risks information leakage and is
# a different experimental setup than HEPT, which does not consume pt. Default sort key is therefore ``eta`` (pure geometry); pt-based
# keys are opt-in and flagged in the config/logs.

def _sort_key_values(coords: torch.Tensor, pt, sort_key: str) -> torch.Tensor:
    """Return a per-point scalar ``[N]`` to sort by (ascending)."""
    eta = coords[..., 0]
    phi = coords[..., 1]
    if sort_key == "original":
        # identity order (row index) -> stable no-op sort
        return torch.arange(coords.shape[0], device=coords.device, dtype=coords.dtype)
    if sort_key == "eta":
        return eta
    if sort_key == "phi":
        return phi
    if sort_key == "r":  # distance from origin in the (eta, phi) plane
        return torch.sqrt(eta ** 2 + phi ** 2)
    if sort_key == "pt":
        if pt is None:
            raise ValueError("sort_key='pt' requires pt to be provided in kwargs")
        return pt.reshape(-1).to(coords.dtype)
    if sort_key == "kt":  # kt = pt * r  (r used as a delta_r-from-origin proxy)
        if pt is None:
            raise ValueError("sort_key='kt' requires pt to be provided in kwargs")
        r = torch.sqrt(eta ** 2 + phi ** 2)
        return pt.reshape(-1).to(coords.dtype) * r
    raise ValueError(f"unknown sort_key: {sort_key}")


VALID_SORT_KEYS = ("original", "eta", "phi", "r", "pt", "kt")


def compute_sort_perm(coords: torch.Tensor, pt, sort_key: str):
    """Return ``(perm, inv_perm)`` index tensors of shape ``[N]``.

    ``x[perm]`` sorts the points; ``sorted_x[inv_perm]`` restores the order.
    """
    key = _sort_key_values(coords, pt, sort_key)
    perm = torch.argsort(key, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
    return perm, inv_perm


# ---------------------------------------------------------------------------
# Delta rule math (pure PyTorch)
# ---------------------------------------------------------------------------
# Convention (matches fla/ops/delta_rule/naive.py):
#   q, k, v : [B, L, D]      (B is the head/batch axis, L the sequence length)
#   beta    : [B, L]         per-position writing strength in (0, 1)
#   q is scaled by D**-0.5; the state S is [B, D, Dv]; output is [B, L, Dv].


def delta_rule_recurrence(q, k, v, beta):
    """Brute-force O(L) reference implementation of the delta rule.

    State update:  S_t = S_{t-1} + k_t (beta_t (v_t - S_{t-1}^T k_t))^T
    Output:        o_t = S_t^T q_t
    This is the ground truth the chunkwise kernel is validated against.
    """
    b, l, d = q.shape
    d_v = v.shape[-1]
    q = q * (d ** -0.5)
    o = torch.zeros(b, l, d_v, device=q.device, dtype=q.dtype)
    S = q.new_zeros(b, d, d_v)
    for i in range(l):
        k_i = k[:, i]                       # [b, d]
        q_i = q[:, i]                       # [b, d]
        v_i = v[:, i].clone()               # [b, d_v]
        beta_i = beta[:, i]                 # [b]
        # error-corrected value: v_i - S^T k_i, scaled by beta
        v_i = (v_i - (S * k_i[..., None]).sum(-2)) * beta_i[..., None]
        S = S + k_i[..., None] * v_i[..., None, :]
        o[:, i] = (q_i[..., None] * S).sum(-2)
    return o


def delta_rule_chunkwise(q, k, v, beta, chunk_size: int = 32):
    """Chunkwise-parallel delta rule (the training path).

    Splits the length into chunks of ``chunk_size``: intra-chunk interactions are
    computed in parallel via the WY-style representation of the product of
    Householder-like updates ``(I - tril(diag(beta) K K^T, -1))^{-1}``; the
    ``[D, Dv]`` state is carried recurrently across chunks. Length must be a
    multiple of ``chunk_size`` (the caller pads).
    """
    b, l, d = q.shape
    d_v = v.shape[-1]
    assert l % chunk_size == 0, f"length {l} not a multiple of chunk_size {chunk_size}"
    c = chunk_size
    n = l // c

    q = q * (d ** -0.5)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]

    q, k, v, k_beta = (rearrange(x, "b (n c) d -> b n c d", c=c) for x in (q, k, v, k_beta))

    # Invert (I - M) where M = tril(diag(beta) K K^T, -1) (strictly lower) via
    # forward substitution -> attn = (I - M)^{-1}.
    tri_strict = torch.tril(torch.ones(c, c, dtype=torch.bool, device=q.device), diagonal=-1)
    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(~tri_strict, 0)
    for i in range(1, c):
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone() * attn[..., :, :i].clone()).sum(-2)
    attn = attn + torch.eye(c, dtype=q.dtype, device=q.device)

    u = attn @ v          # pseudo-values ("new value" targets)
    w = attn @ k_beta     # pseudo-keys used to remove the running state

    S = q.new_zeros(b, d, d_v)
    o = torch.zeros(b, n, c, d_v, device=q.device, dtype=q.dtype)
    causal_strict = torch.triu(torch.ones(c, c, dtype=torch.bool, device=q.device), diagonal=1)
    for idx in range(n):
        q_i, k_i = q[:, idx], k[:, idx]
        u_i = u[:, idx] - w[:, idx] @ S            # correct pseudo-values by prior state
        o_inter = q_i @ S                          # contribution from previous chunks
        a_i = (q_i @ k_i.transpose(-1, -2)).masked_fill(causal_strict, 0)  # causal, incl. diagonal
        o_intra = a_i @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i        # advance the state
        o[:, idx] = o_intra + o_inter
    return rearrange(o, "b n c d -> b (n c) d")


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

class DeltaNetAttention(nn.Module):
    """DeltaNet token-mixer with the same I/O contract as ``HEPTAttention``.

    kwargs (from the yaml ``model_kwargs``):
        h_dim, num_heads            : as in the rest of the backbone
        chunk_size (int, 32)        : chunkwise kernel chunk length
        deltanet_sort_key (str)     : one of VALID_SORT_KEYS (locality order)
        deltanet_direction (str)    : "uni" (causal) or "bi" (forward+backward)
        deltanet_qk_norm (str)      : "l2" or "none"
        deltanet_qk_activation (str): "silu", "relu", "identity"
        deltanet_use_beta (bool)    : learn per-point beta (else beta=1)
        deltanet_allow_neg_eigval   : scale beta by 2 (allows negative eigenvalues)
        deltanet_impl (str)         : "auto" | "torch" | "fla"
        deltanet_kernel (str)       : "standard" | "hept"
        deltanet_hept_backend (str) : "rff" | "exact" | "hash" (local exact buckets)
        deltanet_kernel_features   : even total Fourier feature dimension
        deltanet_kernel_seed       : fixed frequency bank seed, saved in state_dict
    """

    def __init__(self, coords_dim=None, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.chunk_size = kwargs.get("chunk_size", 32)
        self.sort_key = kwargs.get("deltanet_sort_key", "eta")
        self.direction = kwargs.get("deltanet_direction", "bi")
        self.qk_norm = kwargs.get("deltanet_qk_norm", "l2")
        self.qk_activation = kwargs.get("deltanet_qk_activation", "silu")
        self.use_beta = kwargs.get("deltanet_use_beta", True)
        self.allow_neg_eigval = kwargs.get("deltanet_allow_neg_eigval", False)
        self.impl = kwargs.get("deltanet_impl", "auto")
        self.kernel = kwargs.get("deltanet_kernel", "standard")
        self.hept_backend = kwargs.get("deltanet_hept_backend", "rff")
        self.kernel_features = kwargs.get("deltanet_kernel_features", 128)
        self.num_w_per_dist = kwargs.get("num_w_per_dist", 10)
        self.exact_max_points = kwargs.get("deltanet_hept_exact_max_points", 1024)
        if self.kernel not in ("standard", "hept"):
            raise ValueError("deltanet_kernel must be standard or hept")
        if self.hept_backend not in ("rff", "exact", "hash"):
            raise ValueError("deltanet_hept_backend must be rff, exact or hash")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.kernel == "hept":
            if coords_dim is None or coords_dim < 2:
                raise ValueError("HEPT-kernel DeltaNet needs coords_dim >= 2")
            if self.impl != "torch":
                raise ValueError("HEPT-kernel DeltaNet currently requires deltanet_impl=torch")
            # SiLU/ReLU or L2 normalization of augmented q/k changes the target
            # Gaussian. RFF itself is already unit norm; disallow silent changes.
            if self.qk_activation != "identity" or self.qk_norm != "none":
                raise ValueError("HEPT kernel requires deltanet_qk_activation=identity and deltanet_qk_norm=none")
            if self.hept_backend == "rff":
                self.kernel_map = GaussianFourierFeatures(self.num_heads, self.dim_per_head + coords_dim,
                                                          self.kernel_features,
                                                          kwargs.get("deltanet_kernel_seed", 0))
            elif self.hept_backend == "hash":
                self.hash_delta = HashBucketGaussianDelta(
                    self.num_heads, self.dim_per_head + coords_dim,
                    block_size=kwargs.get("block_size", 100), n_hashes=kwargs.get("n_hashes", 3),
                    num_regions=kwargs.get("num_regions", 15),
                    chunk_buckets=kwargs.get("deltanet_hash_chunk_buckets", 64),
                    checkpoint_buckets=kwargs.get("deltanet_hash_checkpoint", True))

        assert self.sort_key in VALID_SORT_KEYS, f"bad sort_key {self.sort_key}"
        assert self.direction in ("uni", "bi")

        # per-head scalar beta from the (projected) value stream
        if self.use_beta:
            self.b_proj = nn.Linear(self.dim_per_head, 1)

        # output projection: concat heads -> h_dim (matches HEPTAttention.out_linear)
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

    # -- feature maps -------------------------------------------------------
    def _act(self, x):
        if self.qk_activation == "silu":
            return F.silu(x)
        if self.qk_activation == "relu":
            return F.relu(x)
        return x

    def _norm_qk(self, x):
        x = self._act(x)
        if self.qk_norm == "l2":
            x = F.normalize(x, p=2, dim=-1)
        return x

    # -- one directional scan ----------------------------------------------
    def _scan(self, q, k, v, beta):
        """q,k,v: [H, N, Dh]; beta: [H, N]. Pads to chunk_size, runs kernel."""
        h, n, d = q.shape
        if self.kernel == "hept" and self.hept_backend == "exact":
            if n > self.exact_max_points:
                raise ValueError(f"Exact Gaussian DeltaNet is quadratic: {n} points exceeds "
                                 f"deltanet_hept_exact_max_points={self.exact_max_points}; use hash or rff for full events")
            return kernel_delta_rule(q, k, v, beta)
        pad = (-n) % self.chunk_size
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            beta = F.pad(beta, (0, pad))
        use_fla = (
            self.impl in ("auto", "fla")
            and q.is_cuda
            and _fla_available()
        )
        if use_fla:
            o = _fla_chunk(q, k, v, beta, self.chunk_size)
        else:
            if self.impl == "fla":
                raise RuntimeError("deltanet_impl='fla' requested but fla/CUDA unavailable")
            o = delta_rule_chunkwise(q, k, v, beta, self.chunk_size)
        return o[:, :n]

    def forward(self, query, key, value, **kwargs):
        # [N, H*Dh] -> [H, N, Dh]
        q = rearrange(query, "n (h d) -> h n d", h=self.num_heads)
        k = rearrange(key, "n (h d) -> h n d", h=self.num_heads)
        v = rearrange(value, "n (h d) -> h n d", h=self.num_heads)

        if self.kernel == "hept":
            rpe = kwargs.get("w_rpe")
            if rpe is None:
                raise ValueError("HEPT-kernel DeltaNet needs the backbone w_rpe module")
            w = rearrange(rpe.weight, "(h d) (r k) -> h d r k", h=self.num_heads,
                          d=self.dim_per_head, k=self.num_w_per_dist)
            q, k = prep_qk(rearrange(q, "h n d -> n h d"), rearrange(k, "h n d -> n h d"),
                            w, kwargs["coords"])
            q, k = [rearrange(a, "n h d -> h n d") for a in (q,k)]
            if self.hept_backend == "rff":
                q, k = self.kernel_map(q), self.kernel_map(k)
                # Existing delta kernels scale queries by D^-0.5. Cancel that
                # only here so effective q.k approximates the HEPT kernel itself.
                q = q * (self.kernel_features ** 0.5)
        else:
            q = self._norm_qk(q)
            k = self._norm_qk(k)

        if self.use_beta:
            beta = self.b_proj(v).squeeze(-1).sigmoid()        # [H, N]
            if self.allow_neg_eigval:
                beta = beta * 2.0
        else:
            beta = torch.ones(q.shape[0], q.shape[1], device=q.device, dtype=q.dtype)

        # locality: sort points into a 1-D order
        coords = kwargs["coords"]
        pt = kwargs.get("pt", None)
        perm, inv_perm = compute_sort_perm(coords, pt, self.sort_key)
        if self.kernel == "hept" and self.hept_backend == "hash":
            # inv_perm gives each original point its causal rank. Query and key
            # bucket permutations differ, so a positional triangle is invalid.
            o = self.hash_delta(q, k, v, beta, coords, inv_perm, self.direction)
            return self.out_linear(rearrange(o, "h n d -> n (h d)"))
        qs, ks, vs, betas = q[:, perm], k[:, perm], v[:, perm], beta[:, perm]

        o = self._scan(qs, ks, vs, betas)
        if self.direction == "bi":
            # backward scan over the same order (flip -> scan -> flip back)
            o_bwd = self._scan(qs.flip(1), ks.flip(1), vs.flip(1), betas.flip(1)).flip(1)
            o = o + o_bwd

        o = o[:, inv_perm]                                     # restore original order
        out = self.out_linear(rearrange(o, "h n d -> n (h d)"))
        return out


# ---------------------------------------------------------------------------
# Optional fla (Triton) fast path
# ---------------------------------------------------------------------------
_FLA_OK = None


def _fla_available():
    global _FLA_OK
    if _FLA_OK is None:
        try:
            from fla.ops.delta_rule import chunk_delta_rule  # noqa: F401
            _FLA_OK = True
        except Exception:
            _FLA_OK = False
    return _FLA_OK


def _fla_chunk(q, k, v, beta, chunk_size):
    """Call fla's chunk_delta_rule. fla expects [B, H, L, D]; we have [H, L, D]."""
    from fla.ops.delta_rule import chunk_delta_rule

    q4 = q.unsqueeze(0)
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    beta3 = beta.unsqueeze(0)
    o, _ = chunk_delta_rule(q4, k4, v4, beta3, chunk_size=chunk_size, head_first=True)
    return o.squeeze(0)
