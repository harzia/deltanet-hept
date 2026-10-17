"""HEPT hash-selected, exact Gaussian delta updates in independent buckets.

Hash tables select query/key blocks as in HEPT. Each key block owns its own
memory; residual writes are causal in the configured event order. Query reads
use event ranks, never query/key positions in their different hash sorts.
Tables are averaged; there is no attention normalization or Fourier map.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .hept import get_geo_shift, sort_to_buckets, unsort_from_buckets
from .hept_delta_kernel import gaussian_kernel
from ..model_utils.hash_utils import (
    E2LSH, get_regions, invert_permutation, lsh_mapping, quantile_partition,
)


def bucket_kernel_delta_rule(query, key, value, beta, query_ranks,
                             key_ranks, key_valid, direction="bi"):
    """Exact per-bucket recurrence; keys must be sorted by increasing rank.

    Inputs have arbitrary batch dimensions followed by [points, dim] (ranks,
    validity and beta omit dim). Queries need not have the same order as keys.
    Padding has zero beta and is excluded from reads in both directions.
    """
    kk = gaussian_kernel(key, key)
    qk = gaussian_kernel(query, key)
    beta = beta * key_valid.to(beta.dtype)
    identity = torch.eye(key.shape[-2], device=key.device, dtype=key.dtype)

    def writes(kernel, gates, values):
        system = torch.tril(gates.unsqueeze(-1) * kernel, diagonal=-1) + identity
        return torch.linalg.solve_triangular(system, gates.unsqueeze(-1) * values, upper=False)

    forward_writes = writes(kk, beta, value)
    forward_mask = (key_ranks.unsqueeze(-2) <= query_ranks.unsqueeze(-1)) & key_valid.unsqueeze(-2)
    output = (qk * forward_mask.to(qk.dtype)) @ forward_writes
    if direction == "bi":
        reverse_writes = writes(kk.flip((-2, -1)), beta.flip(-1), value.flip(-2)).flip(-2)
        reverse_mask = (key_ranks.unsqueeze(-2) >= query_ranks.unsqueeze(-1)) & key_valid.unsqueeze(-2)
        output = output + (qk * reverse_mask.to(qk.dtype)) @ reverse_writes
    elif direction != "uni":
        raise ValueError("direction must be uni or bi")
    return output


class HashBucketGaussianDelta(nn.Module):
    """Parallel local memories with O(R*H*N*B) kernel entries, not N squared.

    Work groups bound temporary matrix allocations. Optional recomputation
    avoids retaining each group's dense kernels for backward. Fixed hash
    projections and coordinate-region parameters are stored in checkpoints.
    """
    def __init__(self, num_heads, input_dim, block_size=100, n_hashes=3,
                 num_regions=15, chunk_buckets=64, checkpoint_buckets=True):
        super().__init__()
        for name, size in (("block_size", block_size), ("n_hashes", n_hashes),
                           ("chunk_buckets", chunk_buckets), ("num_heads", num_heads)):
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_regions < 1:
            raise ValueError("num_regions must be positive")
        self.block_size = block_size
        self.n_hashes = n_hashes
        self.chunk_buckets = chunk_buckets
        self.checkpoint_buckets = checkpoint_buckets
        self.e2lsh = E2LSH(n_hashes, num_heads, input_dim)
        self.register_buffer("regions", get_regions(num_regions, n_hashes, num_heads))

    @torch.no_grad()
    def bucket_positions(self, query, key, coords, order_ranks):
        """Return [tables, heads, padded_points] query/key permutations.

        Membership follows HEPT's separate query/key hash sorts and geographic
        shifts. Reordering keys *within* a block preserves that membership.
        Quantiles and hash ranges see real points only.
        """
        n = query.shape[-2]
        q_hash, k_hash, shift = lsh_mapping(self.e2lsh, query, key)
        regions_h = rearrange(self.regions, "c a h -> a (c h)")
        region_indices = [quantile_partition(coords[:, axis].argsort(stable=True), regions_h[axis, :, None])
                          for axis in range(2)]
        q_shift, k_shift = get_geo_shift(regions_h, rearrange(shift, "c h d -> (c h) d"),
                                       region_indices, self.n_hashes)
        pad = (-n) % self.block_size
        q_positions = F.pad(q_hash + q_shift, (0, pad), value=float("inf")).argsort(dim=-1, stable=True)
        k_positions = F.pad(k_hash + k_shift, (0, pad), value=float("inf")).argsort(dim=-1, stable=True)
        ranks = F.pad(order_ranks, (0, pad), value=n)
        keys = rearrange(k_positions, "c h (b s) -> c h b s", s=self.block_size)
        local_order = ranks[keys].argsort(dim=-1, stable=True)
        k_positions = rearrange(keys.gather(-1, local_order), "c h b s -> c h (b s)")
        return q_positions, k_positions

    def forward(self, query, key, value, beta, coords, order_ranks, direction="bi"):
        n = query.shape[-2]
        if n < 1:
            raise ValueError("hash Gaussian DeltaNet needs a nonempty event")
        q_positions, k_positions = self.bucket_positions(query, key, coords, order_ranks)
        pad = (-n) % self.block_size
        query, key, value = [F.pad(x, (0, 0, 0, pad)) for x in (query, key, value)]
        beta = F.pad(beta, (0, pad))
        ranks = F.pad(order_ranks, (0, pad), value=n)
        s_query, s_key, s_value = [sort_to_buckets(x, positions, self.block_size)
                                  for x, positions in ((query, q_positions), (key, k_positions), (value, k_positions))]
        s_beta = sort_to_buckets(beta.unsqueeze(-1), k_positions, self.block_size).squeeze(-1)
        q_ranks = rearrange(ranks[q_positions], "c h (b s) -> c h b s", s=self.block_size)
        k_ranks = rearrange(ranks[k_positions], "c h (b s) -> c h b s", s=self.block_size)
        valid = rearrange(k_positions < n, "c h (b s) -> c h b s", s=self.block_size)
        table_outputs = []
        for table in range(self.n_hashes):
            groups = []
            for start in range(0, s_query.shape[2], self.chunk_buckets):
                sl = slice(start, start + self.chunk_buckets)
                inputs = (s_query[table, :, sl], s_key[table, :, sl], s_value[table, :, sl],
                          s_beta[table, :, sl], q_ranks[table, :, sl], k_ranks[table, :, sl], valid[table, :, sl])
                if self.checkpoint_buckets and torch.is_grad_enabled() and any(x.requires_grad for x in inputs):
                    group = checkpoint(bucket_kernel_delta_rule, *inputs, direction=direction, use_reentrant=False)
                else:
                    group = bucket_kernel_delta_rule(*inputs, direction=direction)
                groups.append(group)
            table_outputs.append(torch.cat(groups, dim=1))
        output = unsort_from_buckets(torch.stack(table_outputs), invert_permutation(q_positions))
        return output[..., :n, :].mean(dim=0)
