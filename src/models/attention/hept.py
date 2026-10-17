import torch
import torch.nn as nn
from typing import List

from einops import rearrange
from ..model_utils.hash_utils import lsh_mapping, batched_index_select, invert_permutation, E2LSH


def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )


def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)


def qkv_res(s_query, s_key, s_value, key_mask=None, *, normalized=False):
    """Gaussian block attention; optionally return log mass and normalized output.

    The log-space form lets OR tables merge by their true kernel mass, even
    when all unshifted Gaussian weights would underflow. Padding is excluded
    from both the numerator and denominator.
    """
    if s_query.dtype in (torch.float16, torch.bfloat16):
        s_query, s_key, s_value = s_query.float(), s_key.float(), s_value.float()
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    scores = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0)
    if key_mask is not None:
        scores = scores.masked_fill(~key_mask.unsqueeze(-2), float("-inf"))
    if normalized:
        log_mass = torch.logsumexp(scores, dim=-1, keepdim=True)
        # Also define fully masked blocks as zero, rather than NaN.
        finite = torch.isfinite(log_mass)
        safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
        weights = safe_scores.softmax(dim=-1) * finite.to(scores.dtype)
        return log_mass, torch.einsum("...ij,...jd->...id", weights, s_value)
    clustered_dists = scores.exp()

    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists

    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so


def prep_qk(query, key, w, coords):
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    new_qw_expand_dim = torch.cat([qw[:, :1], qw], dim=-1)

    sqrt_w_r = torch.sqrt(2 * new_qw_expand_dim)[None] * coords[:, None]
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)
    return q_hat, k_hat


@torch.no_grad()
def get_geo_shift(regions_h: List[List[int]], hash_shift, region_indices, num_or_hashes):
    region_indices_eta, region_indices_phi = region_indices

    q_hash_shift_eta = region_indices_eta * hash_shift
    k_hash_shift_eta = region_indices_eta * hash_shift

    q_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    k_hash_shift_phi = region_indices_phi * hash_shift * (torch.ceil(regions_h[0][:, None]) + 1)
    res = torch.stack([q_hash_shift_phi + q_hash_shift_eta, k_hash_shift_phi + k_hash_shift_eta], dim=0)
    return rearrange(res, "a (c h) n -> a c h n", c=num_or_hashes)


class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        if min(self.block_size, self.n_hashes, self.num_w_per_dist, self.num_heads, self.dim_per_head) < 1:
            raise ValueError("HEPT dimensions, block_size, n_hashes and num_w_per_dist must be positive")
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, **kwargs):
        # One event per forward; the backbone enforces batch_size=1.
        raw_size = int(kwargs["raw_size"])
        if not 0 < raw_size <= query.shape[0] or query.shape[0] % self.block_size:
            raise ValueError("HEPT needs a nonempty event padded to block_size")
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")
        valid = torch.arange(query.shape[0], device=query.device) < raw_size
        q_hat = q_hat.masked_fill(~valid[None, :, None], 0.0)
        k_hat = k_hat.masked_fill(~valid[None, :, None], 0.0)
        value = value.masked_fill(~valid[None, :, None], 0.0)

        # Dummy points must not change the hash range used to separate regions.
        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat[..., :raw_size, :], k_hat[..., :raw_size, :])
        if raw_size < query.shape[0]:
            pad = query.shape[0] - raw_size
            q_hashed = torch.nn.functional.pad(q_hashed, (0, pad), value=float("inf"))
            k_hashed = torch.nn.functional.pad(k_hashed, (0, pad), value=float("inf"))
        hash_shift = rearrange(hash_shift, "c h d -> (c h) d")
        q_hashed[..., kwargs["raw_size"] :] = float("inf")
        k_hashed[..., kwargs["raw_size"] :] = float("inf")

        q_shifts, k_shifts = get_geo_shift(kwargs["regions_h"], hash_shift, kwargs["region_indices"], self.n_hashes)

        q_hashed = q_hashed + q_shifts
        k_hashed = k_hashed + k_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        s_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        s_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        s_value = sort_to_buckets(value, k_positions, self.block_size)

        s_valid = rearrange(valid[k_positions], "c h (b s) -> c h b s", s=self.block_size)
        log_mass, so = qkv_res(s_query, s_key, s_value, s_valid, normalized=True)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(log_mass, q_rev_positions)
        # Sec. 4.2: sum numerators / sum denominators across OR tables.
        # An unweighted average of the individual table outputs is incorrect.
        out = (o * logits.softmax(dim=0)).sum(dim=0).to(query.dtype)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        return out
