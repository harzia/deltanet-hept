"""The HEPT Gaussian kernel inside DeltaNet's delta-rule update.

The exact dual form is quadratic in event length. Paired random Fourier
features provide a bounded, unit-norm finite feature map for a linear-size
recurrent state. Neither path computes HEPT's normalized LSH attention.
"""
import math
import torch
from torch import nn


def gaussian_kernel(query, key):
    """exp(-||query-key||^2 / 2), with [heads, points, augmented_dim]."""
    scores = query @ key.transpose(-1, -2)
    scores = scores - query.square().sum(-1, keepdim=True) / 2
    scores = scores - key.square().sum(-1).unsqueeze(-2) / 2
    return scores.clamp(max=0).exp()


def kernel_delta_rule(query, key, value, beta):
    """Exact Gaussian RKHS delta rule (no query dimension scaling).

    u_t = beta_t * (v_t - sum_{j<t} kernel(k_t,k_j) * u_j)
    o_t = sum_{j<=t} kernel(q_t,k_j) * u_j

    Solve the causal system for residual writes u in parallel. This is the
    dual representation of the delta-rule state, not ordinary kernel attention.
    """
    kk = gaussian_kernel(key, key)
    qk = gaussian_kernel(query, key)
    system = torch.tril(beta.unsqueeze(-1) * kk, diagonal=-1)
    system = system + torch.eye(key.shape[-2], device=key.device, dtype=key.dtype)
    writes = torch.linalg.solve_triangular(system, beta.unsqueeze(-1) * value, upper=False)
    return torch.tril(qk) @ writes


class GaussianFourierFeatures(nn.Module):
    """phi(x)=[cos(xW),sin(xW)]/sqrt(m), W columns iid N(0,I).

    E[phi(x).phi(y)] = exp(-||x-y||^2/2). Every realized phi(x) has
    norm one, preserving the stable unit-key delta update. Finite features
    may yield negative off-diagonal approximations; that is documented.
    """
    def __init__(self, num_heads, input_dim, feature_dim=128, seed=0):
        super().__init__()
        if feature_dim < 2 or feature_dim % 2:
            raise ValueError("deltanet_kernel_features must be an even integer >= 2")
        self.feature_dim = feature_dim
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("frequencies", torch.randn(num_heads, input_dim, feature_dim // 2,
                                                        generator=generator))

    def forward(self, x):
        angles = x @ self.frequencies
        return torch.cat([angles.cos(), angles.sin()], dim=-1) / math.sqrt(self.feature_dim // 2)
