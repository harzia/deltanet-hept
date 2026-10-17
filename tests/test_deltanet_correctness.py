"""Validate the chunkwise delta rule against the brute-force recurrence.

Run:  .venv/bin/python tests/test_deltanet_correctness.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.attention.deltanet import (  # noqa: E402
    delta_rule_recurrence,
    delta_rule_chunkwise,
    DeltaNetAttention,
    compute_sort_perm,
)


def test_chunkwise_matches_recurrence():
    torch.manual_seed(0)
    B, L, D, Dv = 3, 128, 16, 16
    q = torch.randn(B, L, D)
    k = torch.nn.functional.normalize(torch.randn(B, L, D), p=2, dim=-1)  # bounded keys
    v = torch.randn(B, L, Dv)
    beta = torch.rand(B, L)

    ref = delta_rule_recurrence(q, k, v, beta)
    for cs in (16, 32, 64):
        out = delta_rule_chunkwise(q, k, v, beta, chunk_size=cs)
        err = (ref - out).abs().max().item()
        print(f"  chunk_size={cs:3d}  max_abs_err={err:.2e}")
        assert err < 1e-4, f"chunkwise mismatch at chunk_size={cs}: {err}"
    print("PASS: chunkwise == recurrence")


def test_sort_perm_roundtrip():
    torch.manual_seed(1)
    N = 257
    coords = torch.randn(N, 6)
    pt = torch.rand(N)
    for key in ("original", "eta", "phi", "r", "pt", "kt"):
        perm, inv = compute_sort_perm(coords, pt, key)
        x = torch.randn(N, 4)
        assert torch.allclose(x[perm][inv], x), f"roundtrip failed for {key}"
    print("PASS: sort perm roundtrip for all keys")


def test_layer_shapes_and_padding():
    torch.manual_seed(2)
    N, H, Dh = 100, 4, 8  # N not a multiple of chunk_size -> exercises padding
    kwargs = dict(h_dim=Dh, num_heads=H, chunk_size=32, deltanet_direction="bi",
                  deltanet_sort_key="eta", deltanet_impl="torch")
    layer = DeltaNetAttention(**kwargs)
    q = torch.randn(N, H * Dh)
    k = torch.randn(N, H * Dh)
    v = torch.randn(N, H * Dh)
    coords = torch.randn(N, 6)
    out = layer(q, k, v, coords=coords, pt=torch.rand(N))
    assert out.shape == (N, Dh), out.shape
    assert torch.isfinite(out).all()
    print("PASS: layer output shape", tuple(out.shape), "with internal padding")


if __name__ == "__main__":
    test_chunkwise_matches_recurrence()
    test_sort_perm_roundtrip()
    test_layer_shapes_and_padding()
    print("\nALL TESTS PASSED")
