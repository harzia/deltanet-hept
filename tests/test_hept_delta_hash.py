"""Independent recurrence and integration checks for hash-local DeltaNet."""
import sys
from pathlib import Path

import pytest
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.attention.deltanet import DeltaNetAttention
from models.attention.hept_delta_hash import HashBucketGaussianDelta, bucket_kernel_delta_rule
from models.attention.hept import get_geo_shift
from models.model_utils.hash_utils import lsh_mapping, quantile_partition


def sequential_bucket(q, k, v, beta, qr, kr, valid, direction):
    """Explicit residual writes and rank-based reads, with direct distances."""
    kk = torch.exp(-0.5 * (k.unsqueeze(-2) - k.unsqueeze(-3)).square().sum(-1))
    qk = torch.exp(-0.5 * (q.unsqueeze(-2) - k.unsqueeze(-3)).square().sum(-1))
    output = torch.zeros(*q.shape[:-1], v.shape[-1], dtype=q.dtype, device=q.device)
    for reverse in ([False, True] if direction == "bi" else [False]):
        indices = list(range(k.shape[-2]))
        if reverse:
            indices.reverse()
        writes = []
        for t in indices:
            prediction = torch.zeros_like(v[..., t, :])
            for j, u in writes:
                prediction = prediction + kk[..., t, j, None] * u
            u = (beta[..., t] * valid[..., t])[..., None] * (v[..., t, :] - prediction)
            writes.append((t, u))
            mask = (qr >= kr[..., t, None]) if not reverse else (qr <= kr[..., t, None])
            mask = mask & valid[..., t, None]
            output = output + (qk[..., :, t] * mask)[..., None] * u.unsqueeze(-2)
    return output


@pytest.mark.parametrize("direction", ["uni", "bi"])
def test_bucket_recurrence_and_gradients_with_separate_query_order(direction):
    torch.manual_seed(17)
    q = torch.randn(2, 3, 5, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 6, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 6, 2, dtype=torch.float64, requires_grad=True)
    beta = torch.rand(2, 3, 6, dtype=torch.float64, requires_grad=True)
    qr = torch.tensor([7, 1, 5, 0, 3]).expand(2, 3, -1)
    kr = torch.tensor([0, 2, 4, 6, 7, 9]).expand(2, 3, -1)
    valid = torch.tensor([True, True, True, True, True, False]).expand(2, 3, -1)
    actual = bucket_kernel_delta_rule(q, k, v, beta, qr, kr, valid, direction)
    expected = sequential_bucket(q, k, v, beta, qr, kr, valid, direction)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), (q, k, v, beta), retain_graph=True),
                    torch.autograd.grad(expected.square().sum(), (q, k, v, beta))):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    # Dummy writes cannot affect outputs even if their values are enormous.
    changed = v.detach().clone(); changed[..., -1, :] = 1e12
    torch.testing.assert_close(bucket_kernel_delta_rule(q, k, changed, beta, qr, kr, valid, direction), actual)


@pytest.mark.parametrize("direction", ["uni", "bi"])
def test_one_bucket_matches_full_exact_layer_with_padding(direction):
    torch.manual_seed(3)
    opts = dict(h_dim=4, num_heads=2, coords_dim=3, deltanet_kernel="hept",
                deltanet_impl="torch", deltanet_qk_activation="identity", deltanet_qk_norm="none",
                deltanet_sort_key="eta", deltanet_direction=direction, num_w_per_dist=2,
                block_size=12, n_hashes=3, num_regions=3)
    full = DeltaNetAttention(deltanet_hept_backend="exact", **opts).double()
    local = DeltaNetAttention(deltanet_hept_backend="hash", **opts).double()
    local.load_state_dict(full.state_dict(), strict=False)
    q, k, v = [torch.randn(9, 8, dtype=torch.float64, requires_grad=True) * 0.2 for _ in range(3)]
    coords = torch.randn(9, 3, dtype=torch.float64, requires_grad=True) * 0.1
    rpe = torch.nn.Linear(4, 8).double()
    a = full(q, k, v, coords=coords, w_rpe=rpe)
    b = local(q, k, v, coords=coords, w_rpe=rpe)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    inputs = (q, k, v, coords, rpe.weight)
    for ga, gb in zip(torch.autograd.grad(a.square().sum(), inputs, retain_graph=True),
                      torch.autograd.grad(b.square().sum(), inputs)):
        torch.testing.assert_close(ga, gb, atol=1e-11, rtol=1e-9)


def test_hash_membership_matches_hept_and_multibucket_output_reference():
    torch.manual_seed(13)
    model = HashBucketGaussianDelta(2, 5, block_size=4, n_hashes=3, num_regions=4,
                                   chunk_buckets=1, checkpoint_buckets=False).double()
    n = 11
    q, k = [torch.randn(2, n, 5, dtype=torch.float64) * 0.3 for _ in range(2)]
    v = torch.randn(2, n, 3, dtype=torch.float64)
    beta = torch.rand(2, n, dtype=torch.float64)
    coords = torch.randn(n, 3, dtype=torch.float64)
    ranks = coords[:, 0].argsort().argsort()
    qp, kp = model.bucket_positions(q, k, coords, ranks)
    # Independently reconstruct HEPT selection before the local key reorder.
    qh, kh, shift = lsh_mapping(model.e2lsh, q, k)
    rh = rearrange(model.regions, "c a h -> a (c h)")
    regions = [quantile_partition(coords[:, a].argsort(), rh[a, :, None]) for a in range(2)]
    qs, ks = get_geo_shift(rh, rearrange(shift, "c h d -> (c h) d"), regions, 3)
    expected_qp, expected_kp = [torch.nn.functional.pad(x, (0, 1), value=float("inf")).argsort(stable=True)
                               for x in (qh + qs, kh + ks)]
    assert torch.equal(qp, expected_qp)
    assert torch.equal(kp.reshape(3, 2, 3, 4).sort(-1).values,
                       expected_kp.reshape(3, 2, 3, 4).sort(-1).values)
    for direction in ("uni", "bi"):
        expected = torch.zeros(2, n, 3, dtype=q.dtype)
        for table in range(3):
            for head in range(2):
                for bucket in range(3):
                    qi = qp[table, head, bucket*4:(bucket+1)*4]
                    ki = kp[table, head, bucket*4:(bucket+1)*4]
                    qi, ki = qi[qi < n], ki[ki < n]
                    if qi.numel() == 0:
                        continue
                    out = sequential_bucket(q[head, qi], k[head, ki], v[head, ki], beta[head, ki],
                                            ranks[qi], ranks[ki], torch.ones_like(ki, dtype=torch.bool), direction)
                    expected[head, qi] += out / 3
        torch.testing.assert_close(model(q, k, v, beta, coords, ranks, direction), expected,
                                   atol=1e-12, rtol=1e-10)
    # Replicating a table must not multiply the output magnitude.
    with torch.no_grad():
        model.e2lsh.alpha.copy_(model.e2lsh.alpha[..., :1].expand_as(model.e2lsh.alpha))
        model.regions.copy_(model.regions[:1].expand_as(model.regions))
    a = model(q, k, v, beta, coords, ranks)
    single = HashBucketGaussianDelta(2, 5, 4, 1, 4, checkpoint_buckets=False).double()
    with torch.no_grad():
        single.e2lsh.alpha.copy_(model.e2lsh.alpha[..., :1])
        single.regions.copy_(model.regions[:1])
    torch.testing.assert_close(a, single(q, k, v, beta, coords, ranks))


def test_work_groups_checkpoint_gradients_and_config_validation():
    torch.manual_seed(12)
    a = HashBucketGaussianDelta(2, 4, 4, 2, 3, chunk_buckets=1, checkpoint_buckets=True).double()
    b = HashBucketGaussianDelta(2, 4, 4, 2, 3, chunk_buckets=64, checkpoint_buckets=False).double()
    b.load_state_dict(a.state_dict())
    q, k, v = [torch.randn(2, 13, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    beta = torch.rand(2, 13, dtype=torch.float64, requires_grad=True)
    coords = torch.randn(13, 3, dtype=torch.float64)
    ranks = torch.arange(13)
    out_a, out_b = [model(q, k, v, beta, coords, ranks) for model in (a, b)]
    torch.testing.assert_close(out_a, out_b)
    for ga, gb in zip(torch.autograd.grad(out_a.square().sum(), (q, k, v, beta), retain_graph=True),
                      torch.autograd.grad(out_b.square().sum(), (q, k, v, beta))):
        torch.testing.assert_close(ga, gb)
    for bad in (dict(block_size=0), dict(n_hashes=0), dict(chunk_buckets=0), dict(num_regions=0)):
        with pytest.raises(ValueError, match="positive"):
            HashBucketGaussianDelta(2, 4, **bad)


def test_60k_event_uses_only_bucket_sized_kernel_matrices(monkeypatch):
    import models.attention.hept_delta_hash as module
    original = module.gaussian_kernel
    sizes = []
    def bounded_kernel(q, k):
        sizes.append((q.shape[-2], k.shape[-2]))
        assert q.shape[-2] <= 16 and k.shape[-2] <= 16
        return original(q, k)
    monkeypatch.setattr(module, "gaussian_kernel", bounded_kernel)
    torch.manual_seed(1)
    n = 60000
    model = HashBucketGaussianDelta(1, 3, block_size=16, n_hashes=1, num_regions=150)
    q, k = [torch.randn(1, n, 3) for _ in range(2)]
    with torch.no_grad():
        out = model(q, k, torch.randn(1, n, 2), torch.rand(1, n), torch.randn(n, 3), torch.arange(n))
    assert out.shape == (1, n, 2) and torch.isfinite(out).all() and sizes
