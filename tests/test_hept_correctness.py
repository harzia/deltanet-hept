"""Check Gaussian math and the LSH approximation against independent dense math."""
import sys
from pathlib import Path

import pytest
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.attention.hept import HEPTAttention, prep_qk, get_geo_shift, qkv_res
from models.baselines.transformer import Transformer, prepare_input
from models.model_utils.hash_utils import get_regions, lsh_mapping


def setup_inputs(n=11, block=4, hashes=3, dtype=torch.float64):
    torch.manual_seed(10)
    heads, dim, cd = 2, 3, 3
    options = dict(h_dim=dim, num_heads=heads, block_size=block, n_hashes=hashes, num_w_per_dist=2)
    layer = HEPTAttention(dim + cd, **options).to(dtype=dtype)
    rpe = torch.nn.Linear(4, heads * dim).to(dtype=dtype)
    coords = torch.randn(n, cd, dtype=dtype)
    regions = get_regions(4, hashes, heads).to(dtype=dtype)
    x, _, helper = prepare_input(torch.randn(n, heads * dim, dtype=dtype), coords, None,
                                  torch.zeros(n, dtype=torch.long), "hept",
                                  dict(block_size=block, regions=regions))
    q, k, v = [torch.randn_like(x).requires_grad_() for _ in range(3)]
    helper["w_rpe"] = rpe
    return layer, q, k, v, helper


def dense_reference(layer, q, k, v, helper, sparse):
    n = helper["raw_size"]
    w = rearrange(helper["w_rpe"].weight, "(h d) (r k) -> h d r k", h=layer.num_heads,
                  d=layer.dim_per_head, k=layer.num_w_per_dist)
    qa, ka = prep_qk(q.reshape(-1, layer.num_heads, layer.dim_per_head),
                     k.reshape(-1, layer.num_heads, layer.dim_per_head), w, helper["coords"])
    qa, ka = [rearrange(a[:n], "n h d -> h n d") for a in (qa, ka)]
    values = rearrange(v[:n], "n (h d) -> h n d", h=layer.num_heads)
    counts = q.new_zeros(layer.num_heads, n, n)
    if sparse:
        with torch.no_grad():
            qhash, khash, shift = lsh_mapping(layer.e2lsh, qa, ka)
            qs, ks = get_geo_shift(helper["regions_h"], rearrange(shift, "c h d -> (c h) d"),
                                   [a[..., :n] for a in helper["region_indices"]], layer.n_hashes)
            qp, kp = (qhash + qs).argsort(-1), (khash + ks).argsort(-1)
            for t in range(layer.n_hashes):
                for h in range(layer.num_heads):
                    for start in range(0, n, layer.block_size):
                        rows = qp[t, h, start:start+layer.block_size]
                        cols = kp[t, h, start:start+layer.block_size]
                        counts[h, rows[:, None], cols[None, :]] += 1
    else:
        counts.fill_(1)
    # Independent direct squared differences, rather than norm + matmul identity.
    scores = -0.5 * (qa[:, :, None] - ka[:, None, :]).square().sum(-1) + counts.log()
    out = scores.softmax(-1) @ values
    return layer.out_linear(rearrange(out, "h n d -> n (h d)"))


@pytest.mark.parametrize("n,block,hashes", [(5,16,1),(7,16,3),(11,4,3),(1,4,2),(12,4,2)])
def test_dense_and_sparse_reference(n, block, hashes):
    layer, q, k, v, helper = setup_inputs(n, block, hashes)
    actual = layer(q, k, v, **helper)[:n]
    expected = dense_reference(layer, q, k, v, helper, sparse=n > block)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    params = (q, k, v, helper["w_rpe"].weight)
    actual_grads = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), params)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)


def test_padding_does_not_dilute_values():
    a, q, k, v, helper = setup_inputs(n=5, block=16)
    v = torch.ones_like(v)
    expected = a.out_linear(torch.ones(5, a.num_heads * a.dim_per_head, dtype=q.dtype))
    torch.testing.assert_close(a(q,k,v,**helper)[:5], expected)


def test_underflow_and_empty_blocks():
    q = torch.tensor([[[100.0, 0.0]]], requires_grad=True)
    k = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    v = torch.tensor([[[2.0], [4.0]]], requires_grad=True)
    mass, out = qkv_res(q,k,v,normalized=True)
    torch.testing.assert_close(out, torch.tensor([[[2.7550814]]]))
    assert torch.isfinite(mass).all()
    out.sum().backward()
    assert all(torch.isfinite(a.grad).all() for a in (q,k,v))
    mass, out = qkv_res(q,k,v,torch.zeros(1,2,dtype=torch.bool),normalized=True)
    assert torch.isneginf(mass).all() and (out == 0).all()


def test_backbone_checkpoint_and_forward_do_not_change_setting():
    options = dict(num_heads=2,h_dim=8,n_layers=2,block_size=4,n_hashes=2,num_regions=4,
                   pe_type="none",num_w_per_dist=2,use_ckpt=True)
    model = Transformer("hept",5,3,"tracking",**options)
    data = dict(x=torch.randn(7,5),coords=torch.randn(7,3),edge_index=None,batch=torch.zeros(7,dtype=torch.long))
    model(data)
    assert model.use_ckpt
    from torch_geometric.data import Data
    model(Data(x=data["x"], coords=data["coords"], edge_index=torch.empty(2,0,dtype=torch.long),
               batch=data["batch"])).square().mean().backward()
    assert torch.isfinite(model.feat_encoder[0].weight.grad).all()
    with pytest.raises(ValueError, match="batch_size=1"):
        model(dict(data,batch=torch.tensor([0,0,0,1,1,1,1])))
