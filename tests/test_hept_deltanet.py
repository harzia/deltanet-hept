"""Gaussian-kernel DeltaNet is checked against a direct kernel recurrence."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from models.attention.hept_delta_kernel import GaussianFourierFeatures, gaussian_kernel, kernel_delta_rule
from models.attention.deltanet import DeltaNetAttention
from models.baselines.transformer import Transformer
from models.attention.hept import prep_qk
from einops import rearrange


def direct_kernel_recurrence(q,k,v,beta):
    # Independent squared differences and sequential residual writes.
    kk = torch.exp(-0.5*(k[:,:,None,:]-k[:,None,:,:]).square().sum(-1))
    qk = torch.exp(-0.5*(q[:,:,None,:]-k[:,None,:,:]).square().sum(-1))
    writes,outputs = [],[]
    for t in range(q.shape[1]):
        pred = torch.zeros_like(v[:,t]) if t == 0 else (kk[:,t,:t,None]*torch.stack(writes,dim=1)).sum(1)
        writes.append(beta[:,t,None]*(v[:,t]-pred))
        outputs.append((qk[:,t,:t+1,None]*torch.stack(writes,dim=1)).sum(1))
    return torch.stack(outputs,dim=1)


def test_exact_kernel_delta_outputs_and_gradients():
    torch.manual_seed(4)
    q,k,v = [torch.randn(2,7,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    beta = torch.rand(2,7,dtype=torch.float64,requires_grad=True)
    actual=kernel_delta_rule(q,k,v,beta)
    expected=direct_kernel_recurrence(q,k,v,beta)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-11)
    for a,b in zip(torch.autograd.grad(actual.square().sum(),(q,k,v,beta),retain_graph=True),
                   torch.autograd.grad(expected.square().sum(),(q,k,v,beta))):
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)


def test_gaussian_features_unit_norm_translation_and_kernel():
    fmap=GaussianFourierFeatures(2,4,32768,seed=7).double()
    torch.manual_seed(1)
    x=torch.randn(2,8,4,dtype=torch.float64)*0.3
    y=torch.randn(2,6,4,dtype=torch.float64)*0.3
    fx,fy=fmap(x),fmap(y)
    torch.testing.assert_close(fx.square().sum(-1),torch.ones(2,8,dtype=x.dtype))
    approx=fx@fy.transpose(-1,-2)
    assert (approx-gaussian_kernel(x,y)).abs().max()<0.02
    shift=torch.randn(2,1,4,dtype=x.dtype)
    torch.testing.assert_close(fmap(x+shift)@fmap(y+shift).transpose(-1,-2),approx)


def layer_options(backend):
    return dict(h_dim=2,num_heads=1,coords_dim=3,chunk_size=4,deltanet_kernel="hept",
                deltanet_hept_backend=backend,deltanet_kernel_features=32768,
                deltanet_qk_activation="identity",deltanet_qk_norm="none",
                deltanet_impl="torch",deltanet_sort_key="original",deltanet_direction="uni",
                num_w_per_dist=2,deltanet_kernel_seed=5)


def test_rff_layer_approximates_exact_kernel_delta_without_dimension_shrinkage():
    torch.manual_seed(2)
    exact=DeltaNetAttention(**layer_options("exact")).double()
    rff=DeltaNetAttention(**layer_options("rff")).double()
    rff.load_state_dict(exact.state_dict(),strict=False)
    rpe=torch.nn.Linear(4,2).double()
    q,k,v=[torch.randn(7,2,dtype=torch.float64)*0.2 for _ in range(3)]
    helper=dict(coords=torch.randn(7,3,dtype=torch.float64)*0.1,w_rpe=rpe)
    a=exact(q,k,v,**helper)
    b=rff(q,k,v,**helper)
    assert (a-b).abs().max()<0.015
    # Reference includes HEPT coordinate augmentation, not just raw q/k.
    w=rearrange(rpe.weight,"(h d) (r k) -> h d r k",h=1,d=2,k=2)
    qa,ka=prep_qk(q[:,None],k[:,None],w,helper["coords"])
    beta=exact.b_proj(v[None]).squeeze(-1).sigmoid()
    expected=direct_kernel_recurrence(qa.transpose(0,1),ka.transpose(0,1),v[None],beta)
    torch.testing.assert_close(a,exact.out_linear(expected.squeeze(0)))


@pytest.mark.parametrize("backend",["rff","exact","hash"])
def test_backbone_gaussian_gradient_checkpoint_and_padding(backend):
    from torch_geometric.data import Data
    torch.manual_seed(23)
    options=layer_options(backend)
    options.update(h_dim=8,num_heads=2,n_layers=2,deltanet_kernel_features=64,deltanet_direction="bi",use_ckpt=True)
    if backend == "hash":
        options.update(block_size=4,n_hashes=2,num_regions=3,deltanet_hash_chunk_buckets=1)
    options.pop("coords_dim")
    model=Transformer("deltanet",5,3,"tracking",**options)
    coords=torch.randn(7,3,requires_grad=True)
    data=Data(x=torch.randn(7,5),coords=coords,edge_index=torch.empty(2,0,dtype=torch.long),batch=torch.zeros(7,dtype=torch.long))
    out=model(data)
    assert out.shape==(7,4) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(coords.grad).all() and coords.grad.abs().sum()>0
    grad=model.attns[0].w_rpe.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    state=model.state_dict()
    other=Transformer("deltanet",5,3,"tracking",**options)
    other.load_state_dict(state)
    model.eval();other.eval()
    torch.testing.assert_close(model(data),other(data))


def test_exact_size_guard_and_config_validation():
    options=layer_options("exact");options["deltanet_hept_exact_max_points"]=3
    layer=DeltaNetAttention(**options)
    with pytest.raises(ValueError,match="quadratic"):
        layer(torch.randn(7,2),torch.randn(7,2),torch.randn(7,2),
              coords=torch.randn(7,3),w_rpe=torch.nn.Linear(4,2))
    with pytest.raises(ValueError,match="identity"):
        DeltaNetAttention(**dict(layer_options("rff"),deltanet_qk_activation="silu"))
    with pytest.raises(ValueError,match="torch"):
        DeltaNetAttention(**dict(layer_options("rff"),deltanet_impl="fla"))
