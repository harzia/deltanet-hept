import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.losses import InfoNCELoss, deterministic_scatter
from utils.metrics import acc_and_pr_at_k
from models.attention.deltanet import delta_rule_recurrence, delta_rule_chunkwise


def test_negative_groups_with_gaps():
    src = torch.tensor([2.,3.,4.],requires_grad=True)
    index = torch.tensor([1,1,4])
    torch.testing.assert_close(deterministic_scatter(src,index,"sum",dim_size=6),torch.tensor([0.,5.,0.,0.,4.,0.]))
    x = torch.randn(4,3,requires_grad=True)
    # Anchor 0 has a positive but no negative; anchor 2 has both.
    pairs = torch.tensor([[0,2,2],[1,3,1]])
    ids = torch.tensor([1,1,2,2])
    loss = InfoNCELoss(0.05,"l2_rbf")(x,pairs,ids,torch.ones(4),torch.ones(4))
    sim = torch.exp(-torch.linalg.norm(x[pairs[0]]-x[pairs[1]],dim=-1)/(2*0.75**2))/0.05
    expected = torch.nn.functional.softplus(sim[2]-sim[1])/2
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError,match="no positive"):
        InfoNCELoss(0.05,"l2_rbf")(x,pairs,ids,torch.zeros(4),torch.ones(4))


def test_metric_chunking_and_small_event(monkeypatch):
    x = torch.tensor([[0.],[0.1],[5.],[5.1],[9.],[9.1]])
    ids = torch.tensor([1,1,2,2,3,3])
    original = torch.cdist
    sizes = []
    def record(a,b,**kw):
        sizes.append(a.shape[0]); return original(a,b,**kw)
    monkeypatch.setattr(torch,"cdist",record)
    full = acc_and_pr_at_k(x,ids,torch.ones(6,dtype=torch.bool),"l2_rbf",batch_size=6)
    sizes.clear()
    chunked = acc_and_pr_at_k(x,ids,torch.ones(6,dtype=torch.bool),"l2_rbf",batch_size=2)
    assert full == chunked and chunked[0] == 1.0 and max(sizes) <= 2


def test_deltanet_backward_matches_recurrence():
    torch.manual_seed(2)
    q = torch.randn(2,16,4,dtype=torch.float64,requires_grad=True)
    k = (torch.randn_like(q)*0.1).requires_grad_()
    v = torch.randn_like(q,requires_grad=True)
    beta = torch.rand(2,16,dtype=torch.float64,requires_grad=True)
    ref = delta_rule_recurrence(q,k,v,beta)
    out = delta_rule_chunkwise(q,k,v,beta,chunk_size=4)
    for a,b in zip(torch.autograd.grad(ref.square().sum(),(q,k,v,beta),retain_graph=True),
                   torch.autograd.grad(out.square().sum(),(q,k,v,beta))):
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
