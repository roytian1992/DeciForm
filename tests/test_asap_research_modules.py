import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/formal'))

import pytest
import torch
from asap_research_modules import QueryEvidencePool, laplace_log_probabilities, MedianLaplaceReadout


@pytest.mark.parametrize('support', [list(range(4)), list(range(2,13)), list(range(61)), [0.,.1,.3,.9,1.]])
@pytest.mark.parametrize('location', [.01,.173,.537,.987])
def test_native_median_equals_nearest_latent_median(support, location):
    m = torch.tensor(location, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(.12, dtype=torch.float64, requires_grad=True)
    z = laplace_log_probabilities(m,b,support)
    p = z.softmax(0)
    assert torch.isfinite(z).all() and (p>0).all()
    assert torch.allclose(p.sum(),torch.tensor(1.,dtype=p.dtype))
    median = int(torch.searchsorted(p.cumsum(0),.5))
    native = support[0]+location*(support[-1]-support[0])
    assert median == min(range(len(support)),key=lambda i:abs(support[i]-native))
    (-z[len(z)//2]).backward()
    assert torch.isfinite(m.grad) and torch.isfinite(b.grad)


def test_score_affine_range_invariance():
    m,b=torch.tensor(.42),torch.tensor(.07)
    a=laplace_log_probabilities(m,b,[0,1,2,3,4])
    c=laplace_log_probabilities(m,b,[10,20,30,40,50])
    torch.testing.assert_close(a,c)


def test_cdf_has_nonzero_location_gradient_at_midpoint():
    m=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    z=laplace_log_probabilities(m,torch.tensor(.1,dtype=torch.float64),[0,1])
    z.exp()[0].backward()
    torch.testing.assert_close(m.grad,torch.tensor(-5.,dtype=torch.float64))


def test_pool_excludes_padding_and_preserves_batched_rows():
    torch.manual_seed(7)
    pool=QueryEvidencePool(16,8,rank=4,heads=2)
    x=torch.randn(2,9,16,requires_grad=True);q=torch.randn(2,8,requires_grad=True)
    mask=torch.arange(9)[None]<torch.tensor([5,9])[:,None]
    original=pool(x,q,mask)
    changed=x.detach().clone();changed[0,5:]=torch.randn_like(changed[0,5:])*1000
    torch.testing.assert_close(original,pool(changed,q,mask))
    for i,n in enumerate([5,9]):
        torch.testing.assert_close(original[i],pool(x[i:i+1,:n],q[i:i+1],torch.ones(1,n,dtype=torch.bool))[0])
    original.square().sum().backward()
    assert x.grad[0,5:].abs().sum()==0 and x.grad[0,:5].abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters())


def test_readout_routes_regression_gradients_without_rounding():
    torch.manual_seed(2)
    head=MedianLaplaceReadout(8)
    z,m,b=head(torch.randn(3,8),[list(range(4)),list(range(11)),list(range(61))])
    target=m.detach()+torch.tensor([.02,-.02,.03])
    loss=(m-target).abs().sum()+sum(-v[len(v)//2] for v in z)
    loss.backward()
    assert head.output.weight.grad[0].abs().sum()>0
    assert head.output.weight.grad[1].abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in head.parameters())
