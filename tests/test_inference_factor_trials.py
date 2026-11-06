from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from inference_factor_trials import fold_factor, PrecastFactors


@pytest.mark.parametrize('side',['a','b'])
def test_fold_matches_column_convention_in_real_arithmetic(side):
    torch.manual_seed(21)
    a=torch.randn(3,7,dtype=torch.float64)
    b=torch.randn(5,3,dtype=torch.float64)
    r=torch.randn(3,3,dtype=torch.float64)*.1
    x=torch.randn(7,4,dtype=torch.float64)
    low=a@x
    expected=b@(low+r@low)
    factor=fold_factor(a,b,r,side)
    actual=b@factor@x if side=='a' else factor@a@x
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)


def test_precast_restores_original_parameters_even_on_exception(monkeypatch):
    import inference_factor_trials as module
    from types import SimpleNamespace
    a=torch.nn.Linear(7,3,bias=False)
    b=torch.nn.Linear(3,5,bias=False)
    a.requires_grad_(False);b.requires_grad_(False)
    original_a=a.weight.detach().clone()
    original_b=b.weight.detach().clone()
    fake=SimpleNamespace(active_adapters=['default'],lora_A={'default':a},lora_B={'default':b})
    monkeypatch.setattr(module,'routed_modules',lambda model:[fake])
    cache=PrecastFactors(None)
    with pytest.raises(RuntimeError):
        with cache:
            assert a.weight.dtype==torch.bfloat16
            assert b.weight.dtype==torch.bfloat16
            raise RuntimeError('test cleanup')
    assert torch.equal(a.weight,original_a) and a.weight.dtype==original_a.dtype
    assert torch.equal(b.weight,original_b) and b.weight.dtype==original_b.dtype
