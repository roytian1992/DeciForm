"""Proper-risk derivatives, native support, decoder matching and task isolation."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
import common
from score_distribution import (DistributionObjective, task_data_loss,
    logit_diagnostics, decisions, score_rule_report)
from calibrate_score_objective import uniform_scale_from_counts
from models_score_distribution import FormalModel


@pytest.mark.parametrize('kind',['mean_moment','ordinal_main'])
@pytest.mark.parametrize('values',[[0,1,2,3],[0,1,20,100],[-10,-2,5,80]])
def test_population_risk_stationary_at_true_distribution(kind,values):
    eta=torch.tensor([.1,.2,.3,.4],dtype=torch.float64)
    z=eta.log().requires_grad_()
    objective=DistributionObjective(kind,ordinal_scale=3.038)
    risk=sum(eta[y]*task_data_loss(z,y,'score',objective,values) for y in range(4))
    g=torch.autograd.grad(risk,z)[0]
    torch.testing.assert_close(g,torch.zeros_like(g),atol=1e-12,rtol=0)


@pytest.mark.parametrize('kind',['mean_moment','ordinal_main'])
@pytest.mark.parametrize('task',['choice','noul','score'])
def test_analytic_output_gradients_match_autograd(kind,task):
    z=torch.tensor([.4,-1.2] if task=='noul' else [.4,-1.2,2.,.1],dtype=torch.float64,requires_grad=True)
    values=[0,1,20,100] if task=='score' else None
    objective=DistributionObjective(kind,ordinal_scale=3.1)
    actual=task_data_loss(z,1,task,objective,values)
    grad=torch.autograd.grad(actual,z)[0]
    diag=logit_diagnostics(z,1,task,objective,values)
    assert not diag.requires_grad
    torch.testing.assert_close(diag[1],actual.detach())
    torch.testing.assert_close(diag[3],grad.norm(),rtol=1e-12,atol=1e-12)


def test_moment_uses_predictive_mean_squared_error_and_native_span():
    z=torch.tensor([.2,-.1,.7],dtype=torch.float64,requires_grad=True)
    s=torch.tensor([0.,20.,100.],dtype=torch.float64)
    objective=DistributionObjective('mean_moment')
    ce=F.cross_entropy(z[None],torch.tensor([1]))
    mu=(z.softmax(0)*s).sum()
    actual=task_data_loss(z,1,'score',objective,s.tolist())
    torch.testing.assert_close(actual,ce+.5*(mu-20).square()/100)
    translated=task_data_loss(z,1,'score',objective,(s+17).tolist())
    scaled=task_data_loss(z,1,'score',objective,(s*7+17).tolist())
    torch.testing.assert_close(translated,actual)
    torch.testing.assert_close(scaled-ce,7*(actual-ce))
    # Keep corrective log-score gradients even when the mean saturates at a wrong class.
    extreme=torch.tensor([0.,-1000.,1000.],requires_grad=True)
    loss=task_data_loss(extreme,0,'score',objective,s.tolist())
    g=torch.autograd.grad(loss,extreme)[0]
    assert torch.isfinite(loss) and torch.isfinite(g).all() and g[2]>0 and g[0]<0


@pytest.mark.parametrize('label',[0,1,2,3])
def test_pure_ordinal_retains_finite_corrective_gradients(label):
    z=torch.tensor([-10000.,0.,10000.,-5000.],requires_grad=True)
    loss=task_data_loss(z,label,'score',DistributionObjective('ordinal_main',ordinal_scale=3.),[0,1,20,100])
    grad=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
    if label!=2:
        assert grad[2]>0


@pytest.mark.parametrize('kind',['mean_moment','ordinal_main'])
def test_actual_model_only_changes_score_and_preserves_microbatch_weighting(kind):
    rows=[{'task':'choice','question':{'options':['a','b','c']},'label':'b'},
          {'task':'noul','question':{},'label':True},
          {'task':'score','question':{'scale':{'min':0,'max':4}},'label':3}]
    zs=[torch.randn(k,requires_grad=True) for k in [3,2,5]]
    penalty=torch.tensor(.03,requires_grad=True)
    model=SimpleNamespace(scores=lambda rs,es:es,net=SimpleNamespace(regularization=lambda:penalty.square()),
        distribution_objective=DistributionObjective(kind,ordinal_scale=3.1),collect_loss_diagnostics=False)
    out=FormalModel.losses(model,rows,zs)
    torch.testing.assert_close(out[0],F.cross_entropy(zs[0][None],torch.tensor([1]))+penalty.square(),rtol=0,atol=0)
    torch.testing.assert_close(out[1],F.binary_cross_entropy_with_logits(zs[1][1]-zs[1][0],torch.tensor(1.))+penalty.square(),rtol=0,atol=0)
    split=sum(FormalModel.losses(model,rows[a:b],zs[a:b]).sum()/3 for a,b in [(0,1),(1,3)])
    expected=torch.autograd.grad(out.mean(),zs+[penalty],retain_graph=True)
    actual=torch.autograd.grad(split,zs+[penalty])
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b)


def test_global_ordinal_scale_matches_uniform_rms_without_label_reweighting():
    counts=[{'support':[0,1,2,3,4],'label_index':y,'rows':y+1} for y in range(5)]
    audit=uniform_scale_from_counts(counts)
    torch.testing.assert_close(torch.tensor(audit['scale']*audit['uniform_ordinal_rms_logit_gradient']),
                               torch.tensor(audit['uniform_ce_rms_logit_gradient']))
    repeated=uniform_scale_from_counts([{**r,'rows':r['rows']*10} for r in counts])
    assert repeated['scale']==pytest.approx(audit['scale'])


@pytest.mark.parametrize('support',[[0,1,2],[0,20,100],[-4,0,3]])
@pytest.mark.parametrize('p',[[.4,.35,.25],[.5,0.,.5],[0.,1.,0.],[1.,0.,0.]])
def test_median_minimizes_predicted_absolute_error_on_native_support(support,p):
    d=decisions(support,p)
    median=d['values']['median'];mean=d['values']['mean']
    risks=[sum(prob*abs(a-v) for v,prob in zip(support,p)) for a in support]
    actual=sum(prob*abs(median-v) for v,prob in zip(support,p))
    assert actual==pytest.approx(min(risks))
    mean_mse=sum(prob*(mean-v)**2 for v,prob in zip(support,p))
    assert all(mean_mse<=sum(prob*(a-v)**2 for v,prob in zip(support,p))+1e-10 for a in support)


def test_rules_distinguish_continuous_expectation_and_nearest_native_support():
    d=decisions([0,20,100],[.4,.35,.25])
    assert d['values']=={'argmax':0.,'median':20.,'mean':32.,'mean_nearest':20.}
    assert decisions([0,100],[.5,.5])['indices']['median']==0
    assert decisions([0,100],[.5,.5])['indices']['mean_nearest']==0


def test_replay_accepts_declared_old_and_new_rules_without_modifying_predictions():
    row={'id':'a','source':'toy','task':'score','valid':True,'k':3,'support':[0,20,100],
         'label_index':1,'label':20,'prediction_index':0,'probabilities':[.4,.35,.25]}
    original=copy.deepcopy(row)
    old=score_rule_report([row],'dev')
    assert row==original
    new={**row,'prediction_index':1,'score_decision_rule':'median'}
    report=score_rule_report([new],'dev')
    assert old['rules']==report['rules']
    assert report['primary_mae']==0 and report['rules']['argmax']['mae']==20
    assert report['rules']['mean']['mae']==12
    with pytest.raises(ValueError):score_rule_report([{**new,'prediction_index':0}],'dev')
    with pytest.raises(ValueError):score_rule_report([{**new,'valid':False}],'dev')


def test_actual_prediction_uses_median_only_for_score_without_an_extra_forward():
    rows=[{'task':'choice','question':{'options':['a','b','c']},'label':'b'},
          {'task':'noul','question':{},'label':True},
          {'task':'score','question':{'scale':{'min':0,'max':2}},'label':1}]
    zs=[torch.tensor([.4,.35,.25]).log(),torch.tensor([.6,.4]).log(),torch.tensor([.4,.35,.25]).log()]
    calls=[]
    def scores(rs,es):
        calls.append(1)
        return zs
    model=SimpleNamespace(scores=scores,clear=lambda:None)
    preds=FormalModel.predict(model,rows,[],0)
    assert len(calls)==1
    assert [p['prediction_index'] for p in preds]==[0,0,1]
    assert preds[2]['argmax_prediction_index']==0 and preds[2]['score_decision_rule']=='median'


@pytest.mark.parametrize('values,p',[([0,0],[.5,.5]),([1,0],[.5,.5]),([0,1],[.4,.4]),
                                    ([0,1],[float('nan'),.5]),([0,1],[-.1,1.1])])
def test_invalid_native_distributions_are_rejected(values,p):
    with pytest.raises(ValueError):decisions(values,p)


def test_worker_uses_distribution_model_and_same_rule_reporter():
    import worker_score_distribution as worker
    assert worker.main.__globals__['FormalModel'] is FormalModel
    assert worker.main.__globals__['score_rule_report'] is score_rule_report
