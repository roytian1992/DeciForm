import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from score_specialist import (SpecialistObjective, interpolation, data_loss, encode_score,
                               prediction_row, metrics, decisions, spearman)
from typed_losses import ordinal_log_loss
from test_order_audit import Tokenizer


@pytest.mark.parametrize('values,y',[([0.,.1,.2,.3],.175),(list(range(101)),73.333333),([0,2,5],3.5)])
def test_interpolation_preserves_fractional_target_and_local_support(values,y):
    q=interpolation(values,y)
    assert sum(q)==pytest.approx(1.) and min(q)>=0 and sum(v>0 for v in q)<=2
    assert sum(p*s for p,s in zip(q,values))==pytest.approx(y)


@pytest.mark.parametrize('k',[5,11,51,101])
def test_grid_node_loss_and_gradient_equal_original_discrete_d2(k):
    values=list(range(k));z=torch.linspace(-2,2,k,dtype=torch.float64,requires_grad=True);y=k//2
    obj=SpecialistObjective('soft_cdf')
    new=data_loss(z,values,y,obj);old=obj.ordinal_scale*ordinal_log_loss(z,y,values)
    assert torch.allclose(new,old,atol=1e-10)
    a=torch.autograd.grad(new,z,retain_graph=True)[0];b=torch.autograd.grad(old,z)[0]
    assert torch.allclose(a,b,atol=1e-10)


@pytest.mark.parametrize('kind',['soft_ce','soft_cdf'])
def test_loss_has_correct_gradient_and_is_affine_invariant(kind):
    z=torch.tensor([-.4,.2,1.],dtype=torch.float64,requires_grad=True);obj=SpecialistObjective(kind)
    assert torch.autograd.gradcheck(lambda v:data_loss(v,[0.,1.,2.],.7,obj),(z,))
    assert torch.allclose(data_loss(z,[0.,1.,2.],.7,obj),data_loss(z,[10.,35.,60.],27.5,obj))
    extreme=torch.tensor([-1000.,1000.,0.],requires_grad=True)
    loss=data_loss(extreme,[0.,1.,2.],.1,obj);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()


def test_continuous_encoding_never_embeds_or_rounds_training_label():
    row={'task':'score','label':2.35,'score_support':[0,1,2,3,4,5],
         'state':{'text':'Sentence A: one\nSentence B: two'},
         'question':{'instructions':'Rate 0 to 5; fractional values allowed.'}}
    tok=Tokenizer();a=encode_score(row,tok);b=encode_score({**row,'label':4.123},tok)
    assert a==b and a['labels']==[None] and len(a['opt_idx'][0])==6
    assert not a['state_truncated']


def test_metrics_compare_to_original_decimal_and_macro_sources_equally():
    rows=[]
    for i,(source,values,y) in enumerate([('a',[0,1,2],.4),('b',[0,50,100],10.),('b',[0,50,100],10.)]):
        r={'id':str(i),'task':'score','source_family':source,'group_id':str(i),
           'score_support':values,'label':y};p={'valid':True,'prediction_index':0,'probabilities':[.6,.3,.1]}
        rows.append(prediction_row(r,p))
    report=metrics(rows)
    assert rows[0]['mae']==.4 and rows[0]['label']==.4
    assert report['macro_source_nmae']==pytest.approx(.15)
    assert 'accuracy' not in report['task/score']


def test_bad_or_missing_prediction_cannot_silently_disappear():
    r={'task':'score','score_support':[0,1,2],'label':1.4}
    with pytest.raises(ValueError):interpolation([0,1,2],2.1)
    with pytest.raises(ValueError):prediction_row(r,{'valid':False})


def test_spearman_uses_average_tied_ranks_and_handles_constants():
    assert spearman([1,2,2,4],[4,2,2,1])==pytest.approx(-1.)
    # Centered ranks [-1,-1,.5,1.5] and [-1.5,-.5,1,1].
    assert spearman([1,1,3,4],[1,2,3,3])==pytest.approx(8/9)
    assert spearman([1,1],[2,3]) is None


def test_split_components_link_shared_sentence_and_document_entities():
    from build_score_specialist_data import Components
    c=Components();c.join(['sentence-a','sentence-b']);c.join(['sentence-c','sentence-b'])
    assert c.root('sentence-a')==c.root('sentence-c')
    c.join(['en-de-source','same-document']);c.join(['en-zh-source','same-document'])
    assert c.root('en-de-source')==c.root('en-zh-source')
    assert c.root('unrelated')!=c.root('sentence-a')
