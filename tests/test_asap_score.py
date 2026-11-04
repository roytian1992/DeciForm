from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from asap_score import (RANGES,SpecialistObjective,data_loss,prediction_row,metrics,quadratic_kappa)
from typed_losses import ordinal_log_loss
from build_asap_score_data import instructions


@pytest.mark.parametrize('number',range(1,9))
def test_all_native_scales_exact_ce_and_d2_gradients(number):
    lo,hi=RANGES[number];values=list(range(lo,hi+1));j=len(values)//2
    z=torch.linspace(-2,2,len(values),dtype=torch.float64,requires_grad=True)
    for kind in ('soft_ce','soft_cdf'):
        objective=SpecialistObjective(kind)
        actual=data_loss(z,values,values[j],objective)
        expected=torch.nn.functional.cross_entropy(z[None],torch.tensor([j])) if kind=='soft_ce' else objective.ordinal_scale*ordinal_log_loss(z,j,values)
        assert torch.allclose(actual,expected,atol=1e-10)
        a=torch.autograd.grad(actual,z,retain_graph=True)[0];b=torch.autograd.grad(expected,z,retain_graph=True)[0]
        assert torch.allclose(a,b,atol=1e-10)


def test_qwk_and_metrics_keep_prompt_scale_separate():
    assert quadratic_kappa([0,1,2],[0,1,2],[0,1,2])==pytest.approx(1.)
    assert quadratic_kappa([0,2],[2,0],[0,1,2])==pytest.approx(-1.)
    assert quadratic_kappa([0,0],[0,0],[0,1,2]) is None
    rows=[]
    for number in (1,7):
        lo,hi=RANGES[number];values=list(range(lo,hi+1));p=[0.]*len(values);p[0]=1.
        r={'id':str(number),'essay_set':number,'task':'score','score_support':values,
           'label':lo+1,'source_family':f'asap_set_{number}','group_id':str(number)}
        rows.append(prediction_row(r,{'valid':True,'probabilities':p,'prediction_index':0}))
    report=metrics(rows)
    assert report['macro_prompt_nmae']==pytest.approx((1/10+1/30)/2)
    assert report['source/asap_set_1']['normalized_rps']==pytest.approx(.1)


def test_full_source_rubric_and_adjudication_retained_without_corpus_metadata():
    parts=['Essay Set #3','Grade level:','10','Training set size:','1726','Source Essay',
           'Complete source passage.','Prompt','Full question.','Rubric Guidelines',
           'Score 0: original criteria','Score 3: original criteria','Adjudication Rules','Original resolution rule.']
    text=instructions(parts,3)
    assert '\n'.join(parts[6:]) in text
    assert '1726' not in text and 'grade 10' in text and '0 to 3' in text
