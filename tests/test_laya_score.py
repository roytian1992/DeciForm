import copy,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/formal'))
sys.path.insert(0,str(ROOT/'runs/formal_laya_20261106_v1/vendor/laya_release'))
from laya_score import make_parts,encode,record
from laya_adapter import make_parts as main_parts,encode_parts
from test_laya_baseline import tok,row
from test_score_baselines import row as score_row
from laya.common import collate_items,QTYPES
from laya.train import rlcd_loss

@pytest.mark.parametrize('layout',['sequential','parallel'])
def test_native_score_encoding_matches_existing_laya(tok,layout):
    r=row('score');r['score_support']=list(range(1,6))
    assert encode(make_parts(r,tok),layout)==encode_parts(main_parts(r,tok),layout)

@pytest.mark.parametrize('k',[4,5,6,11,31,61,101])
def test_full_support_and_long_text_keep_every_token(tok,k):
    r=score_row(True,k-1);r['score_support']=list(range(k))
    r['state']['text']='Complete essay. '*5000
    parts=make_parts(r,tok);item=encode(parts)
    assert len(item['markers'])==k and item['target'][-1]==1
    assert parts['suffix'][1:-1]==tok(r['state']['text'],add_special_tokens=False)['input_ids']
    assert len(item['ids'])>8192

@pytest.mark.parametrize('values,y',[(list(range(101)),92.123456789),([i/10 for i in range(51)],3.27)])
def test_continuous_target_is_interpolated_not_rounded(tok,values,y):
    r=score_row(False,y);r['score_support']=values
    p=make_parts(r,tok)
    assert sum(a*b for a,b in zip(p['target_distribution'],values))==pytest.approx(y)
    assert sum(v>0 for v in p['target_distribution'])==2
    changed=make_parts({**r,'label':values[0]},tok)
    for layout in ['sequential','parallel']:
        a,b=encode(p,layout),encode(changed,layout)
        assert {k:v for k,v in a.items() if k!='target'}=={k:v for k,v in b.items() if k!='target'}
        order=list(range(len(values)))[::-1]
        assert encode(p,layout,order)['target']==p['target_distribution'][::-1]
    probs=p['target_distribution']
    rec=record(r,probs)
    assert rec['label']==y and rec['score_decisions']['mean']==pytest.approx(y)
    assert rec['mae']>=0

def test_native_rlcd_accepts_mixed_full_soft_targets(tok):
    rows=[score_row(True,31),score_row(False,92.125)]
    items=[encode(make_parts(r,tok),'parallel') for r in rows]
    b=collate_items([items],tok.pad_token_id)
    z=torch.zeros_like(b['target'],requires_grad=True)
    torch.manual_seed(20261101)
    loss=rlcd_loss(z,b['target'],b['marker_mask'],b['qtype'],sigma=.4,samples=4,w_sph=.75,w_rps=1.)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert torch.all(z.grad[~b['marker_mask']]==0)
    assert torch.all(b['qtype']==QTYPES['score'])
