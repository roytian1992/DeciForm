"""Exact probability enumeration, semantic IDs, complete support and decoder tests."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from zero_shot_logits import (make_trie,split_common,score_batch,dense_scores,answer_strings,
                             prediction,encode,batches)
from test_zero_shot_json import noul
from test_classification_order_audit import choice
from test_score_baselines import row as score_row

class ToyCache:
    def __init__(self,state):self.state=state
    def reorder_cache(self,index):self.state=self.state.index_select(0,index)
class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.vocab=17
    def forward(self,input_ids,attention_mask=None,past_key_values=None,use_cache=False,logits_to_keep=0,**kwargs):
        state=torch.zeros(len(input_ids),dtype=torch.long) if past_key_values is None else past_key_values.state
        outputs=[]
        for i in range(input_ids.shape[1]):
            active=attention_mask[:,-input_ids.shape[1]+i] if attention_mask is not None else torch.ones_like(state)
            state=torch.where(active.bool(),(state*3+input_ids[:,i]+1)%self.vocab,state)
            logits=-(torch.arange(self.vocab)[None,:]-state[:,None]).abs().float()/3
            outputs.append(logits)
        z=torch.stack(outputs,1)
        if logits_to_keep:z=z[:,-logits_to_keep:]
        return SimpleNamespace(logits=z,past_key_values=ToyCache(state))

@pytest.mark.parametrize('tails',[
    [[1,4],[1,2,4],[3,4]],[[1,2,4],[1,3,4],[5,6,7,4],[5,8,4]],
    [[i//10+1,i%10+1,16] for i in range(101)]])
def test_exact_leaf_logp_and_global_probability_match_dense(tails):
    model=ToyLM()
    enc={'ids':[2,4,3],'tails':tails}
    actual=score_batch(model,[enc],0)[0]
    expected=dense_scores(model,enc,list(range(len(tails))))
    assert torch.allclose(torch.tensor(actual['sequence_likelihood']),torch.tensor(expected),atol=2e-6)
    assert torch.tensor(actual['constrained_trie']).exp().sum().item()==pytest.approx(1,abs=2e-6)
    # Repeated parent cache states cannot alias under in-place decoding.
    reverse=score_batch(model,[{**enc,'tails':tails[::-1]}],0)[0]
    assert actual['sequence_likelihood']==reverse['sequence_likelihood'][::-1]

def test_mixed_padding_and_multitoken_batch_preserve_answers():
    model=ToyLM()
    encs=[{'ids':[2,3,4,5],'tails':[[1,4],[2,4]]},
          {'ids':[5],'tails':[[3,1,4],[3,2,4],[6,4]]}]
    batch=score_batch(model,encs,0)
    for i,e in enumerate(encs):assert batch[i]==score_batch(model,[e],0)[0]
    assert not torch.allclose(torch.tensor(batch[1]['sequence_likelihood']).softmax(0),
                              torch.tensor(batch[1]['constrained_trie']).exp(),atol=1e-3)

def test_illegal_prefixes_and_duplicates_rejected():
    for tails in ([[1],[1,2]],[[1,2],[1,2]]):
        with pytest.raises(ValueError):make_trie(tails)
    assert split_common([[1,2,3],[1,2,4]])==([1,2],[[3],[4]])

class CharTokenizer:
    def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=[ord(c) for c in text])
    def decode(self,ids,**kwargs):return ''.join(map(chr,ids))
    def apply_chat_template(self,messages,**kwargs):return [ord(c) for c in '<chat>'+messages[0]['content']+'<assistant>']

@pytest.mark.parametrize('k',[2,21,77])
def test_complete_candidate_id_semantics_gold_independence(k):
    r=choice(k);r['label']=r['question']['option_ids'][0]
    e=encode(r,'choice_order','reversed',CharTokenizer())
    changed=copy.deepcopy(r);changed['label']=r['question']['option_ids'][-1]
    assert encode(changed,'choice_order','reversed',CharTokenizer())==e
    assert len(e['tails'])==k
    assert answer_strings(r)==[json.dumps({'choice_id':f'c{i}'},separators=(',',':')) for i in range(k)]
    scores={'sequence_likelihood':[-50.]*(k-1)+[0.]}
    p=prediction(r,'choice_order','reversed',e,scores,'sequence_likelihood')
    assert p['prediction_index']==k-1
    assert p['support']==r['question']['option_ids']
    assert p['order']==list(range(k))[::-1]

def test_fractional_labels_preserved_and_mean_differs_from_median():
    r=score_row(False,92.123456789)
    e=encode(r,'continuous','original',CharTokenizer())
    s=[-100.]*len(r['score_support']);s[0]=0;s[-1]=-.5
    p=prediction(r,'continuous','original',e,{'sequence_likelihood':s},'sequence_likelihood')
    assert p['label']==92.123456789 and p['prediction_index']==0
    assert p['score_decisions']['mean']>0 and p['prediction_value']==0
    assert len(e['tails'])==len(r['score_support'])==101

def test_noul_original_reverse_keep_true_false_identity():
    r=noul();t=CharTokenizer()
    a=encode(r,'noul_order','original',t);b=encode(r,'noul_order','reversed',t)
    assert a['ids']!=b['ids'] and a['tails']==b['tails']
    for e,v in [(a,'original'),(b,'reversed')]:
        p=prediction(r,'noul_order',v,e,{'sequence_likelihood':[-9.,0.]},'sequence_likelihood')
        assert p['prediction_index']==1 and p['correct']

def test_large_inventory_is_not_truncated_by_batch_budget():
    items=[{'row':{'id':str(i)},'length':6000,'enc':{'tails':[[1]]*101}} for i in range(3)]
    out=list(batches(items,budget=100000))
    assert [len(b) for b in out]==[1,1,1]
    assert all(len(b[0]['enc']['tails'])==101 for b in out)
