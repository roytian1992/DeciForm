"""Full native Score support for upstream Laya, including fractional targets."""
import math
from laya_adapter import encode_parts
from score_specialist import support, interpolation, prediction_row, metrics, decisions


def make_parts(row,tok,state_ids=None,token_cache=None):
    from laya.common import render_options,QTYPES
    values=support(row)
    q={'t':'score','ins':row['question']['instructions'],'crit':[format(v,'.12g') for v in values]}
    def tokens(text):
        text=text.replace(tok.mask_token,' ')
        if token_cache is not None and text in token_cache:return token_cache[text]
        ids=tok(text,add_special_tokens=False,truncation=False)['input_ids']
        if token_cache is not None:token_cache[text]=ids
        return ids
    prefix=[tok.cls_token_id]+tokens(f"score question: {q['ins']}")+[tok.sep_token_id]
    options=[[tok.mask_token_id]+tokens(' '+s) for s in render_options(q)]
    if state_ids is None:state_ids=tokens(row['state']['text'])
    distribution=interpolation(values,row['label'])
    assert len(options)==len(values) and abs(math.fsum(distribution)-1)<1e-8
    return {'prefix':prefix,'options':options,'suffix':[tok.sep_token_id]+state_ids+[tok.sep_token_id],
            'qtype':QTYPES['score'],'label_index':max(range(len(values)),key=distribution.__getitem__),
            'target_distribution':distribution,'length':len(prefix)+sum(map(len,options))+len(state_ids)+2}


def encode(parts,layout='sequential',order=None):
    order=list(range(len(parts['options']))) if order is None else order
    item=encode_parts(parts,layout,order)
    item['target']=[parts['target_distribution'][i] for i in order]
    return item


def record(row,probs):
    if len(probs)!=len(support(row)) or abs(math.fsum(probs)-1)>1e-5:
        raise ValueError('Incomplete probability distribution')
    if any(not math.isfinite(v) or v<0 for v in probs):raise ValueError('Invalid probabilities')
    d=decisions(support(row),probs)
    return prediction_row(row,{'valid':True,'probabilities':probs,'prediction_index':d['indices']['median']})
