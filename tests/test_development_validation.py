import pytest
from deciform.development_validation import select_panels, summarize


def test_panels_are_disjoint_balanced_and_preserve_natural_imbalance():
    rows = [{'id':str(i),'label':'rare' if i<100 else 'common',
             'question':{'options':['rare','common']}} for i in range(1000)]
    panels = select_panels(rows,seed=47,target=200)
    assert panels == select_panels(list(reversed(rows)),seed=47,target=200)
    a, b = panels['natural'], panels['balanced']
    assert not {r['id'] for r in a}&{r['id'] for r in b}
    assert len(a)==200
    assert sum(r['label']=='rare' for r in a) < 50
    assert sum(r['label']=='rare' for r in b)==sum(r['label']=='common' for r in b)


def test_metrics_include_probability_quality_and_empty_abstention_case():
    rows = [{'label':'a','prediction':'a','confidence':.8,'probabilities':{'a':.8,'b':.2}},
            {'label':'b','prediction':'a','confidence':.6,'probabilities':{'a':.6,'b':.4}}]
    m = summarize(rows)
    assert m['accuracy']==.5
    assert m['brier']==pytest.approx(.4)
    assert m['macro_f1']==pytest.approx(1/3)
    assert m['ece']==pytest.approx(.4)
    assert m['fixed_threshold_risk']['0.9']=={'count':0,'coverage':0,'risk':None}
    assert m['accuracy_wilson_95'][0]<.5<m['accuracy_wilson_95'][1]
    with pytest.raises(ValueError):
        summarize([{**rows[0],'probabilities':{'a':2,'b':0}}])
