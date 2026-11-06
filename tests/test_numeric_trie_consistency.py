import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/formal'))
from numeric_trie import ALPHABET, NumericTrieReadout
from models_numeric_trie_consistency import ConsistencyObjective, FormalModel, suffix_tokens

class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [32 + ALPHABET.index(c) for c in text]

class CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 16)
        self.calls = 0
    def forward(self, input_ids, position_ids, attention_mask, use_cache=False):
        self.calls += 1
        h = self.embedding(input_ids) * attention_mask[..., None]
        return SimpleNamespace(last_hidden_state=h.cumsum(1)/(position_ids[..., None]+1))

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_raw = nn.Parameter(torch.zeros(1))
        self.prepared = False
    def prepare(self, *args):
        if self.prepared: raise RuntimeError("Clear routing context before another forward")
        self.prepared = True

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.task_adapter = Adapter()
        self.head = nn.Module()
        self.head.q = nn.Linear(16, 8)
        self.head.geometry_raw = nn.Parameter(torch.zeros(3, 8, 8))
        self.head.register_buffer('task_weights', torch.tensor([.25, .375, .375]))
        self.head.numeric_score = NumericTrieReadout(torch.randn(12,16),torch.randn(12,16),8)
        nn.init.normal_(self.head.numeric_score.readout.weight, std=.1)
    def _pad_rows(self, sequences):
        width = max(len(ids) for ids, _ in sequences)
        ids = torch.zeros(len(sequences), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (row, _) in enumerate(sequences):
            ids[i,:len(row)] = torch.tensor(row)
            mask[i,:len(row)] = 1
        return ids, torch.arange(width)[None].expand_as(ids), mask
    def regularization(self):
        return self.head.q.weight.sum() * 0
    def clear_task_ids(self):
        self.task_adapter.prepared = False

def fixture(kind='trie_token_ce'):
    torch.manual_seed(31)
    model = FormalModel.__new__(FormalModel)
    nn.Module.__init__(model)
    model.arm = 'tcdl'
    model.device = torch.device('cpu')
    model.tok = Tokenizer()
    model.net = Net()
    model.lm = CausalLM()
    model.distribution_objective = ConsistencyObjective(kind, cdf_weight=.5 if kind.endswith('_cdf') else 0.)
    model.collect_loss_diagnostics = False
    rows = [{'task':'score','label':12,'score_support':list(range(31))},
            {'task':'score','label':3.6666666666666665,'score_support':[i/10 for i in range(51)]}]
    encs = [{'ids':[3,4,5],'labels':[None],'tasks':['score']},
            {'ids':[6,7,8,9,10],'labels':[None],'tasks':['score']}]
    return model, rows, encs

@pytest.mark.parametrize('value',[12,0,3.6666666666666665,100.0])
def test_exact_json_suffix_not_rounded(value):
    ids, classes = suffix_tokens({'label':value}, Tokenizer())
    text = ''.join(ALPHABET[c] for c in classes)
    assert text == json.dumps({'score':value},separators=(',',':'))[len('{"score":'):]
    assert ids == [c+32 for c in classes]

def test_shared_forward_and_no_future_label_information():
    model, rows, encs = fixture()
    model.eval()
    baseline = model.scores(rows, encs)
    model.clear()
    model.lm.calls = 0
    actual, token_loss = model.joint_scores_and_token_loss(rows, encs)
    assert model.lm.calls == 1 and torch.isfinite(token_loss).all()
    for a,b in zip(actual,baseline): torch.testing.assert_close(a,b)
    model.clear()
    modified = [{**r,'label':r['score_support'][-1]} for r in rows]
    altered, _ = model.joint_scores_and_token_loss(modified, encs)
    for a,b in zip(actual,altered): torch.testing.assert_close(a,b)
    model.clear()
    assert model.verify_causal_prefix(rows, encs)['max_probability_difference'] < 1e-6

@pytest.mark.parametrize('kind',['trie_token_ce','trie_token_ce_cdf'])
def test_both_objectives_one_backbone_pass_with_gradients(kind):
    model, rows, encs = fixture(kind)
    losses = model.losses(rows, encs)
    assert model.lm.calls == 1 and len(losses) == 2
    losses.mean().backward()
    for p in [model.lm.embedding.weight,model.net.head.q.weight,model.net.head.numeric_score.readout.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0

def test_target_suffix_has_exact_teacher_forcing_alignment():
    model, rows, encs = fixture()
    _, actual = model.joint_scores_and_token_loss(rows, encs)
    expected = []
    for row, enc in zip(rows, encs):
        ids, cls = suffix_tokens(row,model.tok)
        sequence = torch.tensor(enc['ids']+ids[:-1])[None]
        positions = torch.arange(sequence.shape[1])[None]
        hidden = model.lm(sequence,positions,torch.ones_like(sequence)).last_hidden_state[0]
        start = len(enc['ids'])-1
        logits = hidden[start:start+len(cls)] @ model.net.head.numeric_score.output_rows.T
        expected.append(torch.nn.functional.cross_entropy(logits,torch.tensor(cls)))
    torch.testing.assert_close(actual,torch.stack(expected))

def test_gold_label_in_prompt_metadata_rejected():
    model, rows, encs = fixture()
    encs[0]['labels'] = [12]
    with pytest.raises(ValueError,match='answer labels'):
        model.joint_scores_and_token_loss(rows,encs)
