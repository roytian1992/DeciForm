"""Check paired execution parity and continuous-grid model wiring."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
import common  # Establish the frozen vendor import paths before candidate code.
import independent_candidates
import models_independent_score_specialist as candidate_model
from test_independent_candidates import tiny, cpu_reference_kernels, flatten


def encoding(k):
    ids, option_ids, ends = [3, 5, 8], [-1] * 3, []
    for i in range(k):
        ids.extend([21 + i, 41 + i])
        option_ids.extend([i, i])
        ends.append(len(ids) - 1)
    ids.append(2)
    option_ids.append(-2)
    return {'ids': ids, 'opt': option_ids, 'opt_idx': [ends],
            'decide_idx': [len(ids) - 1], 'tasks': ['score'], 'labels': [None],
            'state_truncated': False}


def test_matches_live_asap_chunk_execution_and_gradients_with_dropout():
    # Read only the reference function; importing the live wrapper would launch it.
    path = Path(__file__).resolve().parents[1] / 'scripts/formal/worker_independent_asap_score.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_asap_shared_hidden')
    namespace = {'torch': torch, 'independent_candidates': independent_candidates,
                 '_shared_hidden': independent_candidates.shared_hidden, 'BRANCH_CHUNK_SIZE': 8}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    lm, router = tiny()
    lm.train()
    for layer in lm.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = .05
    params = [p for p in [*lm.parameters(), *router.parameters()] if p.requires_grad]
    torch.manual_seed(19)
    got = flatten(candidate_model.chunked_shared_hidden(lm, [encoding(9)], 0, torch.tensor([2]), router))
    weights = torch.linspace(-.2, .3, len(got))
    grads = torch.autograd.grad((got * weights).sum(), params)
    torch.manual_seed(19)
    ref = flatten(namespace['_asap_shared_hidden'](lm, [encoding(9)], 0, torch.tensor([2]), router))
    ref_grads = torch.autograd.grad((ref * weights).sum(), params)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fractional_support_uses_score_route_and_all_candidates(monkeypatch):
    enc = encoding(3)
    row = {'task': 'score', 'score_support': [0., .1, .2], 'label': .13}
    def hidden(lm, es, pad, task_ids, router):
        assert task_ids.tolist() == [2]
        return torch.ones((1, 4)), [torch.arange(12.).reshape(3, 4)]
    class Head:
        numeric_supports = None
        def padded(self, q, h, valid, task_ids):
            assert self.numeric_supports == [[0., .1, .2]]
            assert valid.tolist() == [[True, True, True]]
            assert task_ids.tolist() == [2]
            return h.sum(-1)
    head = Head()
    model = SimpleNamespace(device='cpu', lm=None, tok=SimpleNamespace(pad_token_id=0),
                            net=SimpleNamespace(head=head, task_adapter=None))
    monkeypatch.setattr(candidate_model, 'chunked_shared_hidden', hidden)
    result = candidate_model.FormalModel.scores(model, [row], [enc])
    assert len(result[0]) == 3 and head.numeric_supports is None
