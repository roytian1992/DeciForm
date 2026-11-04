import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))
from common import sha
from curve_contract import load_subset, require_nested


@pytest.fixture
def fixture(tmp_path):
    rows = [{'task': 'choice', 'source': 'a'}, {'task': 'noul', 'source': 'b'},
            {'task': 'score', 'source': 'c'}]
    path = tmp_path / 'subset.json'
    record = {'status': 'passed', 'source_rows': 3, 'size': 2, 'indices': [0, 2],
              'task_counts': {'choice': 1, 'score': 1}, 'source_counts': {'a': 1, 'c': 1}}
    path.write_text(json.dumps(record))
    return path, rows, record


def test_original_row_order_and_identity(fixture):
    path, rows, _ = fixture
    ids = load_subset(path, sha(path), rows, 2)
    assert ids == [0, 2]
    assert [rows[i] for i in ids] == [rows[0], rows[2]]
    require_nested({2: ids, 3: [0, 1, 2]})


def test_reject_changed_file(fixture):
    path, rows, _ = fixture
    digest = sha(path)
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_subset(path, digest, rows, 2)


@pytest.mark.parametrize('ids', [[0, 0], [-1, 2], [0, 3], [2, 0], [False, 2]])
def test_reject_bad_indices(fixture, ids):
    path, rows, record = fixture
    record['indices'] = ids
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='indices'):
        load_subset(path, sha(path), rows, 2)


def test_reject_changed_source_membership(fixture):
    path, rows, _ = fixture
    rows[2]['source'] = 'other'
    with pytest.raises(ValueError, match='source_counts'):
        load_subset(path, sha(path), rows, 2)


def test_reject_non_nested():
    with pytest.raises(ValueError, match='not strictly nested'):
        require_nested({2: [0, 3], 3: [0, 1, 2]})
