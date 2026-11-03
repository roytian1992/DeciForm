"""One frozen global ordinal scale, derived only from training metadata."""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from common import sha, support, target
from score_distribution import DistributionObjective, logit_diagnostics


def uniform_scale_from_counts(counts):
    rows, ce2, ordinal2 = 0, 0., 0.
    for item in counts:
        values, y, n = item['support'], item['label_index'], item['rows']
        if type(n) is not int or n <= 0:
            raise ValueError('Positive integer count required')
        d = logit_diagnostics(torch.zeros(len(values),dtype=torch.float64),y,'score',
                              DistributionObjective('ordinal_main'),values)
        rows += n
        ce2 += n*float(d[4])**2
        ordinal2 += n*float(d[3])**2
    if not rows or ordinal2 <= 0:
        raise ValueError('Nonempty nondegenerate training score population required')
    return {'rows': rows, 'uniform_ce_rms_logit_gradient': math.sqrt(ce2/rows),
            'uniform_ordinal_rms_logit_gradient': math.sqrt(ordinal2/rows),
            'scale': math.sqrt(ce2/ordinal2)}


def calibrate_ordinal_scale(data):
    data = Path(data)
    audit_path = data/'cache/audit.json'
    audit = json.loads(audit_path.read_text())
    entry = audit['splits']['train']
    source = Path(entry['views_path'])
    assert audit['status']=='passed' and sha(source)==entry['views_sha256']
    counts, seen = Counter(), 0
    for line in source.open():
        row = json.loads(line)
        seen += 1
        if row['task']=='score':
            counts[(tuple(support(row)),target(row))] += 1
    assert seen == entry['rows'] == 200000
    grouped = [{'support':list(values),'label_index':y,'rows':n}
               for (values,y),n in sorted(counts.items())]
    result = uniform_scale_from_counts(grouped)
    assert result['rows']==75000
    result.update(source=str(source),source_sha256=sha(source),population='training score only',
        total_training_rows=seen,counts=grouped,model_inference=False,dev_or_test_used=False,
        scope='single global positive constant; match RMS logit gradients at uniform probabilities',
        warning='not a match of actual backbone gradients and not per-label loss reweighting')
    return result
