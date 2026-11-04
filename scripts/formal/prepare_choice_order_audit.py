"""Freeze the missing Choice panel without changing completed Noul results."""
import collections
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from common import PROJECT, atomic, sha, support
from classification_order_audit import view_orders


def main():
    previous = PROJECT / 'runs/formal_answer_order_20261104_v1'
    root = PROJECT / 'runs/formal_choice_order_20261104_v1'
    manifest = json.loads((previous/'experiment_manifest.json').read_text())
    assert json.loads((previous/'status.json').read_text())['phase'] == 'complete'
    hashes = dict(manifest['code_sha256'])
    for path, digest in hashes.items():
        assert sha(PROJECT/path) == digest, path
    paths = ['scripts/formal/classification_order_audit.py',
             'scripts/formal/worker_choice_order_audit.py',
             'scripts/formal/supervise_choice_order_audit.py',
             'scripts/formal/prepare_choice_order_audit.py',
             'scripts/formal/report_classification_order.py',
             'tests/test_classification_order_audit.py']
    hashes.update({path:sha(PROJECT/path) for path in paths})
    assert sha(manifest['views_path']) == manifest['views_sha256']
    rows = [json.loads(line) for line in open(manifest['views_path'])]
    choices = [row for row in rows if row['task']=='choice']
    assert len(choices)==len({row['id'] for row in choices})==5000
    assert all(row['candidate_mode']=='complete' and len(support(row))>=3 for row in choices)
    created = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    panel = {'native':[row['id'] for row in choices], 'selection':'all frozen in-domain Choice rows',
             'source_k_counts':{f'{s}/K{k}':n for (s,k),n in collections.Counter(
                 (r['source_family'],len(support(r))) for r in choices).items()}}
    old_protocol = json.loads((previous/'protocol.json').read_text())
    protocol = {'created':created,'name':'native_choice_order_20261104_v1',
        'split':'test_in_domain','tasks':['choice'],'rows':5000,
        'panels':['native'],'orders':'original, reversal, three fixed unique shuffles per row',
        'permutation_seed':'SHA256(choice-order-20261104: + row ID); identical across models',
        'position_balance':'not claimed; five orders retain all K options but are not exhaustive',
        'candidate_policy':'full native inventory in every order; no truncation or candidate removal',
        'semantic_alignment':'pointer readouts gathered by stable original ID; LoRA probabilities remapped; JSON keys stay attached to original text',
        'primary_metrics':['accuracy','reverse_flip_rate','any_order_flip_rate'],
        'failure_policy':'invalid JSON remains incorrect and in paired denominators',
        'uncertainty':'2000 paired source/group bootstrap replicates; no cross-seed claim',
        'token_budget':old_protocol['token_budget'],'max_batch_rows':old_protocol['max_batch_rows'],
        'max_new_tokens':old_protocol['max_new_tokens'],'generation':old_protocol['generation'],
        'selection':'frozen model, rows and all five orders; no best-order selection',
        'noul':'reuse previous audit with original applicability and panel labels',
        'score':'excluded from classification order audit by latest user correction'}
    root.mkdir(exist_ok=False)
    atomic(root/'panel.json',panel); atomic(root/'protocol.json',protocol)
    for path,digest in hashes.items():
        dst=root/'code'/path;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/path,dst);assert sha(dst)==digest
    for name,spec in manifest['models'].items():
        assert sha(spec['checkpoint'])==spec['checkpoint_sha256']
        spec['expected_predictions']=sum(len(view_orders(spec['arm'],'native',row)) for row in choices)
        assert spec['expected_predictions']==25000
        (root/name).mkdir()
    manifest.update(created=created,status='prepared',python=sys.executable,code_sha256=hashes,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        panel_sha256=sha(root/'panel.json'),protocol_sha256=sha(root/'protocol.json'),
        previous_noul_root=str(previous),scope='Choice/Noul classification flips; this run fills Choice only',
        launch_command=[sys.executable,str(PROJECT/'scripts/formal/supervise_choice_order_audit.py'),
                        '--root',str(root)])
    atomic(root/'experiment_manifest.json',manifest)
    print(root)
    print(json.dumps(panel['source_k_counts']))


if __name__=='__main__':
    main()
