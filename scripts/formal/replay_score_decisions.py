"""Matched CPU replay; preserves all frozen model predictions and reports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common import PROJECT, sha
from score_distribution import PROTOCOL, score_rule_report

REFERENCES = {
    'A4': 'formal_v6_ablations_20261103_v1/a4_ce_only',
    'Kev': 'formal_main_20261101_v2_one_epoch/kev',
    'ordinary_LoRA': 'formal_main_20261101_v2_one_epoch/lora',
    'full_v6': 'formal_ordinal_20261102_v1',
    'L4_ordinal_auxiliary': 'formal_typed_losses_20261103_v1/l4_score_ordinal',
    'M1_regression_auxiliary': 'formal_decision_mechanisms_20261103_v1/m1_score_regression',
    'M2_conditional_choice': 'formal_decision_mechanisms_20261103_v1/m2_choice_metric',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--split',choices=['dev','test_in_domain'],required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'experiment_manifest.json').read_text())
    protocol_path = root/'evaluation_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['name']==manifest['evaluation_protocol']==PROTOCOL
    assert sha(protocol_path)==manifest['protocol_file_sha256']
    assert protocol['primary_score_rule']=='median'
    output = root/'reference_score_rules'/f'{args.split}.json'
    if output.exists():
        raise FileExistsError(output)
    report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'split':args.split,
              'protocol':protocol,'protocol_sha256':sha(protocol_path),'models':{},
              'scope':'paired full-native saved-probability CPU replay; no checkpoint or original output changes',
              'excluded':'JSON-generative LoRA has no comparable full predictive score distribution'}
    signature = None
    for name,relative in REFERENCES.items():
        directory = PROJECT/'runs'/relative/'evaluations'
        files = sorted(directory.glob(f'epoch_1.{args.split}.rank*.jsonl'))
        rows = [json.loads(line) for path in files for line in path.read_text().splitlines()]
        source_metric = directory/f'epoch_1.{args.split}.json'
        metrics = json.loads(source_metric.read_text())
        expected = 6000 if args.split=='dev' else 15000
        assert metrics['native_full_candidates'] and len(rows)==metrics['rows']==expected
        assert len({r['id'] for r in rows})==expected
        current = sorted((r['id'],r['label'],tuple(r['support'])) for r in rows if r['task']=='score')
        if signature is None:
            signature=current
        assert current==signature, f'Mismatched population: {name}'
        decision_report=score_rule_report(rows,args.split)
        assert abs(decision_report['rules']['argmax']['mae']-metrics['task/score']['mae'])<1e-12
        report['models'][name] = {
            'primary': {'choice_accuracy':metrics['task/choice']['accuracy'],
                        'noul_accuracy':metrics['task/noul']['accuracy'],
                        'score_mae':decision_report['primary_mae'],'score_rule':'median'},
            'score_rules':decision_report,
            'source_sha256':{str(p):sha(p) for p in [*files,source_metric]},
        }
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(output)
    for name,record in report['models'].items():
        print(name,json.dumps(record['score_rules']['rules']))


if __name__=='__main__':
    main()
