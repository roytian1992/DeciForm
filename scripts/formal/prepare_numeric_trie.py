"""Freeze four fresh numeric-trie runs and label-independent prompt caches."""
import copy
from datetime import datetime
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from common import BASE, PROJECT, atomic, sha
from numeric_trie import ALPHABET, NumericTrieReadout, TrieObjective, numeric_tree
from numeric_score_distribution import NumericCDFReadout
from score_baselines import json_prompt


def main():
    torch.set_num_threads(4)
    root = PROJECT/'runs/formal_numeric_trie_20261106_v1'
    cache_root = PROJECT/'data/score_numeric_trie_20261106_v1'
    if root.exists() or cache_root.exists():
        raise FileExistsError('Use a new version; never overwrite a frozen run')
    cache_root.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokens = [tok.encode(c, add_special_tokens=False) for c in ALPHABET]
    assert all(len(t) == 1 for t in tokens)
    token_ids = [t[0] for t in tokens]
    fixed_prefix = tok.encode('{"score":', add_special_tokens=False)
    index_path = BASE/'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    tensors = {}
    source_hashes = {}
    for key, name in [('output_rows','lm_head.weight'),('input_rows','model.language_model.embed_tokens.weight')]:
        path = BASE/index[name]
        if str(path) not in source_hashes:
            source_hashes[str(path)] = sha(path)
        with safe_open(path, framework='pt', device='cpu') as handle:
            sliced = handle.get_slice(name)
            tensors[key] = torch.cat([sliced[t:t+1] for t in token_ids]).clone()
    anchors = cache_root/'numeric_anchors.pt'
    torch.save({**tensors,'alphabet':ALPHABET,'token_ids':token_ids,'base':str(BASE.resolve())},anchors)
    atomic(cache_root/'numeric_anchors.json', {'path':str(anchors),'sha256':sha(anchors),
        'source_files_sha256':source_hashes,'index_sha256':sha(index_path),'alphabet':ALPHABET,
        'token_ids':token_ids,'fixed_json_prefix_ids':fixed_prefix,
        'scope':'Original pretrained Base input/output rows only; no fine-tuned checkpoint.'})
    specs = {
        'asap': ('runs/formal_score_objectives_20261106_v1/r1_cdf_mean',
                 'runs/formal_score_numeric_asap_20261105_v1',
                 'data/score_baseline_encodings_20261105_v1/asap/audit.json',
                 {'train':7784,'dev':2593,'test_in_domain':2598}),
        'continuous': ('runs/formal_score_recheck_20261106_v1/continuous_r1_cdf_mean',
                       'runs/formal_score_numeric_20261105_v1/numeric_cdf',
                       'data/score_baseline_encodings_stsb_enzh_20261106_v1/audit.json',
                       {'train':8129,'dev':2500,'test_in_domain':2379})}
    configs, manifests, audits = {}, {}, {}
    verified_supports = set()
    for panel, (previous, reference, audit_name, counts) in specs.items():
        configs[panel] = json.loads((PROJECT/previous/'config.json').read_text())
        manifests[panel] = json.loads((PROJECT/previous/'experiment_manifest.json').read_text())
        original = json.loads((PROJECT/audit_name).read_text())
        assert original['status'] == 'passed'
        assert {s:d['rows'] for s,d in original['splits'].items()} == counts
        audit = copy.deepcopy(original)
        output = cache_root/panel; output.mkdir()
        for split, desc in audit['splits'].items():
            rows = [json.loads(line) for line in open(desc['views_path'])]
            assert sha(desc['views_path']) == desc['views_sha256']
            entry = desc['arms']['json_sft']
            assert sha(entry['path']) == entry['sha256']
            with open(entry['path'],'rb') as f: encoded = pickle.load(f)
            assert len(rows) == len(encoded) == desc['rows']
            result = []
            for row, old in zip(rows, encoded):
                # Retain the exact complete prompt previously used by JSON LoRA.
                assert old['answer'][:len(fixed_prefix)] == fixed_prefix
                assert json_prompt(row) == json_prompt({**row,'label':row['score_support'][0]})
                strings, _, _, _ = numeric_tree(tuple(row['score_support']))
                # Verify the canonical score suffix agrees with tokenization.
                if strings not in verified_supports:
                    for string in strings:
                        suffix = [token_ids[ALPHABET.index(c)] for c in string]
                        assert tok.encode('{"score":'+string,add_special_tokens=False) == fixed_prefix+suffix
                    verified_supports.add(strings)
                result.append({'ids':old['prompt']+fixed_prefix,'tasks':['score'],'labels':[None]})
            # Full retokenization guards the reused prompt transport, including longest inputs.
            reconstructed = tok([json_prompt(r) for r in rows],add_special_tokens=True,truncation=False,padding=False).input_ids
            assert all(e['ids']==tokens+fixed_prefix for e,tokens in zip(result,reconstructed))
            dest = output/(split+'.pkl')
            with dest.open('wb') as f: pickle.dump(result,f,protocol=pickle.HIGHEST_PROTOCOL)
            lengths = [len(e['ids']) for e in result]
            desc['arms'] = {'kev': {'path':str(dest),'sha256':sha(dest),'max_tokens':max(lengths),
                'forward_tokens':sum(lengths),'longest_indices':sorted(range(len(lengths)),key=lengths.__getitem__,reverse=True)[:10]}}
        audit.update(status='passed',source_audit_path=str(PROJECT/audit_name),source_audit_sha256=sha(PROJECT/audit_name),
            encoding='Exact JSON baseline full prompt + constant score field prefix; no answer tokens',
            labels_in_input=False,complete_retokenization_verified=True,tokenization_no_truncation=True,
            encoder_sha256=sha(Path(__file__)),numeric_anchor_sha256=sha(anchors))
        atomic(output/'audit.json',audit); audits[panel] = output/'audit.json'
        print(json.dumps({'panel':panel,'cache':'passed','counts':counts}),flush=True)
    additions = ['numeric_trie.py','models_numeric_trie.py','worker_numeric_trie_asap.py',
                 'worker_numeric_trie_continuous.py','supervise_numeric_trie.py','prepare_numeric_trie.py',
                 'score_baselines.py','schedule_research_queue.py']
    paths = set().union(*(m['code_sha256'] for m in manifests.values()))
    for manifest in manifests.values():
        for p, digest in manifest['code_sha256'].items():
            assert sha(PROJECT/p) == digest, 'Reference dependency changed: '+p
    paths |= {'scripts/formal/'+p for p in additions} | {'tests/test_numeric_trie.py'}
    hashes = {p:sha(PROJECT/p) for p in sorted(paths)}
    root.mkdir()
    for p in paths:
        target=root/'code'/p; target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PROJECT/p,target)
    count = lambda m:sum(p.numel() for p in m.parameters())
    delta = count(NumericTrieReadout(tensors['output_rows'],tensors['input_rows']))-count(NumericCDFReadout(256))
    arms,queues = {},{}
    created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    specifications=[('asap','n1_trie_ce','0,1','trie_ce'),('asap','n2_trie_ce_cdf','2,3','trie_ce_cdf'),
                    ('continuous','n1_trie_ce','4,5','trie_ce'),('continuous','n2_trie_ce_cdf','6,7','trie_ce_cdf')]
    for panel, suffix, gpus, kind in specifications:
        name=panel+'_'+suffix;out=root/name;out.mkdir()
        _,reference,_,counts=specs[panel]
        objective=TrieObjective(kind,cdf_weight=0. if kind=='trie_ce' else .5)
        cfg=copy.deepcopy(configs[panel])
        cfg.update(output=str(out),label=name,experiment_name=name,distribution_objective=objective.inventory(),
            task_losses={'score':kind},worker_variant=str(PROJECT/f'scripts/formal/worker_numeric_trie_{panel}.py'),
            method_variant='numeric_trie_score_research',score_architecture='pretrained_numeric_root_trie',
            score_head='Frozen pretrained root-digit logits + 256-dimensional GRU decimal trie, exact full-support leaf probabilities',
            objective_description='Soft CE' if kind=='trie_ce' else '0.5 soft CE + 0.5 scaled ordered CDF log loss',
            cache_audit_path=str(audits[panel]),cache_audit_sha256=sha(audits[panel]),
            numeric_anchor_path=str(anchors),numeric_anchor_sha256=sha(anchors),
            query_encoder='Complete JSON baseline prompt + fixed score-field prefix; no gold answer',
            candidate_encoder='All legal numeric strings through compact shared decoder',
            candidate_positions='Native numeric support, not displayed candidate positions',
            classification_ties='Not applicable: fixed native median decoding',
            paired_reference=str(PROJECT/reference))
        assert cfg['parent_checkpoint'] is None and cfg['train_rows']==counts['train'] and cfg['epochs']==1
        atomic(out/'config.json',cfg)
        params=json.loads((PROJECT/reference/'initialization.json').read_text())['trainable_parameters']+delta
        m=copy.deepcopy(manifests[panel]);supervisor=str(PROJECT/'scripts/formal/supervise_numeric_trie.py')
        m.update(created=created,python=sys.executable,config=str(out/'config.json'),config_sha256=sha(out/'config.json'),
            worker=cfg['worker_variant'],supervisor=supervisor,gpus=gpus,code_sha256=hashes,code_snapshot=str(root/'code'),
            experiment_name=name,distribution_objective=objective.inventory(),expected_trainable_parameters=params,
            numeric_initialization_reference=str(PROJECT/reference/'initialization.json'),paired_reference=str(PROJECT/reference),
            numeric_anchor_sha256=sha(anchors),
            tmux_session='deciform_numeric_trie_g'+gpus.replace(',','')+'_20261106',
            launch_command=[sys.executable,supervisor,'--manifest',str(out/'experiment_manifest.json')],
            resume_command=[sys.executable,supervisor,'--manifest',str(out/'experiment_manifest.json'),'--resume'],
            comparison='Same frozen data/order/seed/epoch/base LoRA initialization. New explicit JSON-prefix numeric decoder. No parent checkpoint. Not an isolated loss-only contrast to D2.',
            execution_benchmark_scope='Exact new model passes longest/dense/first-global-batch distributed backward with zero updates.',
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip())
        m.pop('execution_benchmark',None);m.pop('execution_benchmark_sha256',None)
        atomic(out/'experiment_manifest.json',m)
        arms[name]={'manifest':str(out/'experiment_manifest.json'),'gpus':gpus,'parameters':params,'objective':objective.inventory()}
        queues[gpus]=[name]
    atomic(root/'experiment_manifest.json',{'created':created,'python':sys.executable,'arms':arms,'gpu_queues':queues,
        'queue_dependencies':{},'code_sha256':hashes,'fresh_base_only':True,'parent_checkpoint':None,'epochs':1,'global_batch':64,
        'main_d2_unchanged':True,'numeric_anchor_sha256':sha(anchors),
        'criterion':'A common mechanism must lower native test MAE on ASAP, STS-B and WMT en-zh vs fresh JSON replay; do not cherry-pick per-source arms. Single-seed evidence is not cross-seed stability.',
        'scope':'Two mechanisms, each trained separately on ASAP and STS-B/WMT panels, as in existing specialist comparisons.'})
    print(json.dumps({'root':str(root),'arms':arms},indent=2))


if __name__=='__main__':main()
