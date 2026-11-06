"""Small, frozen, full-native end-to-end latency comparison of trained models.

Each arm runs in a fresh process on the SAME physical GPU. No training, input
truncation, prediction reuse, cross-request prefix cache, or weight merging.
"""
import argparse
import collections
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

from common import BASE, PROJECT, TASKS, atomic, encode, json_prompt, sha, support

ROOT = PROJECT / 'runs/formal_inference_timing_20261106_v1'
LAYA = PROJECT / 'runs/formal_laya_20261106_v1'
MAIN = PROJECT / 'runs/formal_main_20261101_v2_one_epoch'
D2 = PROJECT / 'runs/formal_independent_20261105_v1'
ARMS = ('lora', 'kev', 'json_sft', 'd2', 'laya_sequential', 'laya_parallel')
NAMES = dict(zip(ARMS, ('Ordinary LoRA', 'Kev', 'Generative LoRA', 'D2 (independent)', 'Laya sequential', 'Laya parallel')))


def now():
    return datetime.datetime.now().astimezone().isoformat()


def command_output(args):
    return subprocess.check_output(args, text=True).strip()


def select_rows(rows, per_task=50, seed=20261106):
    """Proportional sources; one seeded draw in each equal-size length stratum."""
    rng = random.Random(seed)
    selected = []
    for task in TASKS:
        groups = collections.defaultdict(list)
        for row in rows:
            if row['task'] == task:
                groups[row['source_family']].append(row)
        total = sum(map(len, groups.values()))
        quotas = {s: int(per_task * len(rs) / total) for s, rs in groups.items()}
        residual = sorted(groups, key=lambda s: (-(per_task * len(groups[s]) / total - quotas[s]), s))
        for source in residual[:per_task - sum(quotas.values())]:
            quotas[source] += 1
        for source in sorted(groups):
            ordered = sorted(groups[source], key=lambda r: (len(json_prompt(r)), r['id']))
            n = quotas[source]
            for j in range(n):
                selected.append(rng.choice(ordered[j * len(ordered) // n:(j + 1) * len(ordered) // n]))
    rng.shuffle(selected)
    assert len(selected) == per_task * len(TASKS)
    assert len({r['id'] for r in selected}) == len(selected)
    assert all(r['candidate_mode'] == 'complete' for r in selected)
    return selected


def prepare(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    audit = PROJECT / 'data/formal_200k_20261101_v1/cache/audit.json'
    desc = json.loads(audit.read_text())['splits']['test_in_domain']
    assert desc['native_evaluation'] and sha(desc['views_path']) == desc['views_sha256']
    rows = [json.loads(line) for line in open(desc['views_path'])]
    sample = select_rows(rows)
    sample_path = root / 'sample.jsonl'
    sample_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in sample))
    # Warm the longest sampled row per source, plus every native K represented.
    warm = {}
    for r in sample:
        for key in (r['source_family'], 'k/' + str(len(support(r)))):
            if key not in warm or len(json_prompt(r)) > len(json_prompt(warm[key])):
                warm[key] = r
    warm_ids = sorted({r['id'] for r in warm.values()})
    checkpoints = {a: MAIN / a / 'epoch_1.pt' for a in ARMS[:3]}
    checkpoints['d2'] = D2 / 'epoch_1.pt'
    checkpoints.update({f'laya_{layout}': LAYA / layout / 'checkpoint/model.safetensors' for layout in ('sequential', 'parallel')})
    files = set((PROJECT / 'scripts/formal').glob('*.py'))
    files.update((PROJECT / 'src/deciform').glob('*.py'))
    files.update((PROJECT / 'scripts').glob('kev_ours*.py'))
    files.update((PROJECT / 'runs/kev_ours_v2_20261101_v1/code/vendor/kev').rglob('*.py'))
    files.update((LAYA / 'vendor/laya_release/laya').rglob('*.py'))
    cfg = dict(created_at=now(), seed=20261106, sample_path=str(sample_path), sample_sha256=sha(sample_path),
               population_path=desc['views_path'], population_sha256=desc['views_sha256'], population_rows=len(rows),
               sample_rows=len(sample), counts_by_task=dict(collections.Counter(r['task'] for r in sample)),
               counts_by_source=dict(collections.Counter(r['source_family'] for r in sample)),
               counts_by_task_k=dict(collections.Counter(f"{r['task']}/K{len(support(r))}" for r in sample)),
               sample_selection='proportional source quotas within task; seeded draw from equal-sized character-length strata; no labels/predictions used',
               arms=list(ARMS), repeats=3, batch_size=1, warmup_ids=warm_ids, max_new_tokens=desc['max_json_response_tokens'] + 8,
               checkpoints={a: {'path': str(p), 'sha256': sha(p)} for a, p in checkpoints.items()},
               d2_config=json.loads((D2 / 'config.json').read_text()),
               base=str(BASE), base_config_sha256=sha(BASE / 'config.json'),
               code_hashes={str(p): sha(p) for p in sorted(files)}, python=sys.executable,
               cpu_threads=4, cpu_affinity=sorted(os.sched_getaffinity(0))[:4],
               timing='synchronized wall time; raw row -> tokenization -> GPU transfer -> inference -> CPU answer; excludes model load, warmup, metric computation, disk/network',
               inference='eval, torch.inference_mode, BF16 autocast, existing native implementations, unmerged adapters, no torch.compile, no cross-request cache',
               throughput='serial requests per second, reciprocal of mean latency; not saturated serving throughput',
               git_head=command_output(['git', 'rev-parse', 'HEAD']))
    atomic(root / 'config.json', cfg)
    atomic(root / 'status.json', {'status': 'prepared', 'at': now()})
    print(json.dumps({k: cfg[k] for k in ('sample_rows', 'counts_by_task_k', 'counts_by_source')}, indent=2), flush=True)


def load_model(arm, cfg):
    import torch
    if arm.startswith('laya_'):
        sys.path.insert(0, str(LAYA / 'vendor/laya_release'))
        from laya.common import build_model
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        path = Path(cfg['checkpoints'][arm]['path']).parent
        model_cfg = json.loads((path / 'rl_agent_config.json').read_text())
        tok = AutoTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
        model = build_model(model_cfg, encoder_dir=str(path / 'encoder'), pretrained=False)
        model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        model.encoder.config.reference_compile = False
        model = model.float().cuda()
    else:
        if arm == 'd2':
            from models_independent import FormalModel
            d = cfg['d2_config']
            model = FormalModel('tcdl', BASE, checkpointing=False, ablation=d['ablation'], distribution_objective=d['distribution_objective'])
        else:
            from models import FormalModel
            model = FormalModel(arm, BASE, checkpointing=False)
        state = torch.load(cfg['checkpoints'][arm]['path'], map_location='cpu', weights_only=False)
        inventory = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if set(inventory) != set(state['trainable']):
            raise ValueError(f'{arm}: checkpoint parameter inventory mismatch')
        with torch.no_grad():
            for name, p in inventory.items():
                p.copy_(state['trainable'][name])
        del state
        tok = model.tok
    model.eval().requires_grad_(False)
    return model, tok


def request(model, tok, arm, row, max_new_tokens):
    import torch
    t0 = time.perf_counter()
    if arm.startswith('laya_'):
        from laya_adapter import make_parts, encode_parts
        encoded = encode_parts(make_parts(row, tok), arm.removeprefix('laya_'))
        # Target construction is preparation-only; never fed to model.
        encoded.pop('target')
        token_info = {'total_input_tokens': len(encoded['ids']), 'max_sequence_tokens': len(encoded['ids'])}
    elif arm == 'd2':
        from independent_candidates import split_candidates
        encoded = split_candidates(encode(row, 'kev', tok))
        token_info = {'total_input_tokens': encoded['original_tokens'], 'max_sequence_tokens': len(encoded['prefix']) + max(map(len, encoded['branches'])), 'prefix_tokens': len(encoded['prefix'])}
    elif arm == 'json_sft':
        encoded = {'prompt': tok(json_prompt(row), add_special_tokens=True, truncation=False).input_ids}
        token_info = {'total_input_tokens': len(encoded['prompt']), 'max_sequence_tokens': len(encoded['prompt'])}
    else:
        encoded = encode(row, arm, tok)
        seqs = encoded['ids_list'] if arm == 'lora' else [encoded['ids']]
        token_info = {'total_input_tokens': sum(map(len, seqs)), 'max_sequence_tokens': max(map(len, seqs))}
    t1 = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        if arm.startswith('laya_'):
            from laya.common import collate_items
            batch = collate_items([[encoded]], tok.pad_token_id)
            fields = ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')
            batch = {k: v.cuda() for k, v in batch.items() if k in fields or k in ('position_ids', 'option_ids')}
            extra = {k: batch[k] for k in ('position_ids', 'option_ids') if k in batch}
            z, _ = model(*(batch[k] for k in fields), **extra)
            probs = z[0, :len(support(row))].float().softmax(-1).cpu()
            assert torch.isfinite(probs).all()
            prediction = {'prediction_index': int(probs.argmax()), 'probabilities': probs.tolist(), 'valid': True}
            if row['task'] == 'score':
                from score_distribution import decisions
                prediction['prediction_index'] = decisions(support(row), prediction['probabilities'])['indices']['median']
        else:
            prediction = model.predict([row], [encoded], max_new_tokens)[0]
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return prediction, token_info, {'encode_ms': (t1-t0)*1000, 'inference_ms': (t2-t1)*1000, 'e2e_ms': (t2-t0)*1000}


def describe(values):
    vs = sorted(values)
    n = len(vs)
    return {'count': n, 'mean_ms': statistics.mean(vs), 'median_ms': statistics.median(vs),
            'p95_ms': vs[max(0, math.ceil(.95*n)-1)], 'serial_requests_s': 1000/statistics.mean(vs)}


def summarize(records):
    groups = collections.defaultdict(list)
    for r in records:
        for key in ('all', 'task/' + r['task'], 'source/' + r['source'], f"task_k/{r['task']}/{r['k']}", 'repeat/' + str(r['repeat'])):
            groups[key].append(r)
    return {key: {**describe([r['e2e_ms'] for r in rs]),
                  'mean_encode_ms': statistics.mean(r['encode_ms'] for r in rs),
                  'mean_inference_ms': statistics.mean(r['inference_ms'] for r in rs),
                  'valid_requests': sum(r['prediction']['valid'] for r in rs)} for key, rs in groups.items()}


def worker(root, arm):
    import torch
    import transformers
    cfg = json.loads((root / 'config.json').read_text())
    out = root / arm
    out.mkdir(exist_ok=False)
    status = lambda **kw: atomic(out / 'status.json', {'at': now(), 'arm': arm, **kw})
    try:
        torch.set_num_threads(cfg['cpu_threads'])
        os.sched_setaffinity(0, cfg['cpu_affinity'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.set_device(0)
        assert sha(cfg['sample_path']) == cfg['sample_sha256']
        assert sha(cfg['checkpoints'][arm]['path']) == cfg['checkpoints'][arm]['sha256']
        rows = [json.loads(line) for line in open(cfg['sample_path'])]
        env = {'started_at': now(), 'pid': os.getpid(), 'python': sys.version, 'torch': torch.__version__,
               'transformers': transformers.__version__, 'cuda': torch.version.cuda,
               'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'cpu_affinity': sorted(os.sched_getaffinity(0)),
               'gpu': command_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit', '--format=csv'])}
        atomic(out / 'environment.json', env)
        status(status='loading')
        started = time.perf_counter()
        model, tok = load_model(arm, cfg)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter()-started
        params = sum(p.numel() for p in model.parameters())
        gc.collect()
        torch.cuda.empty_cache()
        records = []
        status(status='warmup', rows=len(cfg['warmup_ids']))
        start_warm = time.perf_counter()
        with torch.inference_mode():
            for row in rows:
                if row['id'] in cfg['warmup_ids']:
                    request(model, tok, arm, row, cfg['max_new_tokens'])
            warm_seconds = time.perf_counter()-start_warm
            torch.cuda.reset_peak_memory_stats()
            baseline_memory = torch.cuda.memory_allocated()
            with open(out / 'requests.jsonl', 'x', buffering=1) as handle:
                for repeat in range(cfg['repeats']):
                    order = list(range(len(rows)))
                    random.Random(cfg['seed']+repeat).shuffle(order)
                    for number, i in enumerate(order):
                        row = rows[i]
                        torch.cuda.synchronize()
                        prediction, tokens, timing = request(model, tok, arm, row, cfg['max_new_tokens'])
                        record = {'id': row['id'], 'task': row['task'], 'source': row['source_family'],
                                  'k': len(support(row)), 'repeat': repeat, 'order': number,
                                  'prediction': prediction, **tokens, **timing}
                        if 'raw_output' in prediction:
                            record['generated_tokens_including_eos'] = len(tok(prediction['raw_output'], add_special_tokens=False).input_ids)
                        records.append(record)
                        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if len(records) % 25 == 0:
                            status(status='measuring', completed=len(records), total=len(rows)*cfg['repeats'],
                                   mean_ms=statistics.mean(r['e2e_ms'] for r in records))
                            print(f'{arm}: {len(records)}/{len(rows)*cfg["repeats"]}', flush=True)
        summary = {'arm': arm, 'name': NAMES[arm], 'completed_at': now(), 'status': 'complete',
                   'unique_rows': len(rows), 'requests': len(records), 'repeats': cfg['repeats'],
                   'load_seconds': load_seconds, 'warmup_seconds': warm_seconds, 'parameters': params,
                   'baseline_allocated_gib': baseline_memory/2**30,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30,
                   'metrics': summarize(records)}
        assert len(records) == len(rows)*cfg['repeats']
        atomic(out / 'summary.json', summary)
        status(status='complete', completed=len(records), total=len(records))
    except Exception:
        status(status='failed', traceback=traceback.format_exc())
        raise


def report(root):
    cfg = json.loads((root / 'config.json').read_text())
    summaries = {a: json.loads((root/a/'summary.json').read_text()) for a in ARMS if (root/a/'summary.json').exists()}
    atomic(root/'summary.json', {'at': now(), 'complete_arms': len(summaries), 'total_arms': len(ARMS), 'arms': summaries})
    text = ['## 小样本推理时间（同一 A100，batch=1）', '',
            '固定主测试集 150 条，Choice/Noul/Score 各 50 条，来源按原比例分配；完整原始候选与文本。每条重复 3 次，先预热，计入分词、GPU 传输、推理、答案解码，不含加载、文件与网络 I/O。保留原生推理实现，BF16 autocast，adapter 未合并。', '',
            '| 模型 | 平均 ms/条 | Choice | Noul | Score | P95 ms | 峰值分配 GiB |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        if arm not in summaries:
            text.append(f'| {NAMES[arm]} | 待完成 | — | — | — | — | — |')
            continue
        s = summaries[arm]; m = s['metrics']
        text.append(f"| {NAMES[arm]} | {m['all']['mean_ms']:.2f} | {m['task/choice']['mean_ms']:.2f} | {m['task/noul']['mean_ms']:.2f} | {m['task/score']['mean_ms']:.2f} | {m['all']['p95_ms']:.2f} | {s['peak_allocated_gib']:.2f} |")
    text += ['', f'完成 {len(summaries)}/{len(ARMS)} 组。每组 150 个唯一请求 × 3 次；不是 450 条独立数据。',
             'Laya 是约 0.42B ModernBERT，其他模型是 Qwen3.5-9B，跨骨干速度不等于机制本身的增益。串行吞吐仅为平均延迟的倒数，不代表批处理服务吞吐。固定小样本用于效率测量，不重新估计主实验准确率；每个请求的概率/原始 JSON、有效性、长度、K 和重复序号均保留。',
             f'协议与原始记录：`{root.relative_to(PROJECT)}`；脚本：`scripts/formal/benchmark_inference.py`。', '']
    md = '\n'.join(text)
    (root/'report.md').write_text(md)
    # Shared master document lock used by all ongoing result writers.
    import fcntl
    master = PROJECT/'docs/experiment_record_final_d2_20261105.md'
    with open(PROJECT/'runs/final_record_update.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = master.read_text()
        start, end = '<!-- INFERENCE_TIMING_START -->', '<!-- INFERENCE_TIMING_END -->'
        block = start+'\n'+md+end
        if start in old:
            left, rest = old.split(start, 1)
            _, right = rest.split(end, 1)
            new = left+block+right
        else:
            new = old.rstrip()+'\n\n'+block+'\n'
        tmp = master.with_suffix('.timing.tmp')
        tmp.write_text(new); tmp.replace(master)


def run(root):
    cfg = json.loads((root/'config.json').read_text())
    for arm in ARMS:
        atomic(root/'status.json', {'status': 'running', 'arm': arm, 'at': now()})
        with open(root/f'{arm}.log', 'x') as log:
            args = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--root', str(root), '--arm', arm]
            code = subprocess.call(args, stdout=log, stderr=subprocess.STDOUT)
        report(root)
        if code:
            atomic(root/'status.json', {'status': 'failed', 'arm': arm, 'returncode': code, 'at': now()})
            raise RuntimeError(f'{arm} failed; see {root/arm/"status.json"}')
    atomic(root/'status.json', {'status': 'complete', 'at': now(), 'arms': list(ARMS)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'run', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.action == 'worker':
        worker(args.root, args.arm)
    else:
        globals()[args.action](args.root)
