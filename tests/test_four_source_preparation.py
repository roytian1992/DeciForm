import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result

def test_jsonl_unicode_line_separator_remains_inside_one_record(tmp_path):
    prepare=module('prepare_four_source_continuation')
    path=tmp_path/'data.jsonl'
    row={'id':'one','state':{'text':'before'+chr(0x2028)+'after'}}
    path.write_text(json.dumps(row,ensure_ascii=False)+chr(10))
    assert prepare.verified({'path':str(path),'count':1,'sha256':prepare.file_sha256(path)})==[row]

def test_source_average_differs_from_sample_average_and_requires_completion():
    report=module('report_full_model_comparison')
    records=[{'source_family':'large','label':'yes','prediction':'yes'} for _ in range(9)]
    records.append({'source_family':'small','label':'yes','prediction':'no'})
    complete=report.accuracy_summary(records,{'large':9,'small':1})
    assert complete['pooled_accuracy']==.9
    assert complete['macro_source_accuracy']==.5
    partial=report.accuracy_summary(records[:-1],{'large':9,'small':1})
    assert partial['sources']['large']['accuracy']==1
    assert partial['sources']['small']['accuracy'] is None
    assert partial['macro_source_accuracy'] is None and partial['pooled_accuracy'] is None
