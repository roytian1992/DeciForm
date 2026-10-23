import json
import pytest
from deciform.joint_data import file_sha256
from deciform.unified_comparison import relationship_inventory,grouped_results,validate_prediction,read_run_predictions
from deciform.unified_native import native_input,option_fields


def row(name="one",source="trained",label="id_b"):
    return {"id":name,"source_family":source,"label":label,"state":{"text":"complete evidence"},
        "question":{"type":"choice","instructions":"pick a category","options":["a","b"],
                    "option_ids":["id_a","id_b"],"option_definitions":["category A definition","category B definition"]}}


def prediction(r):
    return {"id":r["id"],"source_family":r["source_family"],"label":r["label"],
        "prediction":"id_b","probabilities":{"id_a":.2,"id_b":.8},"confidence":.8}


def test_source_relation_comes_from_actual_training_plan(tmp_path):
    plan=tmp_path/"plan.json"
    plan.write_text(json.dumps({"sources":[{"name":"trained","count":7}]}))
    manifest={"training_plans":{"parent":{"path":str(plan),"sha256":file_sha256(plan)}},
        "streams":[{"source_families":["trained"],"training_source_family":True},
                   {"source_families":["new"],"training_source_family":False}]}
    rows=[row(),row("two","new"),row("three","trained")]
    inventory=relationship_inventory(manifest,rows)
    assert inventory["groups"]["trained_source_holdout"]["record_count"]==2
    assert inventory["groups"]["unseen_source"]["record_count"]==1
    results=grouped_results([prediction(rows[1]),prediction(rows[0])],inventory)
    assert results["groups"]["unseen_source"]["metrics"]["accuracy"]==1
    assert results["groups"]["trained_source_holdout"]["metrics"] is None
    assert results["overall"]["metrics"] is None
    manifest["streams"][0]["training_source_family"]=False
    with pytest.raises(ValueError,match="relationship disagrees"):
        relationship_inventory(manifest,rows)


def test_native_models_receive_definitions_without_the_gold_label():
    original=row()
    changed={**original,"label":"id_a"}
    for model in ["kev","open_jev"]:
        assert native_input(original,model)==native_input(changed,model)
    open_input=native_input(original,"open_jev")
    assert open_input["options"]==original["question"]["option_definitions"]
    assert open_input["answer_keys"]==["id_a","id_b"]
    assert native_input(original,"kev")["questions"][0]["options"]==open_input["options"]
    assert option_fields(original)[0]==["id_a","id_b"]


def test_prediction_checks_option_ids_and_probabilities():
    original=row();p=prediction(original)
    validate_prediction(p,original)
    with pytest.raises(ValueError,match="option IDs"):
        validate_prediction({**p,"probabilities":{"a":.2,"b":.8}},original)
    with pytest.raises(ValueError,match="probability distribution"):
        validate_prediction({**p,"probabilities":{"id_a":.2,"id_b":float("nan")}},original)


def test_reader_uses_committed_progress_and_validates_population(tmp_path):
    r=row();p=prediction(r)
    (tmp_path/"rank_0_manifest.json").write_text(json.dumps({"test_manifest_sha256":"locked"}))
    (tmp_path/"rank_0_progress.json").write_text(json.dumps({"rank":0,"status":"running","evaluated_records":1}))
    path=tmp_path/"rank_0_predictions.jsonl"
    path.write_text(json.dumps(p)+"\n"+"unfinished write")
    predictions,status=read_run_predictions(tmp_path,[r],"locked")
    assert predictions==[p] and status["status"]=="running"
    with pytest.raises(ValueError,match="different test population"):
        read_run_predictions(tmp_path,[r],"changed")
    with pytest.raises(ValueError,match="incomplete"):
        read_run_predictions(tmp_path,[r],"locked",require_complete=True)
