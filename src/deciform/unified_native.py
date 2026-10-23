"""Map common decision inputs to native baseline interfaces without gold labels."""
import json


def option_fields(row):
    question = row["question"]
    options = question["options"]
    ids = question.get("option_ids", options)
    definitions = question.get("option_definitions", options)
    if question["type"] != "choice" or not len(ids)==len(definitions)==len(options):
        raise ValueError("unaligned choice options")
    if len(ids)<2 or len(set(ids))!=len(ids) or not all(isinstance(v,str) for v in [*ids,*definitions]):
        raise ValueError("choice options require unique IDs and text definitions")
    return list(ids),list(definitions)


def native_input(row, model_name):
    ids, definitions = option_fields(row)
    state = row["state"]
    text = state["text"] if isinstance(state,dict) and set(state)=={"text"} else state
    if not isinstance(text,str):
        text = json.dumps(text,sort_keys=True,ensure_ascii=False)
    question = row["question"]["instructions"]
    if model_name=="open_jev":
        return {"id":row["id"], "state":text, "kind":"choice", "question":question,
                "options":definitions, "answer_keys":ids}
    if model_name=="kev":
        # Native encoder requires a label slot, but inference never receives gold.
        return {"state":text, "questions":[{"instr":question, "options":definitions, "label":0}]}
    raise ValueError(model_name)


def native_probabilities(model_name, model, tokenizer, rows):
    import torch
    records=[native_input(row,model_name) for row in rows]
    if model_name=="open_jev":
        with torch.inference_mode():
            logits=model(records)
        temperature=float(model.temperature)
        return [torch.softmax(v.float().cpu()/temperature,-1).tolist() for v in logits]
    from kev.model import admit
    encodings=[admit(model,tokenizer,r,truncate=False) for r in records]
    if any(enc.get("state_truncated",False) for enc in encodings):
        raise ValueError("baseline input was truncated")
    with torch.inference_mode():
        logits=model.forward_batch(encodings)
    # Kev's head already applies its frozen temperature.
    return [torch.softmax(v[0].float().cpu(),-1).tolist() for v in logits]
