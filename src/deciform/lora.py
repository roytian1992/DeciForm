"""Standard PEFT LoRA used by the released DeciForm checkpoint."""
import json
from pathlib import Path

from peft import LoraConfig, PeftModel, get_peft_model

STANDARD_PROGRAM = "standard_lora_v1"


def program_metadata(program, rank, target_modules):
    if program != STANDARD_PROGRAM or rank < 1:
        raise ValueError("expected standard LoRA with positive rank")
    return {"adapter_program": program, "rank": rank, "lora_alpha": 2 * rank,
            "target_modules": list(target_modules), "lora_dropout": 0.0}


def initialize_adapter(model, program, rank, targets, parent_dir=None):
    program_metadata(program, rank, targets)
    model.requires_grad_(False)
    if parent_dir is None:
        return get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank,
            lora_dropout=0.0, target_modules=list(targets), bias="none")), {
                "mode": "fresh", "parent": None}
    parent_dir = Path(parent_dir)
    config = json.loads((parent_dir / "adapter_config.json").read_text())
    if config.get("r") != rank or set(config["target_modules"]) != set(targets):
        raise ValueError("checkpoint rank and target modules must match")
    if config.get("use_dora") or config.get("use_rslora") or config.get("bias", "none") != "none":
        raise ValueError("checkpoint must use standard LoRA")
    loaded = PeftModel.from_pretrained(model, parent_dir, is_trainable=True)
    return loaded, {"parent": str(parent_dir), "source_program": STANDARD_PROGRAM,
                    "target_program": program, "rank": rank, "mode": "continue",
                    "max_weight_error": 0.0}
