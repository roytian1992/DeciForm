"""Portable inference with the exact released candidate-scoring interface."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .choice import ChoiceHead, choice_response, encode_prompts, render_choice_prompts
from .joint_data import file_sha256
from .lora import STANDARD_PROGRAM, initialize_adapter


class DeciFormModel:
    def __init__(self, model, tokenizer, head, temperature, device):
        self.model = model
        self.tokenizer = tokenizer
        self.head = head
        self.temperature = temperature
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(cls, base_model, checkpoint, *, device="cuda:0", dtype="fp32"):
        base_model, checkpoint = Path(base_model), Path(checkpoint)
        hashes = json.loads((checkpoint / "hashes.json").read_text())
        for name in ("choice_head.pt", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"):
            if file_sha256(checkpoint / name) != hashes[name]:
                raise ValueError(f"checkpoint hash mismatch: {name}")
        state = torch.load(checkpoint / "choice_head.pt", map_location="cpu", weights_only=True)
        if file_sha256(base_model / "config.json") != state["model_config_sha256"]:
            raise ValueError("backbone configuration differs from this checkpoint")
        torch.set_float32_matmul_precision("highest")
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForImageTextToText.from_pretrained(base_model, local_files_only=True,
            dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[dtype],
            device_map={"": device}, low_cpu_mem_usage=True)
        config = json.loads((checkpoint / "adapter/adapter_config.json").read_text())
        model.model, _ = initialize_adapter(model.model, STANDARD_PROGRAM, config["r"],
                                             config["target_modules"], checkpoint / "adapter")
        model.eval().requires_grad_(False)
        head = ChoiceHead(state["hidden_size"]).to(device).eval()
        head.load_state_dict(state["state_dict"])
        return cls(model, tokenizer, head, state["temperature"], device)

    def predict(self, row, *, batch_size=16, order=None):
        ids, prompts = render_choice_prompts(row, order)
        vectors = encode_prompts(self.model, self.tokenizer, prompts, device=self.device,
                                 batch_size=batch_size, max_prompt_tokens=None,
                                 bucket_by_length=True)
        with torch.inference_mode():
            return choice_response(ids, self.head(vectors.to(self.device)), temperature=self.temperature)
