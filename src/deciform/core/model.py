"""Public facade for Qwen and jointly trained ModernBERT DeciForm models."""

import copy
import json
import math
from pathlib import Path
import torch
from torch import nn
from .schemas import TASKS, normalize, support, canonical_order, semantic_key
from .encoding import qwen_encode, modern_encode, collate_modern
from .objectives import request_loss
from .components.numeric_trie import NumericTrieReadout, ALPHABET
from .components.task_context import task_context
from .components.private_residual import private_context


class DecisionModel(nn.Module):
    def __init__(self, config, device="cpu", tokenizer_source=None):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.backbone = config["backbone"]
        self.device = torch.device(device)
        from transformers import AutoTokenizer

        load = {"local_files_only": config.get("local_files_only", False)}
        if config.get("revision"):
            load["revision"] = config["revision"]
        base = config["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_source) if tokenizer_source else base,
            **({} if tokenizer_source else load),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        token_ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in ALPHABET]
        if any(len(x) != 1 for x in token_ids):
            raise ValueError("Numeric anchor characters must each be one token")
        token_ids = [x[0] for x in token_ids]
        query = config.get("query_size", 256)
        if self.backbone == "qwen":
            from transformers import AutoModelForCausalLM
            from .integrations.kev_encoding import PointerHead
            from .components.shared_adapter import install_shared_adapters
            from .components.adapter_mechanisms import install_output_scales
            from .components.private_residual import install_qwen_private_residual

            dtype = (
                torch.bfloat16
                if self.device.type == "cuda" and config.get("bf16", True)
                else torch.float32
            )
            full = AutoModelForCausalLM.from_pretrained(
                base, dtype=dtype, attn_implementation="sdpa", **load
            )
            self.net = nn.Module()
            self.net.lm = full.model
            self.net.head = PointerHead(self.net.lm.config.hidden_size, query)
            rows = full.get_output_embeddings().weight.detach()[token_ids].float().clone()
            embeds = full.get_input_embeddings().weight.detach()[token_ids].float().clone()
            del full
            self.net.lm.requires_grad_(False)
            self.net.to(self.device)
            self.adapter_inventory = install_shared_adapters(
                self.net.lm, **config.get("adapter", {"rank": 12, "alpha": 24, "dropout": 0.05})
            )
            self.scale_inventory = (
                install_output_scales(self.net.lm) if config.get("output_scale", True) else []
            )
            spec = config.get(
                "private", {"enabled": True, "last_layers": 8, "rank": 2, "scale": 2.0}
            )
            self.private_enabled = spec.get("enabled", True)
            self.private_inventory = []
            if self.private_enabled:
                if not self.scale_inventory:
                    raise ValueError(
                        "Qwen private residual recipe requires output-scale projections"
                    )
                self.private_inventory = install_qwen_private_residual(
                    self.net.lm,
                    spec.get("last_layers", 8),
                    spec.get("rank", 2),
                    spec.get("scale", 2.0),
                    config.get("seed", 20261101),
                )
            self.numeric_score = NumericTrieReadout(rows, embeds, query).to(self.device)
            self.net.lm.config.use_cache = False
            if config.get("gradient_checkpointing", True):
                self.net.lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                self.net.lm.enable_input_require_grads()
                if self.private_enabled:
                    from .components.private_residual import optional_checkpoint

                    for layer in self.net.lm.layers:
                        layer._gradient_checkpointing_func = optional_checkpoint
        elif self.backbone == "modernbert":
            from transformers import AutoModelForMaskedLM
            from .components.modern_core import ModernCore
            from .components.modern_shared import install_shared_adapters
            from .components.modern_scale import install_output_scales
            from .components.modern_private import install_shared_private

            full = AutoModelForMaskedLM.from_pretrained(base, attn_implementation="sdpa", **load)
            self.net = ModernCore(full, token_ids, query)
            self.adapter_inventory = install_shared_adapters(
                self.net.encoder,
                **config.get("adapter", {"rank": 12, "alpha": 24, "dropout": 0.05}),
                freeze_backbone=False,
            )
            self.scale_inventory = (
                install_output_scales(self.net.encoder) if config.get("output_scale", True) else []
            )
            spec = config.get("private", {"enabled": True, "shared_rank": 12, "rank": 2})
            self.private_inventory = []
            if spec.get("enabled", True):
                self.private_inventory = install_shared_private(
                    self.net,
                    spec.get("shared_rank", 12),
                    spec.get("rank", 2),
                    True,
                    config.get("seed", 20261101),
                )
            self.net.encoder.config.reference_compile = False
            if config.get("gradient_checkpointing", True):
                self.net.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            self.net.to(self.device)
        else:
            raise ValueError("backbone must be qwen or modernbert")

    def encode(self, row):
        enc = (qwen_encode if self.backbone == "qwen" else modern_encode)(row, self.tokenizer)
        if (
            self.config.get("max_input_tokens") is not None
            and enc["original_tokens"] > self.config["max_input_tokens"]
        ):
            raise ValueError(
                "Complete input exceeds configured admission bound; no truncation performed"
            )
        return enc

    def forward(self, records, encoded=None):
        rows = [normalize(r) for r in records]
        encs = encoded or [self.encode(r) for r in rows]
        if len(rows) != len(encs):
            raise ValueError("Misaligned encodings")
        if self.backbone == "modernbert":
            batch = collate_modern(encs, self.tokenizer.pad_token_id, self.device)
            zs = self.net(**batch)[0]
            output = [zs[i, : len(support(r))].float() for i, r in enumerate(rows)]
        else:
            from .components.private_residual_independent import shared_hidden
            from .components.shared_adapter import assert_caches_clear

            output = [None] * len(rows)
            ci = [i for i, r in enumerate(rows) if r["task"] != "score"]
            with private_context(self.private_enabled):
                if ci:
                    types = torch.tensor(
                        [TASKS.index(rows[i]["task"]) for i in ci], device=self.device
                    )
                    q, opts = shared_hidden(
                        self.net.lm, [encs[i] for i in ci], self.tokenizer.pad_token_id, types
                    )
                    for j, i in enumerate(ci):
                        output[i] = self.net.head(q[j], opts[j]).float()
                si = [i for i, r in enumerate(rows) if r["task"] == "score"]
                if si:
                    seqs = [encs[i]["ids"] for i in si]
                    l = max(map(len, seqs))
                    ids = torch.full(
                        (len(si), l),
                        self.tokenizer.pad_token_id,
                        device=self.device,
                        dtype=torch.long,
                    )
                    att = torch.zeros_like(ids)
                    pos = torch.zeros_like(ids)
                    for j, s in enumerate(seqs):
                        ids[j, : len(s)] = torch.tensor(s, device=self.device)
                        att[j, : len(s)] = 1
                        pos[j, : len(s)] = torch.arange(len(s), device=self.device)
                    with task_context(
                        torch.full((len(si),), 2, device=self.device, dtype=torch.long)
                    ):
                        hs = self.net.lm(
                            input_ids=ids, position_ids=pos, attention_mask=att, use_cache=False
                        ).last_hidden_state
                    h = hs[
                        torch.arange(len(si), device=self.device),
                        torch.tensor([len(s) - 1 for s in seqs], device=self.device),
                    ].float()
                    scores = self.numeric_score(
                        h, self.net.head.q(h), [support(rows[i]) for i in si]
                    )
                    for i, z in zip(si, scores):
                        output[i] = z
            assert_caches_clear(self.net.lm)
        return [
            z[
                torch.tensor(
                    [e["canonical_to_row"].index(j) for j in range(len(z))], device=z.device
                )
            ]
            for z, e in zip(output, encs)
        ]

    def losses(self, rows, encoded=None):
        return torch.stack(
            [
                request_loss(z, normalize(r, True), self.config.get("objective", {}))
                for r, z in zip(rows, self(rows, encoded))
            ]
        )

    def optimizer_groups(self):
        rates = self.config.get("learning_rates", {})
        groups = {}
        ids = set()
        ratio = float(self.config.get("lora_plus_ratio", 1))
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Positive finite LoRA+ ratio required")
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in ids:
                raise RuntimeError("Duplicate trainable parameter")
            ids.add(id(p))
            if name.endswith("output_gain"):
                key = "output_scale"
                lr = rates.get(key, 1e-3)
            elif "numeric_score." in name:
                key = "numeric_readout"
                lr = rates.get(key, 1e-3)
            elif ".private_residual." in name:
                key = "private"
                lr = rates.get(key, 2e-5)
            elif ".head." in name or name.startswith(("net.q.", "net.k.")):
                key = "pointer"
                lr = rates.get(key, 2e-5)
            elif ".lora_B." in name and self.config.get("lora_plus_ratio", 1) != 1:
                key = "adapter_b"
                lr = rates.get("backbone_adapter", 2e-5) * self.config["lora_plus_ratio"]
            else:
                key = "backbone_adapter"
                lr = rates.get(key, 2e-5)
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError("Positive finite learning rate required")
            groups.setdefault((key, lr), []).append(p)
        return [{"name": k, "lr": lr, "params": ps} for (k, lr), ps in groups.items()]

    def inventory(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "groups": {
                g["name"]: sum(p.numel() for p in g["params"]) for g in self.optimizer_groups()
            },
            "backbone": self.backbone,
            "adapter_sites": len(self.adapter_inventory),
            "output_scale_sites": len(self.scale_inventory),
            "private_sites": len(self.private_inventory),
        }

    @torch.no_grad()
    def predict(self, records):
        old = self.training
        self.eval()
        results = []
        try:
            for row, z in zip(records, self(records)):
                row = normalize(row)
                values = support(row)
                order = canonical_order(row)
                pc = z[torch.tensor(order, device=z.device)].float().softmax(0).cpu().tolist()
                p = [pc[order.index(i)] for i in range(len(order))]
                if row["task"] == "score":
                    c = 0.0
                    index = len(p) - 1
                    for i, prob in enumerate(p):
                        c += prob
                        if c >= 0.5:
                            index = i
                            break
                else:
                    index = min(range(len(p)), key=lambda i: (-p[i], semantic_key(values[i])))
                result = {
                    "task": row["task"],
                    "candidate_values": values,
                    "probabilities": p,
                    "prediction": values[index],
                    "prediction_index": index,
                }
                if "id" in row:
                    result["id"] = row["id"]
                if row["task"] == "score":
                    result["expected_score"] = math.fsum(v * prob for v, prob in zip(values, p))
                    result["decision_rule"] = "median"
                results.append(result)
            return results
        finally:
            self.train(old)
