"""Candidate-isolated ModernBERT with decimal-prefix score readout."""

import torch
from torch import nn
from torch.nn import functional as F
from ..objectives import proper_loss
from .numeric_trie import NumericTrieReadout, TrieObjective, components, ALPHABET


def directed_masks(attention_mask, position_ids, option_ids, sliding_window):
    query, key = option_ids[:, :, None], option_ids[:, None, :]
    # Shared queries read shared keys only. Candidate i reads shared keys and i.
    allowed = (key == 0) | ((query > 0) & (key == query))
    allowed = allowed & attention_mask.bool()[:, None, :]
    result = {"full_attention": allowed[:, None]}
    if sliding_window is not None:
        near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= sliding_window
        result["sliding_attention"] = (allowed & near)[:, None]
    return result


class ModernCore(nn.Module):
    def __init__(self, pretrained_mlm, numeric_token_ids, query_size=256):
        super().__init__()
        self.encoder = pretrained_mlm.model
        self.numeric_transform = pretrained_mlm.head.requires_grad_(False)
        d = self.encoder.config.hidden_size
        self.q = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, query_size, bias=False))
        self.k = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, query_size, bias=False))
        embedding = pretrained_mlm.get_input_embeddings().weight.detach()[numeric_token_ids].float()
        output = pretrained_mlm.get_output_embeddings()
        rows = output.weight.detach()[numeric_token_ids].float()
        bias = (
            output.bias.detach()[numeric_token_ids].float()
            if output.bias is not None
            else rows.new_zeros(len(rows))
        )
        # The extra constant coordinate preserves the actual pretrained MLM bias.
        self.numeric_score = NumericTrieReadout(
            torch.cat([rows, bias[:, None]], 1),
            torch.cat([embedding, embedding.new_zeros(len(embedding), 1)], 1),
            query_size,
        )
        self.scale = query_size**-0.5
        self.trie_objective = TrieObjective("trie_ce_cdf")
        self.class_alpha = 0.75

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        position_ids,
        option_ids,
        items,
    ):
        masks = directed_masks(
            attention_mask, position_ids, option_ids, self.encoder.config.sliding_window
        )
        h = self.encoder(
            input_ids=input_ids, attention_mask=masks, position_ids=position_ids
        ).last_hidden_state
        qs = torch.tensor([it["query_position"] for it in items], device=h.device)
        shared = h[torch.arange(len(items), device=h.device), qs]
        q = self.q(shared)
        result = [None] * len(items)
        score_indices = []
        for i, it in enumerate(items):
            if it["task"] == "score":
                score_indices.append(i)
            else:
                k = len(it["support"])
                candidate = h[i, marker_pos[i, :k]]
                result[i] = (self.k(candidate) @ q[i]).float() * self.scale
        if score_indices:
            hs = self.numeric_transform(shared[score_indices])
            hs = torch.cat([hs.float(), hs.new_ones(len(hs), 1).float()], -1)
            scores = self.numeric_score(
                hs, q[score_indices], [items[i]["support"] for i in score_indices]
            )
            for i, z in zip(score_indices, scores):
                result[i] = z
        logits = torch.stack(
            [F.pad(z, (0, marker_mask.size(1) - len(z)), value=-1e4) for z in result]
        )
        return logits, None

    def losses(self, logits, batch, items):
        values = []
        for i, it in enumerate(items):
            z = logits[i, : len(it["support"])]
            if it["task"] == "score":
                # Preserve exact real-valued supervision, not an argmax label.
                loss = components(z, it["support"], it["score_label"], self.trie_objective)[0]
            else:
                loss = proper_loss(z, int(batch["target"][i].argmax()), self.class_alpha)
            values.append(loss)
        return torch.stack(values)
