"""Original Kev and JSON-generative LoRA for complete Score supports."""
import torch

from models import FormalModel as OriginalModel, pad
from score_specialist import support, decisions, SpecialistObjective, data_loss
from score_baselines import parse_score


def strip_terminal_tokens(text, special_tokens):
    tokens = sorted({t for t in special_tokens if t}, key=len, reverse=True)
    while True:
        token = next((t for t in tokens if text.endswith(t)), None)
        if token is None:
            return text
        text = text[:-len(token)]


class FormalModel(OriginalModel):
    def __init__(self, arm, base):
        if arm not in ('kev', 'json_sft'):
            raise ValueError('Only Kev and JSON-generative LoRA are authorized')
        super().__init__(arm, base)
        self.distribution_objective = SpecialistObjective('soft_ce')

    def set_checkpoint_policy(self, policy):
        from models_ordinal import checkpoint_flags
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        layers = [m for m in self.lm.modules() if isinstance(m, Qwen3_5DecoderLayer)]
        flags = checkpoint_flags(len(layers), policy)
        for layer, flag in zip(layers, flags):
            layer.gradient_checkpointing = flag
        return {'policy':policy, 'layers':len(layers), 'checkpointed_layers':sum(flags)}

    def scores(self, rs, es):
        if self.arm != 'kev':
            raise ValueError('Generative baseline has no candidate probabilities')
        return [z[0] for z in self.net.forward_batch(es)]

    def losses(self, rs, es):
        if self.arm == 'json_sft':
            return super().losses(rs, es)
        return torch.stack([data_loss(z, support(r), r['label'], self.distribution_objective)
                            for r,z in zip(rs,self.scores(rs,es))])

    @torch.no_grad()
    def predict(self, rs, es, max_new_tokens):
        if self.arm == 'json_sft':
            ids, mask = pad([e['prompt'] for e in es], self.tok.pad_token_id, self.device, left=True)
            output = self.generator.generate(input_ids=ids, attention_mask=mask,
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id, logits_to_keep=1)
            texts = self.tok.batch_decode(output[:,ids.shape[1]:], skip_special_tokens=False)
            predictions = []
            for row, raw in zip(rs,texts):
                clean = strip_terminal_tokens(raw, [self.tok.eos_token,self.tok.pad_token])
                value,error = parse_score(clean,row)
                predictions.append({'valid':error is None,'prediction_value':value,
                                    'raw_output':raw,'parse_error':error})
            return predictions
        predictions=[]
        for row,z in zip(rs,self.scores(rs,es)):
            p=z.float().softmax(0).cpu().tolist()
            ds=decisions(support(row),p)
            predictions.append({'valid':True,'prediction_index':ds['indices']['median'],
                                'probabilities':p,'score_decision_rule':'median'})
        return predictions
