"""Fresh-base ablations for the frozen independent-candidate D2 model.

The parent D2 constructor is called with the full architecture flags so that
all random initialization is identical to the completed D2 run.  A single
component is then masked and removed from the optimizer.  Encoding, native
candidate handling and the ordinal distribution objective are unchanged.
"""
import types

import torch

from models_independent import FormalModel as IndependentFormalModel
from models_score_distribution import A4_FLAGS


FULL_FLAGS = dict(A4_FLAGS)
VALID_COMPONENTS = {'task_routing', 'quadratic', 'geometry'}


def _zero_quadratic(module, features):
    return features.new_zeros((*features.shape[:-1], 1))


def install_ablation(net, flags):
    flags = dict(flags)
    if set(flags) != set(FULL_FLAGS):
        raise ValueError('Ablation flags must contain exactly the frozen D2 components')
    if any(type(value) is not bool for value in flags.values()):
        raise ValueError('Ablation flags must be booleans')
    disabled = [name for name in VALID_COMPONENTS if not flags[name]]
    if len(disabled) != 1 or flags['auxiliary_losses'] is not False:
        raise ValueError('Each run disables exactly one architecture component and keeps D2 losses')

    if not flags['task_routing']:
        with torch.no_grad():
            net.task_adapter.matrix_raw.zero_()
        net.task_adapter.matrix_raw.requires_grad_(False)
        net.task_routing_disabled = True
    else:
        net.task_routing_disabled = False

    if not flags['quadratic']:
        for module in (net.head.project, net.head.query_condition, net.head.qf_out):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        # qf_out is initialized at zero in the full D2 model.  Keeping a zero
        # forward makes this branch exact even if a future initializer changes.
        net.head.qf_out.forward = types.MethodType(_zero_quadratic, net.head.qf_out)
        net.quadratic_disabled = True
    else:
        net.quadratic_disabled = False

    if not flags['geometry']:
        with torch.no_grad():
            net.head.geometry_raw.zero_()
        net.head.geometry_raw.requires_grad_(False)
        net.geometry_disabled = True
    else:
        net.geometry_disabled = False

    net.auxiliary_losses_enabled = False
    net.ablation = flags


def inventory(net):
    frozen = []
    if net.task_routing_disabled:
        p = net.task_adapter.matrix_raw
        if p.requires_grad or torch.count_nonzero(p).item():
            raise AssertionError('Task router was not frozen at zero')
        frozen.append('task_adapter.matrix_raw')
    if net.quadratic_disabled:
        for name in ('project', 'query_condition', 'qf_out'):
            module = getattr(net.head, name)
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise AssertionError(f'{name} remains trainable')
            frozen.extend(f'head.{name}.{key}' for key, _ in module.named_parameters())
    if net.geometry_disabled:
        if net.head.geometry_raw.requires_grad or torch.count_nonzero(net.head.geometry_raw).item():
            raise AssertionError('Head geometry was not frozen at zero')
        frozen.append('head.geometry_raw')
    if any(not parameter.requires_grad for parameter in net.head.ordinal.parameters()):
        raise AssertionError('Ordinal readout must remain trainable')
    return {
        'ablation': dict(net.ablation),
        'disabled_component': next(name for name in VALID_COMPONENTS if not net.ablation[name]),
        'frozen_component_parameters': frozen,
        'head_geometry_enabled': not net.geometry_disabled,
        'quadratic_enabled': not net.quadratic_disabled,
        'task_routing_enabled': not net.task_routing_disabled,
        'ordinal_readout_enabled': True,
        'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
    }


class FormalModel(IndependentFormalModel):
    """D2 objective with one architecture component removed."""

    def __init__(self, arm, base, checkpointing=True, ablation=None,
                 distribution_objective=None):
        if dict(ablation or {}) == FULL_FLAGS:
            raise ValueError('Use the completed D2 run for the full architecture')
        # This invokes the exact same random initialization as D2.  The parent
        # requires the full flags; masking happens immediately after it returns.
        super().__init__(arm, base, checkpointing=checkpointing,
                         ablation=FULL_FLAGS,
                         distribution_objective=distribution_objective)
        install_ablation(self.net, ablation)
        self.ablation_inventory = inventory(self.net)
