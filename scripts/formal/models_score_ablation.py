"""Single-component D2 ablations with the frozen ordinal score objective."""

from models_score_distribution import *
from models_score_distribution import FormalModel as D2FormalModel
from models_v6_ablation import FormalModel as AblationFormalModel, ablation_inventory
from score_distribution import DistributionObjective


FULL_FLAGS = {'task_routing': True, 'quadratic': True, 'geometry': True,
              'auxiliary_losses': False}
ABLATIONS = {
    'a1_no_task_routing': {**FULL_FLAGS, 'task_routing': False},
    'a2_no_quadratic': {**FULL_FLAGS, 'quadratic': False},
    'a3_no_head_geometry': {**FULL_FLAGS, 'geometry': False},
}


class FormalModel(D2FormalModel):
    def __init__(self, arm, base, checkpointing=True, ablation=None,
                 distribution_objective=None):
        if ablation not in ABLATIONS.values():
            raise ValueError('Exactly one D2 architectural component must be disabled')
        self.distribution_objective = DistributionObjective(**distribution_objective)
        if self.distribution_objective.kind != 'ordinal_main':
            raise ValueError('D2 ablations require the frozen ordinal objective')
        # The v6 installer admits one disabled switch; D2 already replaces
        # the old auxiliary loss, so mask only the architecture during install.
        install_flags = {**ablation, 'auxiliary_losses': True}
        AblationFormalModel.__init__(self, arm, base, checkpointing=checkpointing,
                                    ablation=install_flags)
        self.net.auxiliary_losses_enabled = False
        self.net.ablation = dict(ablation)
        self.ablation = dict(ablation)
        self.ablation_inventory = ablation_inventory(self.net)
        self.collect_loss_diagnostics = False
        self._loss_diagnostics = None
