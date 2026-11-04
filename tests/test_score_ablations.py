"""The D2 ablations change one architecture switch and retain its objective."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/formal'))

import models_score_ablation as ablation
import worker_score_ablation as worker
from score_distribution import DistributionObjective


def test_every_ablation_disables_only_one_architecture_component():
    assert set(ablation.ABLATIONS) == {
        'a1_no_task_routing', 'a2_no_quadratic', 'a3_no_head_geometry'}
    for name, flags in ablation.ABLATIONS.items():
        assert flags['auxiliary_losses'] is False
        changes = [key for key in ('task_routing', 'quadratic', 'geometry')
                   if flags[key] != ablation.FULL_FLAGS[key]]
        assert changes == [{'a1_no_task_routing': 'task_routing',
                            'a2_no_quadratic': 'quadratic',
                            'a3_no_head_geometry': 'geometry'}[name]]


@pytest.mark.parametrize('name', list(ablation.ABLATIONS))
def test_model_installs_architecture_mask_and_restores_d2_loss_switch(monkeypatch, name):
    flags = ablation.ABLATIONS[name]
    passed = []

    def initialize(self, arm, base, checkpointing, ablation):
        torch.nn.Module.__init__(self)
        passed.append(dict(ablation))
        self.net = torch.nn.Module()
        self.net.ablation = dict(ablation)
        self.net.auxiliary_losses_enabled = True

    monkeypatch.setattr(ablation.AblationFormalModel, '__init__', initialize)
    monkeypatch.setattr(ablation, 'ablation_inventory',
                        lambda net: {'ablation': net.ablation})
    model = ablation.FormalModel('tcdl', 'unused', ablation=flags,
        distribution_objective={'kind': 'ordinal_main', 'ordinal_scale': 3.0383850790577127})
    assert passed == [{**flags, 'auxiliary_losses': True}]
    assert model.ablation_inventory['ablation'] == flags
    assert model.net.auxiliary_losses_enabled is False
    assert model.distribution_objective == DistributionObjective(
        'ordinal_main', ordinal_scale=3.0383850790577127)
    assert model.losses.__func__ is ablation.D2FormalModel.losses
    assert model.predict.__func__ is ablation.D2FormalModel.predict


def test_rejects_full_or_combined_masks_and_non_d2_objectives():
    invalid = [ablation.FULL_FLAGS,
               {**ablation.ABLATIONS['a1_no_task_routing'], 'quadratic': False}]
    for flags in invalid:
        with pytest.raises(ValueError):
            ablation.FormalModel('tcdl', 'unused', ablation=flags,
                                distribution_objective={'kind': 'ordinal_main'})
    with pytest.raises(ValueError):
        ablation.FormalModel('tcdl', 'unused',
            ablation=ablation.ABLATIONS['a1_no_task_routing'],
            distribution_objective={'kind': 'mean_moment'})


def test_worker_binds_new_model_and_same_score_reporter():
    from score_distribution import score_rule_report
    assert worker.main.__globals__['FormalModel'] is ablation.FormalModel
    assert worker.main.__globals__['score_rule_report'] is score_rule_report
