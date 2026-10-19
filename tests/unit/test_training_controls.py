import torch

from neuoct.engine.solver import NeuOctSolver


def test_freeze_batch_norm_statistics_preserves_affine_training():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(4),
    )
    model.train()

    count = NeuOctSolver._freeze_batch_norm_statistics(model)

    assert count == 2
    assert model.training
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            assert not module.training
            assert module.weight.requires_grad
