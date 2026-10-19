import torch

from neuoct.losses import (
    IntersectionObjective,
    IntersectionTarget,
    ObjectiveWeights,
    PNCHLoss,
)
from neuoct.losses.image import (
    color_detail_weights_for_pairs,
    hard_floater_loss,
    hit_map_regularization,
    hit_tversky_loss,
    multiscale_color_laplacian_loss,
    multiscale_normal_gradient_loss,
    soft_hit_boundary_f1_loss,
    soft_hit_maps,
)
from neuoct.models.outputs import IntersectionPrediction, IntersectionStage


def _prediction(hit_probability):
    count = hit_probability.numel()
    stage = IntersectionStage(
        position=torch.full((count, 3), 0.5),
        normal=torch.nn.functional.normalize(torch.ones(count, 3), dim=1),
        color=torch.full((count, 3), 0.25),
        hit_probability=hit_probability,
    )
    return IntersectionPrediction((stage,))


def test_no_hit_objective_returns_finite_maps_and_loss():
    height = width = 16
    count = height * width
    labels = torch.arange(count)
    target = IntersectionTarget(
        position=torch.zeros(count, 3),
        normal=torch.zeros(count, 3),
        color=torch.zeros(count, 3),
        hit=torch.zeros(count, dtype=torch.long),
    )
    result = IntersectionObjective()(
        _prediction(torch.full((count,), 0.1)),
        target,
        labels,
        batch_size=1,
        height=height,
        width=width,
    )
    assert torch.isfinite(result.loss)
    assert not result.predicted_hit_mask.any()
    assert result.predicted_color.shape == (1, height, width, 3)


def test_all_hit_objective_is_differentiable():
    height = width = 16
    count = height * width
    probability = torch.full((count,), 0.9, requires_grad=True)
    prediction = _prediction(probability)
    target = IntersectionTarget(
        position=torch.full((count, 3), 0.5),
        normal=prediction.final.normal.detach().clone(),
        color=torch.full((count, 3), 0.25),
        hit=torch.ones(count, dtype=torch.long),
    )
    result = IntersectionObjective()(
        prediction,
        target,
        torch.arange(count),
        batch_size=1,
        height=height,
        width=width,
    )
    result.loss.backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_geometry_only_objective_is_independent_of_color_targets():
    height = width = 16
    count = height * width
    prediction = _prediction(torch.full((count,), 0.9))
    common = dict(
        position=torch.full((count, 3), 0.5),
        normal=prediction.final.normal.detach().clone(),
        hit=torch.ones(count, dtype=torch.long),
    )
    objective = IntersectionObjective(include_color=False)
    first = objective(
        prediction,
        IntersectionTarget(color=torch.zeros(count, 3), **common),
        torch.arange(count),
        batch_size=1,
        height=height,
        width=width,
    )
    second = objective(
        prediction,
        IntersectionTarget(color=torch.ones(count, 3), **common),
        torch.arange(count),
        batch_size=1,
        height=height,
        width=width,
    )
    torch.testing.assert_close(first.loss, second.loss)
    assert first.components["ssim_color"].item() == 0.0
    assert first.components["smoothness"].item() == 0.0
    assert first.components["constancy"].item() == 0.0


def test_positive_geometry_loss_is_not_diluted_by_negative_pairs():
    prediction = IntersectionStage(
        position=torch.ones((4, 3)),
        normal=torch.ones((4, 3)),
        color=torch.zeros((4, 3)),
        hit_probability=torch.full((4,), 0.5),
    )
    target = IntersectionTarget(
        position=torch.zeros((4, 3)),
        normal=torch.zeros((4, 3)),
        color=torch.zeros((4, 3)),
        hit=torch.tensor([1, 0, 0, 0]),
    )

    legacy = PNCHLoss(include_color=False)(prediction, target)
    normalized = PNCHLoss(include_color=False, positive_geometry=True)(prediction, target)

    torch.testing.assert_close(normalized - legacy, torch.tensor(1.5))


def test_balanced_hit_loss_gives_positive_and_negative_classes_equal_weight():
    probability = torch.tensor([0.1, 0.1, 0.1, 0.1])
    prediction = IntersectionStage(
        position=torch.zeros((4, 3)),
        normal=torch.zeros((4, 3)),
        color=torch.zeros((4, 3)),
        hit_probability=probability,
    )
    target = IntersectionTarget(
        position=torch.zeros((4, 3)),
        normal=torch.zeros((4, 3)),
        color=torch.zeros((4, 3)),
        hit=torch.tensor([1, 0, 0, 0]),
    )

    ordinary = PNCHLoss(include_color=False)(prediction, target)
    balanced = PNCHLoss(include_color=False, hit_loss_mode="balanced_bce")(
        prediction, target
    )

    assert balanced > ordinary


def test_final_emphasis_stage_weights_reach_requested_final_weight():
    weights = IntersectionObjective._stage_weights(
        100,
        100,
        4,
        torch.device("cpu"),
        mode="final_emphasis",
        final_stage_weight=0.5,
    )

    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    torch.testing.assert_close(weights[-1], torch.tensor(0.5))


def test_detail_weights_increase_positive_geometry_penalty():
    stage = IntersectionStage(
        position=torch.tensor([[1.0, 1.0, 1.0], [0.1, 0.1, 0.1]]),
        normal=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        color=torch.zeros((2, 3)),
        hit_probability=torch.full((2,), 0.9),
    )
    target = IntersectionTarget(
        position=torch.zeros((2, 3)),
        normal=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        color=torch.zeros((2, 3)),
        hit=torch.ones(2),
    )
    loss = PNCHLoss(include_color=False, positive_geometry=True)
    ordinary = loss(stage, target)
    weighted = loss(stage, target, torch.tensor([10.0, 1.0]))
    assert weighted > ordinary


def test_ray_distance_loss_supervises_local_one_dimensional_parameter():
    height = width = 16
    count = height * width
    pin = torch.zeros((count, 3))
    pout = torch.zeros((count, 3))
    pout[:, 0] = 1.0
    target_position = torch.zeros((count, 3))
    target_position[:, 0] = 0.75
    predicted_position = torch.zeros((count, 3))
    predicted_position[:, 0] = 0.25
    stage = IntersectionStage(
        position=predicted_position,
        normal=torch.tensor([[0.0, 0.0, 1.0]]).expand(count, -1),
        color=torch.zeros((count, 3)),
        hit_probability=torch.full((count,), 0.9),
    )
    target = IntersectionTarget(
        position=target_position,
        normal=stage.normal.clone(),
        color=torch.zeros((count, 3)),
        hit=torch.ones(count),
        ray_pin=pin,
        ray_pout=pout,
    )
    result = IntersectionObjective(
        ObjectiveWeights(ray_distance=1.0), include_color=False
    )(
        IntersectionPrediction((stage,)),
        target,
        torch.arange(count),
        batch_size=1,
        height=height,
        width=width,
    )
    torch.testing.assert_close(result.components["ray_distance"], torch.tensor(0.5))


def test_geometry_objective_reports_normal_gradient_component():
    height = width = 16
    count = height * width
    probability = torch.full((count,), 0.9)
    prediction = _prediction(probability)
    target_normal = prediction.final.normal.detach().clone()
    target_normal[height * 8 :] = torch.tensor([1.0, 0.0, 0.0])
    target = IntersectionTarget(
        position=torch.full((count, 3), 0.5),
        normal=target_normal,
        color=torch.zeros((count, 3)),
        hit=torch.ones(count),
    )
    result = IntersectionObjective(
        ObjectiveWeights(normal_gradient=0.2),
        include_color=False,
        masked_normal_ssim=True,
        normal_detail_scale=2.0,
    )(
        prediction,
        target,
        torch.arange(count),
        batch_size=1,
        height=height,
        width=width,
    )
    assert result.components["normal_gradient"] > 0
    assert torch.isfinite(result.loss)


def test_absolute_color_loss_stays_bounded_for_dark_targets():
    stage = IntersectionStage(
        position=torch.zeros((2, 3)),
        normal=torch.zeros((2, 3)),
        color=torch.full((2, 3), 0.5),
        hit_probability=torch.full((2,), 0.9),
    )
    target = IntersectionTarget(
        position=torch.zeros((2, 3)),
        normal=torch.zeros((2, 3)),
        color=torch.zeros((2, 3)),
        hit=torch.ones(2),
    )
    stable = PNCHLoss(
        positive_geometry=True,
        color_loss_mode="charbonnier",
        color_weight=2.0,
    )(stage, target)
    unstable = PNCHLoss(
        positive_geometry=True,
        color_loss_mode="relative_mse",
        color_weight=2.0,
    )(stage, target)
    assert stable < 2.0
    assert unstable > 1.0e6


def test_soft_hit_regularization_detects_holes_and_floaters():
    height = width = 7
    labels = torch.arange(height * width)
    probabilities = torch.full((height * width,), 0.01)
    target = torch.zeros((1, height, width), dtype=torch.bool)
    target[:, 2:5, 2:5] = True
    probabilities[target.flatten()] = 0.99
    probabilities[3 * width + 3] = 0.01
    probabilities[0] = 0.99
    soft = soft_hit_maps(
        probabilities,
        labels,
        batch_size=1,
        height=height,
        width=width,
    )
    hole, floater, boundary = hit_map_regularization(soft, target)
    assert hole > 0
    assert floater > 0
    assert boundary > 0


def test_hard_floater_loss_focuses_on_rare_high_confidence_background():
    probability = torch.full((1, 16, 16), 0.001)
    target = torch.zeros((1, 16, 16), dtype=torch.bool)
    target[:, 5:11, 5:11] = True
    probability[target] = 0.99
    probability[:, 0, 0] = 0.9
    hard = hard_floater_loss(probability, target, fraction=0.01)
    probability[:, 0, 0] = 0.01
    clean = hard_floater_loss(probability, target, fraction=0.01)
    assert hard > clean


def test_hard_floater_loss_has_bounded_probability_gradient():
    probability = torch.full((1, 8, 8), 0.001)
    probability[:, 0, 0] = 0.999
    probability.requires_grad_()
    target = torch.zeros((1, 8, 8), dtype=torch.bool)
    target[:, 3:5, 3:5] = True
    loss = hard_floater_loss(probability, target, fraction=0.02)
    loss.backward()

    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()
    assert probability.grad.abs().max() <= 2.0


def test_tversky_hit_loss_prioritizes_holes_and_backpropagates():
    target = torch.zeros((1, 8, 8), dtype=torch.bool)
    target[:, 2:6, 2:6] = True
    perfect = target.float().mul(0.98).add((~target).float().mul(0.02))
    missing = perfect.clone()
    missing[:, 3, 3] = 0.02
    floater = perfect.clone()
    floater[:, 0, 0] = 0.98
    missing.requires_grad_()

    perfect_loss = hit_tversky_loss(perfect, target, alpha=0.3, beta=0.7)
    missing_loss = hit_tversky_loss(missing, target, alpha=0.3, beta=0.7)
    floater_loss = hit_tversky_loss(floater, target, alpha=0.3, beta=0.7)
    missing_loss.backward()

    assert missing_loss > floater_loss > perfect_loss
    assert missing.grad is not None
    assert torch.isfinite(missing.grad).all()


def test_soft_boundary_f1_penalizes_holes_and_floaters_and_backpropagates():
    target = torch.zeros((1, 16, 16), dtype=torch.bool)
    target[:, 3:13, 3:13] = True
    perfect = target.float().mul(0.99).add((~target).float().mul(0.01))
    missing = perfect.clone()
    missing[:, 7:9, 7:9] = 0.01
    floater = perfect.clone()
    floater[:, 0:2, 0:2] = 0.99
    missing.requires_grad_()

    perfect_loss = soft_hit_boundary_f1_loss(perfect, target)
    missing_loss = soft_hit_boundary_f1_loss(missing, target)
    floater_loss = soft_hit_boundary_f1_loss(floater, target)
    missing_loss.backward()

    assert missing_loss > perfect_loss
    assert floater_loss > perfect_loss
    assert missing.grad is not None
    assert torch.isfinite(missing.grad).all()


def test_color_weights_emphasize_edges_and_outliers():
    height = width = 4
    labels = torch.arange(height * width)
    colors = torch.full((height * width, 3), 0.2)
    colors[5] = torch.tensor([1.0, 0.0, 1.0])
    weights = color_detail_weights_for_pairs(
        colors,
        torch.ones(height * width),
        labels,
        batch_size=1,
        height=height,
        width=width,
        detail_scale=1.0,
        rarity_scale=1.0,
    )
    assert weights is not None
    assert weights[5] > weights[0]


def test_multiscale_normal_gradient_loss_detects_coarse_curvature_error():
    target = torch.zeros((1, 16, 16, 3))
    target[..., 2] = 1.0
    target[:, :, 8:, 0] = 0.6
    target[:, :, 8:, 2] = 0.8
    prediction = torch.zeros_like(target)
    prediction[..., 2] = 1.0
    mask = torch.ones((1, 16, 16), dtype=torch.bool)
    loss = multiscale_normal_gradient_loss(prediction, target, mask)
    assert loss > 0
    torch.testing.assert_close(
        multiscale_normal_gradient_loss(target, target, mask), torch.tensor(0.0)
    )


def test_color_laplacian_loss_preserves_texture_and_backpropagates():
    target = torch.zeros((1, 16, 16, 3))
    target[:, ::2, ::2] = 1.0
    prediction = torch.full_like(target, 0.25, requires_grad=True)
    mask = torch.ones((1, 16, 16), dtype=torch.bool)
    loss = multiscale_color_laplacian_loss(prediction, target, mask)
    loss.backward()
    assert loss > 0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
