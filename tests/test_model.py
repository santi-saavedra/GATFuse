"""Architecture invariants that the method depends on."""

import math

import pytest
import torch

from gatfuse.metrics import best_threshold, f1_at_threshold, summarise
from gatfuse.model import FocalLoss, FusionPredictor


def _model(**kwargs):
    torch.manual_seed(0)
    return FusionPredictor(num_node_features=3, num_edge_features=30, **kwargs)


def test_forward_returns_one_logit_per_edge(graph_copy):
    model = _model().eval()
    with torch.no_grad():
        logits = model(graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr)
    assert logits.shape == (graph_copy.edge_index.shape[1],)


def test_evidence_gate_can_only_raise_a_score(graph_copy):
    """The gate is softplus-constrained, so more split reads never lower a logit."""
    model = _model().eval()
    with torch.no_grad():
        base = model(graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr)
        gated = model(
            graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr,
            edge_evidence=graph_copy.edge_evidence,
        )
    assert torch.all(gated >= base - 1e-6)
    assert torch.all(gated[graph_copy.edge_evidence > 0] > base[graph_copy.edge_evidence > 0])


def test_evidence_gate_is_monotone_in_read_support(graph_copy):
    model = _model().eval()
    with torch.no_grad():
        low = model(graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr,
                    edge_evidence=torch.ones_like(graph_copy.edge_evidence))
        high = model(graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr,
                     edge_evidence=torch.full_like(graph_copy.edge_evidence, 5.0))
    assert torch.all(high > low)


def test_classifier_bias_starts_at_the_base_rate():
    prior = 0.002
    model = _model(prior_pos_rate=prior)
    expected = math.log(prior / (1 - prior))
    assert float(model.edge_mlp[-1].bias.item()) == pytest.approx(expected, abs=1e-5)


def test_node_skip_changes_the_classifier_input_width():
    with_skip = _model(node_skip=True)
    without_skip = _model(node_skip=False)
    assert with_skip.edge_mlp[0].in_features > without_skip.edge_mlp[0].in_features


def test_inference_is_deterministic(graph_copy):
    model = _model().eval()
    with torch.no_grad():
        first = model(graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr)
        second = model(graph_copy.x, graph_copy.edge_index, graph_copy.edge_attr)
    assert torch.equal(first, second)


def test_invalid_depth_is_rejected():
    with pytest.raises(ValueError, match="num_gnn_layers"):
        _model(num_gnn_layers=0)


def test_focal_loss_down_weights_easy_examples():
    logits = torch.tensor([5.0, -5.0])       # both confidently correct
    hard = torch.tensor([0.1, -0.1])         # both near the boundary
    targets = torch.tensor([1.0, 0.0])
    loss = FocalLoss(alpha=0.75, gamma=2.0)
    assert loss(logits, targets) < loss(hard, targets)


def test_focal_gamma_zero_reduces_to_weighted_cross_entropy():
    logits = torch.tensor([0.4, -1.2, 2.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    focal = FocalLoss(alpha=0.5, gamma=0.0)(logits, targets)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert float(focal) == pytest.approx(float(bce) * 0.5, rel=1e-5)


def test_metrics_on_a_perfect_ranking():
    probabilities = torch.tensor([0.9, 0.8, 0.2, 0.1])
    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    f1, threshold, precision, recall, auprc = best_threshold(probabilities, targets)
    assert f1 == pytest.approx(1.0, abs=1e-3)
    assert auprc == pytest.approx(1.0, abs=1e-3)
    assert precision == pytest.approx(1.0, abs=1e-3)
    assert threshold == pytest.approx(0.8, abs=1e-6)


def test_metrics_handle_a_set_with_no_positives():
    probabilities = torch.tensor([0.9, 0.1])
    targets = torch.tensor([0.0, 0.0])
    assert best_threshold(probabilities, targets) == (0.0, 0.5, 0.0, 0.0, 0.0)
    assert summarise(probabilities, targets)["num_positive"] == 0


def test_f1_at_threshold_matches_a_hand_computation():
    probabilities = torch.tensor([0.9, 0.6, 0.4])
    targets = torch.tensor([1.0, 0.0, 1.0])
    f1, precision, recall = f1_at_threshold(probabilities, targets, 0.5)
    assert precision == pytest.approx(0.5, abs=1e-4)   # 1 of 2 predicted
    assert recall == pytest.approx(0.5, abs=1e-4)      # 1 of 2 positives
    assert f1 == pytest.approx(0.5, abs=1e-4)
