"""Feature standardisation must be reproducible from the stored statistics."""

import pytest
import torch

from gatfuse.features import continuous_edge_columns
from gatfuse.normalization import apply_edge_norm, compute_edge_norm_stats


def test_statistics_cover_exactly_the_continuous_columns(graph_copy):
    stats = compute_edge_norm_stats([graph_copy], use_known_fusions=False)
    assert stats["continuous_cols"] == continuous_edge_columns(False)
    assert len(stats["mean"]) == len(stats["std"]) == len(stats["continuous_cols"])
    assert all(std > 0 for std in stats["std"])


def test_normalisation_centres_continuous_columns_and_leaves_others_alone(graph_copy):
    original = graph_copy.edge_attr.clone()
    stats = compute_edge_norm_stats([graph_copy], use_known_fusions=False)
    apply_edge_norm([graph_copy], stats)

    for column in stats["continuous_cols"]:
        assert abs(float(graph_copy.edge_attr[:, column].mean())) < 1e-4

    untouched = [c for c in range(original.shape[1]) if c not in stats["continuous_cols"]]
    assert torch.allclose(graph_copy.edge_attr[:, untouched], original[:, untouched])


def test_evidence_tensor_survives_normalisation(graph_copy):
    evidence = graph_copy.edge_evidence.clone()
    stats = compute_edge_norm_stats([graph_copy], use_known_fusions=False)
    apply_edge_norm([graph_copy], stats)
    assert torch.allclose(graph_copy.edge_evidence, evidence)


def test_mismatched_statistics_are_rejected(graph_copy):
    stats = compute_edge_norm_stats([graph_copy], use_known_fusions=False)
    stats["continuous_cols"] = stats["continuous_cols"] + [999]
    stats["mean"].append(0.0)
    stats["std"].append(1.0)
    with pytest.raises(ValueError, match="edge features"):
        apply_edge_norm([graph_copy], stats)
