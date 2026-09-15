"""The feature layout is a contract: the model and the docs both depend on it."""

import pytest

from gatfuse import features


@pytest.mark.parametrize(
    "use_known_fusions, node_width, edge_width", [(False, 3, 30), (True, 5, 32)]
)
def test_layout_widths(use_known_fusions, node_width, edge_width):
    assert features.num_node_features(use_known_fusions) == node_width
    assert features.num_edge_features(use_known_fusions) == edge_width


@pytest.mark.parametrize("use_known_fusions", [False, True])
def test_feature_names_are_unique(use_known_fusions):
    names = features.edge_feature_names(use_known_fusions)
    assert len(names) == len(set(names))


def test_known_fusion_block_shifts_the_aggregate_block():
    """Enabling the catalogue must not silently move a feature under a wrong name."""
    without = features.edge_feature_names(False)
    with_catalogue = features.edge_feature_names(True)
    assert with_catalogue.index("log_pair_junction_count") == (
        without.index("log_pair_junction_count") + 2
    )
    assert "known_pair" not in without


def test_binary_and_one_hot_features_are_not_normalised():
    for use_known_fusions in (False, True):
        schema = features.edge_feature_schema(use_known_fusions)
        for feature in schema:
            if feature.name.startswith(("is_", "junction_type_", "strand_", "known_pair")):
                assert not feature.normalize, f"{feature.name} must not be z-scored"
            if feature.name.endswith("_in_exon"):
                assert not feature.normalize


def test_continuous_columns_match_declared_schema():
    for use_known_fusions in (False, True):
        schema = features.edge_feature_schema(use_known_fusions)
        expected = [i for i, f in enumerate(schema) if f.normalize]
        assert features.continuous_edge_columns(use_known_fusions) == expected


def test_layout_can_be_inferred_from_width():
    assert features.continuous_columns_for_width(32) == features.continuous_edge_columns(True)
    assert features.uses_known_fusions_from_width(5, 32) is True
    assert features.uses_known_fusions_from_width(3, 30) is False
    assert features.uses_known_fusions_from_width(4, 17) is None
    with pytest.raises(ValueError):
        features.continuous_columns_for_width(17)
