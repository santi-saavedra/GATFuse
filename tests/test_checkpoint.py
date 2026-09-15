"""Checkpoint and graph-cache I/O.

The refusal cases matter beyond tidiness: a checkpoint or a cache entry is a
pickle, and GATFuse is meant to be handed model files and cache directories
produced elsewhere. Reading one must never run the code it carries.
"""

import pytest
import torch

from gatfuse import checkpoint as checkpoint_io
from gatfuse.cache import GraphCache
from gatfuse.errors import ModelError
from gatfuse.model import FusionPredictor
from gatfuse.resources import bundled_model_path


class Arbitrary:
    """Stand-in for whatever code a hostile pickle would carry."""

    def __init__(self, marker="unreachable"):
        self.marker = marker


# Every key checkpoint.load needs in order to rebuild the architecture.
CONFIG = {
    "num_node_features": 3,
    "num_edge_features": 30,
    "hidden_dim": 8,
    "edge_embed_dim": 4,
    "heads": 1,
    "num_gnn_layers": 1,
}


def _model():
    torch.manual_seed(0)
    return FusionPredictor(**CONFIG)


def test_round_trip_preserves_weights_config_and_metadata(tmp_path):
    model = _model()
    path = tmp_path / "model.pt"
    checkpoint_io.save(
        path, model, CONFIG,
        metadata={"best_val_thr": 0.42, "graph_params": {"min_split_reads": 3}},
        optimizer_state={"state": {}, "param_groups": []},
    )

    loaded = checkpoint_io.load(path)
    assert loaded.model_config == CONFIG
    assert loaded.threshold == pytest.approx(0.42)
    assert loaded.graph_params.min_split_reads == 3
    assert loaded.optimizer_state == {"state": {}, "param_groups": []}
    saved = model.state_dict().values()
    restored = loaded.model.state_dict().values()
    for before, after in zip(saved, restored, strict=True):
        assert torch.equal(before, after)


def test_the_bundled_model_still_loads():
    """weights_only=True must accept the checkpoint GATFuse ships with."""
    loaded = checkpoint_io.load(bundled_model_path())
    assert loaded.model_config["num_node_features"] > 0
    assert loaded.model.training is False


def test_a_checkpoint_carrying_arbitrary_objects_is_refused(tmp_path):
    path = tmp_path / "hostile.pt"
    torch.save(
        {
            "state_dict": {"w": torch.zeros(2)},
            "config": {"num_node_features": 3, "num_edge_features": 30},
            "metadata": {"payload": Arbitrary()},
        },
        path,
    )
    with pytest.raises(ModelError, match="refuses to unpickle|not tensors"):
        checkpoint_io.load(path)


def test_cache_round_trip_survives_the_safe_loader(tmp_path, sample_graph):
    """A real PyG graph must still come back out of the cache."""
    cache = GraphCache(tmp_path)
    assert cache.store("abc123", sample_graph) is not None

    restored = cache.load("abc123")
    assert restored is not None, (
        "the cached graph was rejected by weights_only=True; a class it is built "
        "from is missing from cache._SAFE_GLOBALS for this torch-geometric version"
    )
    assert torch.equal(restored.x, sample_graph.x)
    assert torch.equal(restored.edge_index, sample_graph.edge_index)
    assert restored.gene_to_node_id == sample_graph.gene_to_node_id
    assert restored.edge_breakpoints == sample_graph.edge_breakpoints


def test_cache_entry_carrying_arbitrary_objects_is_ignored(tmp_path):
    cache = GraphCache(tmp_path)
    torch.save({"payload": Arbitrary()}, cache.path_for("deadbeef"))
    assert cache.load("deadbeef") is None


def test_a_disabled_cache_never_touches_disk():
    cache = GraphCache(None)
    assert cache.enabled is False
    assert cache.path_for("abc123") is None
    assert cache.load("abc123") is None
