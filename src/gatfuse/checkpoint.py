"""Reading and writing model checkpoints."""

import pickle
from dataclasses import dataclass, field

import torch

from . import __version__
from .config import FEATURE_VERSION, GraphParams
from .errors import ModelError
from .features import uses_known_fusions_from_width
from .io.paths import require_file
from .logging_utils import get_logger
from .model import FusionPredictor

log = get_logger("checkpoint")

CHECKPOINT_FORMAT = 1


@dataclass
class Checkpoint:
    """A loaded model plus the context it was trained in."""

    model: FusionPredictor
    model_config: dict
    metadata: dict = field(default_factory=dict)
    optimizer_state: dict = None
    path: str = ""

    @property
    def graph_params(self):
        """Graph-construction parameters recorded at training time."""
        stored = self.metadata.get("graph_params")
        params = GraphParams.from_dict(stored) if stored else GraphParams()
        if stored is None:
            inferred = self.uses_known_fusions
            if inferred is not None and inferred != params.use_known_fusions:
                params = GraphParams.from_dict({**params.to_dict(), "use_known_fusions": inferred})
        return params

    @property
    def uses_known_fusions(self):
        return uses_known_fusions_from_width(
            self.model_config["num_node_features"], self.model_config["num_edge_features"]
        )

    @property
    def edge_norm_stats(self):
        return self.metadata.get("edge_norm_stats")

    @property
    def platt_params(self):
        return self.metadata.get("platt_params")

    @property
    def threshold(self):
        """Operating threshold chosen on validation data, if one was stored."""
        value = self.metadata.get("best_val_thr")
        return float(value) if value is not None else None

    def describe(self):
        """One-line provenance summary for the run log."""
        parts = [f"format={self.metadata.get('checkpoint_format', 'legacy')}"]
        for key, label in (
            ("gatfuse_version", "gatfuse"),
            ("feature_version", "features"),
            ("num_train_graphs", "train_graphs"),
            ("cv_folds", "cv_folds"),
            ("best_val_auprc", "val_auprc"),
        ):
            if self.metadata.get(key) is not None:
                value = self.metadata[key]
                parts.append(f"{label}={value:.4f}" if isinstance(value, float) else f"{label}={value}")
        if self.threshold is not None:
            parts.append(f"threshold={self.threshold:.4f}")
        return ", ".join(parts)


def load(path, map_location="cpu"):
    """Load a checkpoint and instantiate its model."""
    resolved = require_file(path, "model checkpoint")
    try:
        # weights_only=True reads tensors and primitives without executing the
        # pickle stream. Checkpoints are shared between labs, so a model file
        # must never be able to run code just by being loaded.
        raw = torch.load(resolved, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ModelError(
            f"Checkpoint {resolved} holds objects that are not tensors, numbers or "
            f"strings, and GATFuse refuses to unpickle it: {exc}\n"
            "Checkpoints written by 'gatfuse train' contain only such values. "
            "If this file came from elsewhere, treat it as untrusted."
        ) from exc
    except Exception as exc:
        raise ModelError(f"Could not read checkpoint {resolved}: {exc}") from exc

    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise ModelError(
            f"{resolved} is not a GATFuse checkpoint (no 'state_dict' entry). "
            "Checkpoints are produced by 'gatfuse train'."
        )
    config = raw.get("config")
    if not config:
        raise ModelError(
            f"{resolved} has no architecture configuration and cannot be rebuilt. "
            "Retrain with the current version."
        )
    if "edge_embed_dim" not in config:
        raise ModelError(
            f"{resolved} predates the edge-encoder architecture and is not loadable. "
            "Retrain the model with this version of GATFuse."
        )

    try:
        model = FusionPredictor(**config)
        model.load_state_dict(raw["state_dict"])
    except (TypeError, RuntimeError) as exc:
        raise ModelError(f"Checkpoint {resolved} does not match the current architecture: {exc}") from exc
    model.eval()

    checkpoint = Checkpoint(
        model=model,
        model_config=dict(config),
        metadata=dict(raw.get("metadata") or {}),
        optimizer_state=raw.get("optimizer_state_dict"),
        path=str(resolved),
    )
    log.info("Model: %s (%s)", resolved.name, checkpoint.describe())
    return checkpoint


def save(path, model, model_config, metadata=None, optimizer_state=None):
    """Write a checkpoint, stamping it with version and feature provenance."""
    payload = {
        "state_dict": model.state_dict(),
        "config": dict(model_config),
        "metadata": {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "gatfuse_version": __version__,
            "feature_version": FEATURE_VERSION,
            **(metadata or {}),
        },
    }
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    torch.save(payload, path)
    log.info("Checkpoint written: %s", path)
    return path


def check_compatible(checkpoint, graph):
    """Fail with an actionable message when a graph does not fit the model."""
    expected_nodes = checkpoint.model_config["num_node_features"]
    expected_edges = checkpoint.model_config["num_edge_features"]
    if graph.x.shape[1] != expected_nodes or graph.edge_attr.shape[1] != expected_edges:
        expects_catalogue = checkpoint.uses_known_fusions
        hint = ""
        if expects_catalogue is True:
            hint = (
                "\nThis model was trained with known-fusion features; supply the catalogue "
                "with --known-fusions (the bundled one is used by default)."
            )
        elif expects_catalogue is False:
            hint = (
                "\nThis model was trained without known-fusion features; pass "
                "--no-known-fusions."
            )
        raise ModelError(
            f"Feature mismatch: the model expects {expected_nodes} node and {expected_edges} "
            f"edge features, the graph has {graph.x.shape[1]} and {graph.edge_attr.shape[1]}."
            f"{hint}"
        )
