"""Global z-score normalisation of continuous edge features."""

import torch

from .features import continuous_edge_columns
from .logging_utils import get_logger

log = get_logger("normalization")


def compute_edge_norm_stats(graphs, use_known_fusions):
    """Pool mean and standard deviation of the continuous edge features."""
    columns = continuous_edge_columns(use_known_fusions)
    populated = [g.edge_attr for g in graphs if g.edge_attr.shape[0] > 0]
    if not populated:
        raise ValueError("Cannot compute normalisation statistics: no graph has edges.")

    edges = torch.cat(populated, dim=0)
    means, stds = [], []
    for column in columns:
        values = edges[:, column]
        means.append(float(values.mean()))
        stds.append(float(values.std()) + 1e-6)

    log.info(
        "Edge normalisation computed over %d edges (%d continuous columns)",
        edges.shape[0], len(columns),
    )
    return {"continuous_cols": columns, "mean": means, "std": stds}


def apply_edge_norm(graphs, stats):
    """Standardise edge features in place using pre-computed statistics."""
    columns = stats["continuous_cols"]
    means = torch.tensor(stats["mean"], dtype=torch.float32)
    stds = torch.tensor(stats["std"], dtype=torch.float32)
    index = torch.tensor(columns, dtype=torch.long)

    for graph in graphs:
        if graph.edge_attr.shape[0] == 0:
            continue
        if graph.edge_attr.shape[1] <= max(columns, default=-1):
            raise ValueError(
                f"Graph has {graph.edge_attr.shape[1]} edge features but the normalisation "
                f"statistics reference column {max(columns)}."
            )
        graph.edge_attr[:, index] = (graph.edge_attr[:, index] - means) / stds
