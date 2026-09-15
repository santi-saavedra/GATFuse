"""Binary edge labels for training."""

import torch

from .config import BREAKPOINT_LABEL_TOLERANCE
from .logging_utils import get_logger

log = get_logger("labels")


def label_edges(edge_index, fusions, gene_to_node_id, gene_name_to_node_id=None,
                edge_breakpoints=None, tolerance=BREAKPOINT_LABEL_TOLERANCE,
                sample_id=""):
    """Return a float tensor of shape (E,) with 1.0 on confirmed fusion edges."""
    lookup = dict(gene_to_node_id)
    for name, node in (gene_name_to_node_id or {}).items():
        lookup.setdefault(name, node)

    gene_level = set()
    breakpoint_level = []
    unresolved = []

    for fusion in fusions:
        donor, acceptor = fusion[0], fusion[1]
        if donor not in lookup or acceptor not in lookup:
            unresolved.append(f"{donor}:{acceptor}")
            continue
        donor_node, acceptor_node = lookup[donor], lookup[acceptor]

        if len(fusion) == 6 and edge_breakpoints is not None:
            _, _, chr_a, pos_a, chr_b, pos_b = fusion
            breakpoint_level.append((donor_node, acceptor_node, chr_a, pos_a, chr_b, pos_b))
            breakpoint_level.append((acceptor_node, donor_node, chr_b, pos_b, chr_a, pos_a))
        else:
            gene_level.add((donor_node, acceptor_node))
            gene_level.add((acceptor_node, donor_node))

    if unresolved:
        log.warning(
            "%s: %d confirmed fusion(s) absent from this sample's graph (%s)",
            sample_id or "sample", len(unresolved), ", ".join(unresolved[:5]),
        )

    labels = torch.zeros(edge_index.shape[1], dtype=torch.float)
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()

    by_pair = {}
    for entry in breakpoint_level:
        by_pair.setdefault((entry[0], entry[1]), []).append(entry[2:])

    for i, (source, target) in enumerate(zip(sources, targets, strict=True)):
        if (source, target) in gene_level:
            labels[i] = 1.0
            continue
        candidates = by_pair.get((source, target))
        if not candidates or edge_breakpoints is None:
            continue
        breakpoint = edge_breakpoints[i]
        for chr_a, pos_a, chr_b, pos_b in candidates:
            if (
                breakpoint["chr_donor"] == chr_a
                and abs(breakpoint["brkpt_donor"] - pos_a) <= tolerance
                and breakpoint["chr_acceptor"] == chr_b
                and abs(breakpoint["brkpt_acceptor"] - pos_b) <= tolerance
            ):
                labels[i] = 1.0
                break

    positives = int(labels.sum().item())
    log.info(
        "%s: %d positive edge(s), %d negative", sample_id or "sample",
        positives, labels.numel() - positives,
    )
    return labels
