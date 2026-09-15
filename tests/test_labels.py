"""Edge labelling at gene and breakpoint resolution."""

import torch

from gatfuse.labels import label_edges

GENES = {"ENSG_A": 0, "ENSG_B": 1, "ENSG_C": 2}
SYMBOLS = {"BCR": 0, "ABL1": 1, "OTHER": 2}
EDGE_INDEX = torch.tensor([[0, 1, 0], [1, 0, 2]])
BREAKPOINTS = [
    {"chr_donor": "chr1", "brkpt_donor": 1000, "chr_acceptor": "chr2", "brkpt_acceptor": 2000},
    {"chr_donor": "chr2", "brkpt_donor": 2000, "chr_acceptor": "chr1", "brkpt_acceptor": 1000},
    {"chr_donor": "chr1", "brkpt_donor": 9000, "chr_acceptor": "chr3", "brkpt_acceptor": 3000},
]


def test_gene_level_labels_both_directions():
    labels = label_edges(EDGE_INDEX, [("BCR", "ABL1")], GENES, gene_name_to_node_id=SYMBOLS)
    assert labels.tolist() == [1.0, 1.0, 0.0]


def test_ensembl_ids_resolve_too():
    labels = label_edges(EDGE_INDEX, [("ENSG_A", "ENSG_B")], GENES)
    assert labels.tolist() == [1.0, 1.0, 0.0]


def test_breakpoint_labels_only_match_the_reported_coordinates():
    fusion = [("BCR", "ABL1", "chr1", 1000, "chr2", 2000)]
    labels = label_edges(
        EDGE_INDEX, fusion, GENES, gene_name_to_node_id=SYMBOLS, edge_breakpoints=BREAKPOINTS
    )
    assert labels.tolist() == [1.0, 1.0, 0.0]


def test_breakpoint_labels_respect_the_tolerance():
    fusion = [("BCR", "ABL1", "chr1", 1000, "chr2", 2000)]
    within = label_edges(
        EDGE_INDEX, fusion, GENES, gene_name_to_node_id=SYMBOLS,
        edge_breakpoints=BREAKPOINTS, tolerance=100,
    )
    assert within[0] == 1.0

    shifted = [("BCR", "ABL1", "chr1", 1500, "chr2", 2000)]
    outside = label_edges(
        EDGE_INDEX, shifted, GENES, gene_name_to_node_id=SYMBOLS,
        edge_breakpoints=BREAKPOINTS, tolerance=100,
    )
    assert outside.sum() == 0.0


def test_unknown_partner_is_skipped_not_fatal():
    labels = label_edges(EDGE_INDEX, [("BCR", "NOT_EXPRESSED")], GENES,
                         gene_name_to_node_id=SYMBOLS)
    assert labels.sum() == 0.0
