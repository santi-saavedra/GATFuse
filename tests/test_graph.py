"""Graph construction: shapes, feature values and input-mismatch detection."""

import numpy as np
import pytest

from gatfuse.config import GraphParams
from gatfuse.errors import InputError
from gatfuse.features import edge_feature_names, num_edge_features
from gatfuse.graph import build_graph
from gatfuse.io import load_chimeric_junctions, load_discordant_pairs, load_expression


def test_graph_shapes(sample_graph):
    assert sample_graph.x.shape == (5, 3)
    assert sample_graph.edge_attr.shape == (4, 30)
    assert sample_graph.edge_index.shape == (2, 4)
    assert len(sample_graph.edge_breakpoints) == 4


def test_evidence_tensor_holds_unnormalised_split_reads(sample_graph):
    names = edge_feature_names(False)
    column = names.index("log_split_reads")
    assert np.allclose(
        sample_graph.edge_evidence.numpy(), sample_graph.edge_attr[:, column].numpy()
    )
    assert sample_graph.edge_evidence.max().item() == pytest.approx(np.log1p(12), abs=1e-5)


def test_edge_features_of_the_known_fusion(sample_graph):
    names = edge_feature_names(False)
    values = {}
    for i, breakpoint in enumerate(sample_graph.edge_breakpoints):
        if breakpoint["brkpt_donor"] == 3000 and breakpoint["chr_acceptor"] == "chr2":
            values = dict(zip(names, sample_graph.edge_attr[i].tolist(), strict=True))
            evidence = breakpoint
            break
    assert values, "the BCR-ABL1 junction should be present"

    assert values["log_split_reads"] == pytest.approx(np.log1p(12), abs=1e-5)
    assert values["is_interchromosomal"] == 1.0
    assert values["junction_type_1"] == 1.0
    assert values["strand_plus_plus"] == 1.0
    assert values["log_breakpoint_distance"] == pytest.approx(np.log1p(1e7), abs=1e-3)
    # Eight discordant pairs sit within the window of both breakpoints.
    assert evidence["discordant_pairs"] == 8.0
    assert values["log_discordant_pairs"] == pytest.approx(np.log1p(8), abs=1e-5)


def test_known_fusion_features_extend_the_layout(annotation, chimeric_path, counts_path,
                                                 bam_path, known_fusions):
    graph = build_graph(
        load_expression(counts_path),
        load_chimeric_junctions(chimeric_path, min_split_reads=2),
        load_discordant_pairs(bam_path),
        annotation,
        params=GraphParams(use_known_fusions=True),
        known_fusions=known_fusions,
    )
    assert graph.x.shape[1] == 5
    assert graph.edge_attr.shape[1] == num_edge_features(True)

    names = edge_feature_names(True)
    known_pair = graph.edge_attr[:, names.index("known_pair")]
    # BCR-ABL1 and EWSR1-FLI1 are catalogued; the artefact junctions are not.
    assert known_pair.sum().item() == 2


def test_empty_junction_table_yields_a_correctly_shaped_graph(annotation, counts_path, bam_path):
    """A sample with no usable junction must still produce a model-compatible graph."""
    graph = build_graph(
        load_expression(counts_path),
        _empty_junctions(),
        load_discordant_pairs(bam_path),
        annotation,
        params=GraphParams(use_known_fusions=False),
    )
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, num_edge_features(False))
    assert graph.edge_evidence.numel() == 0


def _empty_junctions():
    from gatfuse.io.chimeric import _empty_junction_frame

    return _empty_junction_frame()


def test_contig_name_mismatch_is_reported(annotation, counts_path, bam_path, chimeric_path):
    junctions = load_chimeric_junctions(chimeric_path, min_split_reads=2)
    junctions["chr_donor"] = junctions["chr_donor"].str.replace("chr", "", regex=False)
    junctions["chr_acceptor"] = junctions["chr_acceptor"].str.replace("chr", "", regex=False)
    with pytest.raises(InputError, match="No contig name is shared"):
        build_graph(
            load_expression(counts_path), junctions,
            load_discordant_pairs(bam_path), annotation,
            params=GraphParams(use_known_fusions=False),
        )


def test_annotation_mismatch_is_reported(annotation, chimeric_path, bam_path, counts_path):
    counts = load_expression(counts_path)
    counts["gene_id"] = "UNRELATED_" + counts["gene_id"]
    with pytest.raises(InputError, match="No gene ID"):
        build_graph(
            counts, load_chimeric_junctions(chimeric_path, min_split_reads=2),
            load_discordant_pairs(bam_path), annotation,
            params=GraphParams(use_known_fusions=False),
        )


def test_node_features_are_standardised(sample_graph):
    tpm = sample_graph.x[:, 0].numpy()
    assert abs(float(tpm.mean())) < 1e-5
    assert float(tpm.std()) == pytest.approx(1.0, abs=1e-3)


def test_genes_without_a_symbol_do_not_break_known_fusion_features(
    annotation, chimeric_path, counts_path, bam_path, known_fusions
):
    """A GTF leaves gene_name empty for many genes, and the column is nullable.

    The missing value is pd.NA, which raises instead of being falsy, so every
    run against a real annotation used to die once the catalogue was enabled.
    """
    import pandas as pd

    from gatfuse.io.annotation import Annotation

    genes = annotation.genes.copy()
    genes.loc[genes.index[0], "gene_name"] = pd.NA
    patched = Annotation(genes=genes, exons=annotation.exons, source=annotation.source)

    params = GraphParams(min_split_reads=2, use_known_fusions=True)
    graph = build_graph(
        load_expression(counts_path),
        load_chimeric_junctions(chimeric_path, min_split_reads=2),
        load_discordant_pairs(bam_path),
        patched,
        params=params,
        known_fusions=known_fusions,
    )
    # 3 base node features plus the two contributed by the catalogue.
    assert graph.x.shape == (5, 5)
    assert graph.edge_attr.shape[1] == num_edge_features(True)
