"""Inference: coordinate conventions, deduplication and the reported table."""

import pandas as pd
import pytest

from gatfuse import checkpoint as checkpoint_io
from gatfuse import detect as detection
from gatfuse.config import GraphParams, PostprocessParams
from gatfuse.errors import ModelError
from gatfuse.graph import build_graph
from gatfuse.io import load_chimeric_junctions, load_discordant_pairs, load_expression
from gatfuse.resources import bundled_model_path


@pytest.fixture(scope="module")
def bundled_checkpoint():
    return checkpoint_io.load(bundled_model_path())


@pytest.fixture()
def catalogue_graph(annotation, chimeric_path, counts_path, bam_path, bundled_checkpoint):
    """Built exactly as the checkpoint prescribes, which is what detect() does."""
    from gatfuse.io import load_known_fusions
    from gatfuse.resources import bundled_known_fusions_path

    params = bundled_checkpoint.graph_params
    catalogue = load_known_fusions(
        bundled_known_fusions_path(), use_recurrence=params.known_fusion_recurrence
    )
    return build_graph(
        load_expression(counts_path, library_type=params.library_type),
        load_chimeric_junctions(chimeric_path, min_split_reads=params.min_split_reads),
        load_discordant_pairs(bam_path),
        annotation,
        params=params,
        known_fusions=catalogue,
    )


def test_bundled_model_metadata(bundled_checkpoint):
    assert bundled_checkpoint.uses_known_fusions is True
    assert bundled_checkpoint.threshold == pytest.approx(0.1542, abs=1e-3)
    assert bundled_checkpoint.platt_params is not None
    assert bundled_checkpoint.edge_norm_stats is not None
    # Legacy checkpoints predate the recorded parameters and must fall back to
    # the behaviour they were trained with.
    params = bundled_checkpoint.graph_params
    assert params.min_split_reads == 2
    assert params.biotype_attributes == "ensembl"
    assert params.known_fusion_recurrence is False


def test_end_to_end_detection(catalogue_graph, bundled_checkpoint, annotation):
    result = detection.detect(
        catalogue_graph, bundled_checkpoint, annotation,
        threshold=0.0, postprocess=PostprocessParams(),
    )
    assert list(result.reported.columns) == detection.OUTPUT_COLUMNS
    assert result.reported["score"].is_monotonic_decreasing
    assert result.reported["score"].between(0, 1).all()

    reported = set(zip(result.reported["donor_gene"], result.reported["acceptor_gene"], strict=True))
    assert ("BCR", "ABL1") in reported
    # The intragenic and read-through junctions are filtered out.
    discarded = set(result.discarded["filter_reason"])
    assert {"intragenic", "readthrough"} <= discarded


def test_feature_mismatch_is_reported(sample_graph, bundled_checkpoint):
    """The bundled model needs the catalogue features; say so instead of crashing."""
    with pytest.raises(ModelError, match="--known-fusions"):
        checkpoint_io.check_compatible(bundled_checkpoint, sample_graph)


def test_breakpoints_are_shifted_to_the_last_transcribed_base():
    assert detection._adjust_breakpoint(1000, "+", is_donor=True) == 999
    assert detection._adjust_breakpoint(1000, "-", is_donor=True) == 1001
    assert detection._adjust_breakpoint(2000, "+", is_donor=False) == 2001
    assert detection._adjust_breakpoint(2000, "-", is_donor=False) == 1999


def test_minus_strand_partners_are_swapped_into_transcript_order():
    df = pd.DataFrame([
        {"donor_gene": "A", "acceptor_gene": "B", "chr_donor": "chr1", "chr_acceptor": "chr2",
         "brkpt_donor": 1, "brkpt_acceptor": 2, "strand_donor": "-", "strand_acceptor": "-",
         "donor_region": "exon", "acceptor_region": "intron"},
        {"donor_gene": "C", "acceptor_gene": "D", "chr_donor": "chr3", "chr_acceptor": "chr4",
         "brkpt_donor": 3, "brkpt_acceptor": 4, "strand_donor": "+", "strand_acceptor": "+",
         "donor_region": "exon", "acceptor_region": "exon"},
    ])
    out = detection._swap_to_transcript_orientation(df.copy())
    assert out.loc[0, "donor_gene"] == "B" and out.loc[0, "acceptor_gene"] == "A"
    assert out.loc[0, "donor_region"] == "intron"
    assert out.loc[1, "donor_gene"] == "C"  # plus-strand rows are untouched


def test_complementary_duplicates_are_merged_and_evidence_summed():
    df = pd.DataFrame([
        {"donor_gene": "A", "acceptor_gene": "B", "score": 0.9, "split_reads": 7,
         "discordant_pairs": 2, "chr_donor": "chr1", "brkpt_donor": 100,
         "chr_acceptor": "chr2", "brkpt_acceptor": 200},
        {"donor_gene": "A", "acceptor_gene": "B", "score": 0.4, "split_reads": 5,
         "discordant_pairs": 3, "chr_donor": "chr1", "brkpt_donor": 100,
         "chr_acceptor": "chr2", "brkpt_acceptor": 200},
    ])
    merged = detection._deduplicate_junctions(df)
    assert len(merged) == 1
    assert merged.iloc[0]["score"] == 0.9
    assert merged.iloc[0]["split_reads"] == 12
    assert merged.iloc[0]["discordant_pairs"] == 5


def test_selection_modes():
    df = pd.DataFrame({"score": [0.9, 0.5, 0.1]})
    assert len(detection.select_candidates(df, threshold=0.4)) == 2
    assert len(detection.select_candidates(df, top_k=1)) == 1


def test_a_sample_without_candidates_yields_an_empty_but_valid_table(
    annotation, counts_path, bam_path, bundled_checkpoint
):
    from gatfuse.io.chimeric import _empty_junction_frame

    graph = build_graph(
        load_expression(counts_path), _empty_junction_frame(),
        load_discordant_pairs(bam_path), annotation,
        params=GraphParams(use_known_fusions=False),
    )
    result = detection.detect(graph, bundled_checkpoint, annotation, threshold=0.5)
    assert result.reported.empty
    assert list(result.reported.columns) == detection.OUTPUT_COLUMNS


def test_written_table_keeps_its_header_when_empty(tmp_path):
    path = detection.write_table(detection.empty_result_frame(), tmp_path / "empty.tsv")
    assert path.read_text().strip().split("\t") == detection.OUTPUT_COLUMNS


def test_released_model_scores_are_stable(catalogue_graph, bundled_checkpoint, annotation):
    """Regression guard: pins the distributed model's output on the fixture sample.

    Any unintended change to parsing, feature computation or normalisation moves
    these scores, so this test fails loudly rather than silently altering results.
    """
    result = detection.detect(
        catalogue_graph, bundled_checkpoint, annotation,
        threshold=0.0, postprocess=PostprocessParams(),
    )
    scores = dict(
        zip(zip(result.reported["donor_gene"], result.reported["acceptor_gene"], strict=True),
            result.reported["score"], strict=True)
    )
    assert scores[("BCR", "ABL1")] == pytest.approx(0.3203, abs=5e-4)
    assert scores[("EWSR1", "FLI1")] == pytest.approx(0.0349, abs=5e-4)
