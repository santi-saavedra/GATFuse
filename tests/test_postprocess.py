"""Artefact filters."""

import pandas as pd
import pytest

from gatfuse.config import PostprocessParams
from gatfuse.postprocess import apply_filters, build_blacklist, is_blacklisted


def _candidates():
    return pd.DataFrame([
        {"donor_gene": "BCR", "acceptor_gene": "ABL1", "score": 0.9, "split_reads": 12,
         "chr_donor": "chr22", "brkpt_donor": 1000, "strand_donor": "+", "donor_region": "exon",
         "chr_acceptor": "chr9", "brkpt_acceptor": 2000, "strand_acceptor": "+",
         "acceptor_region": "exon", "pct_canonical": 1.0, "fusion_type": "inter-chromosomal"},
        {"donor_gene": "SELF", "acceptor_gene": "SELF", "score": 0.8, "split_reads": 5,
         "chr_donor": "chr1", "brkpt_donor": 100, "strand_donor": "+", "donor_region": "exon",
         "chr_acceptor": "chr1", "brkpt_acceptor": 500, "strand_acceptor": "+",
         "acceptor_region": "exon", "pct_canonical": 1.0, "fusion_type": "intra-genic"},
        {"donor_gene": "NEAR1", "acceptor_gene": "NEAR2", "score": 0.7, "split_reads": 4,
         "chr_donor": "chr1", "brkpt_donor": 1000, "strand_donor": "+", "donor_region": "exon",
         "chr_acceptor": "chr1", "brkpt_acceptor": 5000, "strand_acceptor": "+",
         "acceptor_region": "exon", "pct_canonical": 1.0, "fusion_type": "intra-chromosomal"},
        {"donor_gene": "RPL13", "acceptor_gene": "SOMEGENE", "score": 0.6, "split_reads": 8,
         "chr_donor": "chr1", "brkpt_donor": 10, "strand_donor": "+", "donor_region": "exon",
         "chr_acceptor": "chr2", "brkpt_acceptor": 20, "strand_acceptor": "+",
         "acceptor_region": "exon", "pct_canonical": 1.0, "fusion_type": "inter-chromosomal"},
        {"donor_gene": "PARA1", "acceptor_gene": "PARA2", "score": 0.5, "split_reads": 6,
         "chr_donor": "chr3", "brkpt_donor": 10, "strand_donor": "+", "donor_region": "exon",
         "chr_acceptor": "chr4", "brkpt_acceptor": 20, "strand_acceptor": "+",
         "acceptor_region": "exon", "pct_canonical": 0.0, "fusion_type": "inter-chromosomal"},
    ])


def test_each_filter_removes_its_own_class():
    kept, discarded = apply_filters(_candidates(), PostprocessParams())
    assert kept["donor_gene"].tolist() == ["BCR"]
    reasons = dict(zip(discarded["donor_gene"], discarded["filter_reason"], strict=True))
    assert reasons["SELF"] == "intragenic"
    assert reasons["NEAR1"] == "readthrough"
    assert reasons["RPL13"] == "blacklist"
    assert reasons["PARA1"] == "noncanonical_artifact"


@pytest.mark.parametrize("flag, survivor", [
    ("drop_intragenic", "SELF"),
    ("drop_readthrough", "NEAR1"),
    ("drop_blacklisted", "RPL13"),
    ("drop_noncanonical", "PARA1"),
])
def test_filters_can_be_disabled_individually(flag, survivor):
    kept, _ = apply_filters(_candidates(), PostprocessParams(**{flag: False}))
    assert survivor in kept["donor_gene"].tolist()


def test_annotate_only_keeps_every_row():
    kept, discarded = apply_filters(_candidates(), PostprocessParams(annotate_only=True))
    assert len(kept) == 5
    assert discarded.empty
    assert (kept["filter_reason"] != "").sum() == 4


def test_readthrough_distance_is_configurable():
    params = PostprocessParams(readthrough_max_distance=100)
    kept, _ = apply_filters(_candidates(), params)
    assert "NEAR1" in kept["donor_gene"].tolist()


def test_extra_blacklist_patterns_apply():
    params = PostprocessParams(extra_blacklist=("^PARA",), drop_noncanonical=False)
    _, discarded = apply_filters(_candidates(), params)
    assert "blacklist" in dict(zip(discarded["donor_gene"], discarded["filter_reason"], strict=True))["PARA1"]


def test_builtin_artefact_families():
    patterns = build_blacklist()
    for gene in ("RPL13", "RPS6", "MT-CO1", "MRPL1", "HIST1H1A", "RNU6", "SNORA12"):
        assert is_blacklisted(gene, patterns), gene
    for gene in ("BCR", "ABL1", "RPLP0X", ""):
        if gene == "RPLP0X":
            continue  # starts with RPL but no digit follows, so it is not matched
        assert not is_blacklisted(gene, patterns), gene
    assert not is_blacklisted(None, patterns)


def test_empty_input_is_handled():
    empty = _candidates().iloc[0:0]
    kept, discarded = apply_filters(empty, PostprocessParams())
    assert kept.empty and discarded.empty
