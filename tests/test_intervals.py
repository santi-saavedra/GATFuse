"""Genomic interval lookups underpin every breakpoint-to-gene assignment."""

import pandas as pd

from gatfuse.intervals import ExonIntervalIndex, GeneIntervalIndex, annotate_region


def _genes():
    return pd.DataFrame({
        "chr": ["chr1", "chr1", "chr2"],
        "start": [1000, 1500, 100],
        "end": [2000, 1800, 500],
        "gene_id": ["CODING", "LNC", "OTHER"],
        "gene_biotype": ["protein_coding", "lncRNA", "protein_coding"],
    })


def test_overlap_lookup():
    index = GeneIntervalIndex(_genes())
    assert index.overlapping("chr1", 1200) == ["CODING"]
    assert index.overlapping("chr1", 2500) == []
    assert index.overlapping("chrX", 1200) == []
    assert index.contigs == {"chr1", "chr2"}


def test_protein_coding_genes_shadow_overlapping_noncoding_genes():
    index = GeneIntervalIndex(_genes())
    assert index.overlapping("chr1", 1600, prefer_coding=True) == ["CODING"]
    assert sorted(index.overlapping("chr1", 1600, prefer_coding=False)) == ["CODING", "LNC"]


def test_annotation_without_biotype_applies_no_preference():
    genes = _genes()
    genes["gene_biotype"] = None
    index = GeneIntervalIndex(genes)
    assert sorted(index.overlapping("chr1", 1600)) == ["CODING", "LNC"]


def test_exon_lookup_and_region_annotation():
    genes = _genes()
    exons = pd.DataFrame({
        "chr": ["chr1", "chr1"],
        "start": [1000, 1900],
        "end": [1100, 2000],
        "gene_id": ["CODING", "CODING"],
    })
    gene_index = GeneIntervalIndex(genes)
    exon_index = ExonIntervalIndex(exons)

    assert exon_index.covers("CODING", 1050)
    assert not exon_index.covers("CODING", 1500)
    assert not exon_index.covers("MISSING", 1050)

    assert annotate_region(gene_index, exon_index, "chr1", 1050) == "exon"
    assert annotate_region(gene_index, exon_index, "chr1", 1300) == "intron"
    assert annotate_region(gene_index, exon_index, "chr1", 5000) == "intergenic"


def test_empty_exon_table_is_supported():
    index = ExonIntervalIndex(pd.DataFrame(columns=["chr", "start", "end", "gene_id"]))
    assert not index.covers("CODING", 10)
    assert len(index.genes_with_exon_at("chr1", 10)) == 0
