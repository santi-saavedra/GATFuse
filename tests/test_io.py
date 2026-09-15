"""Parser behaviour, including the failure modes users actually hit."""

import pytest

from gatfuse.errors import InputError
from gatfuse.io import (
    load_annotation,
    load_chimeric_junctions,
    load_expression,
    load_known_fusions,
    load_manifest,
)


def test_annotation_tables(annotation):
    assert len(annotation.genes) == 5
    assert set(annotation.genes["gene_name"]) == {"BCR", "EWSR1", "ABL1", "FLI1", "TESTLNC"}
    assert annotation.genes["gene_length"].min() > 0
    assert len(annotation.exons) == 15
    assert annotation.has_biotype


def test_annotation_reads_gencode_gene_type(tmp_path):
    """GENCODE names the attribute gene_type; only 'auto' should read it."""
    gtf = tmp_path / "gencode.gtf"
    gtf.write_text(
        'chr1\ttest\tgene\t1\t100\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding"; gene_name "A";\n'
    )
    assert load_annotation(gtf, biotype_attributes="auto").has_biotype
    assert not load_annotation(gtf, biotype_attributes="ensembl").has_biotype


def test_annotation_rejects_file_without_genes(tmp_path):
    gtf = tmp_path / "no_genes.gtf"
    gtf.write_text('chr1\ttest\texon\t1\t100\t.\t+\t.\tgene_id "G1";\n')
    with pytest.raises(InputError, match="No 'gene' records"):
        load_annotation(gtf)


def test_chimeric_aggregates_reads_per_breakpoint(chimeric_path):
    junctions = load_chimeric_junctions(chimeric_path, min_split_reads=2)
    assert len(junctions) == 4
    strongest = junctions.iloc[0]
    assert strongest["num_split_reads"] == 12
    assert strongest["chr_donor"] == "chr1"
    assert strongest["brkpt_donor"] == 3000
    # min(50M, 50M) over the supporting reads
    assert strongest["max_balanced_anchor"] == 50


def test_chimeric_min_split_reads_filters(chimeric_path):
    assert len(load_chimeric_junctions(chimeric_path, min_split_reads=6)) == 1
    assert len(load_chimeric_junctions(chimeric_path, min_split_reads=100)) == 0


def test_chimeric_skips_star_header_line(tmp_path, chimeric_path):
    """STAR >= 2.7.1 writes a non-comment column header; it must not become a row."""
    annotated = tmp_path / "with_header.junction"
    annotated.write_text(
        "chr_donorA\tbrkpt_donorA\tstrand_donorA\tchr_acceptorB\tbrkpt_acceptorB\t"
        "strand_acceptorB\tjunction_type\trepeat_left_lenA\trepeat_right_lenB\tread_name\t"
        "start_alnA\tcigarA\tstart_alnB\tcigarB\n" + chimeric_path.read_text()
    )
    assert len(load_chimeric_junctions(annotated, min_split_reads=2)) == 4


def test_expression_drops_star_summary_rows(counts_path):
    counts = load_expression(counts_path)
    assert len(counts) == 5
    assert not counts["gene_id"].str.startswith("N_").any()


def test_expression_rejects_unknown_library_type(counts_path):
    with pytest.raises(InputError, match="Unknown library type"):
        load_expression(counts_path, library_type="stranded")


def test_expression_library_type_selects_column(counts_path):
    unstranded = load_expression(counts_path, "unstranded")["count"].sum()
    forward = load_expression(counts_path, "stranded_forward")["count"].sum()
    assert unstranded == 2 * forward


def test_known_fusions_recurrence_toggle(tmp_path):
    catalogue = tmp_path / "known.tsv"
    catalogue.write_text("gene_a\tgene_b\tn_cases\nBCR\tABL1\t333\nEWSR1\tFLI1\t10\n")

    weighted = load_known_fusions(catalogue)
    assert weighted.pair_count("BCR", "ABL1") == 333
    # Order must not matter.
    assert weighted.pair_count("ABL1", "BCR") == 333

    unweighted = load_known_fusions(catalogue, use_recurrence=False)
    assert unweighted.pair_count("BCR", "ABL1") == 1
    assert unweighted.partner_count("BCR") == 1
    assert not unweighted.is_fusion_gene("NOTAGENE")


def test_manifest_accepts_legacy_column_names(tmp_path, bam_path, chimeric_path, counts_path):
    manifest = tmp_path / "legacy.tsv"
    manifest.write_text(
        "bam_file\tchimeric_file\treads_per_gene_file\tpositive_fusions\n"
        f"{bam_path}\t{chimeric_path}\t{counts_path}\tBCR:ABL1;EWSR1:FLI1\n"
    )
    rows = load_manifest(manifest, default_annotation=str(chimeric_path))
    assert len(rows) == 1
    assert rows[0].fusions == [("BCR", "ABL1"), ("EWSR1", "FLI1")]


def test_manifest_resolves_paths_relative_to_itself(tmp_path, bam_path, chimeric_path, counts_path):
    """A manifest and its data must move between machines together."""
    import shutil

    for source in (bam_path, chimeric_path, counts_path):
        shutil.copy(source, tmp_path / source.name)
    manifest = tmp_path / "relative.tsv"
    manifest.write_text(
        "bam\tchimeric_junctions\tgene_counts\tannotation\tfusions\n"
        f"{bam_path.name}\t{chimeric_path.name}\t{counts_path.name}\t{chimeric_path.name}\tBCR:ABL1\n"
    )
    rows = load_manifest(manifest)
    assert rows[0].bam == tmp_path / bam_path.name


def test_manifest_parses_breakpoint_labels(tmp_path, bam_path, chimeric_path, counts_path):
    manifest = tmp_path / "breakpoints.tsv"
    manifest.write_text(
        "bam\tchimeric_junctions\tgene_counts\tannotation\tfusions\n"
        f"{bam_path}\t{chimeric_path}\t{counts_path}\t{chimeric_path}\t"
        "BCR:ABL1@chr1:3000-chr2:2000\n"
    )
    assert load_manifest(manifest)[0].fusions == [("BCR", "ABL1", "chr1", 3000, "chr2", 2000)]


def test_manifest_reports_missing_columns(tmp_path):
    manifest = tmp_path / "bad.tsv"
    manifest.write_text("bam\tfusions\nx\tBCR:ABL1\n")
    with pytest.raises(InputError, match="missing required column"):
        load_manifest(manifest, default_annotation="ref.gtf")


def test_manifest_reports_missing_file(tmp_path, chimeric_path, counts_path):
    manifest = tmp_path / "missing.tsv"
    manifest.write_text(
        "bam\tchimeric_junctions\tgene_counts\tannotation\tfusions\n"
        f"/nowhere/sample.bam\t{chimeric_path}\t{counts_path}\t{chimeric_path}\tBCR:ABL1\n"
    )
    with pytest.raises(InputError, match="bam not found"):
        load_manifest(manifest)
