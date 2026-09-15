"""Construction of the per-sample gene/junction graph."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .config import GraphParams
from .errors import InputError
from .features import edge_feature_schema, num_edge_features
from .intervals import ExonIntervalIndex, GeneIntervalIndex
from .logging_utils import get_logger

log = get_logger("graph")


# --- Feature helpers ---


def _relative_position(gene_bounds, gene_id, position):
    """Where the breakpoint falls within the gene body, on a 0-1 scale."""
    bounds = gene_bounds.get(gene_id)
    if bounds is None:
        return 0.5
    start, end = bounds
    length = end - start + 1
    if length <= 0:
        return 0.5
    return max(0.0, min(1.0, (position - start) / length))


def _one_hot_junction(junction_type):
    """One-hot encode STAR's junction type; -1 (mate-spanning) encodes as zeros."""
    vector = np.zeros(4)
    value = int(junction_type)
    if 0 <= value < 4:
        vector[value] = 1.0
    return vector


def _encode_strand(donor, acceptor):
    """Three-class encoding of the donor/acceptor strand combination."""
    if donor == "+" and acceptor == "-":
        return [1.0, 0.0, 0.0]
    if donor == "+" and acceptor == "+":
        return [0.0, 1.0, 0.0]
    if donor == "-" and acceptor == "-":
        return [0.0, 0.0, 1.0]
    return [0.0, 0.0, 0.0]


def _compute_tpm(counts, genes):
    """Transcripts per million, using gene lengths from the annotation."""
    lengths = (
        genes[["gene_id", "gene_length"]]
        .drop_duplicates("gene_id")
        .set_index("gene_id")["gene_length"]
    )
    table = counts.copy()
    table["gene_length"] = table["gene_id"].map(lengths).fillna(1000)
    table["rpk"] = table["count"] / (table["gene_length"] / 1000.0)
    scale = table["rpk"].sum() / 1e6
    table["tpm"] = table["rpk"] / scale if scale > 0 else 0.0
    return table


class _DiscordantIndex:
    """Positions of discordant pairs, grouped by chromosome pair."""

    def __init__(self, pairs, window):
        self.window = window
        self._index = {}
        if pairs is None or pairs.empty or "pos_A" not in pairs.columns:
            return

        grouped = pd.DataFrame(
            {
                "key": list(zip(pairs["chr_A"].astype(str), pairs["chr_B"].astype(str), strict=True)),
                "pos_a": pairs["pos_A"].to_numpy(dtype=np.int64),
                "pos_b": pairs["pos_B"].to_numpy(dtype=np.int64),
            }
        )
        buckets = {}
        for key, group in grouped.groupby("key", sort=False):
            forward = np.stack([group["pos_a"].to_numpy(), group["pos_b"].to_numpy()], axis=1)
            reverse = forward[:, ::-1]
            buckets.setdefault(key, []).append(forward)
            buckets.setdefault((key[1], key[0]), []).append(reverse)
        self._index = {key: np.vstack(chunks) for key, chunks in buckets.items()}

    def count_near(self, chr_donor, pos_donor, chr_acceptor, pos_acceptor):
        positions = self._index.get((chr_donor, chr_acceptor))
        if positions is None or len(positions) == 0:
            return 0.0
        near = (np.abs(positions[:, 0] - pos_donor) <= self.window) & (
            np.abs(positions[:, 1] - pos_acceptor) <= self.window
        )
        return float(near.sum())


# --- Validation ------------------------------------------------------------


def _check_chromosome_naming(junctions, gene_index):
    """Fail early when the chimeric file and the GTF use different contig names."""
    if junctions.empty:
        return
    junction_contigs = set(junctions["chr_donor"].astype(str)) | set(
        junctions["chr_acceptor"].astype(str)
    )
    annotation_contigs = gene_index.contigs
    if junction_contigs & annotation_contigs:
        return
    raise InputError(
        "No contig name is shared between the chimeric junction file and the annotation.\n"
        f"  Junction file uses e.g.: {sorted(junction_contigs)[:5]}\n"
        f"  Annotation uses e.g.:    {sorted(annotation_contigs)[:5]}\n"
        "Supply the same GTF that was used to build the STAR index (UCSC 'chr1' and "
        "Ensembl '1' style names cannot be mixed)."
    )


def _check_annotation_match(counts, genes):
    """Warn when gene counts and annotation come from different releases."""
    annotated = set(genes["gene_id"].dropna())
    counted = set(counts["gene_id"].dropna())
    if not counted:
        return
    overlap = len(counted & annotated) / len(counted)
    if overlap == 0:
        raise InputError(
            "No gene ID in the STAR gene count file appears in the annotation. "
            "The GTF must be the one supplied to STAR when the sample was aligned."
        )
    if overlap < 0.5:
        log.warning(
            "Only %.0f%% of counted gene IDs are present in the annotation; gene lengths "
            "for the rest fall back to 1 kb. Are the count file and GTF from the same release?",
            overlap * 100,
        )


# --- Graph construction ----------------------------------------------------


def build_graph(counts, junctions, discordant_pairs, annotation,
                params=None, known_fusions=None):
    """Assemble the sample graph.

    Args:
        counts: table from ``load_expression``.
        junctions: table from ``load_chimeric_junctions``.
        discordant_pairs: table from ``load_discordant_pairs``.
        annotation: ``Annotation`` from ``load_annotation``.
        params: ``GraphParams``; defaults are used when omitted.
        known_fusions: optional ``KnownFusionCatalogue`` enabling the
            prior-knowledge node and edge features.

    Returns:
        A PyG ``Data`` object with ``x``, ``edge_index``, ``edge_attr``,
        ``edge_evidence`` (un-normalised split-read evidence), plus the
        ``gene_to_node_id``, ``gene_name_to_node_id`` and ``edge_breakpoints``
        bookkeeping attributes.
    """
    params = params or GraphParams()
    genes = annotation.genes
    use_known_fusions = known_fusions is not None
    schema_width = num_edge_features(use_known_fusions)

    _check_annotation_match(counts, genes)

    # --- Nodes: one per expressed gene ---
    expression = _compute_tpm(counts, genes)
    expression = expression[expression["tpm"] > 0].reset_index(drop=True)
    if expression.empty:
        raise InputError("No gene has a positive expression value; the graph would have no nodes.")

    unique_genes = expression["gene_id"].to_numpy(dtype=object)
    unique_genes = pd.unique(unique_genes)
    gene_to_node_id = {gene: node for node, gene in enumerate(unique_genes)}

    log_tpm = np.log1p(expression["tpm"].to_numpy(dtype=np.float64))
    log_length = np.log1p(expression["gene_length"].to_numpy(dtype=np.float64))
    log_tpm = (log_tpm - log_tpm.mean()) / (log_tpm.std() + 1e-6)
    log_length = (log_length - log_length.mean()) / (log_length.std() + 1e-6)

    biotypes = (
        genes[["gene_id", "gene_biotype"]]
        .drop_duplicates("gene_id")
        .set_index("gene_id")["gene_biotype"]
    )
    is_protein_coding = np.array(
        [1.0 if biotypes.get(gene, "") == "protein_coding" else 0.0 for gene in unique_genes],
        dtype=np.float32,
    )

    gene_names = (
        genes[["gene_id", "gene_name"]]
        .drop_duplicates("gene_id")
        .set_index("gene_id")["gene_name"]
    )

    node_columns = [log_tpm, log_length, is_protein_coding]
    if use_known_fusions:
        symbols = [gene_names.get(gene, "") or "" for gene in unique_genes]
        node_columns.append(
            np.array([1.0 if known_fusions.is_fusion_gene(s) else 0.0 for s in symbols],
                     dtype=np.float32)
        )
        node_columns.append(
            np.log1p(np.array([float(known_fusions.partner_count(s)) for s in symbols],
                              dtype=np.float32))
        )

    x = torch.tensor(np.stack(node_columns, axis=1), dtype=torch.float)
    log.info("Graph nodes: %d expressed genes (%d features)", x.shape[0], x.shape[1])

    # --- Lookups shared by both passes over the junctions ---
    gene_index = GeneIntervalIndex(genes)
    _check_chromosome_naming(junctions, gene_index)

    exon_index = ExonIntervalIndex(annotation.exons)
    gene_bounds = {
        gene_id: (start, end)
        for gene_id, start, end in genes.drop_duplicates("gene_id")[
            ["gene_id", "start", "end"]
        ].itertuples(index=False, name=None)
    }
    gene_chrom = dict(
        genes.drop_duplicates("gene_id")[["gene_id", "chr"]].itertuples(index=False, name=None)
    )
    gene_strand = dict(
        genes.drop_duplicates("gene_id")[["gene_id", "strand"]].itertuples(index=False, name=None)
    )
    tpm_by_gene = dict(
        expression[["gene_id", "tpm"]].itertuples(index=False, name=None)
    )

    discordant_index = _DiscordantIndex(discordant_pairs, params.discordant_window)

    # Fusion fragments per million, normalising evidence across sequencing depth.
    total_mapped = float(counts["count"].sum()) if "count" in counts.columns else 0.0
    ffpm_denominator = max(total_mapped / 1e6, 1e-6)

    def is_read_through(gene_a, gene_b):
        if gene_a == gene_b:
            return 0.0
        chrom_a, chrom_b = gene_chrom.get(gene_a), gene_chrom.get(gene_b)
        if chrom_a is None or chrom_a != chrom_b:
            return 0.0
        strand_a, strand_b = gene_strand.get(gene_a), gene_strand.get(gene_b)
        if not strand_a or strand_a != strand_b:
            return 0.0
        bounds_a, bounds_b = gene_bounds.get(gene_a), gene_bounds.get(gene_b)
        if bounds_a is None or bounds_b is None:
            return 0.0
        gap = max(bounds_a[0], bounds_b[0]) - min(bounds_a[1], bounds_b[1])
        return 1.0 if gap <= params.readthrough_max_distance else 0.0

    breakpoint_gene_cache = {}

    def genes_at(chrom, position):
        key = (chrom, int(position))
        cached = breakpoint_gene_cache.get(key)
        if cached is None:
            cached = gene_index.overlapping(chrom, position)
            breakpoint_gene_cache[key] = cached
        return cached

    chr_donor = junctions["chr_donor"].astype(str).to_numpy(dtype=object)
    chr_acceptor = junctions["chr_acceptor"].astype(str).to_numpy(dtype=object)
    brkpt_donor = junctions["brkpt_donor"].to_numpy(dtype=np.int64)
    brkpt_acceptor = junctions["brkpt_acceptor"].to_numpy(dtype=np.int64)
    strand_donor = junctions["strand_donor"].astype(str).to_numpy(dtype=object)
    strand_acceptor = junctions["strand_acceptor"].astype(str).to_numpy(dtype=object)
    split_reads_col = junctions["num_split_reads"].to_numpy(dtype=np.float64)
    junction_type_col = junctions["junction_type"].to_numpy(dtype=np.int64)
    repeat_left_col = junctions["repeat_left_len"].to_numpy(dtype=np.float64)
    repeat_right_col = junctions["repeat_right_len"].to_numpy(dtype=np.float64)
    anchor_col = (
        junctions["max_balanced_anchor"].to_numpy(dtype=np.float64)
        if "max_balanced_anchor" in junctions.columns
        else np.zeros(len(junctions))
    )

    # Pass 1: per-gene chimeric background, per-pair junction statistics and
    # per-gene partner promiscuity.
    partners = {}
    pair_stats = {}
    chimeric_background = {}

    for i in range(len(junctions)):
        donor_genes = genes_at(chr_donor[i], brkpt_donor[i])
        acceptor_genes = genes_at(chr_acceptor[i], brkpt_acceptor[i])
        split_reads = float(split_reads_col[i])

        for gene in donor_genes:
            node = gene_to_node_id.get(gene)
            if node is not None:
                chimeric_background[node] = chimeric_background.get(node, 0.0) + split_reads
        for gene in acceptor_genes:
            node = gene_to_node_id.get(gene)
            if node is not None:
                chimeric_background[node] = chimeric_background.get(node, 0.0) + split_reads

        if not donor_genes or not acceptor_genes:
            continue

        for donor in donor_genes:
            if donor in gene_to_node_id:
                bucket = partners.setdefault(donor, set())
                bucket.update(g for g in acceptor_genes if g != donor)
        for acceptor in acceptor_genes:
            if acceptor in gene_to_node_id:
                bucket = partners.setdefault(acceptor, set())
                bucket.update(g for g in donor_genes if g != acceptor)

        is_canonical = int(junction_type_col[i]) in (1, 2, 3)
        for donor in donor_genes:
            donor_node = gene_to_node_id.get(donor)
            if donor_node is None:
                continue
            for acceptor in acceptor_genes:
                acceptor_node = gene_to_node_id.get(acceptor)
                if acceptor_node is None:
                    continue
                stats = pair_stats.setdefault(
                    (donor_node, acceptor_node),
                    {"junctions": 0, "max_split": 0.0, "canonical": 0},
                )
                stats["junctions"] += 1
                stats["max_split"] = max(stats["max_split"], split_reads)
                stats["canonical"] += is_canonical

    promiscuity = {gene: len(found) for gene, found in partners.items()}

    # Pass 2: one edge per breakpoint per gene-pair assignment.
    edge_index_rows = []
    edge_attr_rows = []
    breakpoints = []

    for i in range(len(junctions)):
        donor_genes = genes_at(chr_donor[i], brkpt_donor[i])
        acceptor_genes = genes_at(chr_acceptor[i], brkpt_acceptor[i])
        if not donor_genes or not acceptor_genes:
            continue

        split_reads = float(split_reads_col[i])
        discordant = discordant_index.count_near(
            chr_donor[i], brkpt_donor[i], chr_acceptor[i], brkpt_acceptor[i]
        )
        interchromosomal = chr_donor[i] != chr_acceptor[i]
        distance = 1e7 if interchromosomal else abs(brkpt_donor[i] - brkpt_acceptor[i])
        ffpm = split_reads / ffpm_denominator
        junction_one_hot = _one_hot_junction(junction_type_col[i])
        strand_encoding = _encode_strand(strand_donor[i], strand_acceptor[i])

        for donor in donor_genes:
            donor_node = gene_to_node_id.get(donor)
            if donor_node is None:
                continue
            for acceptor in acceptor_genes:
                acceptor_node = gene_to_node_id.get(acceptor)
                if acceptor_node is None:
                    continue

                donor_exonic = 1.0 if exon_index.covers(donor, brkpt_donor[i]) else 0.0
                acceptor_exonic = 1.0 if exon_index.covers(acceptor, brkpt_acceptor[i]) else 0.0
                tpm_donor = tpm_by_gene.get(donor, 0.0)
                tpm_acceptor = tpm_by_gene.get(acceptor, 0.0)

                values = np.concatenate([
                    [
                        np.log1p(split_reads),
                        np.log1p(discordant),
                        split_reads / (discordant + 1.0),
                        np.log1p(distance),
                        np.log1p((tpm_donor + 1e-6) / (tpm_acceptor + 1e-6)),
                        1.0 if interchromosomal else 0.0,
                    ],
                    junction_one_hot,
                    strand_encoding,
                    [np.log1p(repeat_left_col[i]), np.log1p(repeat_right_col[i])],
                    [
                        donor_exonic,
                        acceptor_exonic,
                        _relative_position(gene_bounds, donor, brkpt_donor[i]),
                        _relative_position(gene_bounds, acceptor, brkpt_acceptor[i]),
                    ],
                    [
                        np.log1p(ffpm * 1000.0),
                        np.log1p(anchor_col[i]),
                        np.log1p(float(promiscuity.get(donor, 0))),
                        np.log1p(float(promiscuity.get(acceptor, 0))),
                        is_read_through(donor, acceptor),
                    ],
                ])

                if use_known_fusions:
                    recurrence = float(
                        known_fusions.pair_count(
                            gene_names.get(donor, "") or "", gene_names.get(acceptor, "") or ""
                        )
                    )
                    values = np.concatenate(
                        [values, [1.0 if recurrence > 0 else 0.0, np.log1p(recurrence)]]
                    )

                stats = pair_stats[(donor_node, acceptor_node)]
                canonical_fraction = (
                    stats["canonical"] / stats["junctions"] if stats["junctions"] else 0.0
                )
                values = np.concatenate([
                    values,
                    [
                        np.log1p(stats["junctions"]),
                        np.log1p(stats["max_split"]),
                        canonical_fraction,
                        np.log1p(chimeric_background.get(donor_node, 0.0)),
                        np.log1p(chimeric_background.get(acceptor_node, 0.0)),
                        canonical_fraction * donor_exonic * acceptor_exonic,
                    ],
                ])

                edge_index_rows.append([donor_node, acceptor_node])
                edge_attr_rows.append(values)
                breakpoints.append({
                    "chr_donor": chr_donor[i],
                    "brkpt_donor": int(brkpt_donor[i]),
                    "strand_donor": strand_donor[i],
                    "chr_acceptor": chr_acceptor[i],
                    "brkpt_acceptor": int(brkpt_acceptor[i]),
                    "strand_acceptor": strand_acceptor[i],
                    "split_reads": split_reads,
                    "discordant_pairs": discordant,
                    "pct_canonical": canonical_fraction,
                })

    if edge_index_rows:
        edge_index = torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.asarray(edge_attr_rows, dtype=np.float32))
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, schema_width), dtype=torch.float)

    if edge_attr.shape[1] != schema_width:
        raise RuntimeError(
            f"Built {edge_attr.shape[1]} edge features but the schema declares {schema_width}."
        )

    log.info("Graph edges: %d chimeric junctions (%d features)", edge_index.shape[1], schema_width)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    evidence_column = next(
        i for i, f in enumerate(edge_feature_schema(use_known_fusions))
        if f.name == "log_split_reads"
    )
    graph.edge_evidence = (
        edge_attr[:, evidence_column].clone()
        if edge_attr.shape[0]
        else torch.zeros((0,), dtype=torch.float32)
    )
    graph.gene_to_node_id = gene_to_node_id
    graph.gene_name_to_node_id = {
        name: node
        for gene, node in gene_to_node_id.items()
        if (name := gene_names.get(gene)) is not None and not pd.isna(name)
    }
    graph.edge_breakpoints = breakpoints
    return graph
