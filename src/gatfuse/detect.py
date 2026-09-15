"""Inference: score the chimeric junctions of one sample and report fusions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import checkpoint as checkpoint_io
from .intervals import ExonIntervalIndex, GeneIntervalIndex, annotate_region
from .io.paths import prepare_output
from .logging_utils import get_logger
from .model import calibrate
from .normalization import apply_edge_norm
from .postprocess import apply_filters

log = get_logger("detect")

OUTPUT_COLUMNS = [
    "donor_gene", "acceptor_gene", "score", "split_reads", "discordant_pairs",
    "chr_donor", "brkpt_donor", "strand_donor", "donor_region",
    "chr_acceptor", "brkpt_acceptor", "strand_acceptor", "acceptor_region",
    "pct_canonical", "fusion_type",
]


@dataclass
class DetectionResult:
    reported: pd.DataFrame
    discarded: pd.DataFrame
    candidates_evaluated: int
    threshold: float = None
    top_k: int = None


def empty_result_frame():
    """An empty table with the full output schema, so downstream parsers work."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in OUTPUT_COLUMNS})


def _adjust_breakpoint(position, strand, is_donor):
    """Convert STAR's coordinate to the last transcribed base."""
    if is_donor:
        return position - 1 if strand == "+" else position + 1
    return position + 1 if strand == "+" else position - 1


def _swap_to_transcript_orientation(df):
    """Order partners 5' to 3' in transcript terms."""
    minus = df["strand_donor"] == "-"
    if not minus.any():
        return df
    pairs = [
        ("donor_gene", "acceptor_gene"),
        ("chr_donor", "chr_acceptor"),
        ("brkpt_donor", "brkpt_acceptor"),
        ("strand_donor", "strand_acceptor"),
        ("donor_region", "acceptor_region"),
    ]
    for left, right in pairs:
        if left in df.columns and right in df.columns:
            buffer = df.loc[minus, left].to_numpy().copy()
            df.loc[minus, left] = df.loc[minus, right].to_numpy()
            df.loc[minus, right] = buffer
    return df


def _deduplicate_junctions(df):
    """Merge the two strand-complementary rows STAR emits per physical junction."""
    if len(df) <= 1:
        return df
    key = ["chr_donor", "brkpt_donor", "chr_acceptor", "brkpt_acceptor"]
    df = df.sort_values("score", ascending=False)
    totals = df.groupby(key, sort=False)[["split_reads", "discordant_pairs"]].sum()
    totals.columns = ["split_reads_total", "discordant_pairs_total"]
    merged = df.drop_duplicates(subset=key, keep="first").join(totals, on=key)
    merged["split_reads"] = merged["split_reads_total"].astype(int)
    merged["discordant_pairs"] = merged["discordant_pairs_total"].astype(int)
    return merged.drop(columns=["split_reads_total", "discordant_pairs_total"])


def score_graph(graph, checkpoint, device=None):
    """Calibrated fusion probability for every edge of *graph*."""
    checkpoint_io.check_compatible(checkpoint, graph)

    stats = checkpoint.edge_norm_stats
    if stats is not None and graph.edge_attr.shape[0] > 0:
        apply_edge_norm([graph], stats)
        log.info("Applied training-time feature normalisation (%d columns)",
                 len(stats["continuous_cols"]))
    else:
        log.warning(
            "Checkpoint carries no normalisation statistics; features are scored raw. "
            "Scores from this model are not comparable to those of a normalised one."
        )

    model = checkpoint.model
    if device is not None:
        model = model.to(device)
        graph = graph.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(
            graph.x, graph.edge_index, graph.edge_attr,
            edge_evidence=getattr(graph, "edge_evidence", None),
        )
    platt = checkpoint.platt_params
    if platt:
        logits = calibrate(logits, platt)
        log.info("Applied probability calibration (A=%.4f, B=%.4f)", platt["A"], platt["B"])
    else:
        log.warning("Checkpoint has no calibration parameters; scores are ranks, not probabilities.")
    return torch.sigmoid(logits).reshape(-1).cpu().numpy()


def build_candidate_table(graph, scores, annotation):
    """Assemble the scored, annotated and deduplicated candidate table."""
    gene_index = GeneIntervalIndex(annotation.genes)
    exon_index = ExonIntervalIndex(annotation.exons)

    names = (
        annotation.genes.drop_duplicates("gene_id").set_index("gene_id")["gene_name"].to_dict()
    )
    node_to_gene = {node: gene for gene, node in graph.gene_to_node_id.items()}

    def label(node):
        gene = node_to_gene.get(int(node))
        symbol = names.get(gene)
        if symbol and not pd.isna(symbol):
            return symbol
        return gene if gene is not None else str(int(node))

    breakpoints = graph.edge_breakpoints
    donor_positions = [
        _adjust_breakpoint(bp["brkpt_donor"], bp["strand_donor"], is_donor=True)
        for bp in breakpoints
    ]
    acceptor_positions = [
        _adjust_breakpoint(bp["brkpt_acceptor"], bp["strand_acceptor"], is_donor=False)
        for bp in breakpoints
    ]
    donor_chroms = [bp["chr_donor"] for bp in breakpoints]
    acceptor_chroms = [bp["chr_acceptor"] for bp in breakpoints]

    log.info("Annotating %d breakpoint pairs", len(breakpoints))
    region_cache = {}

    def region(chrom, position):
        key = (chrom, int(position))
        if key not in region_cache:
            region_cache[key] = annotate_region(gene_index, exon_index, chrom, position)
        return region_cache[key]

    edges = graph.edge_index.cpu().numpy()
    df = pd.DataFrame({
        "donor_gene": [label(node) for node in edges[0]],
        "acceptor_gene": [label(node) for node in edges[1]],
        "score": np.round(scores, 4),
        "split_reads": [int(bp["split_reads"]) for bp in breakpoints],
        "discordant_pairs": [int(bp.get("discordant_pairs", 0)) for bp in breakpoints],
        "chr_donor": donor_chroms,
        "brkpt_donor": donor_positions,
        "strand_donor": [bp["strand_donor"] for bp in breakpoints],
        "donor_region": [region(c, p) for c, p in zip(donor_chroms, donor_positions, strict=True)],
        "chr_acceptor": acceptor_chroms,
        "brkpt_acceptor": acceptor_positions,
        "strand_acceptor": [bp["strand_acceptor"] for bp in breakpoints],
        "acceptor_region": [region(c, p) for c, p in zip(acceptor_chroms, acceptor_positions, strict=True)],
        "pct_canonical": [round(bp.get("pct_canonical", 0.0), 3) for bp in breakpoints],
    })

    same_gene = df["donor_gene"].to_numpy() == df["acceptor_gene"].to_numpy()
    same_chrom = df["chr_donor"].to_numpy() == df["chr_acceptor"].to_numpy()
    df["fusion_type"] = np.where(
        same_gene, "intra-genic", np.where(same_chrom, "intra-chromosomal", "inter-chromosomal")
    )

    df = _swap_to_transcript_orientation(df)
    df = _deduplicate_junctions(df)
    return df.sort_values("score", ascending=False).reset_index(drop=True)[OUTPUT_COLUMNS]


def select_candidates(df, threshold=None, top_k=None):
    """Apply the reporting rule: top-k when given, otherwise a score threshold."""
    if top_k is not None:
        selected = df.head(top_k).copy()
        lowest = selected["score"].min() if not selected.empty else float("nan")
        log.info("Reporting the top %d candidate(s) (lowest score %.4f)", top_k, lowest)
        return selected
    selected = df[df["score"] >= threshold].copy()
    log.info("Reporting %d candidate(s) scoring >= %.4f", len(selected), threshold)
    return selected


def write_table(df, path, description="results"):
    """Write a TSV, always including the header so empty results stay parseable."""
    destination = prepare_output(path, description)
    df.to_csv(destination, sep="\t", index=False)
    log.info("Wrote %d row(s) to %s", len(df), destination)
    return destination


def detect(graph, checkpoint, annotation, threshold=None, top_k=None,
           postprocess=None, device=None):
    """Score, annotate, filter and select the fusions of one sample."""
    if graph.edge_index.shape[1] == 0:
        log.warning(
            "No chimeric junction could be mapped to a gene pair; no candidate to score."
        )
        return DetectionResult(empty_result_frame(), empty_result_frame(), 0, threshold, top_k)

    scores = score_graph(graph, checkpoint, device=device)
    candidates = build_candidate_table(graph, scores, annotation)

    discarded = empty_result_frame()
    if postprocess is not None:
        candidates, discarded = apply_filters(candidates, postprocess)

    log.info("Candidate junctions evaluated: %d", len(candidates))
    reported = select_candidates(candidates, threshold=threshold, top_k=top_k)
    return DetectionResult(reported, discarded, len(candidates), threshold, top_k)


__all__ = [
    "DetectionResult",
    "OUTPUT_COLUMNS",
    "detect",
    "build_candidate_table",
    "score_graph",
    "select_candidates",
    "write_table",
    "empty_result_frame",
]
