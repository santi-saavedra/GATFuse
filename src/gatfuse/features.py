"""Explicit declaration of the node and edge feature layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    # Continuous features are z-scored; binary and one-hot blocks are not.
    normalize: bool = False


# --- Node features ---------------------------------------------------------

NODE_FEATURES_BASE = (
    Feature("log_tpm", "log1p(TPM), z-scored within the sample", normalize=True),
    Feature("log_gene_length", "log1p(gene length in bp), z-scored within the sample", normalize=True),
    Feature("is_protein_coding", "1 if the gene biotype is protein_coding"),
)

NODE_FEATURES_KNOWN_FUSIONS = (
    Feature("is_known_fusion_gene", "1 if the gene appears in the known-fusion catalogue"),
    Feature("log_partner_count", "log1p(number of distinct catalogued fusion partners)"),
)


# --- Edge features ---------------------------------------------------------

# Tier 1-3 evidence computed per chimeric breakpoint.
EDGE_FEATURES_JUNCTION = (
    Feature("log_split_reads", "log1p(split reads supporting the junction)", normalize=True),
    Feature("log_discordant_pairs", "log1p(discordant pairs near both breakpoints)",
            normalize=True),
    Feature("split_to_discordant_ratio", "split_reads / (discordant_pairs + 1)", normalize=True),
    Feature("log_breakpoint_distance",
            "log1p(distance between breakpoints; 1e7 if interchromosomal)", normalize=True),
    Feature("log_expression_ratio", "log1p(TPM_donor / TPM_acceptor)", normalize=True),
    Feature("is_interchromosomal", "1 if donor and acceptor are on different chromosomes"),
    Feature("junction_type_0", "STAR junction type 0 (non-canonical motif)"),
    Feature("junction_type_1", "STAR junction type 1 (GT/AG)"),
    Feature("junction_type_2", "STAR junction type 2 (CT/AC)"),
    Feature("junction_type_3", "STAR junction type 3 (reserved; always 0 for current STAR)"),
    Feature("strand_plus_minus", "donor + / acceptor - orientation"),
    Feature("strand_plus_plus", "donor + / acceptor + orientation"),
    Feature("strand_minus_minus", "donor - / acceptor - orientation"),
    Feature("log_repeat_left", "log1p(repeat length left of the breakpoint)",
            normalize=True),
    Feature("log_repeat_right", "log1p(repeat length right of the breakpoint)",
            normalize=True),
)

EDGE_FEATURES_BREAKPOINT_CONTEXT = (
    Feature("donor_in_exon", "1 if the donor breakpoint falls inside an annotated exon"),
    Feature("acceptor_in_exon",
            "1 if the acceptor breakpoint falls inside an annotated exon"),
    Feature("donor_relative_position", "donor breakpoint position within the gene body (0-1)",
            normalize=True),
    Feature("acceptor_relative_position", "acceptor breakpoint position within the gene body (0-1)",
            normalize=True),
)

EDGE_FEATURES_BACKGROUND = (
    Feature("log_ffpm", "log1p(1000 x fusion fragments per million mapped reads)",
            normalize=True),
    Feature("log_max_balanced_anchor", "log1p(longest balanced anchor over supporting reads)",
            normalize=True),
    Feature("log_donor_promiscuity", "log1p(distinct chimeric partners of the donor gene)",
            normalize=True),
    Feature("log_acceptor_promiscuity", "log1p(distinct chimeric partners of the acceptor gene)",
            normalize=True),
    Feature("is_read_through",
            "1 if the genes are adjacent, same-strand and within the read-through distance"),
)

EDGE_FEATURES_KNOWN_FUSIONS = (
    Feature("known_pair", "1 if the gene pair appears in the known-fusion catalogue"),
    Feature("log_known_pair_recurrence", "log1p(catalogued case count for the pair)", normalize=True),
)

EDGE_FEATURES_PAIR_AGGREGATE = (
    Feature("log_pair_junction_count", "log1p(distinct junctions between this gene pair)",
            normalize=True),
    Feature("log_pair_max_split_reads", "log1p(split reads of the strongest junction of the pair)",
            normalize=True),
    Feature("pair_canonical_fraction", "fraction of the pair's junctions with a canonical motif",
            normalize=True),
    Feature("log_donor_chimeric_background", "log1p(total chimeric reads involving the donor gene)",
            normalize=True),
    Feature("log_acceptor_chimeric_background",
            "log1p(total chimeric reads involving the acceptor gene)", normalize=True),
    Feature(
        "canonical_fraction_x_both_exonic",
        "pair_canonical_fraction x donor_in_exon x acceptor_in_exon; the canonical "
        "motif is only informative at exon-exon junctions",
    ),
)


def node_feature_schema(use_known_fusions):
    """Return the ordered node feature list."""
    schema = list(NODE_FEATURES_BASE)
    if use_known_fusions:
        schema += list(NODE_FEATURES_KNOWN_FUSIONS)
    return tuple(schema)


def edge_feature_schema(use_known_fusions):
    """Return the ordered edge feature list."""
    schema = list(EDGE_FEATURES_JUNCTION)
    schema += list(EDGE_FEATURES_BREAKPOINT_CONTEXT)
    schema += list(EDGE_FEATURES_BACKGROUND)
    if use_known_fusions:
        schema += list(EDGE_FEATURES_KNOWN_FUSIONS)
    schema += list(EDGE_FEATURES_PAIR_AGGREGATE)
    return tuple(schema)


def num_node_features(use_known_fusions):
    return len(node_feature_schema(use_known_fusions))


def num_edge_features(use_known_fusions):
    return len(edge_feature_schema(use_known_fusions))


def edge_feature_names(use_known_fusions):
    return [f.name for f in edge_feature_schema(use_known_fusions)]


def continuous_edge_columns(use_known_fusions):
    """Indices of the edge features that are z-scored."""
    return [i for i, f in enumerate(edge_feature_schema(use_known_fusions)) if f.normalize]


def continuous_columns_for_width(num_features):
    """Infer the continuous columns from a feature-matrix width."""
    for use_known in (False, True):
        if num_edge_features(use_known) == num_features:
            return continuous_edge_columns(use_known)
    raise ValueError(
        f"{num_features} edge features do not match any known layout "
        f"({num_edge_features(False)} without or {num_edge_features(True)} with known-fusion features)."
    )


def uses_known_fusions_from_width(num_node, num_edge):
    """Infer whether a checkpoint was trained with the known-fusion features."""
    for use_known in (False, True):
        if num_node_features(use_known) == num_node and num_edge_features(use_known) == num_edge:
            return use_known
    return None
