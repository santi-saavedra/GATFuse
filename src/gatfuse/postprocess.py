"""Rule-based filtering of scored candidates."""

import re

import pandas as pd

from .config import PostprocessParams
from .logging_utils import get_logger

log = get_logger("postprocess")

# Artefact-prone gene families.
_ARTEFACT_FAMILIES = (
    r"^RPL\d",       # cytoplasmic ribosomal, large subunit
    r"^RPS\d",       # cytoplasmic ribosomal, small subunit
    r"^MRPL\d",      # mitochondrial ribosomal, large subunit
    r"^MRPS\d",      # mitochondrial ribosomal, small subunit
    r"^MT-",         # mitochondrially encoded
    r"^HIST\d",      # histones
    r"^RNU\d",       # small nuclear RNA
    r"^SNOR[AD]\d",  # small nucleolar RNA
    r"^RN7S",        # 7SK / 7SL RNA
    r"^RNA5",        # 5S / 5.8S ribosomal RNA
    r"^TRNA",        # transfer RNA
)


def build_blacklist(extra_patterns=()):
    """Compile the artefact-family patterns, plus any user-supplied ones."""
    return [re.compile(pattern) for pattern in (*_ARTEFACT_FAMILIES, *extra_patterns)]


def is_blacklisted(gene_name, patterns):
    if not isinstance(gene_name, str) or not gene_name:
        return False
    return any(pattern.match(gene_name) for pattern in patterns)


def _is_read_through(row, max_distance):
    """Adjacent, co-oriented genes on one chromosome: transcription, not fusion."""
    if row["chr_donor"] != row["chr_acceptor"]:
        return False
    if row["donor_gene"] == row["acceptor_gene"]:
        return False
    if abs(int(row["brkpt_donor"]) - int(row["brkpt_acceptor"])) > max_distance:
        return False
    return bool(row.get("strand_donor") == row.get("strand_acceptor"))


def _is_noncanonical_artefact(row, min_canonical_fraction):
    """Inter-chromosomal exon-exon junction with no canonical splice signal."""
    if row.get("pct_canonical", 1.0) >= min_canonical_fraction:
        return False
    if row.get("fusion_type") != "inter-chromosomal":
        return False
    return row.get("donor_region") == "exon" and row.get("acceptor_region") == "exon"


def apply_filters(df, params=None):
    """Split scored candidates into kept and discarded rows.

    Returns:
        (kept, discarded) — the discarded frame carries a ``filter_reason``
        column naming every rule that matched.
    """
    params = params or PostprocessParams()
    if df is None or df.empty:
        empty = df.iloc[0:0].copy() if df is not None else pd.DataFrame()
        return df, empty

    patterns = build_blacklist(params.extra_blacklist)
    out = df.copy()

    def reasons_for(row):
        reasons = []
        if params.drop_intragenic and row["donor_gene"] == row["acceptor_gene"]:
            reasons.append("intragenic")
        if params.drop_blacklisted and (
            is_blacklisted(row["donor_gene"], patterns)
            or is_blacklisted(row["acceptor_gene"], patterns)
        ):
            reasons.append("blacklist")
        if params.drop_readthrough and _is_read_through(row, params.readthrough_max_distance):
            reasons.append("readthrough")
        if params.drop_noncanonical and _is_noncanonical_artefact(
            row, params.noncanonical_min_canonical_fraction
        ):
            reasons.append("noncanonical_artifact")
        return ";".join(reasons)

    out["filter_reason"] = out.apply(reasons_for, axis=1)

    if params.annotate_only:
        log.info("Post-processing: annotate-only, %d row(s) flagged",
                 int((out["filter_reason"] != "").sum()))
        return out, out.iloc[0:0].copy()

    keep = out["filter_reason"] == ""
    kept = out[keep].drop(columns=["filter_reason"]).reset_index(drop=True)
    discarded = out[~keep].reset_index(drop=True)

    if len(discarded):
        breakdown = ", ".join(
            f"{reason}={count}"
            for reason, count in discarded["filter_reason"].value_counts().items()
        )
        log.info("Post-processing: removed %d candidate(s) (%s)", len(discarded), breakdown)
    else:
        log.info("Post-processing: no candidate removed")
    return kept, discarded
